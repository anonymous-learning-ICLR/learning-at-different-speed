"""Epoch-based TC training with task/expert feedback and optional input acceleration."""
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from paper_pipeline import (FORMAT, seed_all, make_model, make_dataset, loader,
    split_manifest, unpack_batch, absolute_outputs, objective, collect_statistics,
    load_checkpoint, jsonable, source_hashes)
from taer_feedback import FeedbackController, VARIANTS
from checkpoint_compat import compatible_sources
import builtins
import copy
import logging
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from unittest.mock import patch
import paper_pipeline
from TCNM.data import paper_data


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--split-policy', choices=['existing', 'paper_years'], default='existing',
                   help='Use the existing train/val/test folders; paper_years additionally enforces temporal bounds')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--variant', choices=VARIANTS, default='full')
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--areas', nargs='+', default=['EP', 'NA', 'NI', 'SI', 'SP', 'WP'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--obs-len', type=int, default=8)
    p.add_argument('--pred-len', type=int, default=4)
    p.add_argument('--experts', type=int, default=6)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--grad-clip', type=float, default=2.0)
    p.add_argument('--window', type=int, default=3)
    p.add_argument('--ema', type=float, default=0.8)
    p.add_argument('--tau-u', type=float, default=0.05)
    p.add_argument('--alpha-min', type=float, default=0.3)
    p.add_argument('--alpha-max', type=float, default=1.7)
    p.add_argument('--gamma', type=float, default=0.1)
    p.add_argument('--tau-r', type=float, default=1.0)
    p.add_argument('--beta-load', type=float, default=0.0)
    p.add_argument('--beta-reg', type=float, default=0.0)
    p.add_argument('--feature-mode', choices=['legacy', 'strict'], default='legacy',
                   help='legacy exactly matches TC3 timestamp.npy lookup and zero/partial fallback')
    p.add_argument('--feature-reduction', choices=['none', 'spatial_mean'], default='none',
                   help='Only available with --feature-mode strict; legacy never pools spatial maps')
    p.add_argument('--resume', help='Resume an epoch-boundary last.pt from this implementation')
    p.add_argument('--preflight-only', action='store_true')
    return p


def train_epoch(model, batches, optimizer, config, device):
    model.train()
    totals = torch.zeros(2, dtype=torch.float64)
    count = 0
    for batch in batches:
        inputs, target, mask = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        result = model(*inputs)
        prediction, _ = absolute_outputs(result, inputs[0][-1])
        loss, tasks = objective(model, result, prediction, target, mask,
                                config['beta_load'], config['beta_reg'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        totals += tasks.detach().double().cpu() * target.shape[1]
        count += target.shape[1]
    if not count:
        raise ValueError('Empty training loader')
    return (totals / count).tolist()


def run_training(args=None):
    args = parser().parse_args(args)
    c = vars(args).copy()
    if c['epochs'] < 1 or c['batch_size'] < 1 or c['lr'] <= 0 or c['grad_clip'] <= 0:
        raise ValueError('Invalid training settings')
    if c['beta_load'] < 0 or c['beta_reg'] < 0 or c['weight_decay'] < 0:
        raise ValueError('Regularization coefficients must be nonnegative')
    if c['feature_mode'] == 'legacy' and c['feature_reduction'] != 'none':
        raise ValueError('Use --feature-reduction none for legacy feature parity')
    torch.set_num_threads(c['threads'])
    seed_all(c['seed'])
    device = torch.device(c['device'])
    manifest = split_manifest(c['data_root'], c['areas'], ['train', 'val'], policy=c['split_policy'])
    train_data = make_dataset(c['data_root'], c['areas'], 'train', c)
    val_data = make_dataset(c['data_root'], c['areas'], 'val', c)
    generator = torch.Generator().manual_seed(c['seed'])
    batches = loader(train_data, c, True, generator)
    train_probe, val_probe = loader(train_data, c), loader(val_data, c)
    model = make_model(c).to(device)
    control = FeedbackController(experts=c['experts'], variant=c['variant'], window=c['window'],
                                 ema=c['ema'], tau_u=c['tau_u'], alpha_min=c['alpha_min'],
                                 alpha_max=c['alpha_max'], gamma=c['gamma'], tau_r=c['tau_r'])
    control.apply(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    epoch0, best = 0, float('inf')
    hashes = source_hashes()
    if args.resume:
        saved = load_checkpoint(args.resume)
        mutable = {'epochs', 'resume', 'output_dir', 'data_root', 'device', 'threads', 'preflight_only'}
        for key, value in c.items():
            old_default = {'feature_mode': 'strict', 'split_policy': 'paper_years'}.get(key)
            if key not in mutable and saved['config'].get(key, old_default) != value:
                raise ValueError('Resume configuration mismatch: ' + key)
        if not compatible_sources(saved['sources'], hashes) or saved['manifest'] != manifest:
            raise ValueError('Source code or split manifest changed since checkpoint')
        model.load_state_dict(saved['model'], strict=True)
        control.load_state_dict(saved['controller'])
        control.apply(model)
        optimizer.load_state_dict(saved['optimizer'])
        epoch0, best = saved['epoch'], saved['best_score']
        random.setstate(saved['rng']['python'])
        np.random.set_state(saved['rng']['numpy'])
        torch.set_rng_state(saved['rng']['torch'])
        if torch.cuda.is_available() and saved['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        generator.set_state(saved['rng']['loader'])
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists() and not args.resume:
        raise FileExistsError('Run already exists; use a new directory or --resume')
    (out / 'config.json').write_text(json.dumps(c, indent=2))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if args.preflight_only:
        inputs, target, mask = unpack_batch(next(iter(train_probe)), device)
        model.eval()
        result = model(*inputs)
        pred, _ = absolute_outputs(result, inputs[0][-1])
        loss, _ = objective(model, result, pred, target, mask)
        loss.backward()
        print(json.dumps(dict(status='PASS', train_windows=len(train_data), val_windows=len(val_data),
                              prediction_shape=list(pred.shape), loss=float(loss.detach()))))
        return
    for epoch in range(epoch0 + 1, c['epochs'] + 1):
        train_loss = train_epoch(model, batches, optimizer, c, device)
        # Same parameters and preceding controller state for both panels.
        train_stats = collect_statistics(model, train_probe)
        val_stats = collect_statistics(model, val_probe)
        old_alpha, old_penalty = model.task_alpha.clone(), model.gap_penalty.clone()
        feedback = control.update(val_stats['task_loss'], train_stats['expert_error'], val_stats['expert_error'])
        control.apply(model)  # Commit BOTH new signals only after both complete passes.
        # Score the EXACT state we save, including its updated routing bias.
        selected_stats = collect_statistics(model, val_probe)
        score = selected_stats['metrics']['ade_km']
        improved = score < best
        best = min(best, score)
        row = dict(epoch=epoch, train_loss=train_loss, preceding_alpha=old_alpha,
                   preceding_penalty=old_penalty, train_stats=train_stats, val_stats=val_stats,
                   feedback=feedback, selected_state_val=selected_stats)
        with (out / 'feedback.jsonl').open('a') as f:
            f.write(json.dumps(jsonable(row), allow_nan=False) + '\n')
        state = dict(format=FORMAT, config=c, sources=hashes, manifest=manifest,
                     model=model.state_dict(), controller=control.state_dict(),
                     optimizer=optimizer.state_dict(), epoch=epoch, best_score=best,
                     selected_state_val=jsonable(selected_stats),
                     rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                              torch=torch.get_rng_state(),
                              cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                              loader=generator.get_state()))
        torch.save(state, out / 'last.tmp')
        os.replace(out / 'last.tmp', out / 'last.pt')
        if improved:
            torch.save(state, out / 'best.pt')
        print(json.dumps(dict(epoch=epoch, variant=c['variant'], alpha=feedback['alpha'],
                              val=selected_stats['metrics']), allow_nan=False), flush=True)



@contextmanager
def cached_inputs(frames):
    """Bounded caches for unchanged data files; callers always get fresh copies."""
    if not frames:
        yield
        return
    original_np = paper_data.np
    original_image = paper_data.TrajectoryDataset.img_read

    @lru_cache(maxsize=frames)
    def environment(path):
        return original_np.load(path, allow_pickle=True)

    @lru_cache(maxsize=frames)
    def image(dataset, path):
        return original_image(dataset, path)

    class NumpyProxy:
        def __getattr__(self, name):
            return getattr(original_np, name)

        def load(self, path, *args, **kwargs):
            if isinstance(path, (str, Path)) and not args and kwargs == {'allow_pickle': True}:
                # Object arrays contain mutable dictionaries/arrays; shallow
                # copies would let one sample corrupt another sample's cache.
                return copy.deepcopy(environment(str(path)))
            return original_np.load(path, *args, **kwargs)

    def read_image(dataset, path):
        return image(dataset, path).copy()

    try:
        with patch.object(paper_data, 'np', NumpyProxy()), \
             patch.object(paper_data.TrajectoryDataset, 'img_read', read_image):
            yield
    finally:
        environment.cache_clear()
        image.cache_clear()


class ParallelSamples:
    """Read independent, deterministic dataset samples concurrently, in order."""
    def __init__(self, dataset, pool):
        self.dataset, self.pool = dataset, pool

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.pool.submit(self.dataset.__getitem__, index)


class PrefetchBatches:
    """Overlap CPU loading with training, preserving the original DataLoader RNG."""
    def __init__(self, batches, ahead):
        self.batches, self.ahead = batches, ahead

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        # Construct on the main thread: consume exactly the original base seed
        # inside collect_statistics' RNG scope. Dataset reads never draw RNG.
        iterator = iter(self.batches)
        end = object()
        pending = deque()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='batch-prefetch') as producer:
            try:
                for _ in range(min(self.ahead, len(self))):
                    pending.append(producer.submit(next, iterator, end))
                while pending:
                    batch = pending.popleft().result()
                    if batch is end:
                        break
                    pending.append(producer.submit(next, iterator, end))
                    yield batch
            finally:
                for future in pending:
                    future.cancel()


def unpack_batch_async(batch, device):
    """Same tensors and operations as paper_pipeline.unpack_batch; async copies."""
    obs = torch.cat([batch[0], batch[7]], dim=-1).to(device, non_blocking=True)
    rel = torch.cat([batch[2], batch[9]], dim=-1).to(device, non_blocking=True)
    target = torch.cat([batch[1], batch[8]], dim=-1).to(device, non_blocking=True)
    mask = batch[5][:, obs.shape[0]:].to(device, non_blocking=True)
    return (obs, rel, batch[6].to(device, non_blocking=True),
            batch[13].to(device, non_blocking=True),
            {k: v.to(device, non_blocking=True) for k, v in batch[-2].items()}), target, mask


class ProgressBatches:
    def __init__(self, batches, label, every, emit=None, detail=None):
        self.batches, self.label, self.every = batches, label, every
        self.emit = emit or (lambda text: print(text, flush=True))
        self.detail = detail

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        total = len(self)
        started, data_seconds = time.perf_counter(), 0.0
        self.emit('[{}] Starting {} batches; waiting for first batch...'.format(self.label, total))
        iterator = iter(self.batches)
        completed = 0
        while True:
            loading = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            data_seconds += time.perf_counter() - loading
            yield batch
            completed += 1
            if completed == 1 or completed % self.every == 0 or completed == total:
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * max(total - completed, 0)
                detail = ', ' + self.detail() if self.detail else ''
                self.emit('[{}] batch {}/{}{} | elapsed {:.1f}s | data wait {:.1f}s | ETA {:.1f}s'.format(
                    self.label, completed, total, detail, elapsed, data_seconds, eta))


def run_accelerated(training, argv=None):
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument('--log-every', '--print-every', '--print_every', type=int, default=5)
    options.add_argument('--io-workers', type=int,
                         help='Parallel reader threads; leaves saved training settings unchanged')
    options.add_argument('--prefetch-factor', type=int, default=2)
    options.add_argument('--pin-memory', action='store_true')
    options.add_argument('--cache-frames', type=int, default=0,
                         help='LRU capacity for observed images and environment files (0 disables)')
    argv = list(sys.argv[1:] if argv is None else argv)
    extra, remaining = options.parse_known_args(argv)
    if extra.log_every < 1:
        options.error('--log-every must be positive')
    if extra.io_workers is not None and extra.io_workers < 0:
        options.error('--io-workers must be nonnegative')
    if extra.prefetch_factor < 1:
        options.error('--prefetch-factor must be positive')
    if extra.cache_frames < 0:
        options.error('--cache-frames must be nonnegative')
    if '--help' in remaining or '-h' in remaining:
        print('Progress option: --log-every N (default 5); alias --print_every. '
              'Console output is also appended to OUTPUT_DIR/out.log.\n'
              'Optional I/O: --io-workers N --prefetch-factor N (default 2) --pin-memory --cache-frames N. '
              'These options do not change saved model/training settings.')
        return training.run_training(remaining)
    io_options = ('--log-every', '--print-every', '--print_every', '--io-workers',
                  '--prefetch-factor', '--pin-memory', '--cache-frames')
    if not any(arg.split('=', 1)[0] in io_options for arg in argv):
        return training.run_training(remaining)
    config = vars(training.parser().parse_args(remaining))
    out = Path(config['output_dir'])
    if (out / 'last.pt').exists() and not config['resume']:
        raise FileExistsError('Run already exists; use a new directory or --resume')
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.Logger('train', level=logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    logfile = logging.FileHandler(str(out / 'out.log'), mode='a', encoding='utf-8')
    for handler in (console, logfile):
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)

    emit = logger.info
    original_train = training.train_epoch
    original_collect = training.collect_statistics
    original_dataset = training.make_dataset
    original_objective = training.objective
    original_load = training.load_checkpoint
    original_loader = training.loader
    original_unpack = training.unpack_batch
    io_workers = extra.io_workers or 0
    readers = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix='data-reader') if io_workers else None
    transfer = unpack_batch_async if extra.pin_memory else original_unpack
    epoch = [0]
    statistic = [0]
    current_loss = [None]
    epoch_loss = [None]
    epoch_started = [None]
    best = [float('inf')]
    device_logged = [False]

    def loader(dataset, config, shuffle=False, generator=None):
        # Threaded file reads avoid CUDA/OpenMP fork hazards and shared-memory
        # limits. Keep the original sampler, collator and epoch seed lifecycle.
        runtime = dict(config, workers=0) if readers else config
        data = ParallelSamples(dataset, readers) if readers else dataset
        result = original_loader(data, runtime, shuffle, generator)
        result.pin_memory = extra.pin_memory
        if readers:
            collate = result.collate_fn
            result.collate_fn = lambda samples: collate([sample.result() for sample in samples])
            return PrefetchBatches(result, extra.prefetch_factor)
        return result

    def load(path):
        saved = original_load(path)
        epoch[0], best[0] = saved['epoch'], saved['best_score']
        emit('[Resume] Read {}; completed epoch {}; best validation ADE={:.6f} km'.format(
            path, epoch[0], best[0]))
        return saved

    def objective(*args, **kwargs):
        loss, tasks = original_objective(*args, **kwargs)
        # Observe the existing calculation; never rerun a forward/backward pass.
        current_loss[0] = (loss.detach(), tasks.detach())
        return loss, tasks

    def loss_detail():
        loss, tasks = current_loss[0]
        return 'loss={:.6f}, track_loss={:.6f}, intensity_loss={:.6f}'.format(
            float(loss), float(tasks[0]), float(tasks[1]))

    def dataset(root, areas, split, config):
        started = time.perf_counter()
        emit('[Data] Loading {} windows...'.format(split))
        result = original_dataset(root, areas, split, config)
        emit('[Data] {}: {} windows in {:.1f}s'.format(split, len(result), time.perf_counter() - started))
        return result

    def train(model, batches, optimizer, config, device):
        # Use the actual parameter device, not just the command-line request.
        epoch[0] += 1
        statistic[0] = 0
        epoch_started[0] = time.perf_counter()
        if not device_logged[0]:
            actual = next(model.parameters()).device
            hardware = torch.cuda.get_device_name(actual) if actual.type == 'cuda' else 'CPU'
            emit('[Device] {} | {} | batch_size={} | io_workers={} | pin_memory={}'.format(
                actual, hardware, config['batch_size'], io_workers, extra.pin_memory))
            device_logged[0] = True
        emit('Starting epoch {}/{}, lr={:.8g}'.format(
            epoch[0], config['epochs'], optimizer.param_groups[0]['lr']))
        emit('[Routing] alpha_track={:.6f}, alpha_intensity={:.6f}, gamma={:.6f}, max_penalty={:.6f}'.format(
            float(model.task_alpha[0]), float(model.task_alpha[1]),
            float(model.gap_gamma), float(model.gap_penalty.max())))
        wrapped = ProgressBatches(batches, 'Train {}/{}'.format(epoch[0], config['epochs']),
                                  extra.log_every, emit, loss_detail)
        result = original_train(model, wrapped, optimizer, config, device)
        epoch_loss[0] = result
        return result

    def collect(model, batches):
        labels = ('Train expert statistics', 'Validation expert statistics', 'Validation after feedback')
        label = '{} {}/{}'.format(labels[statistic[0] % len(labels)], epoch[0], config['epochs'])
        statistic[0] += 1
        return original_collect(model, ProgressBatches(batches, label, extra.log_every, emit))

    def summary(*values, **kwargs):
        # Only replace the driver's final epoch print, after its checkpoint saves.
        # feedback.jsonl and all serialized checkpoint fields stay untouched.
        try:
            row = json.loads(values[0]) if len(values) == 1 and isinstance(values[0], str) else None
        except (TypeError, ValueError):
            row = None
        if not isinstance(row, dict) or not {'epoch', 'alpha', 'val'}.issubset(row):
            return builtins.print(*values, **kwargs)
        val = row['val']
        improved = val['ade_km'] < best[0]
        best[0] = min(best[0], val['ade_km'])
        emit('[Epoch {}/{}] track_loss={:.6f}, intensity_loss={:.6f}, '
             'ADE={:.4f} km, pressure={:.4f} hPa, wind={:.4f} m/s, best={}, seconds={:.1f}'.format(
                 row['epoch'], config['epochs'], *epoch_loss[0], val['ade_km'],
                 val['pressure_mae_hpa'], val['wind_mae_ms'], improved,
                 time.perf_counter() - epoch_started[0]))
        emit('[TSI-UPDATE] alpha_track={:.6f}, alpha_intensity={:.6f}'.format(*row['alpha']))
        emit('[Saved] {}'.format(out / 'last.pt'))
        if improved:
            emit('[Saved] {} (new best validation ADE)'.format(out / 'best.pt'))

    # Restore originals on completion/errors, including when this entry point is imported.
    try:
        emit('Using random seed {} | variant={} | log_every={}'.format(
            config['seed'], config['variant'], extra.log_every))
        emit('[Log] {}'.format(out / 'out.log'))
        emit('[Input] io_workers={}, prefetch_factor={}, pin_memory={}, async_transfer={}, '
             'worker_start={}, cache_frames={}; batch size and sample order unchanged'.format(
                 io_workers, extra.prefetch_factor if io_workers else None,
                 extra.pin_memory, extra.pin_memory, 'threads' if io_workers else 'original loader',
                 extra.cache_frames))
        with cached_inputs(extra.cache_frames), \
             patch.object(training, 'make_dataset', dataset), \
             patch.object(training, 'loader', loader), \
             patch.object(training, 'unpack_batch', transfer), \
             patch.object(paper_pipeline, 'unpack_batch', transfer), \
             patch.object(training, 'train_epoch', train), \
             patch.object(training, 'collect_statistics', collect), \
             patch.object(training, 'objective', objective), \
             patch.object(training, 'load_checkpoint', load), \
             patch.object(training, 'print', summary, create=True):
            return training.run_training(remaining)
    finally:
        if readers:
            readers.shutdown(wait=True)
        for handler in (console, logfile):
            logger.removeHandler(handler)
            handler.close()


def main(argv=None):
    return run_accelerated(sys.modules[__name__], argv)


if __name__ == '__main__':
    main()
