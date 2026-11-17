"""Train the explicitly reconstructed Eq. (3)-(9) TC implementation."""
import argparse
import json
from pathlib import Path
import torch
from paper_pipeline import (FORMAT, seed_all, make_model, make_dataset, loader,
    split_manifest, unpack_batch, absolute_outputs, objective, collect_statistics,
    load_checkpoint, jsonable, source_hashes)
from taer_feedback import FeedbackController, VARIANTS
from training_checkpoints import (CheckpointWriter, compatible_sources, rng_state, restore_rng,
                                  atomic_save, freeze)

# Kept separate so progress logging distinguishes interval validation from the
# three epoch-end passes. Interval validation never updates the controller.
validate_for_checkpoint = collect_statistics


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--split-policy', choices=['existing', 'paper_years'], default='existing',
                   help='Use the existing train/val/test folders; paper_years additionally enforces temporal bounds')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--variant', choices=VARIANTS, default='full')
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--areas', nargs='+', default=['EP', 'NA', 'NI', 'SI', 'SP', 'WP'])
    p.add_argument('--epochs', '--num_epochs', type=int, default=101)
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
    p.add_argument('--checkpoint-every', '--checkpoint_every', type=int, default=3000)
    p.add_argument('--resume', help='Resume last.pt or a numbered training checkpoint')
    p.add_argument('--preflight-only', action='store_true')
    return p


def train_epoch(model, batches, optimizer, config, device, progress=None,
                on_batch=None, resume_rng=None):
    model.train()
    progress = progress if progress is not None else {}
    totals = torch.as_tensor(progress.get('train_totals', [0., 0.]), dtype=torch.float64).clone()
    count = progress.get('train_count', 0)
    skipped = progress.get('batch_in_epoch', 0)
    iterator = iter(batches)
    # Reconstruct the original shuffled epoch using its saved loader seed.
    # These batches are only read; no model/optimizer updates are repeated.
    for _ in range(skipped):
        next(iterator)
    if resume_rng is not None:
        restore_rng(resume_rng)
    if skipped and progress.get('global_step', 0) >= progress.get('total_steps', float('inf')):
        if skipped == len(batches):
            # Complete the sampler's normal end-of-epoch RNG bookkeeping.
            next(iterator, None)
        return (totals / count).tolist()
    for batch_index, batch in enumerate(iterator, skipped + 1):
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
        progress.update(batch_in_epoch=batch_index, train_totals=totals.tolist(), train_count=count,
                        global_step=progress.get('global_step', 0) + 1)
        if on_batch is not None:
            on_batch(progress)
        if progress['global_step'] >= progress.get('total_steps', float('inf')) and batch_index < len(batches):
            break
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
    if c['checkpoint_every'] < 1:
        raise ValueError('checkpoint_every must be positive')
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
    # Match train_github.py exactly: floor the fractional batch count * epochs,
    # rather than ceil(dataset/batch_size) * epochs.
    total_steps = int(len(train_data) / c['batch_size'] * c['epochs'])
    if total_steps < 1:
        raise ValueError('Training sample/batch/epoch settings produce zero total steps')
    model = make_model(c).to(device)
    control = FeedbackController(experts=c['experts'], variant=c['variant'], window=c['window'],
                                 ema=c['ema'], tau_u=c['tau_u'], alpha_min=c['alpha_min'],
                                 alpha_max=c['alpha_max'], gamma=c['gamma'], tau_r=c['tau_r'])
    control.apply(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    epoch0, best, global_step = 0, float('inf'), 0
    saved, partial = None, None
    hashes = source_hashes()
    if args.resume:
        saved = load_checkpoint(args.resume)
        mutable = {'epochs', 'resume', 'output_dir', 'data_root', 'device', 'threads',
                   'preflight_only', 'checkpoint_every'}
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
        global_step = saved.get('global_step', epoch0 * len(batches))
        restore_rng(saved['rng'], generator)
        if saved.get('progress', {}).get('phase') == 'training':
            partial = dict(saved['progress'])
            epoch0 -= 1
        if total_steps < global_step:
            raise ValueError('Configured total steps are below the resumed global step')
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists() and not args.resume:
        raise FileExistsError('Run already exists; use a new directory or --resume')
    if saved is not None and not (out / 'best.pt').exists():
        # Carry the best weights when continuing in a new directory. Never claim
        # a historical best score without retaining the corresponding weights.
        source_best = Path(args.resume).resolve().parent / 'best.pt'
        candidate = load_checkpoint(source_best) if source_best.exists() else saved
        if (candidate['selected_state_val']['metrics']['ade_km'] != best or
                candidate['config']['seed'] != c['seed'] or candidate['manifest'] != manifest):
            raise ValueError('The matching historical best.pt is required to resume in a new directory')
        atomic_save(freeze(candidate), out / 'best.pt')
    (out / 'config.json').write_text(json.dumps(c, indent=2))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    writer = CheckpointWriter(out, best, saved.get('best_record') if saved else None)
    print(json.dumps(dict(event='training_plan', train_windows=len(train_data),
                          batch_size=c['batch_size'], epochs=c['epochs'],
                          total_steps=total_steps, completed_steps=global_step,
                          checkpoint_every=c['checkpoint_every'])), flush=True)
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
        if global_step >= total_steps and partial is None:
            break
        if partial is not None:
            progress = partial
            generator.set_state(progress['epoch_start_loader_rng'])
            resume_rng = saved['rng']
            print(json.dumps(dict(event='resume_position', epoch=epoch,
                                  skip_batches=progress['batch_in_epoch'], global_step=global_step)), flush=True)
        else:
            progress = dict(batch_in_epoch=0, train_totals=[0., 0.], train_count=0,
                            epoch_start_loader_rng=generator.get_state().clone())
            resume_rng = None
        progress.update(phase='training', global_step=global_step, total_steps=total_steps)

        def snapshot(stats):
            return dict(format=FORMAT, config=c, sources=hashes, manifest=manifest,
                        model=model.state_dict(), controller=control.state_dict(),
                        optimizer=optimizer.state_dict(), epoch=epoch,
                        global_step=progress['global_step'], total_steps=total_steps,
                        progress=progress, selected_state_val=jsonable(stats),
                        rng=rng_state(generator))

        def checkpoint_batch(position):
            step = position['global_step']
            if step % c['checkpoint_every']:
                return
            stats = validate_for_checkpoint(model, val_probe)
            improved = writer.save(snapshot(stats), archive=True)
            print(json.dumps(dict(event='checkpoint', epoch=epoch, global_step=step,
                                  total_steps=total_steps, best=improved, best_score=writer.best_score,
                                  val=stats['metrics'])), flush=True)

        train_loss = train_epoch(model, batches, optimizer, c, device, progress=progress,
                                 on_batch=checkpoint_batch, resume_rng=resume_rng)
        global_step = progress['global_step']
        partial = None
        # Same parameters and preceding controller state for both panels.
        train_stats = collect_statistics(model, train_probe)
        val_stats = collect_statistics(model, val_probe)
        old_alpha, old_penalty = model.task_alpha.clone(), model.gap_penalty.clone()
        feedback = control.update(val_stats['task_loss'], train_stats['expert_error'], val_stats['expert_error'])
        control.apply(model)  # Commit BOTH new signals only after both complete passes.
        # Score the EXACT state we save, including its updated routing bias.
        selected_stats = collect_statistics(model, val_probe)
        row = dict(epoch=epoch, train_loss=train_loss, preceding_alpha=old_alpha,
                   preceding_penalty=old_penalty, train_stats=train_stats, val_stats=val_stats,
                   feedback=feedback, selected_state_val=selected_stats)
        with (out / 'feedback.jsonl').open('a') as f:
            f.write(json.dumps(jsonable(row), allow_nan=False) + '\n')
        progress['phase'] = 'epoch_end'
        improved = writer.save(snapshot(selected_stats))
        print(json.dumps(dict(epoch=epoch, variant=c['variant'], alpha=feedback['alpha'],
                              global_step=global_step, total_steps=total_steps, best=improved,
                              best_score=writer.best_score, val=selected_stats['metrics']), allow_nan=False), flush=True)



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
    original_interval_collect = training.validate_for_checkpoint
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
        if saved.get('progress', {}).get('phase') == 'training':
            epoch[0] -= 1
        emit('[Resume] Read {}; completed epochs {}; best validation ADE={:.6f} km'.format(
            path, epoch[0], best[0]))
        return saved

    def objective(*args, **kwargs):
        loss, tasks = original_objective(*args, **kwargs)
        # Observe the existing calculation; never rerun a forward/backward pass.
        current_loss[0] = (loss.detach(), tasks.detach())
        return loss, tasks

    def loss_detail():
        if current_loss[0] is None:
            return 'restoring data order (no optimizer update)'
        loss, tasks = current_loss[0]
        return 'loss={:.6f}, track_loss={:.6f}, intensity_loss={:.6f}'.format(
            float(loss), float(tasks[0]), float(tasks[1]))

    def dataset(root, areas, split, config):
        started = time.perf_counter()
        emit('[Data] Loading {} windows...'.format(split))
        result = original_dataset(root, areas, split, config)
        emit('[Data] {}: {} windows in {:.1f}s'.format(split, len(result), time.perf_counter() - started))
        return result

    def train(model, batches, optimizer, config, device, **kwargs):
        # Use the actual parameter device, not just the command-line request.
        epoch[0] += 1
        statistic[0] = 0
        current_loss[0] = None
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
        result = original_train(model, wrapped, optimizer, config, device, **kwargs)
        epoch_loss[0] = result
        return result

    def collect(model, batches):
        labels = ('Train expert statistics', 'Validation expert statistics', 'Validation after feedback')
        label = '{} {}/{}'.format(labels[statistic[0] % len(labels)], epoch[0], config['epochs'])
        statistic[0] += 1
        return original_collect(model, ProgressBatches(batches, label, extra.log_every, emit))

    def interval_collect(model, batches):
        return original_interval_collect(model, ProgressBatches(
            batches, 'Checkpoint validation (epoch {})'.format(epoch[0]), extra.log_every, emit))

    def summary(*values, **kwargs):
        # Only replace the driver's final epoch print, after its checkpoint saves.
        # feedback.jsonl and all serialized checkpoint fields stay untouched.
        try:
            row = json.loads(values[0]) if len(values) == 1 and isinstance(values[0], str) else None
        except (TypeError, ValueError):
            row = None
        if isinstance(row, dict) and row.get('event') == 'training_plan':
            emit('[Plan] {} training windows | batch_size={} | epochs={} | total_steps={} | '
                 'completed_steps={} | checkpoint_every={}'.format(
                     row['train_windows'], row['batch_size'], row['epochs'], row['total_steps'],
                     row['completed_steps'], row['checkpoint_every']))
            return
        if isinstance(row, dict) and row.get('event') == 'resume_position':
            emit('[Resume] Restoring epoch {} data order; skipping {} completed batches; global_step={}'.format(
                row['epoch'], row['skip_batches'], row['global_step']))
            return
        if isinstance(row, dict) and row.get('event') == 'checkpoint':
            best[0] = row['best_score']
            emit('[Checkpoint] step {}/{} | validation ADE={:.6f} km | best={}'.format(
                row['global_step'], row['total_steps'], row['val']['ade_km'], row['best']))
            emit('[Saved] {}'.format(out / 'checkpoint_with_model_{:05d}.pt'.format(row['global_step'])))
            emit('[Saved] {}'.format(out / 'last.pt'))
            if row['best']:
                emit('[Saved] {} | {}'.format(out / 'best.pt', out / 'best.txt'))
            return
        if not isinstance(row, dict) or not {'epoch', 'alpha', 'val'}.issubset(row):
            return builtins.print(*values, **kwargs)
        val = row['val']
        improved = row.get('best', val['ade_km'] < best[0])
        best[0] = row.get('best_score', min(best[0], val['ade_km']))
        emit('[Epoch {}/{}] track_loss={:.6f}, intensity_loss={:.6f}, '
             'ADE={:.4f} km, pressure={:.4f} hPa, wind={:.4f} m/s, best={}, seconds={:.1f}'.format(
                 row['epoch'], config['epochs'], *epoch_loss[0], val['ade_km'],
                 val['pressure_mae_hpa'], val['wind_mae_ms'], improved,
                 time.perf_counter() - epoch_started[0]))
        emit('[FEEDBACK-UPDATE] alpha_track={:.6f}, alpha_intensity={:.6f}'.format(*row['alpha']))
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
             patch.object(training, 'validate_for_checkpoint', interval_collect), \
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
