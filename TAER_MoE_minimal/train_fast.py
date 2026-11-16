"""Optional input acceleration and progress logs for the epoch-based driver.

Accepts all train_paper.py arguments, plus --log-every (default: 5),
--io-workers, --prefetch-factor and --pin-memory for optional input pipelining.
--cache-frames caches immutable observed frames without changing their values.
This entry point does not change checkpoint configuration or model source hashes.
"""
import argparse
import builtins
import copy
import json
import logging
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import torch
import paper_pipeline
import train_paper as training
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


def main(argv=None):
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument('--log-every', '--print-every', '--print_every', type=int, default=5)
    options.add_argument('--io-workers', type=int,
                         help='Parallel reader threads; leaves saved training settings unchanged')
    options.add_argument('--prefetch-factor', type=int, default=2)
    options.add_argument('--pin-memory', action='store_true')
    options.add_argument('--cache-frames', type=int, default=0,
                         help='LRU capacity for observed images and environment files (0 disables)')
    extra, remaining = options.parse_known_args(sys.argv[1:] if argv is None else argv)
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
        return training.main(remaining)
    config = vars(training.parser().parse_args(remaining))
    out = Path(config['output_dir'])
    if (out / 'last.pt').exists() and not config['resume']:
        raise FileExistsError('Run already exists; use a new directory or --resume')
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.Logger('train_fast', level=logging.INFO)
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
            return training.main(remaining)
    finally:
        if readers:
            readers.shutdown(wait=True)
        for handler in (console, logfile):
            logger.removeHandler(handler)
            handler.close()


if __name__ == '__main__':
    main()
