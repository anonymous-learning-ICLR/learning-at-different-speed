"""Evaluate one trained checkpoint; print only ALL and regional error tables."""
import argparse
import contextlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from paper_pipeline import (TRACK, absolute_outputs, check_mask, load_checkpoint,
                            loader, make_dataset, make_model, seed_all, unpack_batch)
from TCNM.losses import toNE

AREAS = ('EP', 'NA', 'NI', 'SI', 'SP', 'WP')


def physical_values(value):
    xy, intensity = toNE(value[..., :2].clone(), value[..., 2:].clone())
    return torch.cat([xy, intensity], -1).permute(1, 0, 2).cpu().numpy()


def error_sums(prediction, truth, mask):
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError('Nonfinite model predictions or labels')
    dx = (prediction[..., 0] - truth[..., 0]) / 10 * 111 * np.cos(np.deg2rad(truth[..., 1] / 10))
    dy = (prediction[..., 1] - truth[..., 1]) / 10 * 111
    errors = np.stack([np.hypot(dx, dy), np.abs(prediction[..., 2] - truth[..., 2]),
                       np.abs(prediction[..., 3] - truth[..., 3])], axis=-1)
    if mask.shape != errors.shape[:2] or not np.isin(mask, [0, 1]).all():
        raise ValueError('Invalid evaluation mask')
    return ((errors * mask[..., None]).sum(0, dtype=np.float64),
            mask.sum(0, dtype=np.float64))


def summarize(count, sums, mass):
    if count == 0 or np.any(mass <= 0):
        raise ValueError('No valid observations for one or more forecast steps')
    steps, avg = sums / mass[:, None], sums.sum(0) / mass.sum()
    return dict(count=count, ade=float(avg[0]), fde=float(steps[-1, 0]),
                steps=steps, avg=avg, sums=sums, mass=mass)


def report_text(reports):
    combined = summarize(sum(r['count'] for r in reports.values()),
                         np.sum([r['sums'] for r in reports.values()], axis=0),
                         np.sum([r['mass'] for r in reports.values()], axis=0))
    bar, blocks = '-' * 72, []
    for area, report in [('ALL', combined)] + list(reports.items()):
        lines = ['Area: {} (N={})'.format(area, report['count']),
                 'Overall ADE (km): {:.6f}'.format(report['ade']),
                 'Overall FDE (km): {:.6f}'.format(report['fde']), bar,
                 'Step     | TDE (km)        | Press (hPa)     | Wind (m/s)', bar]
        for step, values in enumerate(report['steps'], 1):
            lines.append('Step %-3d | %-15.4f | %-15.4f | %-15.4f' % (step, *values))
        lines.extend([bar, 'AVG      | %-15.4f | %-15.4f | %-15.4f' % tuple(report['avg']), bar])
        blocks.append('\n'.join(lines))
    return '\n\n'.join(blocks) + '\n'


@torch.no_grad()
def evaluate(checkpoint_path, data_root, areas=None, device='cpu', batch_size=None,
             threads=4, workers=0):
    checkpoint_path = Path(checkpoint_path)
    with checkpoint_path.open('rb') as stream:
        if stream.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('Checkpoint is a Git LFS pointer. Run git lfs pull to download it.')
    checkpoint = load_checkpoint(checkpoint_path)
    config = dict(checkpoint['config'], workers=workers)
    config['batch_size'] = batch_size or config['batch_size']
    if config['batch_size'] < 1 or threads < 1 or workers < 0:
        raise ValueError('Invalid batch size, thread count or worker count')
    root = Path(data_root)
    if areas is None:
        selected = [a for a in AREAS if any((root / TRACK / a / 'test').glob('*.txt'))]
    else:
        selected = [a for a in AREAS if a in areas]
    if not selected:
        raise ValueError('No test tracks found in the selected data directory')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu or a CUDA-enabled environment')
    torch.set_num_threads(threads)
    seed_all(config['seed'])
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    reports = OrderedDict()
    for area in selected:
        dataset = make_dataset(root, [area], 'test', config)
        sums = np.zeros((config['pred_len'], 3), dtype=np.float64)
        mass = np.zeros(config['pred_len'], dtype=np.float64)
        count = 0
        for batch in loader(dataset, config):
            inputs, truth, mask = unpack_batch(batch, device)
            check_mask(mask)
            result = model(*inputs)
            prediction, _ = absolute_outputs(result, inputs[0][-1])
            batch_sums, batch_mass = error_sums(physical_values(prediction),
                                               physical_values(truth), mask.cpu().numpy())
            sums += batch_sums
            mass += batch_mass
            count += truth.shape[1]
        reports[area] = summarize(count, sums, mass)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', '--ckpt_path', default='ckpt.pt')
    parser.add_argument('--data-dir', '--data-root', '--data_root', dest='data_root', default='data_test')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--gpu-num', '--gpu_num', dest='gpu_num')
    parser.add_argument('--areas', nargs='+', choices=AREAS)
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--output', '--output-dir', '--output_dir', default='test_report')
    args = parser.parse_args(argv)
    if args.threads < 1 or args.workers < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error('Batch size and threads must be positive; workers must be nonnegative')
    if args.gpu_num is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        reports = evaluate(args.checkpoint, args.data_root, args.areas, args.device,
                           args.batch_size, args.threads, args.workers)
    text = report_text(reports)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.txt').write_text(text, encoding='utf-8')
    print(text, end='', flush=True)
    return reports


if __name__ == '__main__':
    main()
