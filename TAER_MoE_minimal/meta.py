"""Fit saved per-area fusion rules and embed all member weights into one ckpt.pt.

--fit-split test explicitly reproduces the historical test-fit/test-eval protocol.
Use --fit-split val for separate calibration and test data. Testing never refits.
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

from meta_ensemble import (AREAS, snapshot_members, validate_bundle, prediction_panels,
                           fit_rule, apply_rule, metrics, atomic_save, read_checkpoint,
                           split_manifest)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir')
    p.add_argument('--checkpoint-paths', nargs='+')
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True, help='Final self-contained ckpt.pt file')
    p.add_argument('--fit-split', choices=('val', 'test'), default='val')
    p.add_argument('--areas', nargs='+', choices=AREAS)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--io-workers', type=int, default=1)
    p.add_argument('--cache-frames', type=int, default=4096)
    args = p.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        p.error('Output already exists; choose a new path to preserve the existing checkpoint')
    if not args.checkpoint_dir and not args.checkpoint_paths:
        p.error('Supply --checkpoint-dir or --checkpoint-paths')
    if min(args.batch_size, args.threads) < 1 or min(args.io_workers, args.cache_frames) < 0:
        p.error('Invalid resource settings')
    emit = lambda text: print('[Meta] ' + text, flush=True)
    bundle = snapshot_members(args.checkpoint_dir, args.checkpoint_paths)
    validate_bundle(bundle, fitted=False)
    areas = args.areas or bundle['model_config']['areas']
    if len(set(areas)) != len(areas) or not set(areas).issubset(bundle['model_config']['areas']):
        p.error('Select unique areas included in the source configuration')
    areas = [a for a in AREAS if a in areas]
    manifest = split_manifest(args.data_root, areas, ['train', 'val', 'test'],
                              policy=bundle['model_config']['split_policy'])
    bundle['evaluation_protocol'].update(meta_fit_split=args.fit_split, metric_split='test',
        independent_test=(args.fit_split != 'test'))
    bundle['fit_data'] = dict(split=args.fit_split,
        track_manifest_sha256=hashlib.sha256(json.dumps(manifest[args.fit_split], sort_keys=True).encode()).hexdigest(),
        counts={})
    bundle['evaluation_defaults'] = dict(batch_size=args.batch_size, threads=args.threads,
                                         io_workers=args.io_workers, cache_frames=args.cache_frames)
    bundle['fit_metrics'] = OrderedDict()
    emit('{} distinct frozen models: {}'.format(len(bundle['checkpoint_keys']),
        ', '.join('{} ({})'.format(v['global_step'], '/'.join(v['source_files'])) for v in bundle['base_checkpoints'].values())))
    emit('Fusion fit split: {}; final evaluation split: test'.format(args.fit_split))
    # Send progress to the real stdout while suppressing legacy constructor prints.
    stream = sys.stdout
    progress = lambda text: print('[Meta] ' + text, file=stream, flush=True)
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
        for area, pred, truth, mask in prediction_panels(bundle, args.data_root, args.fit_split, areas,
                device=args.device, batch_size=args.batch_size, threads=args.threads,
                io_workers=args.io_workers, cache_frames=args.cache_frames, progress=progress):
            rule = fit_rule(pred, truth, mask)
            bundle['area_models'][area] = rule
            bundle['fit_data']['counts'][area] = len(truth)
            bundle['fit_metrics'][area] = metrics(apply_rule(rule, pred), truth, mask)
    validate_bundle(bundle)
    atomic_save(bundle, output)
    restored = read_checkpoint(output)
    validate_bundle(restored)
    if restored['area_models'] != bundle['area_models']:
        raise RuntimeError('Fusion rules changed during serialization')
    emit('Saved ' + str(output.resolve()))


if __name__ == '__main__':
    main()
