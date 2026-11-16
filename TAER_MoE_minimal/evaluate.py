"""Evaluate epoch-model checkpoints, original snapshots, or existing ensembles.

Example: python evaluate.py --checkpoint /path/to/seed3047ckpt.pt
         --data-dir /path/to/TropiCycloneNet1 --device cuda --seed 3407

The checkpoint format selects the evaluator automatically. Epoch-model checkpoints
use frozen feedback and --output metrics.json; ensembles retain their table output.

Ensembles apply their saved per-area coefficients without refitting. Their original
Best-of-K protocol uses true future tracks to select candidates. Historical fusion
coefficients fitted on test data do not constitute an independent held-out score.
"""
import argparse
import contextlib
import hashlib
import inspect
import json
import os
import random
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from TCNM.data.trajectoriesWithMe_unet_training_4to4 import TrajectoryDataset, seq_collate
from TCNM.models_prior_unet4to4_tsi_router import TrajectoryGenerator
from TCNM.losses import toNE
from TCNM.utils import relative_to_abs
from paper_pipeline import (FORMAT, make_model, make_dataset, loader,
                            collect_statistics, split_manifest)

# Architecture defaults formerly imported from the historical training script.
LEGACY_MODEL_DEFAULTS = {'obs_len': 8,
 'pred_len': 4,
 'embedding_dim': 32,
 'mlp_dim': 128,
 'num_layers': 1,
 'noise_dim': (16,),
 'noise_type': 'gaussian',
 'noise_mix_type': 'ped',
 'pooling_type': None,
 'pool_every_timestep': False,
 'dropout': 0,
 'bottleneck_dim': 16,
 'batch_norm': 0,
 'neighborhood_size': 2.0,
 'grid_size': 8,
 'encoder_h_dim_g': 64,
 'decoder_h_dim_g': 64}

AREAS = ('EP', 'NA', 'NI', 'SI', 'SP', 'WP')
TRACK = 'bst_divi10_train_val_test_inlcude15_2023_new'


def load_checkpoint(path):
    kwargs = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(str(path), **kwargs)


def model_config(saved):
    args = dict(LEGACY_MODEL_DEFAULTS)
    args.update(vars(saved) if hasattr(saved, '__dict__') else saved)
    keys = ('obs_len', 'pred_len', 'embedding_dim', 'mlp_dim', 'num_layers', 'noise_dim',
            'noise_type', 'noise_mix_type', 'pooling_type', 'pool_every_timestep',
            'dropout', 'bottleneck_dim', 'batch_norm', 'neighborhood_size', 'grid_size')
    result = {key: args[key] for key in keys}
    result.update(encoder_h_dim=args.get('encoder_h_dim', args['encoder_h_dim_g']),
                  decoder_h_dim=args.get('decoder_h_dim', args['decoder_h_dim_g']))
    result['noise_dim'] = tuple(result['noise_dim'])
    return result


def unpack(checkpoint, path, state_key='auto'):
    kind = checkpoint.get('format')
    if kind in ('tc3_tsi_router_meta_ensemble', 'tc3-release-ensemble-v1'):
        if state_key != 'auto':
            raise ValueError('An ensemble supplies its own states; omit --state-key')
        if kind == 'tc3_tsi_router_meta_ensemble' and checkpoint.get('format_version') != 1:
            raise ValueError('Unsupported checkpoint version')
        if checkpoint['evaluation_protocol']['base_prediction_selection'] != 'best_of_k_by_ground_truth_track_distance':
            raise ValueError('Unsupported selection protocol')
        saved = checkpoint.get('model_config', checkpoint.get('args', {}))
        defaults = checkpoint.get('evaluation_defaults', checkpoint.get('args', {}))
        return (model_config(saved), checkpoint['base_checkpoints'], checkpoint['checkpoint_keys'],
                checkpoint['area_models'], defaults)
    if 'args' not in checkpoint and 'model_config' not in checkpoint:
        raise ValueError('Checkpoint lacks model metadata; use an original training or ensemble checkpoint')
    if state_key == 'auto':
        state_key = next((key for key in ('g_best_state', 'g_state', 'generator_state_dict')
                          if isinstance(checkpoint.get(key), dict)), None)
    if state_key is None or not isinstance(checkpoint.get(state_key), dict):
        raise ValueError('Checkpoint has no selected generator weights')
    saved = checkpoint.get('model_config', checkpoint.get('args', {}))
    match = re.search(r'_with_model_(\d+)\.pt$', Path(path).name)
    key = int(match[1]) if match else Path(path).name
    bases = {key: {'state_dict': checkpoint[state_key]}}
    rules = {area: {'mode': 'single_checkpoint', 'keys': [key]} for area in AREAS}
    defaults = vars(saved) if hasattr(saved, '__dict__') else saved
    return model_config(saved), bases, [key], rules, defaults


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prediction_seed(seed, key, area):
    return int.from_bytes(hashlib.sha256('{}|{}|{}'.format(seed, key, area).encode()).digest()[:4], 'little')


def test_loader(root, config, area, batch_size, limit=None, workers=0):
    root = Path(root)
    paths = dict(root=str(root / TRACK), type='test', sort_files=True, observed_only=True,
                 env_root=str(root / 'all_area_correct_location_includeunder15_2023'),
                 image_root=str(root / 'all_ocean_gph500_2023'),
                 feature6_root=str(root / 'features6'), feature10_root=str(root / 'features10'))
    dataset = TrajectoryDataset(paths, obs_len=config['obs_len'], pred_len=config['pred_len'],
                                skip=1, delim='\t', other_modal='gph', areas=[area])
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=seq_collate)


@torch.no_grad()
def predict(model, batches, device, num_samples):
    predictions, targets = [], []
    for batch in batches:
        obs = torch.cat([batch[0], batch[7]], dim=2).to(device)
        rel = torch.cat([batch[2], batch[9]], dim=2).to(device)
        xy, intensity = batch[1].to(device), batch[8].to(device)
        env = {key: batch[-2][key].to(device) for key in model.env_net_chooser.data_embed}
        relative, _, _, _ = model(obs, rel, batch[6].to(device), batch[13].to(device),
                                  env, num_samples=num_samples, all_g_out=False)
        absolute = relative_to_abs(relative, obs[-1])
        distance = ((absolute[:, :, :, :2] - xy.unsqueeze(1)) ** 2).sum(3).sqrt().sum(0)
        winner = distance.argmin(0)
        selected = torch.stack([absolute[:, winner[b], b] for b in range(absolute.shape[2])], dim=1)
        px, pv = toNE(selected[:, :, :2].clone(), selected[:, :, 2:].clone())
        tx, tv = toNE(xy.clone(), intensity.clone())
        predictions.append(torch.cat([px, pv], dim=2).permute(1, 0, 2).cpu().numpy())
        targets.append(torch.cat([tx, tv], dim=2).permute(1, 0, 2).cpu().numpy())
    if not predictions:
        raise ValueError('No complete test windows')
    return np.concatenate(predictions), np.concatenate(targets)


def fuse(rule, predictions):
    values = [predictions[key] for key in rule['keys']]
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise ValueError('Inconsistent member prediction shapes')
    if rule['mode'] == 'single_checkpoint':
        if len(values) != 1:
            raise ValueError('An identity rule must reference exactly one member')
        return values[0].copy()
    if rule['mode'] not in ('meta_fusion', 'linear_fusion'):
        raise ValueError('Unsupported fusion mode')
    stacked = np.stack(values)
    result = np.empty(shape, dtype=np.float32)
    for feature in range(4):
        saved = rule['model_info']['feat_{}'.format(feature)]
        coefficients = np.asarray(saved['coef'], dtype=saved['coef_dtype'])
        design = stacked[:, :, :, feature].reshape(len(values), -1).T
        result[:, :, feature] = (design @ coefficients + saved['intercept']).reshape(shape[:2])
    return result


def metrics(prediction, truth):
    dx = (prediction[:, :, 0] - truth[:, :, 0]) / 10 * 111 * np.cos(np.deg2rad(truth[:, :, 1] / 10))
    dy = (prediction[:, :, 1] - truth[:, :, 1]) / 10 * 111
    tde = np.sqrt(dx ** 2 + dy ** 2)
    press, wind = np.abs(prediction[:, :, 2] - truth[:, :, 2]), np.abs(prediction[:, :, 3] - truth[:, :, 3])
    return dict(count=len(prediction), ade=float(tde.mean()), fde=float(tde[:, -1].mean()),
                steps=[[float(a[:, i].mean()) for a in (tde, press, wind)] for i in range(tde.shape[1])],
                avg=[float(a.mean()) for a in (tde, press, wind)])


def table(area, values):
    bar = '-' * 72
    lines = ['Area: {}'.format('ALL' if area == 'GLOBAL' else area), 'Overall ADE (km): {:.6f}'.format(values['ade']),
             'Overall FDE (km): {:.6f}'.format(values['fde']), bar,
             'Step     | TDE (km)        | Press (hPa)     | Wind (m/s)', bar]
    for i, row in enumerate(values['steps'], 1):
        lines.append('Step %-3d | %-15.4f | %-15.4f | %-15.4f' % (i, *row))
    lines.extend([bar, 'AVG      | %-15.4f | %-15.4f | %-15.4f' % tuple(values['avg']), bar])
    return '\n'.join(lines)


def run(args, checkpoint=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num or '0'
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu')
    torch.set_num_threads(args.threads)
    if checkpoint is None:
        checkpoint = load_checkpoint(args.checkpoint)
    config, bases, keys, rules, defaults = unpack(checkpoint, args.checkpoint, args.state_key)
    batch_size, samples = args.batch_size or defaults.get('batch_size', 64), args.num_samples or defaults.get('num_samples', 20)
    seed = defaults.get('seed', 3407) if args.seed is None else args.seed
    selected = args.areas or [a for a in AREAS if (Path(args.data_dir) / TRACK / a / 'test').is_dir()]
    if not selected or len(set(selected)) != len(selected) or any(a not in rules for a in selected):
        raise ValueError('Select available areas without duplicates')
    areas = [area for area in AREAS if area in selected]
    needed = {key for area in areas for key in rules[area]['keys']}
    if not needed or not needed.issubset(bases) or not needed.issubset(keys) or len(set(keys)) != len(keys):
        raise ValueError('Missing or duplicate parameter sets')
    batches = {area: test_loader(args.data_dir, config, area, batch_size, args.limit, args.workers) for area in areas}
    seed_all(seed)
    generator = TrajectoryGenerator(**config).to(args.device).eval()
    predictions, targets = {area: {} for area in areas}, {}
    for key in keys:
        if key not in needed:
            continue
        generator.load_state_dict(bases[key]['state_dict'], strict=True)
        for area in areas:
            if key not in rules[area]['keys']:
                continue
            seed_all(prediction_seed(seed, key, area))
            prediction, truth = predict(generator, batches[area], args.device, samples)
            if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
                raise ValueError('Non-finite predictions or labels')
            if area in targets and not np.array_equal(targets[area], truth):
                raise ValueError('Sample order changed between members')
            predictions[area][key], targets[area] = prediction, truth
    results = OrderedDict((area, (fuse(rules[area], predictions[area]), targets[area])) for area in areas)
    all_pred = np.concatenate([pred for pred, _ in results.values()])
    all_true = np.concatenate([truth for _, truth in results.values()])
    if not np.isfinite(all_pred).all():
        raise ValueError('Non-finite fused predictions')
    reports = OrderedDict([('GLOBAL', metrics(all_pred, all_true))])
    reports.update((area, metrics(*values)) for area, values in results.items())
    text = '\n\n'.join(table(area, report) for area, report in reports.items()) + '\n'
    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'summary.txt').write_text(text, encoding='utf-8')
        if args.save_npz:
            # Keep native tenths-of-degree positions, matching the original exporter.
            for area, (pred, truth) in results.items():
                np.savez_compressed(out / (area + '_results.npz'), pred=pred, gt=truth)
    return text, reports


def run_paper(args, saved):
    """Evaluate the saved epoch-model state without fitting or feedback updates."""
    c = dict(saved['config'])
    if args.areas is not None:
        if len(set(args.areas)) != len(args.areas) or not set(args.areas).issubset(c['areas']):
            raise ValueError('Select unique areas included in the saved configuration')
        c['areas'] = list(args.areas)
    for key in ('threads', 'workers', 'batch_size', 'seed'):
        value = getattr(args, key)
        if value is not None:
            c[key] = value
    if args.gpu_num is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu')
    torch.set_num_threads(c['threads'])
    seed_all(c['seed'])
    if not args.demo_only:
        split_manifest(args.data_dir, c['areas'], ['train', 'val', 'test'],
                       policy=c.get('split_policy', 'paper_years'))
    model = make_model(c).to(args.device)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    stats = collect_statistics(model, loader(make_dataset(args.data_dir, c['areas'], 'test', c), c))
    result = dict(format=saved['format'], variant=c['variant'], seed=c['seed'],
                  epoch=saved['epoch'], demo_only=args.demo_only, metrics=stats['metrics'], count=stats['count'])
    text = json.dumps(result, indent=2, allow_nan=False)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text + '\n', result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', '--ckpt_path', required=True)
    parser.add_argument('--data-dir', '--data-root', '--data_root', default='data_test')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--gpu_num', help='Optional CUDA visibility override; legacy default: 0')
    parser.add_argument('--state-key', default='auto')
    parser.add_argument('--areas', nargs='+', choices=AREAS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch-size', '--batch_size', type=int)
    parser.add_argument('--num-samples', '--num_samples', type=int)
    parser.add_argument('--workers', '--loader_num_workers', type=int, default=None)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--limit', type=int, help='Development only: first N windows per area')
    parser.add_argument('--output', '--output_dir', help='Legacy: directory for summary.txt; epoch model: JSON file path (required)')
    parser.add_argument('--save_npz', action='store_true', help='With --output, save native-unit prediction arrays')
    parser.add_argument('--demo-only', action='store_true', help='Epoch model only: skip split-overlap audit for test-only demonstrations')
    args = parser.parse_args(argv)
    if any(n is not None and n < 1 for n in (args.batch_size, args.num_samples, args.limit, args.threads)) or (args.workers is not None and args.workers < 0):
        parser.error('Batch/sample/thread counts must be positive; workers must be nonnegative')
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error('seed must be between 0 and 2**32 - 1')
    if args.save_npz and not args.output:
        parser.error('--save_npz requires --output')
    checkpoint = load_checkpoint(args.checkpoint)
    is_paper = checkpoint.get('format') == FORMAT
    if is_paper:
        if not args.output:
            parser.error('Epoch-model evaluation requires --output /path/to/metrics.json')
        if args.state_key != 'auto' or args.num_samples is not None or args.limit is not None or args.save_npz:
            parser.error('--state-key/--num-samples/--limit/--save_npz are legacy-only options')
    else:
        if args.demo_only:
            parser.error('--demo-only is for epoch-model checkpoints only')
        args.threads = 2 if args.threads is None else args.threads
        args.workers = 0 if args.workers is None else args.workers
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        text, reports = run_paper(args, checkpoint) if is_paper else run(args, checkpoint)
    print(text, end='', flush=True)
    return reports


if __name__ == '__main__':
    main()
