"""Test the supplied checkpoint and print only ALL and regional error tables.

The original checkpoint format identifiers are retained for loading compatibility.
Candidate selection uses true future trajectories as specified by the checkpoint.
This file applies saved parameters only; it does not fit or export checkpoints.
"""
import argparse
import contextlib
import hashlib
import inspect
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from paper_pipeline import make_model, make_dataset, loader, unpack_batch, seed_all, check_mask
from TCNM.losses import toNE

# These serialized identifiers must match the unchanged supplied ckpt.pt.
FORMAT = 'taer-paper-expert-bestofk-meta-v1'
SELECTION = 'best_of_k_sampled_experts_by_ground_truth_track_distance'
SAMPLING = 'cpu_multinomial_per_member_area_window_v1'
AREAS = ('EP', 'NA', 'NI', 'SI', 'SP', 'WP')



def read_checkpoint(path):
    with Path(path).open('rb') as stream:
        if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('Run git lfs pull to download checkpoint weights')
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    # An open descriptor remains a consistent snapshot if last.pt is replaced.
    with Path(path).open('rb') as stream:
        return torch.load(stream, **kwargs)


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_structure(bundle, fitted=True):
    if bundle.get('format') != FORMAT or bundle.get('format_version') != 1:
        raise ValueError('Unsupported supplied checkpoint format')
    keys, members = bundle['checkpoint_keys'], bundle['base_checkpoints']
    if not keys or len(set(keys)) != len(keys) or set(keys) != set(members):
        raise ValueError('Missing or duplicate embedded model keys')
    for key in keys:
        if state_digest(members[key]['state_dict']) != members[key]['state_sha256']:
            raise ValueError('Embedded model checksum mismatch: ' + key)
    if fitted:
        rules = bundle['area_models']
        if not rules or any(area not in AREAS for area in rules):
            raise ValueError('No valid fitted area rules')
        for area, rule in rules.items():
            if rule['keys'] != keys or rule['mode'] not in ('single_checkpoint', 'meta_fusion'):
                raise ValueError('Invalid area rule: ' + area)
            if rule['mode'] == 'single_checkpoint':
                if len(keys) != 1:
                    raise ValueError('Identity fusion needs exactly one member')
                continue
            for feature in range(4):
                info = rule['model_info']['feat_' + str(feature)]
                if info['coef_dtype'] not in ('float32', 'float64'):
                    raise ValueError('Unsupported coefficient dtype')
                coef = np.asarray(info['coef'], dtype=info['coef_dtype'])
                if coef.shape != (len(keys),) or not np.isfinite(coef).all() or not np.isfinite(info['intercept']):
                    raise ValueError('Invalid fusion coefficients')


def validate_bundle(bundle, fitted=True):
    if bundle.get('format') != FORMAT:
        raise ValueError('Expected a taer-paper-expert-bestofk-meta-v1 checkpoint')
    validate_structure(bundle, fitted=fitted)
    p = bundle['evaluation_protocol']
    expected = {
        'base_prediction_selection': SELECTION, 'candidate_sampling': SAMPLING,
        'candidate_source': 'existing_deterministic_expert_trajectories',
        'sampling_with_replacement': True, 'router': 'saved_corrected_track_router',
        'selected_expert_outputs': 'same_expert_for_longitude_latitude_pressure_wind',
        'selection_distance': 'sum_valid_steps_l2_normalized_xy',
        'latent_noise_dim': 0, 'oracle_selection': True,
        'requires_ground_truth': True, 'independent_test': False,
        'legacy_noise_model_reproduced': False,
        'max_distinct_candidates': bundle['model_config']['experts'],
        'sampling_seed': bundle['model_config']['seed'],
    }
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Unsupported or inconsistent Best-of-K protocol metadata')
    if not isinstance(p.get('num_samples'), int) or p['num_samples'] < 1:
        raise ValueError('Invalid candidate count')


def sample_experts(probabilities, num_samples, seed, member, area, window_start):
    """Stable CPU draws independent of basin order, batching and I/O concurrency."""
    probabilities = probabilities.detach().to(device='cpu', dtype=torch.float32)
    if probabilities.ndim != 2 or not torch.isfinite(probabilities).all():
        raise ValueError('Expected finite [batch, experts] router probabilities')
    if (probabilities < 0).any() or (probabilities.sum(1) <= 0).any():
        raise ValueError('Invalid router probabilities')
    draws = []
    for i, row in enumerate(probabilities):
        identity = json.dumps([int(seed), member, area, int(window_start + i)], separators=(',', ':'))
        local_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little') % (2**63)
        rng = torch.Generator(device='cpu').manual_seed(local_seed)
        draws.append(torch.multinomial(row, num_samples, replacement=True, generator=rng))
    return torch.stack(draws)


def choose_best(experts_absolute, target, mask, sampled_indices):
    """Use the old summed track distance; retain the winner's full 4-D output."""
    if experts_absolute.ndim != 4 or experts_absolute.shape[-1] != 4:
        raise ValueError('Expected expert predictions [horizon, expert, batch, 4]')
    horizon, experts, batch, _ = experts_absolute.shape
    if target.shape != (horizon, batch, 4) or mask.shape != (batch, horizon):
        raise ValueError('Target/mask shape does not match expert predictions')
    if sampled_indices.ndim != 2 or sampled_indices.shape[0] != batch or sampled_indices.shape[1] < 1:
        raise ValueError('Expected nonempty [batch, candidates] indices')
    if not torch.isfinite(experts_absolute).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite predictions or labels')
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all() or (mask.sum(1) <= 0).any():
        raise ValueError('Expected nonempty binary target masks')
    sampled = sampled_indices.to(device=experts_absolute.device, dtype=torch.long)
    if (sampled < 0).any() or (sampled >= experts).any():
        raise ValueError('Sampled expert index out of range')
    batch_indices = torch.arange(batch, device=experts_absolute.device)
    candidates = experts_absolute[:, sampled.T, batch_indices, :]
    distance = ((candidates[..., :2] - target[:, None, :, :2]) ** 2).sum(3).sqrt()
    winner = (distance * mask.T[:, None, :]).sum(0).argmin(0)
    return candidates[:, winner, batch_indices, :]


def apply_rule(rule, predictions):
    values = [np.asarray(predictions[key]) for key in rule['keys']]
    if not values or any(v.shape != values[0].shape for v in values):
        raise ValueError('Member predictions have different shapes')
    shape = values[0].shape
    if len(shape) != 3 or shape[-1] != 4 or not all(np.isfinite(v).all() for v in values):
        raise ValueError('Expected finite [N, horizon, 4] predictions')
    if rule['mode'] == 'single_checkpoint':
        if len(values) != 1:
            raise ValueError('Identity rule with multiple members')
        return values[0].copy()
    if rule['mode'] != 'meta_fusion':
        raise ValueError('Unknown fusion rule')
    stack, fused = np.stack(values), np.empty(shape, dtype=np.float32)
    for feature in range(4):
        info = rule['model_info']['feat_' + str(feature)]
        coef = np.asarray(info['coef'], dtype=info['coef_dtype'])
        x = stack[..., feature].reshape(len(values), -1).T
        fused[..., feature] = (x @ coef + info['intercept']).reshape(shape[:2])
    if not np.isfinite(fused).all():
        raise ValueError('Nonfinite fused predictions')
    return fused


@torch.no_grad()
def prediction_panels(bundle, data_root, split, areas, device='cpu', batch_size=None,
                      threads=2, io_workers=1, cache_frames=4096, progress=None):
    """Read each batch once, then evaluate all frozen embedded models on it."""
    from train import ParallelSamples, PrefetchBatches, cached_inputs, unpack_batch_async
    c = dict(bundle['model_config'], workers=0)
    c['batch_size'] = batch_size or c['batch_size']
    if c['batch_size'] < 1 or threads < 1 or io_workers < 0 or cache_frames < 0:
        raise ValueError('Invalid inference resource settings')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.set_num_threads(threads)
    seed_all(c['seed'])
    models = OrderedDict()
    for key in bundle['checkpoint_keys']:
        model = make_model(c).to(device)
        model.load_state_dict(bundle['base_checkpoints'][key]['state_dict'], strict=True)
        models[key] = model.eval()
    with contextlib.ExitStack() as stack:
        stack.enter_context(cached_inputs(cache_frames))
        readers = stack.enter_context(ThreadPoolExecutor(max_workers=io_workers)) if io_workers else None
        for area in areas:
            dataset = make_dataset(data_root, [area], split, c)
            data = ParallelSamples(dataset, readers) if readers else dataset
            batches = loader(data, c)
            batches.pin_memory = device.type == 'cuda'
            if readers:
                collate = batches.collate_fn
                batches.collate_fn = lambda samples, fn=collate: fn([sample.result() for sample in samples])
                batches = PrefetchBatches(batches, 2)
            outputs = {key: [] for key in models}
            truths, masks = [], []
            if progress:
                progress('{} {}: {} windows, {} batches, {} models'.format(split, area, len(dataset), len(batches), len(models)))
            for index, batch in enumerate(batches, 1):
                transfer = unpack_batch_async if device.type == 'cuda' else unpack_batch
                inputs, target, mask = transfer(batch, device)
                check_mask(mask)
                tx, tv = toNE(target[..., :2].clone(), target[..., 2:].clone())
                truth = torch.cat([tx, tv], -1).permute(1, 0, 2).cpu().numpy()
                truths.append(truth)
                masks.append(mask.cpu().numpy())
                for key, model in models.items():
                    result = model(*inputs)
                    experts = result['experts_relative'].cumsum(0) + inputs[0][-1][None, None, :, :]
                    protocol = bundle['evaluation_protocol']
                    sampled = sample_experts(result['weights'][:, 0], protocol['num_samples'],
                        protocol['sampling_seed'], key, area, (index - 1) * c['batch_size'])
                    prediction = choose_best(experts, target, mask, sampled)
                    px, pv = toNE(prediction[..., :2].clone(), prediction[..., 2:].clone())
                    values = torch.cat([px, pv], -1).permute(1, 0, 2).cpu().numpy()
                    if not np.isfinite(values).all() or not np.isfinite(truth).all():
                        raise ValueError('Nonfinite model predictions or labels')
                    outputs[key].append(values)
                if progress and (index == 1 or index % 25 == 0 or index == len(batches)):
                    progress('{} {}: batch {}/{}'.format(split, area, index, len(batches)))
            if not truths:
                raise ValueError('No complete windows for ' + area)
            yield area, OrderedDict((key, np.concatenate(v)) for key, v in outputs.items()), np.concatenate(truths), np.concatenate(masks)


def metrics(prediction, truth, mask):
    dx = (prediction[..., 0] - truth[..., 0]) / 10 * 111 * np.cos(np.deg2rad(truth[..., 1] / 10))
    dy = (prediction[..., 1] - truth[..., 1]) / 10 * 111
    error = np.stack([np.sqrt(dx * dx + dy * dy), np.abs(prediction[..., 2] - truth[..., 2]),
                      np.abs(prediction[..., 3] - truth[..., 3])], axis=-1)
    if not np.isfinite(error).all() or mask.shape != error.shape[:2] or not np.isin(mask, [0, 1]).all():
        raise ValueError('Invalid metric inputs')
    mass = mask.sum(0, dtype=np.float64)
    if np.any(mass == 0):
        raise ValueError('No valid observations for a forecast step')
    sums = (error * mask[..., None]).sum(0, dtype=np.float64)
    steps, avg = sums / mass[:, None], sums.sum(0) / mass.sum()
    return dict(count=len(truth), ade=float(avg[0]), fde=float(steps[-1, 0]),
                steps=steps.tolist(), avg=avg.tolist(), sums=sums.tolist(), mass=mass.tolist())


def aggregate(reports):
    sums = np.sum([r['sums'] for r in reports.values()], axis=0)
    mass = np.sum([r['mass'] for r in reports.values()], axis=0)
    steps, avg = sums / mass[:, None], sums.sum(0) / mass.sum()
    return dict(count=sum(r['count'] for r in reports.values()), ade=float(avg[0]),
                fde=float(steps[-1, 0]), steps=steps.tolist(), avg=avg.tolist())


def report_text(reports, global_name='Area: ALL'):
    bar = '-' * 72
    blocks = []
    for area, report in [('GLOBAL', aggregate(reports))] + list(reports.items()):
        name = global_name if area == 'GLOBAL' else 'Area: ' + area
        lines = ['{} (N={})'.format(name, report['count']),
                 'Overall ADE (km): {:.6f}'.format(report['ade']),
                 'Overall FDE (km): {:.6f}'.format(report['fde']), bar,
                 'Step     | TDE (km)        | Press (hPa)     | Wind (m/s)', bar]
        for i, values in enumerate(report['steps'], 1):
            lines.append('Step %-3d | %-15.4f | %-15.4f | %-15.4f' % (i, *values))
        lines.extend([bar, 'AVG      | %-15.4f | %-15.4f | %-15.4f' % tuple(report['avg']), bar])
        blocks.append('\n'.join(lines))
    return '\n\n'.join(blocks) + '\n'


def evaluate_bundle(bundle, data_root, areas=None, split='test', **runtime):
    validate_bundle(bundle)
    selected = list(bundle['area_models']) if areas is None else list(areas)
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(bundle['area_models']):
        raise ValueError('Select unique areas with saved fusion rules')
    selected = [area for area in AREAS if area in selected]
    reports = OrderedDict()
    for area, pred, truth, mask in prediction_panels(bundle, data_root, split, selected, **runtime):
        reports[area] = metrics(apply_rule(bundle['area_models'][area], pred), truth, mask)
    return report_text(reports), reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', '--ckpt_path', default='ckpt.pt')
    parser.add_argument('--data-dir', '--data-root', '--data_root', dest='data_root', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--gpu-num', '--gpu_num', dest='gpu_num')
    parser.add_argument('--areas', nargs='+', choices=AREAS)
    parser.add_argument('--seed', type=int, help='Must match the saved checkpoint seed')
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--io-workers', '--workers', '--loader_num_workers', dest='io_workers', type=int)
    parser.add_argument('--cache-frames', type=int)
    parser.add_argument('--output', '--output-dir', '--output_dir', dest='output', default='./test_report')
    parser.add_argument('--progress-log', help='Optional separate log; terminal output remains tables only')
    args = parser.parse_args(argv)
    if args.gpu_num is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    if any(v is not None and v < 1 for v in (args.batch_size, args.threads)):
        parser.error('Batch size and thread count must be positive')
    if any(v is not None and v < 0 for v in (args.io_workers, args.cache_frames)):
        parser.error('Reader count and cache capacity must be nonnegative')
    output = Path(args.output)
    if output.exists() and not output.is_dir():
        parser.error('--output must name a directory')
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        bundle = read_checkpoint(args.checkpoint)
    if args.seed is not None and args.seed != bundle['model_config']['seed']:
        parser.error('--seed must match the saved checkpoint seed')
    defaults = dict(batch_size=64, threads=4, io_workers=1, cache_frames=4096)
    defaults.update(bundle.get('evaluation_defaults', {}))
    runtime = {key: defaults[key] if getattr(args, key) is None else getattr(args, key)
               for key in ('batch_size', 'threads', 'io_workers', 'cache_frames')}
    with contextlib.ExitStack() as stack:
        quiet = stack.enter_context(open(os.devnull, 'w'))
        stack.enter_context(contextlib.redirect_stdout(quiet))
        stack.enter_context(contextlib.redirect_stderr(quiet))
        progress = None
        if args.progress_log:
            path = Path(args.progress_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            log = stack.enter_context(path.open('w'))
            progress = lambda line: print(line, file=log, flush=True)
        text, reports = evaluate_bundle(bundle, args.data_root, areas=args.areas,
                                       device=args.device, progress=progress, **runtime)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.txt').write_text(text, encoding='utf-8')
    print(text, end='', flush=True)
    return reports


if __name__ == '__main__':
    main()
