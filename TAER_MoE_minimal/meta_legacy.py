"""Adapt legacy Best-of-K + regression fusion to trained PaperGenerator experts.

Sample K expert indices with replacement from the saved track router, select one
whole four-variable trajectory per window using the true normalized future XY,
then fit per-area nonnegative coordinate regression and intensity Ridge(alpha=1).

These checkpoints have no latent-noise layer. K=20 draws therefore contain at most
E distinct expert trajectories, not 20 independent noise-perturbed predictions.
This is an oracle evaluation protocol, not label-free forecasting. Training/model
weights and feedback buffers are kept unchanged. Only the evaluation rule changes.
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch

import meta_ensemble as base

FORMAT = 'taer-paper-expert-bestofk-meta-v1'
SELECTION = 'best_of_k_sampled_experts_by_ground_truth_track_distance'
SAMPLING = 'cpu_multinomial_per_member_area_window_v1'


def configure(bundle, num_samples=20):
    """Change the protocol, never the architecture or embedded parameter values."""
    if num_samples < 1:
        raise ValueError('num_samples must be positive')
    bundle['format'] = FORMAT
    bundle['evaluation_protocol'].update(
        base_prediction_selection=SELECTION, num_samples=int(num_samples),
        candidate_source='existing_deterministic_expert_trajectories',
        candidate_sampling=SAMPLING, sampling_seed=int(bundle['model_config']['seed']),
        sampling_with_replacement=True, router='saved_corrected_track_router',
        selected_expert_outputs='same_expert_for_longitude_latitude_pressure_wind',
        selection_distance='sum_valid_steps_l2_normalized_xy',
        latent_noise_dim=0, max_distinct_candidates=int(bundle['model_config']['experts']),
        oracle_selection=True, requires_ground_truth=True, independent_test=False,
        legacy_noise_model_reproduced=False)
    return bundle


def validate_bundle(bundle, fitted=True):
    if bundle.get('format') != FORMAT:
        raise ValueError('Expected a taer-paper-expert-bestofk-meta-v1 checkpoint')
    # Reuse structural/weight integrity checks without mislabelling the saved file.
    base.validate_bundle(dict(bundle, format=base.FORMAT), fitted=fitted)
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


def prediction_panels(bundle, data_root, split, areas, window_offsets=None, **runtime):
    p = bundle['evaluation_protocol']
    offsets = window_offsets or {}
    def select(result, last, target, mask, member, area, window_start):
        experts = result['experts_relative'].cumsum(0) + last[None, None, :, :]
        sampled = sample_experts(result['weights'][:, 0], p['num_samples'],
                                 p['sampling_seed'], member, area,
                                 offsets.get(area, 0) + window_start)
        return choose_best(experts, target, mask, sampled)
    return base.prediction_panels(bundle, data_root, split, areas,
                                  prediction_selector=select, **runtime)


def evaluate_bundle(bundle, data_root, areas=None, split='test', **runtime):
    validate_bundle(bundle)
    selected = list(bundle['area_models']) if areas is None else list(areas)
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(bundle['area_models']):
        raise ValueError('Select unique areas with saved fusion rules')
    selected = [a for a in base.AREAS if a in selected]
    reports = OrderedDict()
    for area, pred, truth, mask in prediction_panels(bundle, data_root, split, selected, **runtime):
        reports[area] = base.metrics(base.apply_rule(bundle['area_models'][area], pred), truth, mask)
    return base.report_text(reports), reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', '--checkpoint_dir')
    parser.add_argument('--checkpoint-paths', nargs='+')
    parser.add_argument('--data-root', '--data_root', required=True)
    parser.add_argument('--output', required=True, help='New self-contained ckpt.pt file')
    parser.add_argument('--fit-split', choices=('test', 'val'), default='test')
    parser.add_argument('--num-samples', '--num_samples', type=int, default=20)
    parser.add_argument('--areas', nargs='+', choices=base.AREAS)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch-size', '--batch_size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--io-workers', type=int, default=2)
    parser.add_argument('--cache-frames', type=int, default=4096)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; use a new path')
    if not args.checkpoint_dir and not args.checkpoint_paths:
        parser.error('Supply --checkpoint-dir or --checkpoint-paths')
    if min(args.num_samples, args.batch_size, args.threads) < 1 or min(args.io_workers, args.cache_frames) < 0:
        parser.error('Invalid candidate/resource settings')
    bundle = configure(base.snapshot_members(args.checkpoint_dir, args.checkpoint_paths), args.num_samples)
    validate_bundle(bundle, fitted=False)
    areas = args.areas or bundle['model_config']['areas']
    if not areas or len(set(areas)) != len(areas) or not set(areas).issubset(bundle['model_config']['areas']):
        parser.error('Select unique areas from the saved model configuration')
    areas = [a for a in base.AREAS if a in areas]
    manifest = base.split_manifest(args.data_root, areas, ['train', 'val', 'test'],
                                   policy=bundle['model_config']['split_policy'])
    bundle['evaluation_protocol'].update(meta_fit_split=args.fit_split, metric_split='test')
    bundle['fit_data'] = dict(split=args.fit_split, counts={},
        track_manifest_sha256=hashlib.sha256(json.dumps(manifest[args.fit_split], sort_keys=True).encode()).hexdigest())
    runtime = {k: getattr(args, k) for k in ('batch_size', 'threads', 'io_workers', 'cache_frames')}
    bundle['evaluation_defaults'] = runtime
    bundle['fit_metrics'] = OrderedDict()
    stream = sys.stdout
    emit = lambda line: print('[Meta] ' + line, file=stream, flush=True)
    emit('{} embedded models; {} sampled expert candidates per window; at most {} distinct trajectories.'.format(
        len(bundle['checkpoint_keys']), args.num_samples, bundle['model_config']['experts']))
    emit('Oracle Best-of-K uses true future XY. Existing weights have no latent noise.')
    emit('Fusion fit split: {}. Evaluation requires future labels even when fit split is val.'.format(args.fit_split))
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
        for area, pred, truth, mask in prediction_panels(bundle, args.data_root, args.fit_split,
                areas, device=args.device, progress=emit, **runtime):
            bundle['area_models'][area] = base.fit_rule(pred, truth, mask)
            bundle['fit_data']['counts'][area] = len(truth)
            bundle['fit_metrics'][area] = base.metrics(base.apply_rule(bundle['area_models'][area], pred), truth, mask)
    validate_bundle(bundle)
    base.atomic_save(bundle, output)
    output.chmod(0o644)
    restored = base.read_checkpoint(output)
    validate_bundle(restored)
    if restored['area_models'] != bundle['area_models']:
        raise RuntimeError('Saved fusion rules changed during serialization')
    # These are fit-split results. A test-fit run can report them directly;
    # test.py separately reruns predictions from the self-contained checkpoint.
    report = output.parent / (args.fit_split + '_fit_summary.txt')
    text = base.report_text(restored['fit_metrics'])
    report.write_text(text, encoding='utf-8')
    emit('Saved ' + str(output.resolve()))
    print(text, end='', flush=True)


if __name__ == '__main__':
    main()
