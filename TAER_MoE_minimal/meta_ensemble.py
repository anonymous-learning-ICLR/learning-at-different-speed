"""Self-contained prediction-level ensembles of epoch-model checkpoints.

Fusion matches the legacy seed256 recipe, but the embedded networks are
PaperGenerator models with their saved task/expert feedback buffers.
"""
import contextlib
import hashlib
import inspect
import json
import os
import re
import tempfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from paper_pipeline import (FORMAT as BASE_FORMAT, make_model, make_dataset,
                            loader, unpack_batch, absolute_outputs, seed_all,
                            split_manifest, check_mask)
from TCNM.losses import toNE

FORMAT = 'taer-paper-meta-ensemble-v1'
AREAS = ('EP', 'NA', 'NI', 'SI', 'SP', 'WP')
MODEL_SOURCES = ('TCNM/paper_model.py', 'TCNM/data/paper_data.py',
                 'TCNM/models_prior_unet4to4_tsi_router.py',
                 'TCNM/Unet3D_merge_tiny4to4_dynamic.py',
                 'TCNM/env_net_transformer_gphsplit.py', 'TCNM/losses.py')
CONFIG_KEYS = ('obs_len', 'pred_len', 'experts', 'feature_mode',
               'feature_reduction', 'split_policy', 'variant')


def read_checkpoint(path):
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    # An open descriptor remains a consistent snapshot if last.pt is replaced.
    with Path(path).open('rb') as stream:
        return torch.load(stream, **kwargs)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    os.close(handle)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def snapshot_members(directory=None, paths=None):
    candidates = list(paths or [])
    if directory is not None:
        root = Path(directory)
        candidates.extend(root.glob('checkpoint_with_model_*.pt'))
        candidates.extend(root / n for n in ('best.pt', 'last.pt') if (root / n).is_file())
    candidates = list(dict.fromkeys(Path(p).resolve() for p in candidates))
    def order(path):
        match = re.search(r'_with_model_(\d+)\.pt$', path.name)
        return (int(match[1]) if match else 10**18, path.name)
    candidates.sort(key=order)
    if not candidates:
        raise ValueError('No input checkpoints found')
    members, seen, config = OrderedDict(), {}, None
    source_root = Path(__file__).resolve().parent
    for path in candidates:
        saved = read_checkpoint(path)
        if saved.get('format') != BASE_FORMAT:
            raise ValueError('Expected epoch-model checkpoint: ' + path.name)
        current = saved['config']
        for name in MODEL_SOURCES:
            actual = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            if saved['sources'].get(name) != actual:
                raise ValueError('Model/input source mismatch: ' + name)
        if config is None:
            config = {k: current[k] for k in CONFIG_KEYS}
            config.update(seed=current['seed'], areas=list(current['areas']),
                          batch_size=current['batch_size'], threads=current['threads'], workers=0)
        if any(current.get(k) != config[k] for k in CONFIG_KEYS):
            raise ValueError('Incompatible model or input configurations: ' + path.name)
        if current['areas'] != config['areas']:
            raise ValueError('Input checkpoints must have the same configured areas')
        digest = state_digest(saved['model'])
        if digest in seen:
            members[seen[digest]]['source_files'].append(path.name)
            continue
        state = OrderedDict((n, t.detach().cpu().clone()) for n, t in saved['model'].items())
        step = int(saved.get('global_step', 0))
        key = 'step{:06d}_{}'.format(step, digest[:12])
        members[key] = dict(state_dict=state, state_sha256=digest,
                            source_files=[path.name], global_step=step, epoch=saved['epoch'],
                            seed=current['seed'])
        seen[digest] = key
    return dict(format=FORMAT, format_version=1, model_class='TCNM.paper_model.PaperGenerator',
                model_config=config, checkpoint_keys=list(members), base_checkpoints=members,
                area_models=OrderedDict(), feature_order=['longitude', 'latitude', 'pressure', 'wind'],
                prediction_space='absolute toNE; longitude/latitude in 0.1 degree; pressure hPa; wind m/s',
                evaluation_protocol=dict(base_prediction_selection='deterministic_task_expert_mixture',
                    num_samples=1, position_regression='LinearRegression(fit_intercept=False, positive=True)',
                    intensity_regression='Ridge(alpha=1.0)'))


def validate_bundle(bundle, fitted=True):
    if bundle.get('format') != FORMAT or bundle.get('format_version') != 1:
        raise ValueError('Unsupported epoch-model ensemble format')
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


def fit_rule(predictions, truth, mask):
    # Lazy import: scikit-learn is needed only for fitting, never for testing.
    from sklearn.linear_model import LinearRegression, Ridge
    keys = list(predictions)
    if not keys:
        raise ValueError('Cannot fit an empty ensemble')
    if len(keys) == 1:
        return dict(mode='single_checkpoint', keys=keys)
    stack = np.stack([predictions[key] for key in keys])
    if stack.shape[1:] != truth.shape or mask.shape != truth.shape[:2]:
        raise ValueError('Prediction/label/mask alignment mismatch')
    if not np.isfinite(stack).all() or not np.isfinite(truth).all() or not np.any(mask):
        raise ValueError('Invalid fitting data')
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('Expected a binary timestep mask')
    valid = mask.astype(bool).reshape(-1)
    info = {}
    for feature in range(4):
        x = stack[..., feature].reshape(len(keys), -1).T[valid]
        y = truth[..., feature].reshape(-1)[valid]
        reg = LinearRegression(fit_intercept=False, positive=True) if feature < 2 else Ridge(alpha=1.0)
        reg.fit(x, y)
        info['feat_' + str(feature)] = dict(coef=reg.coef_.tolist(),
            coef_dtype=str(reg.coef_.dtype), intercept=float(reg.intercept_))
    return dict(mode='meta_fusion', keys=keys, model_info=info)


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
                      threads=2, io_workers=1, cache_frames=4096, progress=None,
                      prediction_selector=None):
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
                    if prediction_selector is None:
                        prediction, _ = absolute_outputs(result, inputs[0][-1])
                    else:
                        prediction = prediction_selector(result, inputs[0][-1], target, mask,
                            key, area, (index - 1) * c['batch_size'])
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


def report_text(reports, global_name='GLOBAL'):
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
