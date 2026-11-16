"""Shared losses, frozen-state statistics, data checks and checkpoint helpers."""
import hashlib
import inspect
import json
import random
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from TCNM.data.paper_data import TrajectoryDataset, seq_collate
from TCNM.losses import toNE
from TCNM.paper_model import PaperGenerator
from taer_feedback import RouteAccumulator

FORMAT = 'taer-paper-reconstruction-v1'
TRACK = 'bst_divi10_train_val_test_inlcude15_2023_new'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_model(config):
    return PaperGenerator(obs_len=config['obs_len'], pred_len=config['pred_len'],
                          embedding_dim=32, encoder_h_dim=64, decoder_h_dim=64,
                          mlp_dim=128, num_layers=1, noise_dim=(0,),
                          pooling_type=None, batch_norm=False, num_gs=config['experts'])


def split_manifest(root, areas, splits, policy='paper_years'):
    """Fail on storm-name overlap, duplicate track files, and missing splits.

    Filenames follow the released basin/year/storm naming convention. This does
    not certify scientific preprocessing or temporal independence of raw fields.
    """
    if policy not in ('existing', 'paper_years'):
        raise ValueError('Unknown split policy: ' + policy)
    root = Path(root)
    result, seen_names, seen_hashes = {}, {}, {}
    for split in splits:
        files = []
        for area in areas:
            directory = root / TRACK / area / split
            if not directory.is_dir():
                raise FileNotFoundError(str(directory))
            found = sorted(directory.glob('*.txt'))
            if not found:
                raise ValueError('Empty split: ' + str(directory))
            for path in found:
                match = re.match(r'^[A-Za-z]{2}(\d{4})', path.name)
                if match is None:
                    raise ValueError('Cannot verify storm year from filename: ' + path.name)
                year = int(match.group(1))
                low, high = (2017, 2023) if split == 'test' else (1950, 2016)
                if policy == 'paper_years' and not low <= year <= high:
                    raise ValueError('Storm year outside paper split: ' + str(path))
                key = area + '/' + path.name.lower()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if key in seen_names and seen_names[key] != split:
                    raise ValueError('Storm appears in multiple splits: ' + key)
                if digest in seen_hashes and seen_hashes[digest] != split:
                    raise ValueError('Duplicate track content across splits: ' + key)
                seen_names[key], seen_hashes[digest] = split, split
                files.append(dict(path=str(path.relative_to(root)), sha256=digest))
        result[split] = files
    return result


def make_dataset(root, areas, split, config):
    root = Path(root)
    paths = dict(root=str(root / TRACK), type=split, sort_files=True, observed_only=True,
                 # Checkpoints from the original reconstruction used strict features.
                 feature_mode=config.get('feature_mode', 'strict'),
                 feature_reduction=config['feature_reduction'],
                 env_root=str(root / 'all_area_correct_location_includeunder15_2023'),
                 image_root=str(root / 'all_ocean_gph500_2023'),
                 feature6_root=str(root / 'features6'), feature10_root=str(root / 'features10'))
    return TrajectoryDataset(paths, obs_len=config['obs_len'], pred_len=config['pred_len'],
                             skip=1, delim='\t', other_modal='gph', areas=areas)


def loader(dataset, config, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle,
                      num_workers=config['workers'], collate_fn=seq_collate,
                      generator=generator)


def unpack_batch(batch, device):
    obs = torch.cat([batch[0], batch[7]], dim=-1).to(device)
    rel = torch.cat([batch[2], batch[9]], dim=-1).to(device)
    target = torch.cat([batch[1], batch[8]], dim=-1).to(device)
    mask = batch[5][:, obs.shape[0]:].to(device)
    return (obs, rel, batch[6].to(device), batch[13].to(device),
            {k: v.to(device) for k, v in batch[-2].items()}), target, mask


def absolute_outputs(result, last):
    prediction = result['relative'].cumsum(0) + last.unsqueeze(0)
    experts = result['experts_relative'].cumsum(0) + last.unsqueeze(0).unsqueeze(0)
    return prediction, experts


def task_losses(prediction, target, mask):
    """Per-window losses [B,2] or [B,2,K], sum channels / valid timesteps.

    Task 0 is (longitude, latitude); task 1 is (pressure, wind). Values remain in
    the released dataset's normalized units, consistently for train and feedback.
    """
    if prediction.ndim == 4:
        error = (prediction - target.unsqueeze(1)).square()  # P,K,B,D
        groups = torch.stack([error[..., :2].sum(-1), error[..., 2:].sum(-1)], dim=-1)
        weighted = groups * mask.T[:, None, :, None]
        return (weighted.sum(0) / mask.sum(-1)[None, :, None]).permute(1, 2, 0)
    error = (prediction - target).square()
    groups = torch.stack([error[..., :2].sum(-1), error[..., 2:].sum(-1)], dim=-1)
    return (groups * mask.T[..., None]).sum(0) / mask.sum(-1)[:, None]


def check_mask(mask):
    if not torch.isfinite(mask).all() or (mask < 0).any() or (mask.sum(-1) <= 0).any():
        raise ValueError('Invalid or empty target mask')


def physical_errors(prediction, target):
    px, pv = toNE(prediction[..., :2].clone(), prediction[..., 2:].clone())
    tx, tv = toNE(target[..., :2].clone(), target[..., 2:].clone())
    dx = (px[..., 0] - tx[..., 0]) / 10 * 111 * torch.cos(torch.deg2rad(tx[..., 1] / 10))
    dy = (px[..., 1] - tx[..., 1]) / 10 * 111
    return torch.stack([(dx.square() + dy.square()).sqrt(),
                        (pv[..., 0] - tv[..., 0]).abs(), (pv[..., 1] - tv[..., 1]).abs()], dim=-1)


@torch.no_grad()
def collect_statistics(model, batches):
    """Eval mode, unchanged old feedback for the entire pass; restore model mode.

    Full panels, no batch averaging of Eq. (5) ratios. Does not update controller.
    """
    training = model.training
    device = next(model.parameters()).device
    routes = RouteAccumulator(2, model.num_gs)
    losses_sum = torch.zeros(2, dtype=torch.float64)
    physical_sum = torch.zeros(3, dtype=torch.float64)
    count, valid_steps = 0, 0.0
    # Statistics must not advance the training RNG (including DataLoader seeds).
    devices = [device.index or 0] if device.type == 'cuda' else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            for batch in batches:
                inputs, target, mask = unpack_batch(batch, device)
                check_mask(mask)
                result = model(*inputs)
                prediction, experts = absolute_outputs(result, inputs[0][-1])
                loss = task_losses(prediction, target, mask)
                individual = task_losses(experts, target, mask)
                if not torch.isfinite(loss).all():
                    raise ValueError('Nonfinite predictions during statistics')
                routes.add(individual, result['weights'])
                losses_sum += loss.double().sum(0).cpu()
                physical_sum += (physical_errors(prediction, target) * mask.T[..., None]).double().sum((0, 1)).cpu()
                count += target.shape[1]
                valid_steps += float(mask.sum())
        if count == 0:
            raise ValueError('Empty statistics loader')
        return dict(task_loss=losses_sum / count, expert_error=routes.errors(),
                    route_mass=routes.mass, count=count,
                    metrics=dict(zip(('ade_km', 'pressure_mae_hpa', 'wind_mae_ms'),
                                     (physical_sum / valid_steps).tolist())))
    finally:
        model.train(training)


def objective(model, result, prediction, target, mask, beta_load=0.0, beta_reg=0.0):
    check_mask(mask)
    per_task = task_losses(prediction, target, mask).mean(0)
    supervised = (per_task * model.task_alpha.detach()).sum()
    usage = result['weights'].mean(0)
    load_loss = (model.num_gs * usage.square().sum(-1) - 1).mean()
    reg_loss = torch.zeros((), device=prediction.device)
    if beta_reg:
        params = list(model.parameters())
        reg_loss = sum(p.square().sum() for p in params) / sum(p.numel() for p in params)
    return supervised + beta_load * load_loss + beta_reg * reg_loss, per_task


def load_checkpoint(path):
    # Only load trusted local checkpoints. Explicitly reject the legacy format.
    kwargs = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    value = torch.load(path, **kwargs)
    if value.get('format') != FORMAT:
        raise ValueError('Not a reconstructed-model checkpoint; legacy weights cannot verify this implementation')
    return value


def jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def source_hashes():
    root = Path(__file__).resolve().parent
    names = ['train.py', 'paper_pipeline.py', 'taer_feedback.py', 'TCNM/paper_model.py',
             'TCNM/data/paper_data.py', 'TCNM/models_prior_unet4to4_tsi_router.py',
             'TCNM/Unet3D_merge_tiny4to4_dynamic.py', 'TCNM/env_net_transformer_gphsplit.py',
             'TCNM/losses.py', 'checkpoint_compat.py']
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
