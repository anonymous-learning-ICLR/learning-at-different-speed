"""Train the explicitly reconstructed Eq. (3)-(9) TC implementation."""
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


def main(args=None):
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
        if saved['sources'] != hashes or saved['manifest'] != manifest:
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


if __name__ == '__main__':
    main()
