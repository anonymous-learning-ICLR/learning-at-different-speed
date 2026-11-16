"""Fixed-checkpoint evaluation; no controller update, oracle selection, or fitting."""
import argparse
import json
from pathlib import Path
import torch
from paper_pipeline import (load_checkpoint, make_model, make_dataset, loader,
                            collect_statistics, seed_all, split_manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--demo-only', action='store_true', help='Skip train/val overlap audit for test-only demo; NOT a benchmark score')
    args = p.parse_args()
    saved = load_checkpoint(args.checkpoint)
    c = saved['config']
    torch.set_num_threads(c['threads'])
    seed_all(c['seed'])
    if not args.demo_only:
        split_manifest(args.data_root, c['areas'], ['train', 'val', 'test'],
                       policy=c.get('split_policy', 'paper_years'))
    model = make_model(c).to(args.device)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    # The chosen checkpoint already contains the frozen expert/task penalty.
    stats = collect_statistics(model, loader(make_dataset(args.data_root, c['areas'], 'test', c), c))
    result = dict(format=saved['format'], variant=c['variant'], seed=c['seed'],
                  epoch=saved['epoch'], demo_only=args.demo_only, metrics=stats['metrics'], count=stats['count'])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
