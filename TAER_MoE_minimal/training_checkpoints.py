"""Independent snapshots, atomic writes and exact training-progress metadata."""
import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from checkpoint_compat import compatible_sources


def freeze(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: freeze(v) for k, v in value.items()}
    if isinstance(value, list):
        return [freeze(v) for v in value]
    if isinstance(value, tuple):
        return tuple(freeze(v) for v in value)
    return copy.deepcopy(value)


def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                loader=generator.get_state())


def restore_rng(state, generator=None):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
    if generator is not None:
        generator.set_state(state['loader'])


def atomic_save(state, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_text(text, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    try:
        tmp.write_text(text, encoding='utf-8')
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class CheckpointWriter:
    def __init__(self, out, best_score=float('inf'), best_record=None):
        self.out = Path(out)
        self.best_score = best_score
        self.best_record = best_record

    def save(self, state, archive=False):
        score = state['selected_state_val']['metrics']['ade_km']
        if not math.isfinite(score):
            raise ValueError('Nonfinite checkpoint validation ADE')
        improved = score < self.best_score
        step = state['global_step']
        filename = 'checkpoint_with_model_{:05d}.pt'.format(step) if archive else 'last.pt'
        if improved:
            self.best_score = score
            self.best_record = dict(metric='validation_ade_km', value=score,
                                    epoch=state['epoch'], global_step=step,
                                    batch_in_epoch=state['progress']['batch_in_epoch'],
                                    phase=state['progress']['phase'], checkpoint='best.pt')
        state = freeze(dict(state, best_score=self.best_score, best_record=self.best_record))
        if archive:
            atomic_save(state, self.out / filename)
        if improved:
            atomic_save(state, self.out / 'best.pt')
            record = self.best_record
            atomic_text('Checkpoint: best.pt\nGlobal step: {global_step}\nEpoch: {epoch}\n'
                        'Batch in epoch: {batch_in_epoch}\nPhase: {phase}\n'
                        'Validation ADE (km): {value:.6f}\n'.format(**record), self.out / 'best.txt')
        atomic_save(state, self.out / 'last.pt')
        event = dict(global_step=step, epoch=state['epoch'], phase=state['progress']['phase'],
                     checkpoint=filename, validation_ade_km=score, best=improved,
                     best_record=self.best_record)
        with (self.out / 'checkpoints.jsonl').open('a') as f:
            f.write(json.dumps(event, allow_nan=False) + '\n')
        return improved
