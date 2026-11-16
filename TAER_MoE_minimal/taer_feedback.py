"""Reconstructed equations (3)-(7), independent of the legacy TSI heuristic."""
from collections import deque
import copy
import torch

VARIANTS = ('mmoe', 'weighting_only', 'routing_only', 'full')


def bounded_allocation(progress, temperature, lower, upper):
    """Eq. (4): solve the scalar log(C), NOT softmax followed by clipping."""
    u = torch.as_tensor(progress, dtype=torch.float64).detach()
    if u.ndim != 1 or not torch.isfinite(u).all():
        raise ValueError('Expected a finite progress vector')
    if not (temperature > 0 and 0 < lower <= 1 <= upper):
        raise ValueError('Require tau_u>0 and 0<alpha_min<=1<=alpha_max')
    if lower == 1 or upper == 1:
        return torch.ones_like(u)
    scores = (u - u.max()) / temperature
    lo = torch.log(torch.tensor(lower, dtype=u.dtype, device=u.device)) - scores.max() - 1
    hi = torch.log(torch.tensor(upper, dtype=u.dtype, device=u.device)) - scores.min() + 1
    for _ in range(100):
        mid = (lo + hi) / 2
        allocation = torch.exp((scores + mid).clamp(max=700)).clamp(lower, upper)
        if allocation.sum() > len(u):
            hi = mid
        else:
            lo = mid
    return torch.exp((scores + (lo + hi) / 2).clamp(max=700)).clamp(lower, upper)


def normalized_gap(train_error, val_error, eps=1e-8):
    """Eq. (6); conventional midpoint median for an even number of experts."""
    gap = torch.as_tensor(val_error, dtype=torch.float64) - torch.as_tensor(train_error, dtype=torch.float64)
    if gap.ndim != 2 or not torch.isfinite(gap).all():
        raise ValueError('Expected finite [tasks, experts] errors')
    median = torch.quantile(gap, 0.5, dim=-1, keepdim=True)
    mad = torch.quantile((gap - median).abs(), 0.5, dim=-1, keepdim=True)
    return gap, ((gap - median) / (mad + eps)).clamp_min(0)


class RouteAccumulator:
    """Eq. (5): sum weighted individual losses and route mass over ALL batches."""
    def __init__(self, tasks, experts, eps=1e-8):
        self.numerator = torch.zeros(tasks, experts, dtype=torch.float64)
        self.mass = torch.zeros_like(self.numerator)
        self.eps = eps

    def add(self, losses, weights):
        # Both arrays: [sample, task, expert]. Feedback never receives gradients.
        loss = losses.detach().to(device='cpu', dtype=torch.float64)
        w = weights.detach().to(device='cpu', dtype=torch.float64)
        if loss.shape != w.shape or loss.shape[1:] != self.mass.shape:
            raise ValueError('Route loss/weight shape mismatch')
        if not torch.isfinite(loss).all() or not torch.isfinite(w).all() or (w < 0).any():
            raise ValueError('Nonfinite losses or invalid route probabilities')
        if not torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6):
            raise ValueError('Routes must sum to one per task')
        self.numerator += (loss * w).sum(0)
        self.mass += w.sum(0)

    def errors(self):
        if self.mass.sum() == 0:
            raise ValueError('Empty statistics panel')
        return self.numerator / (self.mass + self.eps)


class FeedbackController:
    def __init__(self, tasks=2, experts=6, variant='full', window=3, ema=0.8,
                 tau_u=0.05, alpha_min=0.3, alpha_max=1.7, gamma=0.1,
                 tau_r=1.0, eps=1e-8):
        if variant not in VARIANTS or tasks < 1 or experts < 1 or window < 1:
            raise ValueError('Invalid controller configuration')
        if not (0 <= ema < 1 and gamma >= 0 and tau_r > 0 and eps > 0):
            raise ValueError('Invalid smoothing/routing configuration')
        bounded_allocation(torch.zeros(tasks), tau_u, alpha_min, alpha_max)
        self.config = dict(tasks=tasks, experts=experts, variant=variant, window=window,
                           ema=ema, tau_u=tau_u, alpha_min=alpha_min, alpha_max=alpha_max,
                           gamma=gamma, tau_r=tau_r, eps=eps)
        self.history = deque(maxlen=window)
        self.progress = torch.zeros(tasks, dtype=torch.float64)
        self.alpha = torch.ones(tasks, dtype=torch.float64)
        self.penalty = torch.zeros(tasks, experts, dtype=torch.float64)
        self.updates = 0

    def update(self, val_loss, train_errors, val_errors):
        c = self.config
        val = torch.as_tensor(val_loss, dtype=torch.float64).detach().cpu()
        if val.shape != self.alpha.shape or not torch.isfinite(val).all() or (val < 0).any():
            raise ValueError('Invalid task validation losses')
        gap, penalty = normalized_gap(train_errors, val_errors, c['eps'])
        if penalty.shape != self.penalty.shape:
            raise ValueError('Expert/task count mismatch')
        # Compute everything BEFORE committing the new state.
        ready = len(self.history) == c['window']
        descent = torch.zeros_like(val)
        progress = self.progress.clone()
        if ready:
            descent = 1 - val / (torch.stack(list(self.history)).mean(0) + c['eps'])
            progress = c['ema'] * progress + (1 - c['ema']) * descent.clamp_min(0)
        alpha = bounded_allocation(progress, c['tau_u'], c['alpha_min'], c['alpha_max'])
        if c['variant'] not in ('full', 'weighting_only'):
            alpha.fill_(1)
        if c['variant'] not in ('full', 'routing_only'):
            penalty.zero_()
        self.progress, self.alpha, self.penalty = progress, alpha, penalty
        self.history.append(val.clone())
        self.updates += 1
        return dict(update=self.updates, task_warmup=not ready, descent=descent.tolist(),
                    progress=progress.tolist(), alpha=alpha.tolist(), gap=gap.tolist(),
                    penalty=penalty.tolist())

    def apply(self, model):
        model.set_feedback(self.alpha, self.penalty, self.config['gamma'], self.config['tau_r'])

    def state_dict(self):
        return dict(config=copy.deepcopy(self.config), history=[v.clone() for v in self.history],
                    progress=self.progress.clone(), alpha=self.alpha.clone(),
                    penalty=self.penalty.clone(), updates=self.updates)

    def load_state_dict(self, state):
        if state['config'] != self.config:
            raise ValueError('Controller config differs from checkpoint')
        self.history = deque([v.detach().cpu().double() for v in state['history']],
                             maxlen=self.config['window'])
        for key in ('progress', 'alpha', 'penalty'):
            setattr(self, key, state[key].detach().cpu().double().clone())
        self.updates = int(state['updates'])
