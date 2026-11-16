import copy
import io
import unittest
import torch
from taer_feedback import bounded_allocation, normalized_gap, RouteAccumulator, FeedbackController, VARIANTS
from paper_pipeline import (make_model, task_losses, absolute_outputs, objective, collect_statistics, seed_all)
from train import train_epoch


def synthetic_batch(model, batch=2):
    o, p = model.obs_len, model.pred_len
    obs, target = torch.randn(o, batch, 4) * .1, torch.randn(p, batch, 4) * .1
    rel = torch.randn_like(obs) * .01
    env = {key: torch.randn(batch, o, layer.in_features)
           for key, layer in model.env_net_chooser.data_embed.items()}
    return (obs[..., :2], target[..., :2], rel[..., :2], torch.zeros(p, batch, 2),
            torch.zeros(batch), torch.ones(batch, o+p), torch.tensor([[i, i+1] for i in range(batch)]),
            obs[..., 2:], target[..., 2:], rel[..., 2:], torch.zeros(p, batch, 2),
            torch.zeros(o, batch, 4), torch.zeros(p, batch, 4),
            torch.randn(batch, 1, o, 64, 64) * .1, torch.zeros(batch, 1, p, 64, 64), env, [])


class Equations(unittest.TestCase):
    def test_bounded_entropy_solution(self):
        u = torch.tensor([0., .05, .2], dtype=torch.float64)
        a = bounded_allocation(u, .1, .3, 1.7)
        self.assertAlmostEqual(a.sum().item(), 3, places=10)
        self.assertTrue(bool(((a >= .3) & (a <= 1.7)).all()))
        free = (a > .3000001) & (a < 1.6999999)
        values = a[free].log() - u[free] / .1
        self.assertLess(float(values.max() - values.min()), 1e-9)
        torch.testing.assert_close(bounded_allocation(torch.zeros(3), .1, .3, 1.7), torch.ones(3, dtype=torch.float64))
        torch.testing.assert_close(bounded_allocation(torch.tensor([0., 1e6]), 1e-3, .3, 1.7), torch.tensor([.3, 1.7], dtype=torch.float64))

    def test_weighted_individual_errors_not_mixture_or_batch_means(self):
        loss = torch.tensor([[[2., 9.]], [[4., 1.]], [[8., 3.]]])
        w = torch.tensor([[[.8, .2]], [[.2, .8]], [[.5, .5]]])
        a, b = RouteAccumulator(1, 2), RouteAccumulator(1, 2)
        a.add(loss, w)
        b.add(loss[:1], w[:1]); b.add(loss[1:], w[1:])
        expected = (loss.double() * w.double()).sum(0) / (w.double().sum(0) + 1e-8)
        torch.testing.assert_close(a.errors(), expected)
        torch.testing.assert_close(b.errors(), expected)

    def test_median_mad_and_degenerate_gap(self):
        gap, p = normalized_gap(torch.zeros(1, 4), torch.tensor([[0., 1., 2., 7.]]))
        torch.testing.assert_close(p, torch.tensor([[0., 0., .5, 5.5]], dtype=torch.float64), atol=1e-7, rtol=1e-7)
        _, flat = normalized_gap(torch.ones(2, 4), torch.ones(2, 4))
        self.assertEqual(float(flat.sum()), 0)

    def test_positive_descent_then_ema_and_sum_two(self):
        c = FeedbackController(window=1, ema=.5, experts=3)
        e = torch.zeros(2, 3)
        c.update([10., 10.], e, e)
        state = c.update([8., 12.], e, e)
        torch.testing.assert_close(c.progress, torch.tensor([.1, 0.], dtype=torch.float64), atol=1e-8, rtol=1e-8)
        self.assertGreater(c.alpha[0], c.alpha[1])
        self.assertAlmostEqual(float(c.alpha.sum()), 2)
        self.assertFalse(state['task_warmup'])

    def test_all_four_variants_and_roundtrip(self):
        train = torch.zeros(2, 3)
        val = torch.tensor([[0., 1., 5.], [5., 1., 0.]])
        for variant in VARIANTS:
            c = FeedbackController(experts=3, variant=variant, window=1)
            c.update([10., 10.], train, val)
            c.update([5., 10.], train, val)
            self.assertEqual(bool((c.alpha != 1).any()), variant in ('full', 'weighting_only'))
            self.assertEqual(bool(c.penalty.any()), variant in ('full', 'routing_only'))
            restored = FeedbackController(experts=3, variant=variant, window=1)
            restored.load_state_dict(c.state_dict())
            a = c.update([4., 9.], train, val)
            b = restored.update([4., 9.], train, val)
            self.assertEqual(a, b)

    def test_masked_loss(self):
        pred = torch.tensor([[[1., 2., 3., 4.]], [[99., 99., 99., 99.]]])
        target, mask = torch.zeros_like(pred), torch.tensor([[1., 0.]])
        torch.testing.assert_close(task_losses(pred, target, mask), torch.tensor([[5., 25.]]))
        experts = pred.unsqueeze(1).repeat(1, 3, 1, 1)
        torch.testing.assert_close(task_losses(experts, target, mask), torch.tensor([[[5.,5.,5.],[25.,25.,25.]]]))


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        seed_all(100)
        cls.config = dict(obs_len=8, pred_len=4, experts=3, beta_load=0., beta_reg=0., grad_clip=2.)

    def test_routing_formula_gradients_and_checkpoint(self):
        model = make_model(self.config).eval()
        x = torch.randn(4, 64)
        before, raw = model.route_weights(x)
        penalty = torch.tensor([[0., 1., 3.], [3., 1., 0.]], requires_grad=True)
        model.set_feedback([1.2, .8], penalty, .5, .9)
        after, raw_after = model.route_weights(x)
        torch.testing.assert_close(after, ((raw_after - .5 * penalty.detach()) / .9).softmax(-1))
        self.assertFalse(torch.allclose(before, after))
        after[:, 0, 0].sum().backward()
        self.assertIsNone(penalty.grad)
        self.assertGreater(float(model.net_chooser[-1].weight.grad.abs().sum()), 0)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer); buffer.seek(0)
        restored = make_model(self.config).eval()
        restored.load_state_dict(torch.load(buffer, map_location='cpu'))
        torch.testing.assert_close(after, restored.route_weights(x)[0])

    def test_real_network_forward_backward_and_frozen_statistics(self):
        from paper_pipeline import unpack_batch
        model = make_model(self.config)
        batch = synthetic_batch(model)
        inputs, target, mask = unpack_batch(batch, 'cpu')
        result = model(*inputs)
        pred, experts = absolute_outputs(result, inputs[0][-1])
        self.assertEqual(tuple(pred.shape), (4, 2, 4))
        # Eq. (8) is also true in absolute-coordinate output space.
        torch.testing.assert_close(pred[..., :2], torch.einsum('bk,pkbd->pbd', result['weights'][:, 0], experts[..., :2]))
        loss, _ = objective(model, result, pred, target, mask)
        loss.backward()
        for expert in model.gs:
            self.assertGreater(float(expert.hidden2pos.weight.grad.abs().sum()), 0)
        for router in (model.net_chooser, model.intensity_router):
            self.assertGreater(float(router[-1].weight.grad.abs().sum()), 0)
        snapshot = {k: v.clone() for k,v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        stats = collect_statistics(model, [batch])
        self.assertTrue(model.training)
        torch.testing.assert_close(rng, torch.get_rng_state())
        for key, value in snapshot.items():
            torch.testing.assert_close(value, model.state_dict()[key])
        self.assertEqual(stats['count'], 2)
        self.assertEqual(tuple(stats['expert_error'].shape), (2, 3))
        c = FeedbackController(experts=3, window=1)
        c.update(stats['task_loss'], stats['expert_error'] * .5, stats['expert_error'])
        c.apply(model)
        before = model.net_chooser[-1].weight.detach().clone()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        train_epoch(model, [batch], opt, self.config, 'cpu')
        self.assertFalse(torch.equal(before, model.net_chooser[-1].weight))

    def test_feedback_affects_predictions_and_resume_training(self):
        from paper_pipeline import unpack_batch
        seed_all(37)
        m = make_model(self.config).eval()
        batch = synthetic_batch(m)
        inputs, target, mask = unpack_batch(batch, 'cpu')
        first = m(*inputs)['relative'].detach()
        m.set_feedback([1.,1.], [[0.,2.,4.],[4.,2.,0.]], .5, 1.)
        second = m(*inputs)['relative'].detach()
        self.assertFalse(torch.allclose(first, second))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        train_epoch(m, [batch], opt, self.config, 'cpu')
        state, optim_state = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict())
        rng = torch.get_rng_state().clone()
        train_epoch(m, [batch], opt, self.config, 'cpu')
        resumed = make_model(self.config)
        resumed.load_state_dict(state)
        resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
        resumed_opt.load_state_dict(optim_state)
        torch.set_rng_state(rng)
        train_epoch(resumed, [batch], resumed_opt, self.config, 'cpu')
        for key, val in m.state_dict().items():
            torch.testing.assert_close(val, resumed.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
