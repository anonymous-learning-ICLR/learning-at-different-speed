import unittest

import torch

from meta_legacy import choose_best, sample_experts


class LegacySelection(unittest.TestCase):
    def test_matches_original_loop_and_keeps_winners_intensity(self):
        rng = torch.Generator().manual_seed(124)
        experts = torch.randn(4, 6, 3, 4, generator=rng)
        target = torch.randn(4, 3, 4, generator=rng)
        indices = torch.randint(6, (3, 20), generator=rng)
        candidates = torch.stack([torch.stack([experts[:, indices[b, k], b]
            for b in range(3)], dim=1) for k in range(20)], dim=1)
        distance = ((candidates[..., :2] - target[:, None, :, :2]) ** 2).sum(3).sqrt().sum(0)
        winner = distance.argmin(0)
        expected = torch.stack([candidates[:, winner[b], b] for b in range(3)], dim=1)
        actual = choose_best(experts, target, torch.ones(3, 4), indices)
        self.assertTrue(torch.equal(expected, actual))
        # Intensity labels do not influence which trajectory is selected.
        target[..., 2:] += 10000
        self.assertTrue(torch.equal(actual, choose_best(experts, target, torch.ones(3, 4), indices)))

    def test_only_sampled_experts_are_eligible_and_masks_are_honored(self):
        experts = torch.zeros(2, 3, 1, 4)
        experts[:, 0, :, 0] = 0  # Globally best, but absent from the samples.
        experts[:, 1, :, 0] = 2
        experts[0, 2, :, 0] = 1
        experts[1, 2, :, 0] = 100
        experts[:, 1, :, 2:] = 30
        experts[:, 2, :, 2:] = 60
        indices = torch.tensor([[1, 2, 2]])
        target = torch.zeros(2, 1, 4)
        self.assertTrue(torch.equal(choose_best(experts,target,torch.ones(1,2),indices),experts[:,1]))
        self.assertTrue(torch.equal(choose_best(experts,target,torch.tensor([[1.,0.]]),indices),experts[:,2]))
        with self.assertRaises(ValueError):
            choose_best(experts,target,torch.zeros(1,2),indices)

    def test_sampling_repeats_and_is_independent_of_batch_partition(self):
        probabilities = torch.tensor([[.1,.3,.6],[.2,.3,.5],[1.,0.,0.],[.5,.4,.1]])
        full = sample_experts(probabilities,20,200,'member1','WP',37)
        parts = torch.cat([sample_experts(probabilities[:1],20,200,'member1','WP',37),
                           sample_experts(probabilities[1:],20,200,'member1','WP',38)])
        self.assertTrue(torch.equal(full,parts))
        self.assertTrue(torch.equal(full,sample_experts(probabilities,20,200,'member1','WP',37)))
        self.assertTrue((full[2] == 0).all())
        self.assertFalse(torch.equal(full,sample_experts(probabilities,20,201,'member1','WP',37)))


if __name__ == '__main__':
    unittest.main()
