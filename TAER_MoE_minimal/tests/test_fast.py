"""Check that input pipelining preserves the actual training and resume states."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_fast
import train_paper
from paper_pipeline import load_checkpoint, make_model, seed_all
from test_paper import synthetic_batch


class FastParity(unittest.TestCase):
    def same(self, a, b):
        if torch.is_tensor(a):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.same(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.same(x, y)
        else:
            self.assertEqual(a, b)

    def test_two_epochs_and_original_checkpoint_resume(self):
        torch.set_num_threads(2)
        seed_all(83)
        model = make_model(dict(obs_len=8, pred_len=4, experts=3))
        # Different deterministic samples exercise the original shuffle and RNG.
        dataset = [synthetic_batch(model, batch=1) for _ in range(2)]

        def batches(data, config, shuffle=False, generator=None):
            return DataLoader(data, batch_size=1, shuffle=shuffle, num_workers=0,
                              collate_fn=lambda samples: samples[0], generator=generator)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = ['--data-root', '/synthetic', '--experts', '3', '--window', '1',
                      '--device', 'cpu', '--batch-size', '1', '--workers', '0']
            fast = ['--io-workers', '1', '--prefetch-factor', '2', '--cache-frames', '4',
                    '--pin-memory', '--log-every', '100']
            with patch.object(train_paper, 'split_manifest', return_value={'train': [], 'val': []}), \
                 patch.object(train_paper, 'make_dataset', return_value=dataset), \
                 patch.object(train_paper, 'loader', batches):
                train_paper.main(common + ['--epochs', '2', '--output-dir', str(root/'original')])
                train_fast.main(common + fast + ['--epochs', '2', '--output-dir', str(root/'fast')])
                train_paper.main(common + ['--epochs', '1', '--output-dir', str(root/'resumed')])
                train_fast.main(common + fast + ['--epochs', '2', '--output-dir', str(root/'resumed'),
                    '--resume', str(root/'resumed'/'last.pt')])
            expected = load_checkpoint(root/'original'/'last.pt')
            for name in ('fast', 'resumed'):
                actual = load_checkpoint(root/name/'last.pt')
                self.assertEqual(expected.keys(), actual.keys())
                for key in expected:
                    if key != 'config':
                        self.same(expected[key], actual[key])
                a, b = expected['config'].copy(), actual['config'].copy()
                for key in ('output_dir', 'resume'):
                    a.pop(key); b.pop(key)
                self.same(a, b)
                self.assertTrue((root/name/'out.log').is_file())


if __name__ == '__main__':
    unittest.main()
