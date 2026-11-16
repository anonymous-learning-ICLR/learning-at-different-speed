import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import train_paper
import evaluate_paper
from paper_pipeline import make_model, seed_all, load_checkpoint, split_manifest, TRACK
from test_paper import synthetic_batch


class Driver(unittest.TestCase):
    def test_actual_driver_save_resume_and_evaluate(self):
        torch.set_num_threads(2)
        seed_all(72)
        model = make_model(dict(obs_len=8, pred_len=4, experts=3))
        batch = synthetic_batch(model)
        fake_manifest = {'train': [], 'val': []}
        with tempfile.TemporaryDirectory() as tmp:
            uninterrupted, resumed = Path(tmp)/'uninterrupted', Path(tmp)/'resumed'
            common = ['--data-root', '/synthetic', '--experts', '3', '--window', '1', '--device', 'cpu', '--batch-size', '1']
            with patch.object(train_paper, 'split_manifest', return_value=fake_manifest), \
                 patch.object(train_paper, 'make_dataset', return_value=[batch]), \
                 patch.object(train_paper, 'loader', return_value=[batch]):
                train_paper.main(common + ['--output-dir', str(uninterrupted), '--epochs', '2'])
                train_paper.main(common + ['--output-dir', str(resumed), '--epochs', '1'])
                train_paper.main(common + ['--output-dir', str(resumed), '--epochs', '2',
                                          '--resume', str(resumed/'last.pt')])
            a, b = load_checkpoint(uninterrupted/'last.pt'), load_checkpoint(resumed/'last.pt')
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
            self.assertEqual(a['selected_state_val'], b['selected_state_val'])
            self.assertEqual(a['controller']['updates'], 2)
            best = load_checkpoint(resumed/'best.pt')
            with patch.object(evaluate_paper, 'make_dataset', return_value=[batch]), \
                 patch.object(evaluate_paper, 'loader', return_value=[batch]), \
                 patch.object(sys, 'argv', ['evaluate_paper.py', '--checkpoint', str(resumed/'best.pt'),
                     '--data-root', '/synthetic', '--device', 'cpu', '--demo-only',
                     '--output', str(resumed/'test.json')]):
                evaluate_paper.main()
            metrics = json.loads((resumed/'test.json').read_text())
            self.assertEqual(metrics['metrics'], best['selected_state_val']['metrics'])

    def test_split_guard_and_missing_image(self):
        from TCNM.data.paper_data import TrajectoryDataset
        with tempfile.TemporaryDirectory() as tmp:
            for split in ('train', 'val'):
                directory = Path(tmp)/TRACK/'WP'/split
                directory.mkdir(parents=True)
                (directory/'WP2010BSTSAME.txt').write_text('same storm')
            with self.assertRaises(ValueError):
                split_manifest(tmp, ['WP'], ['train','val'])
            obj = TrajectoryDataset.__new__(TrajectoryDataset)
            with self.assertRaises(FileNotFoundError):
                obj.img_read(str(Path(tmp)/'missing.npy'))


if __name__ == '__main__':
    unittest.main()
