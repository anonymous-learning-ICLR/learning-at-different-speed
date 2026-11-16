import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import train
import evaluate
from checkpoint_compat import ORIGINAL_EPOCH_SOURCES, compatible_sources
from paper_pipeline import make_model, seed_all, load_checkpoint, split_manifest, source_hashes, TRACK
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
            with patch.object(train, 'split_manifest', return_value=fake_manifest), \
                 patch.object(train, 'make_dataset', return_value=[batch]), \
                 patch.object(train, 'loader', return_value=[batch]):
                train.main(common + ['--output-dir', str(uninterrupted), '--epochs', '2'])
                train.main(common + ['--output-dir', str(resumed), '--epochs', '1'])
                # Simulate a checkpoint from the original, source-identical epoch driver.
                previous = load_checkpoint(resumed/'last.pt')
                previous['sources'] = dict(ORIGINAL_EPOCH_SOURCES)
                torch.save(previous, resumed/'last.pt')
                train.main(common + ['--output-dir', str(resumed), '--epochs', '2',
                                          '--resume', str(resumed/'last.pt')])
            a, b = load_checkpoint(uninterrupted/'last.pt'), load_checkpoint(resumed/'last.pt')
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
            self.assertEqual(a['selected_state_val'], b['selected_state_val'])
            self.assertEqual(a['controller']['updates'], 2)
            best = load_checkpoint(resumed/'best.pt')
            with patch.object(evaluate, 'make_dataset', return_value=[batch]), \
                 patch.object(evaluate, 'loader', return_value=[batch]), \
                 patch.object(sys, 'argv', ['evaluate.py', '--checkpoint', str(resumed/'best.pt'),
                     '--data-root', '/synthetic', '--device', 'cpu', '--demo-only',
                     '--output', str(resumed/'test.json')]):
                evaluate.main()
            metrics = json.loads((resumed/'test.json').read_text())
            self.assertEqual(metrics['metrics'], best['selected_state_val']['metrics'])

    def test_source_migration_is_limited_to_the_audited_refactor(self):
        current = source_hashes()
        self.assertTrue(compatible_sources(current, current))
        self.assertTrue(compatible_sources(ORIGINAL_EPOCH_SOURCES, current))
        altered = dict(current, **{'train.py': 'changed-training-logic'})
        self.assertFalse(compatible_sources(ORIGINAL_EPOCH_SOURCES, altered))
        old = dict(ORIGINAL_EPOCH_SOURCES, **{'train_paper.py': 'another-version'})
        self.assertFalse(compatible_sources(old, current))

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
