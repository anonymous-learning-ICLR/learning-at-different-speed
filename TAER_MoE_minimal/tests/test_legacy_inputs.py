import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from TCNM.data.paper_data import load_legacy_feature_pair
from paper_pipeline import TRACK, make_dataset, split_manifest
from train import parser


class LegacyInputs(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.timestamp = '2010090818'

    def tearDown(self):
        self.temporary.cleanup()

    def asset(self, width, value, storm='MERANTI', suffix='', year='2010'):
        directory = self.root / ('features%d' % width) / 'WP' / year / storm
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (self.timestamp + suffix + '.npy')
        np.save(path, value)
        return path

    def pair(self, year='2010', storm='MERANTI'):
        return load_legacy_feature_pair(str(self.root / 'features6'), str(self.root / 'features10'),
                                        'WP', year, storm, self.timestamp)

    def test_plain_vectors_and_original_case_priority(self):
        self.asset(6, np.arange(6, dtype=np.float32))
        self.asset(6, np.full(6, 99.), storm='Meranti')
        self.asset(10, np.arange(10, dtype=np.float64), storm='Meranti')
        a, b = self.pair()
        np.testing.assert_array_equal(a, np.arange(6))
        np.testing.assert_array_equal(b, np.arange(10))
        self.assertEqual(a.dtype, np.float32)

    def test_suffix_maps_are_ignored_and_plain_maps_are_not_pooled(self):
        for width in (6, 10):
            self.asset(width, np.full((width, 2, 3), 5.), suffix='_features%d' % width)
        a, b = self.pair()
        np.testing.assert_array_equal(a, np.zeros(6))
        np.testing.assert_array_equal(b, np.zeros(10))
        value = np.arange(36).reshape(6, 2, 3)
        self.asset(6, value)
        np.testing.assert_array_equal(self.pair()[0], value)

    def test_pre2002_and_missing_files_use_original_zeros(self):
        self.asset(6, np.ones(6), year='2001')
        with patch('TCNM.data.paper_data.np.load', side_effect=AssertionError('must not load')):
            a, b = self.pair(year='2001')
        np.testing.assert_array_equal(a, np.zeros(6))
        np.testing.assert_array_equal(b, np.zeros(10))
        a, b = self.pair(year='2012')
        np.testing.assert_array_equal(a, np.zeros(6))
        np.testing.assert_array_equal(b, np.zeros(10))

    def test_original_shared_try_and_partial_load_semantics(self):
        first = self.asset(6, np.arange(6))
        second = self.asset(10, np.arange(10))
        second.write_bytes(b'corrupt')
        a, b = self.pair()
        np.testing.assert_array_equal(a, np.arange(6))
        np.testing.assert_array_equal(b, np.zeros(10))
        first.write_bytes(b'corrupt')
        self.asset(10, np.arange(10))
        a, b = self.pair()
        np.testing.assert_array_equal(a, np.zeros(6))
        np.testing.assert_array_equal(b, np.zeros(10))

    def test_existing_split_keeps_cross_split_duplicate_checks(self):
        for split, name, text in [('train', 'WP2017BSTONE.txt', 'one'),
                                  ('val', 'WP2018BSTTWO.txt', 'two'),
                                  ('test', 'WP1950BSTTHREE.txt', 'three')]:
            directory = self.root / TRACK / 'WP' / split
            directory.mkdir(parents=True)
            (directory / name).write_text(text)
        result = split_manifest(self.root, ['WP'], ['train', 'val', 'test'], policy='existing')
        self.assertEqual([len(result[s]) for s in ['train', 'val', 'test']], [1, 1, 1])
        with self.assertRaisesRegex(ValueError, 'outside paper split'):
            split_manifest(self.root, ['WP'], ['train', 'val'], policy='paper_years')
        (self.root / TRACK / 'WP' / 'val' / 'WP2018BSTTWO.txt').write_text('one')
        with self.assertRaisesRegex(ValueError, 'Duplicate track content'):
            split_manifest(self.root, ['WP'], ['train', 'val'], policy='existing')

    def test_new_defaults_and_old_checkpoint_feature_contract(self):
        args = parser().parse_args(['--data-root', str(self.root), '--output-dir', 'unused'])
        self.assertEqual((args.feature_mode, args.feature_reduction, args.split_policy),
                         ('legacy', 'none', 'existing'))
        config = dict(obs_len=8, pred_len=4, feature_reduction='spatial_mean')
        with patch('paper_pipeline.TrajectoryDataset') as dataset:
            make_dataset(self.root, ['WP'], 'test', config)
            self.assertEqual(dataset.call_args[0][0]['feature_mode'], 'strict')
        config.update(feature_mode='legacy', feature_reduction='none')
        with patch('paper_pipeline.TrajectoryDataset') as dataset:
            make_dataset(self.root, ['WP'], 'test', config)
            self.assertEqual(dataset.call_args[0][0]['feature_mode'], 'legacy')


if __name__ == '__main__':
    unittest.main()
