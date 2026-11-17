import contextlib
import io
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge

import test as evaluate
import meta_ensemble as meta
from paper_pipeline import make_model, seed_all, source_hashes, FORMAT
from train import parser


class Fusion(unittest.TestCase):
    def test_legacy_regressions_and_saved_coefficient_roundtrip(self):
        rng = np.random.RandomState(81)
        predictions = OrderedDict((str(i), rng.rand(23, 4, 4).astype(np.float32)) for i in range(3))
        truth = .2 * predictions['0'] + .7 * predictions['1'] + .1 * predictions['2']
        truth[..., 2:] += .8
        rule = meta.fit_rule(predictions, truth, np.ones((23, 4)))
        fused = meta.apply_rule(rule, predictions)
        for feature in range(4):
            x = np.stack(list(predictions.values()))[..., feature].reshape(3, -1).T
            y = truth[..., feature].reshape(-1)
            reg = LinearRegression(fit_intercept=False, positive=True) if feature < 2 else Ridge(alpha=1.)
            reg.fit(x, y)
            np.testing.assert_allclose(fused[..., feature].reshape(-1), reg.predict(x), rtol=1e-5, atol=1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'rule.pt'
            meta.atomic_save(rule, path)
            np.testing.assert_array_equal(fused, meta.apply_rule(meta.read_checkpoint(path), predictions))

    def test_global_is_weighted_by_samples_and_time(self):
        reports = OrderedDict()
        for area, n, error in [('EP', 1, 1.), ('WP', 4, 3.)]:
            truth = np.zeros((n, 4, 4), dtype=np.float32)
            pred = truth.copy(); pred[..., 1] = error
            reports[area] = meta.metrics(pred, truth, np.ones((n, 4)))
        result = meta.aggregate(reports)
        self.assertEqual(result['count'], 5)
        self.assertAlmostEqual(result['ade'], (1 + 4*3) / 5 * 11.1, places=5)
        self.assertAlmostEqual(result['ade'], result['avg'][0])
        self.assertTrue(meta.report_text(reports).startswith('GLOBAL (N=5)'))

    def test_masks_and_identity(self):
        x = np.ones((3, 4, 4), dtype=np.float32)
        rule = meta.fit_rule({'one': x}, x, np.ones((3,4)))
        np.testing.assert_array_equal(meta.apply_rule(rule, {'one': x}), x)
        with self.assertRaises(ValueError):
            meta.metrics(x, x, np.zeros((3,4)))
        with self.assertRaises(ValueError):
            meta.apply_rule({'mode':'single_checkpoint', 'keys':['a','b']}, {'a':x,'b':x})


class EmbeddedModels(unittest.TestCase):
    def test_original_paths_unneeded_and_feedback_buffers_preserved(self):
        torch.set_num_threads(2); seed_all(13)
        config = vars(parser().parse_args(['--data-root','/not-shipped','--output-dir','/not-shipped',
            '--device','cpu','--experts','3','--batch-size','1','--areas','WP']))
        model = make_model(config)
        data = Path(__file__).resolve().parents[1]/'data_demo'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root/'checkpoint_with_model_03000.pt'
            second = root/'best.pt'
            def save(path, step):
                torch.save(dict(format=FORMAT, model=model.state_dict(), config=config,
                                sources=source_hashes(), epoch=2, global_step=step), path)
            save(first,3000)
            model.set_feedback([1.2,.8], [[0.,2.,4.],[4.,2.,0.]], .2, 1.)
            save(second,4000); save(root/'last.pt',4000)
            bundle = meta.snapshot_members(root)
            self.assertEqual(len(bundle['checkpoint_keys']),2)
            meta.validate_bundle(bundle,fitted=False)
            panels = list(meta.prediction_panels(bundle,data,'test',['WP'],batch_size=1,threads=2,io_workers=1,cache_frames=8))
            area, predictions, truth, mask = panels[0]
            bundle['area_models'][area] = meta.fit_rule(predictions,truth,mask)
            bundle['evaluation_protocol'].update(meta_fit_split='test',metric_split='test',independent_test=False)
            expected = meta.apply_rule(bundle['area_models'][area],predictions)
            path=root/'ckpt.pt';meta.atomic_save(bundle,path)
            # All original checkpoint paths are now absent. Only ckpt.pt remains.
            first.unlink();second.unlink();(root/'last.pt').unlink()
            loaded=meta.read_checkpoint(path);meta.validate_bundle(loaded)
            panels=list(meta.prediction_panels(loaded,data,'test',['WP'],batch_size=1,threads=2,io_workers=0,cache_frames=0))
            actual=meta.apply_rule(loaded['area_models'][area],panels[0][1])
            np.testing.assert_array_equal(expected,actual)
            self.assertNotIn('data_root',loaded['model_config'])
            self.assertNotIn('output_dir',loaded['model_config'])
            # The test entry must only apply rules, never fit on evaluation labels.
            out=io.StringIO()
            with patch.object(meta,'fit_rule',side_effect=AssertionError('evaluation must not fit')), contextlib.redirect_stdout(out):
                reports=evaluate.main(['--checkpoint',str(path),'--data-root',str(data),'--areas','WP',
                    '--device','cpu','--threads','2','--batch-size','1','--output',str(root/'report')])
            self.assertEqual(reports['WP']['count'],1)
            self.assertTrue(out.getvalue().startswith('Area: ALL (N=1)'))
            self.assertNotIn('Running evaluation',out.getvalue())
            self.assertNotIn('Unet3D',out.getvalue())
            self.assertTrue((root/'report'/'summary.txt').is_file())
            self.assertEqual(out.getvalue(),(root/'report'/'summary.txt').read_text())


if __name__ == '__main__':
    unittest.main()
