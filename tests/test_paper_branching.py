"""Tiny synthetic CPU state tests only; no paper scene or long reconstruction."""
import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from simulations.ProPtyNet_paper import Cfg
from functions.paperrepro.branching import run_branch, load_checkpoint, resume_config
from simulations.collect_paper_branches import collect


def toy_scene(cfg, device):
    rng = np.random.default_rng(5)
    m, n = cfg.obj_size, cfg.N
    obj = (rng.uniform(.2, 1., (m, m))*np.exp(1j*rng.uniform(-.5, .5, (m, m)))).astype(np.complex64)
    probe = np.ones((n, n), np.complex64)
    intensity = torch.tensor(rng.uniform(.1, 1., (4, n, n)), dtype=torch.float32, device=device)
    return SimpleNamespace(obj=obj, probe=probe, P0=probe,
                           post=torch.tensor([[0, 0], [0, 2], [2, 0], [2, 2]], device=device),
                           Q=torch.ones((n, n), dtype=torch.complex64, device=device),
                           Imt=intensity, Iclt=intensity, sqrtIm=intensity.sqrt(),
                           roi=(slice(0, m), slice(0, m)))


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='branch-test-', dir=Path(__file__).resolve().parent)
        self.root = Path(self.tmp.name)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.save_mock = patch('functions.paperrepro.branching._save')
        self.save_mock.start()

    def tearDown(self):
        self.save_mock.stop()
        torch.set_num_threads(self.threads)
        self.tmp.cleanup()

    def config(self, name, steps):
        cfg = Cfg(N=8, obj_size=16, grid=2, step_px=2, base_ch=2,
                  probe_diam_um=10000, eval_size=16, eval_every=2,
                  tgv_amp=.1, tgv_inner_steps=1, iters=steps, device='cpu',
                  outdir=str(self.root/name), checkpoint_out=str(self.root/name/'checkpoint.pt'))
        cfg.quad_sign = -1.
        return cfg

    def resume(self, parent, name, branch='continue', steps=2, **extras):
        return resume_config(argparse.Namespace(
            resume=str(parent), branch=branch, iters=steps, outdir=str(self.root/name),
            checkpoint_out=None, mode='net', **extras), Cfg)

    def test_split_equals_uninterrupted_and_branches_share_state(self):
        full = self.config('full', 4)
        prefix = self.config('prefix', 2)
        run_branch(full, toy_scene)
        run_branch(prefix, toy_scene)
        resume = self.resume(prefix.checkpoint_out, 'continued')
        run_branch(resume, lambda *_: self.fail('resume must not regenerate data'))
        uninterrupted = load_checkpoint(full.checkpoint_out)
        continued = load_checkpoint(resume.checkpoint_out)
        for key in ('object', 'probe', 'tgv_vector'):
            torch.testing.assert_close(uninterrupted[key], continued[key], rtol=0, atol=0)
        for key, value in uninterrupted['network'].items():
            torch.testing.assert_close(value, continued['network'][key], rtol=0, atol=0)
        for branch in ('A', 'B', 'C', 'D'):
            cfg = self.resume(prefix.checkpoint_out, branch, branch=branch, steps=1)
            history = run_branch(cfg, lambda *_: self.fail('must not simulate'))
            state = load_checkpoint(cfg.checkpoint_out)
            self.assertEqual(state['representation'], 'pixel' if branch in ('C', 'D') else 'net')
            self.assertEqual(history[0]['it'], 3)
            self.assertTrue(np.isfinite(history[0]['data_loss']))
            self.assertEqual(state['tgv_vector'] is None, branch in ('B', 'D'))
            self.assertTrue(state['optimizer_object']['state'])

    def test_no_overwrite_and_no_scene_override(self):
        cfg = self.config('prefix', 1)
        run_branch(cfg, toy_scene)
        with self.assertRaises(FileExistsError):
            run_branch(cfg, toy_scene)
        with self.assertRaises(ValueError):
            self.resume(cfg.checkpoint_out, 'invalid', step_px=35)
        with self.assertRaises(ValueError):
            self.resume(cfg.checkpoint_out, 'invalid_lr', lr_obj=.5)

    def test_real_exporter_and_checkpoint_survives_plot_failure(self):
        # Do not mock away the reporting interface: this caught the Colab
        # KeyError('real') that state-only tests previously missed.
        from functions.paperrepro.report import _save as real_save
        cfg = self.config('export_failure', 1)
        with patch('functions.paperrepro.branching._save', side_effect=KeyError('real')):
            with self.assertRaises(KeyError):
                run_branch(cfg, toy_scene)
        state = load_checkpoint(cfg.checkpoint_out)
        self.assertEqual(state['step'], 1)
        self.assertNotIn('real', state['history'][-1])
        for branch in ('continue', 'C'):
            follow = self.resume(cfg.checkpoint_out, 'export_'+branch, branch=branch, steps=1)
            with patch('functions.paperrepro.branching._save', real_save):
                run_branch(follow, toy_scene)
            tag = 'ad' if branch == 'C' else 'net'
            self.assertTrue((Path(follow.outdir)/(tag+'_result.png')).is_file())
            with np.load(Path(follow.outdir)/(tag+'_result.npz'), allow_pickle=False) as z:
                history = json.loads(str(z['hist']))
                self.assertIn('data_loss', history[-1])
                self.assertNotIn('real', history[-1])

    def test_reject_unsupported_schedule(self):
        cfg = self.config('bad', 1)
        cfg.lr_cosine = True
        with self.assertRaises(ValueError):
            run_branch(cfg, toy_scene)

    def test_support_is_inherited_not_added_and_cli_resume(self):
        cfg = self.config('support_prefix', 1)
        cfg.probe_mode = 'support'
        cfg.probe_diam_um = 2000.
        cfg.__post_init__()
        run_branch(cfg, toy_scene)
        follow = self.resume(cfg.checkpoint_out, 'support_C', branch='C', steps=1)
        run_branch(follow, toy_scene)
        state = load_checkpoint(follow.checkpoint_out)
        self.assertEqual(state['cfg']['probe_mode'], 'support')
        self.assertEqual(state['probe'][0, 0], 0)
        from simulations import ProPtyNet_paper as entry
        with patch('sys.argv', ['ProPtyNet_paper.py', 'net', '--resume', cfg.checkpoint_out,
                               '--branch', 'B', '--iters', '1', '--outdir', str(self.root/'cli')]), \
             patch.object(entry, 'run_net') as runner:
            entry.main()
        resumed_cfg = runner.call_args.args[0]
        self.assertEqual(resumed_cfg.tgv_amp, 0.)
        self.assertEqual(resumed_cfg.probe_mode, 'support')
        self.assertEqual(resumed_cfg.quad_sign, -1.)

    def test_collector_checks_common_parent(self):
        for label in ('continue', 'A', 'B', 'C', 'D'):
            folder = self.root/label
            folder.mkdir()
            meta = dict(branch=label, representation='net', parent_sha256='same',
                        additional_steps=1, object_lr=.005, probe_lr=.02,
                        elapsed_s=1., training_s=.5, cumulative_elapsed_s=2.)
            (folder/'branch_metadata.json').write_text(json.dumps(meta), encoding='utf-8')
            record = dict(it=1001, branch_step=1, psnr_amp=20., ssim_amp=.8, ssim_phs=.7,
                          relerr=.1, relerr_p=.2, data_loss=.001,
                          object_relative_update=.01, probe_relative_update=.02)
            np.savez(folder/'net_result.npz', hist=json.dumps([record]),
                     cfg=json.dumps(dict(probe_mode='pixel', tgv_amp=.1)))
        collect(self.root)
        self.assertTrue((self.root/'summary.csv').is_file())
        self.assertTrue((self.root/'branch_curves.png').is_file())
        with self.assertRaises(FileExistsError):
            collect(self.root)
        meta['parent_sha256'] = 'different'
        (self.root/'D/branch_metadata.json').write_text(json.dumps(meta), encoding='utf-8')
        with self.assertRaises(ValueError):
            collect(self.root)


if __name__ == '__main__':
    unittest.main()
