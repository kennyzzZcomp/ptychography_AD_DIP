"""Tiny CPU wiring check for optional TGV in paper run; no full simulation."""
import contextlib
import io
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from simulations import ProPtyNet_paper as entry
from simulations.ProPtyNet_paper import Cfg
from functions.paperrepro import solvers


class PaperRunTGVTests(unittest.TestCase):
    def test_cli_allows_run_amplitude_tgv(self):
        with patch.object(entry, 'run') as runner, \
             patch.object(entry.os, 'makedirs'), \
             patch.object(sys, 'argv', ['ProPtyNet_paper.py', 'run',
                 '--obj-size', '624', '--grid', '8', '--step-px', '12',
                 '--tgv-amp', '0.1']):
            entry.main()
        self.assertEqual(runner.call_args.args[0].tgv_amp, .1)

    def test_optional_amplitude_tgv_and_timing_record(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        pos = np.array([[2, 2], [2, 6], [6, 2], [6, 6]])
        yy, xx = np.mgrid[:16, :16]
        obj = ((.5 + .01 * xx) * np.exp(.02j * yy)).astype(np.complex64)
        im = torch.full((4, 8, 8), .3)
        scene = SimpleNamespace(obj=obj, probe=np.ones((8, 8), np.complex64),
            pos=pos, roi=(slice(4, 12), slice(4, 12)), Q=None,
            post=torch.tensor(pos), Imt=im, Iclt=im, S1t=torch.ones((8, 8)))

        def forward(o, p, positions, q, n, chunk=0):
            return torch.stack([torch.fft.fft2(o[a:a+n, b:b+n] * p, norm='ortho').abs().square()
                                for a, b in positions.tolist()])

        def exercise(weight):
            with tempfile.TemporaryDirectory() as td:
                cfg = Cfg(N=8, obj_size=16, grid=2, step_px=4, iters=2,
                          base_ch=2, eval_every=1, eval_size=8, device='cpu',
                          scale_cal=False, tgv_amp=weight, outdir=td)
                cfg.probe_diam_px = 6
                with patch.object(solvers, 'build_scene', return_value=scene), \
                     patch.object(solvers, 'forward_ptycho', forward), \
                     patch.object(solvers, '_report_device'), \
                     patch.object(solvers, 'evaluate', return_value={
                         'ssim_amp': .5, 'ssim_phs': .5, 'relerr': .5, 'psnr_amp': 10}), \
                     patch.object(solvers, 'seam_diag', return_value=(0., 0.)), \
                     patch.object(solvers, 'probe_relerr', return_value=0.), \
                     patch.object(solvers, '_save') as save, \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    solvers.run(cfg)
                hist = save.call_args.args[5]
                self.assertEqual(save.call_args.kwargs['tag'], 'paper')
                self.assertIn('[paper] 用时', output.getvalue())
                self.assertGreater(hist[-1]['train_elapsed_s'], 0)
                self.assertAlmostEqual(hist[-1]['mean_iteration_s'],
                                       hist[-1]['train_elapsed_s'] / 2)
                self.assertEqual((Path(td) / 'tgv_aux.npz').exists(), weight > 0)
                return hist

        off, on = exercise(0), exercise(.1)
        self.assertNotIn('tgv_amp', off[-1])
        self.assertIn('tgv_amp', on[-1])
        for row in on:
            self.assertAlmostEqual(row['loss'], row['paper_loss'] + row['tgv_weighted'], places=4)
            self.assertAlmostEqual(row['tgv_weighted'], .1 * row['tgv_amp'], places=6)


if __name__ == '__main__':
    unittest.main()
