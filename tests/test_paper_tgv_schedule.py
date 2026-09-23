"""Tiny synthetic CPU checks only; no paper simulation."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from simulations.ProPtyNet_paper import Cfg
from functions.paperrepro import solvers_addip
from functions.paperrepro.tgv_schedule import parse_tgv_schedule, tgv_weight


class ScheduleTests(unittest.TestCase):
    def test_boundaries_and_validation(self):
        stages = parse_tgv_schedule('1000:0.01,1500:0', .1, 2000)
        self.assertEqual([tgv_weight(.1, stages, i) for i in (0, 999, 1000, 1499, 1500)],
                         [.1, .1, .01, .01, 0])
        for spec in ('bad', '0:.01', '2000:.01', '2:.01,1:.02', '1:nan', '1:-1'):
            with self.assertRaises(ValueError):
                parse_tgv_schedule(spec, .1, 2000)
        with self.assertRaises(ValueError):
            parse_tgv_schedule('1:.01', 0, 3)

    def test_continuous_net_wiring_and_unchanged_constant_schedule(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        pos = np.array([[2, 2], [2, 6], [6, 2], [6, 6]])
        y, x = np.mgrid[:16, :16]
        gt = ((.4+.02*x)*np.exp(.03j*y)).astype(np.complex64)
        measured = torch.linspace(.1, 1, 4*8*8).reshape(4, 8, 8)
        scene = SimpleNamespace(obj=gt, probe=np.ones((8, 8), np.complex64),
            P0=np.ones((8, 8), np.complex64), pos=pos, post=torch.tensor(pos),
            roi=(slice(4, 12), slice(4, 12)), Q=None,
            sqrtIm=measured.sqrt(), Iclt=measured, Imt=measured)

        def forward(cfg, obj, probe, positions, q):
            return torch.stack([torch.fft.fft2(obj[a:a+8, b:b+8]*probe, norm='ortho')
                                for a, b in positions.tolist()])

        def run(spec):
            with tempfile.TemporaryDirectory() as td:
                cfg = Cfg(N=8, obj_size=16, grid=2, step_px=4, iters=3,
                          base_ch=2, eval_every=1, eval_size=8, device='cpu',
                          probe_mode='pixel', tgv_amp=.1, tgv_amp_schedule=spec,
                          outdir=td)
                cfg.probe_diam_px = 6
                adam = torch.optim.Adam
                with patch.object(solvers_addip, 'build_scene', return_value=scene), \
                     patch.object(solvers_addip, '_fwd', forward), \
                     patch.object(solvers_addip, '_save') as save, \
                     patch.object(solvers_addip, '_report_device'), \
                     patch('torch.optim.Adam', wraps=adam) as factory, \
                     contextlib.redirect_stdout(io.StringIO()):
                    hist = solvers_addip.run_net(cfg)
                self.assertEqual(factory.call_count, 3)  # net, probe, auxiliary: no reset
                rec = save.call_args.args[1].copy()
                if spec:
                    log = json.loads((Path(td)/'tgv_amp_schedule.json').read_text())
                    self.assertFalse(log['optimizer_reset'])
                    self.assertEqual([r['tgv_amp_weight'] for r in log['steps']],
                                     [r['tgv_amp_weight'] for r in hist])
                return hist, rec

        baseline, rec = run('')
        unchanged, same = run('1:0.1')
        self.assertEqual(baseline, unchanged)
        np.testing.assert_array_equal(rec, same)
        changed, _ = run('1:0.01')
        self.assertEqual([r['tgv_amp_weight'] for r in changed], [.1, .01, .01])
        for row in changed:
            self.assertAlmostEqual(row['tgv_weighted'],
                row['tgv_amp_weight']*float(measured.mean())*row['tgv_amp'], places=7)


if __name__ == '__main__':
    unittest.main()
