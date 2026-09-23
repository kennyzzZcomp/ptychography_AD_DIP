"""Tiny synthetic wiring tests only; no full simulation or GPU training."""
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
from simulations.run_paper_progressive import build_command
from functions.paperrepro import solvers_addip
from functions.paperrepro.lr_schedule import parse_lr_schedule, learning_rates


class ProgressiveTests(unittest.TestCase):
    def test_lr_schedule_boundaries_validation(self):
        stages = parse_lr_schedule("1000:8e-4:2e-2", .005, .02, 4000)
        self.assertEqual(learning_rates(.005, .02, stages, 999), (.005, .02))
        self.assertEqual(learning_rates(.005, .02, stages, 1000), (.0008, .02))
        for spec in ("bad", "0:.1:.2", "4000:.1:.2", "2:.1:.2,1:.1:.2",
                     "1:nan:.2", "1:.1:-1", "1:.1:inf"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_lr_schedule(spec, .005, .02, 4000)
        with self.assertRaisesRegex(ValueError, "cosine"):
            Cfg(lr_schedule="1000:.0008:.02", lr_cosine=True)

    def test_launcher_settings(self):
        cmd = build_command()
        opts = dict(zip(cmd[4::2], cmd[5::2]))
        self.assertEqual(opts["--measurement-schedule"], "0:3,1000:1")
        self.assertEqual(opts["--lr-schedule"], "1000:0.0008:0.02")
        self.assertEqual(opts["--input-policy"], "follow_measurements")
        cfg = Cfg(**{k[2:].replace('-', '_'): v for k, v in {
            "--grid": 10, "--step-px": 12, "--obj-size": 624,
            "--iters": 4000, "--measurement-schedule": opts["--measurement-schedule"],
            "--lr-schedule": opts["--lr-schedule"], "--tgv-amp": .1,
            "--tgv-amp-schedule": "1000:0", "--input-policy": "follow_measurements",
        }.items()})
        self.assertEqual(cfg.n_pat, 100)

    def test_net_joint_transition_retains_state_and_stops_tgv(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        pos = np.array([(r, c) for r in range(10) for c in range(10)])
        y, x = np.mgrid[:24, :24]
        gt = ((.4+.01*x)*np.exp(.02j*y)).astype(np.complex64)
        measured = torch.linspace(.1, 1, 100*8*8).reshape(100, 8, 8)
        scene = SimpleNamespace(obj=gt, probe=np.ones((8, 8), np.complex64),
            P0=np.ones((8, 8), np.complex64), pos=pos, post=torch.tensor(pos),
            roi=(slice(8, 16), slice(8, 16)), Q=None,
            sqrtIm=measured.sqrt(), Iclt=measured, Imt=measured)
        optimizers, input_counts, forward_counts = [], [], []
        original_adam = torch.optim.Adam
        original_install = solvers_addip.install_selected_input
        original_tgv_call = solvers_addip.ObjectAmplitudeTGV.__call__

        def adam(*args, **kw):
            opt = original_adam(*args, **kw)
            optimizers.append(opt)
            return opt

        def install(net):
            layer = original_install(net)
            layer.register_forward_pre_hook(lambda m, a: input_counts.append(
                len(m.indices) if m.indices is not None else a[0].shape[1]))
            return layer

        def forward(cfg, obj, probe, positions, q):
            forward_counts.append(len(positions))
            return torch.stack([torch.fft.fft2(obj[a:a+8, b:b+8]*probe, norm='ortho')
                                for a, b in positions.tolist()])

        with tempfile.TemporaryDirectory() as td:
            cfg = Cfg(N=8, obj_size=24, grid=10, step_px=1, iters=4,
                      base_ch=1, eval_every=1, eval_size=8, device='cpu',
                      probe_mode='pixel', tgv_amp=.1, tgv_amp_schedule='2:0',
                      tgv_inner_steps=1, lr_net=.005, lr_probe=.02,
                      lr_schedule='2:.0008:.02', measurement_schedule='0:3,2:1',
                      input_policy='follow_measurements', outdir=td)
            cfg.probe_diam_px = 6
            with patch.object(solvers_addip, 'build_scene', return_value=scene), \
                 patch.object(solvers_addip, '_fwd', forward), \
                 patch.object(solvers_addip, '_save'), \
                 patch.object(solvers_addip, '_report_device'), \
                 patch.object(solvers_addip, 'install_selected_input', install), \
                 patch.object(solvers_addip.ObjectAmplitudeTGV, '__call__',
                              autospec=True, side_effect=original_tgv_call) as tgv_call, \
                 patch('torch.optim.Adam', side_effect=adam), \
                 contextlib.redirect_stdout(io.StringIO()):
                hist = solvers_addip.run_net(cfg)
            self.assertEqual(tgv_call.call_count, 2)
            self.assertEqual(len(optimizers), 3)  # TGV/net/probe created once, no reset
            self.assertEqual(input_counts, [16, 16, 100, 100])
            self.assertEqual(forward_counts, [16, 100, 16, 100, 100, 100])
            self.assertEqual([r['lr_net'] for r in hist], [.005, .005, .0008, .0008])
            self.assertEqual([r['lr_probe'] for r in hist], [.02]*4)
            self.assertEqual([r['tgv_amp_active'] for r in hist], [True, True, False, False])
            self.assertEqual([r['tgv_amp_weight'] for r in hist], [.1, .1, 0, 0])
            for row in hist[2:]:
                self.assertEqual(row['loss'], row['data_loss'])
                self.assertEqual(row['tgv_weighted'], 0)
            for opt in optimizers[1:]:
                self.assertTrue(all(s['step'].item() == 4 for s in opt.state.values()))
            log = json.loads((Path(td)/'lr_schedule.json').read_text())
            self.assertFalse(log['optimizer_reset'])
            self.assertEqual([r['lr_net'] for r in log['steps']], [.005, .005, .0008, .0008])


if __name__ == '__main__':
    unittest.main()
