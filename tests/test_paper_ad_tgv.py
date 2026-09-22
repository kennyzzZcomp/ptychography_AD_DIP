"""Tiny CPU checks: no paper scene or full reconstruction."""
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from simulations.ProPtyNet_paper import Cfg
from simulations.paper_overlap_colab import _row_from_npz, _tgv_settings, TGV_DEFAULTS
from functions.paperrepro import solvers_addip
from functions.paperrepro.tgv import ObjectAmplitudeTGV


def config(**kwargs):
    args = dict(N=8, obj_size=16, grid=2, step_px=4, iters=3,
                eval_every=1, eval_size=8, device="cpu", probe_mode="truth")
    args.update(kwargs)
    cfg = Cfg(**args)
    cfg.probe_diam_px = 6
    return cfg


class ADTGVTests(unittest.TestCase):
    def test_ad_wiring_truth_and_blind(self):
        pos = np.array([[2, 2], [2, 6], [6, 2], [6, 6]])
        y, x = np.mgrid[:16, :16]
        gt = ((.4 + .02*x)*np.exp(.03j*y)).astype(np.complex64)
        measured = torch.linspace(.1, 1, 4*8*8).reshape(4, 8, 8)
        scene = SimpleNamespace(obj=gt, probe=np.ones((8, 8), np.complex64),
            P0=np.ones((8, 8), np.complex64), pos=pos, post=torch.tensor(pos),
            roi=(slice(4, 12), slice(4, 12)), Q=None,
            sqrtIm=measured.sqrt(), Iclt=measured)

        def toy_forward(cfg, obj, probe, positions, q):
            return torch.stack([torch.fft.fft2(obj[a:a+8, b:b+8]*probe, norm="ortho")
                                for a, b in positions.tolist()])

        with tempfile.TemporaryDirectory() as td, \
             patch.object(solvers_addip, "build_scene", return_value=scene), \
             patch.object(solvers_addip, "_fwd", toy_forward), \
             patch.object(solvers_addip, "_save") as save, \
             patch.object(solvers_addip, "_report_device"), \
             contextlib.redirect_stdout(io.StringIO()):
            with patch.object(solvers_addip, "ObjectAmplitudeTGV") as factory:
                off = solvers_addip.run_ad(config(outdir=td))
                factory.assert_not_called()
            for probe_mode in ("truth", "pixel", "support"):
                on = solvers_addip.run_ad(config(outdir=td, tgv_amp=.1, probe_mode=probe_mode))
                self.assertEqual(on[0]["tgv_amp"], 0)
                self.assertGreater(on[-1]["tgv_amp"], 0)
                for row in on:
                    self.assertTrue(all(np.isfinite(value) for value in row.values()))
                    self.assertAlmostEqual(row["loss"], row["data_loss"] + row["tgv_weighted"], places=6)
                self.assertEqual(save.call_args.kwargs["tag"], "ad")
            again = solvers_addip.run_ad(config(outdir=td, tgv_amp=0))
            self.assertEqual(off, again)
            self.assertNotIn("tgv_amp", off[-1])
            with np.load(Path(td)/"tgv_aux.npz", allow_pickle=False) as z:
                self.assertEqual(z["vector"].shape[0], 2)
                self.assertTrue(z["mask"].any())

    def test_amplitude_prior_independent_of_phase_and_probe(self):
        torch.manual_seed(2)
        cfg = config()
        pos = np.array([[4, 4]])
        amp = (.2 + torch.rand(16, 16)).requires_grad_()
        phase = torch.randn(16, 16, requires_grad=True)
        probe = torch.randn(8, 8, dtype=torch.complex64, requires_grad=True)
        reg_ad = ObjectAmplitudeTGV(cfg, pos, "cpu")
        reg_net = ObjectAmplitudeTGV(cfg, pos, "cpu")
        a = reg_ad(torch.polar(amp, phase).abs())[0]
        b = reg_net(amp)[0]
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        da, dp, dprobe = torch.autograd.grad(a, (amp, phase, probe), allow_unused=True)
        self.assertTrue(torch.isfinite(da).all())
        self.assertLess(dp.abs().max().item(), 1e-6)
        self.assertIsNone(dprobe)

    def test_zero_complex_magnitude_gradient_is_finite(self):
        obj = torch.zeros(16, 16, dtype=torch.complex64, requires_grad=True)
        reg = ObjectAmplitudeTGV(config(), np.array([[4, 4]]), "cpu")
        loss = reg(obj.abs())[0]
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(obj.grad).all())

    def test_phase_rejected_in_ad(self):
        with self.assertRaisesRegex(ValueError, "amplitude TGV only"):
            solvers_addip.run_ad(config(tgv_phase=.1))

    def test_batch_runner_keeps_ad_opt_in(self):
        args = SimpleNamespace(**{**TGV_DEFAULTS, "tgv_amp": .1, "tgv_phase": .01})
        self.assertEqual(_tgv_settings(args, "ad")["tgv_amp"], 0)
        args.ad_tgv_amp = .2
        self.assertEqual(_tgv_settings(args, "ad")["tgv_amp"], .2)
        self.assertEqual(_tgv_settings(args, "ad")["tgv_phase"], 0)
        self.assertEqual(_tgv_settings(args, "net")["tgv_amp"], .1)

    def test_collector_labels_ad_tgv(self):
        with tempfile.TemporaryDirectory() as td:
            labels = []
            for weight in (0, .1):
                path = Path(td)/str(weight)/"ad_result.npz"
                path.parent.mkdir()
                np.savez(path, cfg=json.dumps(asdict(config(tgv_amp=weight))),
                         hist=json.dumps([{"it": 3, "loss": .02, "tgv_weighted": .001}]),
                         roi=np.array([4, 12, 4, 12]))
                row = _row_from_npz(path, Path(td))
                labels.append(row["condition_label"])
                self.assertEqual(row["tgv_amp"], weight)
            self.assertNotEqual(*labels)
            self.assertIn("TGV(lambda=0.1)", labels[1])


if __name__ == "__main__":
    unittest.main()
