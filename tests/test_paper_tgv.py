"""Small CPU checks only: no training on the paper simulation."""
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
import torch.nn.functional as F

from functions.paperrepro.tgv import (
    ObjectAmplitudeTGV, ObjectPhaseTGV, phase_from_head, scan_domain, tgv2_terms,
)
from functions.paperrepro import solvers_addip
from simulations.ProPtyNet_paper import Cfg


def small_cfg(**kw):
    settings = dict(N=8, obj_size=16, grid=2, step_px=4, iters=2,
                    eval_every=1, eval_size=8, device="cpu", probe_mode="truth")
    settings.update(kw)
    cfg = Cfg(**settings)
    cfg.probe_diam_px = 6.0
    return cfg


class TGVTests(unittest.TestCase):
    def test_phase_wrapped_affine_nullspace(self):
        y, x = torch.meshgrid(torch.arange(9, dtype=torch.float64),
                              torch.arange(10, dtype=torch.float64), indexing="ij")
        phi = 3 + .2*y - .1*x
        wrapped = torch.atan2(torch.sin(phi), torch.cos(phi))
        v = torch.stack([torch.full_like(phi, .2), torch.full_like(phi, -.1)])
        mask = torch.ones_like(phi, dtype=torch.bool)
        self.assertLess(abs(tgv2_terms(wrapped, v, mask, wrap_phase=True)[0].item()), 1e-12)
        self.assertGreater(tgv2_terms(wrapped, v, mask)[0].item(), .1)

    def test_phase_periodicity_offset_and_gradcheck(self):
        torch.manual_seed(5)
        phi = (torch.randn(5, 6, dtype=torch.float64) * .2).requires_grad_()
        v = torch.randn(2, 5, 6, dtype=torch.float64, requires_grad=True)
        mask = torch.ones_like(phi, dtype=torch.bool)
        f = lambda a, b: tgv2_terms(a, b, mask, wrap_phase=True)[0]
        shifted = phi + 1.7 + 2*torch.pi*torch.randint(-3, 4, phi.shape)
        torch.testing.assert_close(f(phi, v), f(shifted, v), atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.autograd.gradcheck(f, (phi, v)))

    def test_phase_auxiliary_and_zero_head_gradients(self):
        cfg = small_cfg(tgv_phase=.01)
        reg = ObjectPhaseTGV(cfg, np.array([[4, 4]]), "cpu")
        head = torch.zeros(2, 16, 16, requires_grad=True)
        phase = phase_from_head(head)
        loss = reg(phase)[0]
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(head.grad).all())
        # Nonconstant phase has finite nonzero gradients to phase heads only.
        torch.manual_seed(8)
        c = torch.ones(16, 16)
        s = (.2 * torch.randn(16, 16)).requires_grad_()
        loss = reg(phase_from_head(torch.stack([c, s])))[0]
        loss.backward()
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertGreater(s.grad.abs().sum().item(), 0)

    def test_phase_config_validation(self):
        for bad in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                small_cfg(tgv_phase=bad)

    def test_affine_nullspace(self):
        y, x = torch.meshgrid(torch.arange(9, dtype=torch.float64),
                              torch.arange(10, dtype=torch.float64), indexing="ij")
        u = 3 + .2*y - .1*x
        v = torch.stack([torch.full_like(u, .2), torch.full_like(u, -.1)])
        mask = torch.ones_like(u, dtype=torch.bool)
        total, _, _ = tgv2_terms(u, v, mask)
        self.assertLess(abs(total.item()), 1e-12)
        self.assertGreater(tgv2_terms(u, torch.zeros_like(v), mask)[0].item(), .1)

    def test_symmetric_cross_derivative(self):
        y, x = torch.meshgrid(torch.arange(8, dtype=torch.float64),
                              torch.arange(8, dtype=torch.float64), indexing="ij")
        mask = torch.ones_like(x, dtype=torch.bool)
        # Rotation has antisymmetric gradient, so E(v) is zero.
        self.assertLess(tgv2_terms(x, torch.stack([-x, y]), mask)[2].item(), 1e-12)
        shear = tgv2_terms(x, torch.stack([x, torch.zeros_like(x)]), mask)[2]
        self.assertGreater(shear.item(), .7)

    def test_autograd_and_mask_boundary(self):
        torch.manual_seed(3)
        u = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)
        v = torch.randn(2, 5, 6, dtype=torch.float64, requires_grad=True)
        mask = torch.ones(5, 6, dtype=torch.bool)
        mask[0] = False
        self.assertTrue(torch.autograd.gradcheck(
            lambda a, b: tgv2_terms(a, b, mask)[0], (u, v)))
        baseline = tgv2_terms(u, v, mask)[0]
        altered = u.detach().clone()
        altered[0] += 1000
        torch.testing.assert_close(tgv2_terms(altered, v, mask)[0], baseline)

    def test_normalized_amp_scale_and_finite_gradients(self):
        cfg = small_cfg(tgv_amp=.001)
        positions = np.array([[2, 2], [2, 6], [6, 2], [6, 6]])
        torch.manual_seed(7)
        raw = torch.randn(16, 16, requires_grad=True)
        amp = F.softplus(raw)
        a, b = ObjectAmplitudeTGV(cfg, positions, "cpu"), ObjectAmplitudeTGV(cfg, positions, "cpu")
        reg_a = a(amp)[0]
        reg_b = b(amp * 7)[0]
        torch.testing.assert_close(reg_a, reg_b, rtol=1e-5, atol=1e-6)
        grad, = torch.autograd.grad(reg_a, raw)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(a.v).all())
        # Evaluation window does not affect the regularizer's domain.
        cfg.eval_size = 12
        other_roi, other_mask = scan_domain(cfg, positions)
        self.assertEqual(other_roi, a.roi)
        np.testing.assert_array_equal(other_mask, a.mask.numpy())

    def test_constant_amp_stays_zero(self):
        cfg = small_cfg()
        regularizer = ObjectAmplitudeTGV(cfg, np.array([[4, 4]]), "cpu")
        amp = torch.ones(16, 16, requires_grad=True)
        loss = regularizer(amp)[0]
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(amp.grad).all())
        self.assertEqual(amp.grad.abs().sum().item(), 0)

    def test_net_wiring_with_tiny_mock_scene(self):
        # Test actual run_net control flow with a 16x16 toy network and FFT.
        # No physical simulator, large U-Net, reconstruction report or GT assets.
        class TinyNet(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.e1 = torch.nn.Module()
                self.e1.f = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 1))
                self.head_amp = torch.nn.Conv2d(4, 1, 1)
                self.head_phs = torch.nn.Conv2d(4, 2, 1)

            def forward(self, x):
                x = self.e1.f(x)
                return self.head_amp(x)[0], self.head_phs(x)[0]

        rng = np.random.default_rng(0)
        gt = (1 + .1 * rng.random((16, 16))) * np.exp(1j * rng.random((16, 16)))
        meas = torch.linspace(.1, 1, 4*8*8).reshape(4, 8, 8)
        scene = SimpleNamespace(
            obj=gt.astype(np.complex64), probe=np.ones((8, 8), np.complex64),
            pos=np.array([[2, 2], [2, 6], [6, 2], [6, 6]]),
            roi=(slice(4, 12), slice(4, 12)), Q=None,
            post=torch.tensor([[2, 2], [2, 6], [6, 2], [6, 6]]),
            sqrtIm=meas.sqrt(), Iclt=meas, Imt=meas,
        )

        def toy_forward(cfg, obj, probe, post, q):
            field = obj[4:12, 4:12] * probe
            return torch.fft.fft2(field, norm="ortho").expand(len(post), -1, -1)

        with tempfile.TemporaryDirectory() as td:
            with patch.object(solvers_addip, "AddipUNet", TinyNet), \
                 patch.object(solvers_addip, "build_scene", return_value=scene), \
                 patch.object(solvers_addip, "_fwd", toy_forward), \
                 patch.object(solvers_addip, "_save"), \
                 patch.object(solvers_addip, "_report_device"), \
                 contextlib.redirect_stdout(io.StringIO()):
                off = solvers_addip.run_net(small_cfg(outdir=td))
                on = solvers_addip.run_net(small_cfg(outdir=td, tgv_amp=.01))
                phase_only = solvers_addip.run_net(small_cfg(outdir=td, tgv_phase=.01))
                both = solvers_addip.run_net(small_cfg(outdir=td, tgv_amp=.01, tgv_phase=.02))
                timed = solvers_addip.run_net(small_cfg(outdir=td, tgv_amp=.01, tgv_phase=.02,
                                                       timing_warmup=1))
                again = solvers_addip.run_net(small_cfg(outdir=td))
                scheduled = solvers_addip.run_net(small_cfg(outdir=td, tgv_amp=.01,
                                                             measurement_schedule="0:2,1:1"))
                full = solvers_addip.run_net(small_cfg(outdir=td, tgv_amp=.01,
                                                       measurement_schedule="0:1"))
                joint = solvers_addip.run_net(small_cfg(outdir=td, tgv_amp=.01,
                    measurement_schedule="0:2,1:1", input_policy="follow_measurements"))
                full_joint = solvers_addip.run_net(small_cfg(outdir=td, tgv_amp=.01,
                    measurement_schedule="0:1", input_policy="follow_measurements"))
            self.assertNotIn("tgv_amp", off[-1])
            self.assertEqual([r["active_patterns"] for r in scheduled], [1, 4])
            self.assertEqual(scheduled[-1]["training_patterns_cumulative"], 5)
            self.assertEqual(scheduled[-1]["evaluation_patterns_cumulative"], 4)
            self.assertAlmostEqual(scheduled[-1]["full_data_loss"], scheduled[-1]["data_loss"])
            self.assertEqual(full[-1]["training_patterns_cumulative"], 8)
            self.assertEqual([r["active_input_channels"] for r in joint], [1, 4])
            self.assertEqual([r["active_input_channels"] for r in scheduled], [4, 4])
            for reference, recorded in zip(on, full_joint):
                for key in reference:
                    self.assertEqual(reference[key], recorded[key])
            for reference, recorded in zip(on, full):
                for key in reference:
                    self.assertEqual(reference[key], recorded[key])
            self.assertEqual(off, again)
            self.assertEqual(both, timed)
            timing = json.loads((Path(td) / "net_timing.json").read_text())
            self.assertEqual(timing["measured_iterations"], 1)
            for stage in ("network_decode", "physics_and_data_loss", "amplitude_tgv",
                          "phase_tgv", "outer_backward", "optimizer", "evaluation_and_logging"):
                self.assertIn(stage, timing["stages"])
            self.assertIn("tgv_weighted", on[-1])
            for row in on:
                self.assertAlmostEqual(row["loss"], row["data_loss"] + row["tgv_weighted"], places=6)
                self.assertTrue(np.isfinite(row["tgv_amp"]))
            self.assertTrue((Path(td) / "tgv_aux.npz").exists())
            self.assertTrue((Path(td) / "tgv_phase_aux.npz").exists())
            self.assertNotIn("tgv_amp", phase_only[-1])
            for row in phase_only + both:
                self.assertAlmostEqual(row["loss"], row["data_loss"] +
                                       row.get("tgv_weighted", 0) + row["tgv_phase_weighted"], places=6)
                self.assertTrue(np.isfinite(row["tgv_phase"]))

if __name__ == "__main__":
    unittest.main()
