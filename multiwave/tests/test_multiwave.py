import math
import unittest
from dataclasses import replace
import numpy as np
import torch

from multiwave.config import Config, METHODS
from multiwave.physics import asm_transfer, propagate_padded, amplitude_loss, MultiwaveOperator
from multiwave.scene import simulate
from multiwave.models import ObjectModel
from multiwave.reconstruct import field_metrics, reconstruct
from multiwave.run_simulation import audit_scene


class MultiwaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = Config.preset("smoke")
        cls.scene = simulate(cls.cfg)

    def test_zero_distance_is_identity_on_padded_grid(self):
        torch.manual_seed(4)
        field = torch.randn(2, 8, 8, dtype=torch.complex128)
        h = asm_transfer(16, 4e-6, 515e-9, 0, dtype=torch.complex128)
        torch.testing.assert_close(propagate_padded(field, h), field, rtol=1e-12, atol=1e-12)

    def test_plane_wave_matches_analytic_phase(self):
        n, dx, wavelength, z = 32, 4e-6, 633e-9, .2e-3
        a = torch.arange(n, dtype=torch.float64)
        wave = torch.exp(2j*math.pi*(a[:, None]+2*a[None, :])/n)
        h = asm_transfer(n, dx, wavelength, z, dtype=torch.complex128)
        frequency_squared = 5/(n*dx)**2
        analytic_phase = 2*math.pi*z/wavelength*(math.sqrt(1-wavelength**2*frequency_squared)-1)
        expected = wave*np.exp(1j*analytic_phase)
        got = torch.fft.ifft2(torch.fft.fft2(wave)*h)
        torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-10)

    def test_mixing_is_incoherent_and_invariant_to_independent_pistons(self):
        s = self.scene
        components = s.operator.components(s.objects)
        expected = (components*torch.tensor(self.cfg.weights)[:, None, None, None]).sum(0)
        torch.testing.assert_close(s.clean, expected)
        pistons = torch.exp(1j*torch.tensor([.8, -1.1]))[:, None, None]
        torch.testing.assert_close(s.operator(s.objects*pistons), s.clean, rtol=1e-5, atol=1e-8)

    def test_wavelengths_use_different_transfers_but_same_pixels(self):
        h = self.scene.operator.transfer
        self.assertGreater(float((h[0]-h[1]).abs().mean()), .01)
        self.assertEqual(h.shape[-1], self.cfg.patch_size*self.cfg.pad_factor)
        self.assertEqual(self.scene.clean.shape[-1], self.cfg.patch_size)

    def test_same_opd_produces_inverse_wavelength_phase(self):
        s = self.scene
        # Phantom phase remains within (-pi,pi), so this comparison is unambiguous.
        product = s.objects.angle()*torch.tensor(self.cfg.wavelengths_nm)[:, None, None]
        torch.testing.assert_close(product[0], product[1], rtol=2e-5, atol=1e-4)

    def test_absorption_and_dispersion_are_not_silently_shared(self):
        s = self.scene
        self.assertGreater(float((s.objects[0].abs()-s.objects[1].abs()).abs().max()), .15)
        dispersive = simulate(replace(self.cfg, scene="dispersive"))
        self.assertGreater(float((dispersive.opd_um[0]-dispersive.opd_um[1]).abs().max()), .02)
        shared = simulate(replace(self.cfg, scene="shared_complex"))
        torch.testing.assert_close(shared.objects[0], shared.objects[1])

    def test_probe_energy_and_padding_convergence(self):
        audit = audit_scene(self.cfg, self.scene)
        np.testing.assert_allclose(audit["probe_powers"], 1, rtol=1e-6)
        self.assertLess(audit["padding_relative_intensity_difference"], .002)

    def test_autograd_opd_direction_matches_finite_difference(self):
        s, cfg = self.scene, self.cfg
        op = MultiwaveOperator(cfg, s.operator.probes.to(torch.complex128), s.operator.positions)
        torch.manual_seed(7)
        direction = torch.randn(cfg.object_size, cfg.object_size, dtype=torch.float64)*.03
        wave_um = torch.tensor(cfg.wavelengths_nm, dtype=torch.float64)[:, None, None]*1e-3
        base = s.objects.to(torch.complex128)
        target = s.clean.to(torch.float64)*.98
        def fun(t):
            obj = base*torch.exp(2j*math.pi*t*direction[None]/wave_um)
            return amplitude_loss(op(obj), target)
        t = torch.tensor(.17, dtype=torch.float64, requires_grad=True)
        grad, = torch.autograd.grad(fun(t), t)
        eps = 1e-5
        finite = (fun(t.detach()+eps)-fun(t.detach()-eps))/(2*eps)
        torch.testing.assert_close(grad, finite, rtol=1e-5, atol=1e-9)

    def test_holdout_never_enters_input(self):
        s = self.scene
        self.assertEqual(len(set(s.train.tolist()) & set(s.holdout.tolist())), 0)
        pad = (self.cfg.object_size-self.cfg.patch_size)//2
        cropped = s.input_stack[0, :, pad:pad+self.cfg.patch_size, pad:pad+self.cfg.patch_size]
        torch.testing.assert_close(cropped, s.measured[s.train]/s.measured[s.train].max())
        self.assertEqual(s.input_stack.shape[1], len(s.train))

    def test_noise_reproducible_and_does_not_change_source_scale(self):
        cfg = replace(self.cfg, photons_per_scan=50000)
        a, b = simulate(cfg), simulate(cfg)
        torch.testing.assert_close(a.measured, b.measured, rtol=0, atol=0)
        torch.testing.assert_close(a.clean, self.scene.clean, rtol=0, atol=0)
        self.assertGreater(float((a.measured-a.clean).abs().max()), 0)
        self.assertLess(abs(float(a.measured.sum()/a.clean.sum())-1), .02)

    def test_initial_fields_identical_and_both_heads_receive_gradients(self):
        fields = []
        for method in METHODS:
            torch.manual_seed(3)
            model = ObjectModel(self.cfg, method, self.scene.input_stack)
            obj, _, _ = model()
            fields.append(obj.detach())
            loss = amplitude_loss(self.scene.operator(obj, self.scene.train), self.scene.measured[self.scene.train])
            loss.backward()
            amp_grad = model.net.head_amp.weight.grad if model.net else model.raw_tau.grad
            phase_grad = model.net.head_phs.weight.grad if model.net else model.raw_opd.grad
            self.assertGreater(float(amp_grad.abs().sum()), 0)
            self.assertGreater(float(phase_grad.abs().sum()), 0)
        for field in fields[1:]:
            torch.testing.assert_close(field, fields[0], rtol=0, atol=0)

    def test_truth_metrics_and_marker_matrix(self):
        s = self.scene
        metrics = field_metrics(s.objects, s.optical_depth, s.opd_um, s, self.cfg)
        self.assertLess(metrics["mean_complex_relative_error"], 1e-6)
        np.testing.assert_allclose(metrics["marker_transfer_matrix"], np.eye(2), atol=1e-6)
        # Suppressing every marker is not a successful crosstalk result.
        flat = field_metrics(s.objects, torch.ones_like(s.optical_depth)*.1, s.opd_um, s, self.cfg)
        self.assertGreater(flat["marker_identity_rmse"], .6)

    def test_three_wavelength_forward_and_model(self):
        cfg = replace(self.cfg, wavelengths_nm=(450., 532., 633.), weights=(.2, .5, .3))
        s = simulate(cfg)
        model = ObjectModel(cfg, "unet_coupled", s.input_stack)
        obj, _, opd = model()
        self.assertEqual(obj.shape, s.objects.shape)
        torch.testing.assert_close(opd[0], opd[2])
        self.assertEqual(s.operator(obj).shape, s.measured.shape)

    def test_invalid_weights_and_geometry_rejected(self):
        for cfg in (replace(self.cfg, weights=(.1, .4)),
                    replace(self.cfg, wavelengths_nm=(515., 515.)),
                    replace(self.cfg, step=16), replace(self.cfg, pixel_um=-1),
                    replace(self.cfg, holdout_fraction=0), replace(self.cfg, pad_factor=1)):
            with self.assertRaises(ValueError):
                cfg.validate()

    def test_training_and_post_update_final_fields(self):
        cfg = replace(self.cfg, iterations=6, eval_every=6)
        for method in ("pixel_coupled", "unet_coupled"):
            result = reconstruct(cfg, self.scene, method)
            self.assertLess(result["final"]["train_observed_amplitude_nrmse"],
                            result["initial"]["train_observed_amplitude_nrmse"])
            self.assertEqual(result["history"][-1]["iteration"], cfg.iterations)
            model = ObjectModel(cfg, method, self.scene.input_stack)
            # Input intentionally excluded from checkpoint; reconstruct from saved data/split.
            model.load_state_dict(result["state_dict"], strict=False)
            with torch.no_grad():
                final, _, _ = model()
            np.testing.assert_allclose(final.numpy(), result["objects"], rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
