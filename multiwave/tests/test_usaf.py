import unittest
from dataclasses import replace
import torch

from multiwave.config import Config, METHODS
from multiwave.models import SharedAmplitudeModel, amplitude_tv
from multiwave.scene import simulate
from multiwave.physics import amplitude_loss
from multiwave.reconstruct import field_metrics, reconstruct


class USAFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = replace(Config.preset("smoke"), spectral_mode="weighted", probe_mode="known")
        cls.scene = simulate(cls.cfg)

    def test_default_is_exactly_shared_zero_phase(self):
        self.assertEqual(self.cfg.scene, "usaf_zero_phase")
        s = self.scene
        self.assertEqual(float(s.objects.imag.abs().max()), 0)
        torch.testing.assert_close(s.objects[0], s.objects[1], rtol=0, atol=0)
        self.assertLess(float(s.objects.real.min()), .1)
        self.assertGreater(float(s.objects.real.max()), .9)
        self.assertEqual(s.markers.shape[0], 0)

    def test_only_one_trainable_amplitude_no_phase_heads(self):
        outputs = []
        for method in METHODS:
            model = SharedAmplitudeModel(self.cfg, method, self.scene.input_stack)
            self.assertFalse(any('phs' in name or 'opd' in name for name, _ in model.named_parameters()))
            obj, _, opd = model()
            self.assertEqual(float(obj.imag.detach().abs().max()), 0)
            self.assertEqual(float(opd.abs().max()), 0)
            torch.testing.assert_close(obj[0], obj[1], rtol=0, atol=0)
            amplitude_loss(self.scene.operator(obj), self.scene.measured).backward()
            grad = model.raw_amp.grad if model.net is None else model.net.head_amp.weight.grad
            self.assertGreater(float(grad.abs().sum()), 0)
            outputs.append(obj.detach())
        torch.testing.assert_close(*outputs, rtol=0, atol=0)

    def test_shared_constraint_survives_training_and_checkpoint(self):
        cfg = replace(self.cfg, iterations=8, eval_every=8)
        for method in METHODS:
            result = reconstruct(cfg, self.scene, method)
            self.assertEqual(result['final']['max_object_imaginary_abs'], 0)
            self.assertEqual(result['final']['max_interwavelength_object_difference'], 0)
            self.assertLess(result['final']['train_observed_amplitude_nrmse'], result['initial']['train_observed_amplitude_nrmse'])
            model = SharedAmplitudeModel(cfg, method, self.scene.input_stack)
            model.load_state_dict(result['state_dict'], strict=False)
            torch.testing.assert_close(model()[0].detach(), torch.tensor(result['objects']))

    def test_single_wavelength_has_same_target_and_incident_budget(self):
        cfg = replace(self.cfg, wavelengths_nm=(515.,), weights=(1.,))
        s = simulate(cfg)
        torch.testing.assert_close(s.objects[0], self.scene.objects[0], rtol=0, atol=0)
        torch.testing.assert_close(s.operator.probes[0], self.scene.operator.probes[0])
        torch.testing.assert_close(s.operator.positions, self.scene.operator.positions)
        torch.testing.assert_close(s.clean, self.scene.operator.components(self.scene.objects)[0])

    def test_amplitude_evaluation_and_prior(self):
        s = self.scene
        metrics = field_metrics(s.objects, s.optical_depth, s.opd_um, s, self.cfg)
        self.assertEqual(metrics['shared_amplitude_rmse'], 0)
        self.assertNotIn('marker_transfer_matrix', metrics)
        flat = torch.ones_like(s.objects.real, requires_grad=True)
        tv = amplitude_tv(flat, s.roi)
        self.assertLess(float(tv.detach().abs()), 1e-6)
        tv.backward()
        self.assertTrue(torch.isfinite(flat.grad).all())


if __name__ == '__main__':
    unittest.main()
