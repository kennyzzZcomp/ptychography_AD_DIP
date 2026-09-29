import unittest
from dataclasses import replace
import torch

from multiwave.config import Config
from multiwave.scene import simulate
from multiwave.probes import PixelProbes, initial_probe_field
from multiwave.models import SharedAmplitudeModel
from multiwave.physics import amplitude_loss
from multiwave.reconstruct import reconstruct


class BlindProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = Config.preset('smoke')
        cls.scene = simulate(cls.cfg)

    def test_initialization_is_zero_phase_and_not_truth(self):
        initial = initial_probe_field(self.cfg)
        self.assertEqual(float(initial.imag.abs().max()), 0)
        torch.testing.assert_close(initial[0], initial[1], rtol=0, atol=0)
        self.assertGreater(float((initial-self.scene.operator.probes).abs().norm()), .1)
        torch.testing.assert_close(initial.abs().square().sum((-1, -2)), torch.full((2,), .5))

    def test_unweighted_sum_and_fixed_total_incident_power(self):
        s = self.scene
        torch.testing.assert_close(s.clean, s.operator.components(s.objects).sum(0))
        self.assertAlmostEqual(float(s.operator.probes.abs().square().sum()), 1, places=6)
        single = simulate(replace(self.cfg, wavelengths_nm=(515.,)))
        torch.testing.assert_close(single.objects[0], s.objects[0])
        torch.testing.assert_close(single.operator.probes[0], s.operator.probes[0]*2**.5)

    def test_override_does_not_read_true_probe_values(self):
        s = simulate(self.cfg)
        probes = PixelProbes(self.cfg)()
        expected = s.operator(s.objects, probes=probes)
        s.operator.probes.fill_(complex(float('nan'), 0))
        actual = s.operator(s.objects, probes=probes)
        torch.testing.assert_close(expected, actual)
        self.assertTrue(torch.isfinite(actual).all())

    def test_complex_pixel_gradients_and_energy_after_update(self):
        model = PixelProbes(self.cfg)
        s = self.scene
        objects = SharedAmplitudeModel(self.cfg, 'pixel_shared_amp', s.input_stack)()[0]
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg.lr_probe)
        loss = amplitude_loss(s.operator(objects, s.train, probes=model()), s.measured[s.train])
        loss.backward()
        for grad in (model.real.grad, model.imag.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertTrue((grad.abs().sum((-1, -2)) > 0).all())
        opt.step()
        torch.testing.assert_close(model().abs().square().sum((-1, -2)), torch.full((2,), .5))
        self.assertGreater(float(model().imag.detach().abs().sum()), 0)

    def test_training_saves_updated_probes_and_reload(self):
        cfg = replace(self.cfg, iterations=5, eval_every=5)
        result = reconstruct(cfg, self.scene, 'unet_shared_amp')
        self.assertLess(result['final']['train_observed_amplitude_nrmse'], result['initial']['train_observed_amplitude_nrmse'])
        self.assertGreater(float(torch.tensor(result['probes']-result['initial_probes']).abs().norm()), .001)
        model = PixelProbes(cfg)
        model.load_state_dict(result['probe_state_dict'])
        torch.testing.assert_close(model().detach(), torch.tensor(result['probes']))
        self.assertEqual(result['final']['max_object_imaginary_abs'], 0)


if __name__ == '__main__':
    unittest.main()
