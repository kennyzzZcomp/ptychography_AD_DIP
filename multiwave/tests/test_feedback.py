from dataclasses import replace
import unittest
import torch
from multiwave.config import Config
from multiwave.scene import simulate
from multiwave.probes import PixelProbes
from multiwave.feedback import FeedbackAmplitudeModel
from multiwave.losses import poisson_loss
from multiwave.reconstruct import reconstruct


class FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = replace(Config.preset('smoke'), loss='poisson', photons_per_scan=20000,
                          iterations=3, eval_every=3, chunk=3)
        cls.scene = simulate(cls.cfg)

    def test_physical_gradient_and_initial_identity(self):
        cfg, scene = self.cfg, self.scene
        model = FeedbackAmplitudeModel(cfg, 'feedback_shared_amp', scene.input_stack)
        probes = PixelProbes(cfg)().detach()
        a = model.amplitude.clone().requires_grad_()
        loss = poisson_loss(scene.operator(model.fields(a)[0], scene.train, probes=probes),
                            scene.measured[scene.train], cfg.photons_per_scan)
        reference, = torch.autograd.grad(loss, a)
        model.prepare(scene, probes)
        inputs, direction, support = model.context
        torch.testing.assert_close(direction * model.gradient_rms, reference, rtol=5e-4, atol=1e-7)
        expected = (a.detach() - cfg.feedback_step * direction).clamp_min(0)
        torch.testing.assert_close(model()[0][0].real, expected)
        self.assertFalse(inputs.requires_grad)
        self.assertFalse(direction.requires_grad)

    def test_training_only_inputs_and_coverage(self):
        cfg, scene = self.cfg, self.scene
        model = FeedbackAmplitudeModel(cfg, 'feedback_shared_amp', scene.input_stack)
        probes = PixelProbes(cfg)().detach()
        model.prepare(scene, probes)
        before = model.context[0].clone()
        changed = replace(scene, measured=scene.measured.clone(), objects=torch.zeros_like(scene.objects),
                          roi=torch.zeros_like(scene.roi), input_stack=scene.input_stack * 0)
        changed.measured[scene.holdout] = 100
        model.prepare(changed, probes)
        torch.testing.assert_close(model.context[0], before, rtol=0, atol=0)

    def test_learning_state_reload_and_zero_phase(self):
        result = reconstruct(self.cfg, self.scene, 'feedback_shared_amp')
        self.assertGreater(float(result['state_dict']['net.head_amp.weight'].abs().sum()), 0)
        model = FeedbackAmplitudeModel(self.cfg, 'feedback_shared_amp', self.scene.input_stack)
        model.load_state_dict(result['state_dict'])
        torch.testing.assert_close(model()[0], torch.from_numpy(result['objects']))
        self.assertEqual(result['final']['max_object_imaginary_abs'], 0)
        self.assertEqual(result['final']['max_interwavelength_object_difference'], 0)
        self.assertEqual(result['training_data_gradient_passes'], 6)

    def test_identity_control_does_not_train_network(self):
        cfg = replace(self.cfg, feedback_mode='identity', probe_mode='known')
        result = reconstruct(cfg, self.scene, 'feedback_shared_amp')
        self.assertEqual(float(result['state_dict']['net.head_amp.weight'].abs().sum()), 0)
        self.assertEqual(result['final']['feedback_weight_mean'], 1)


if __name__ == '__main__':
    unittest.main()
