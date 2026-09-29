import unittest
from dataclasses import replace
import torch
from multiwave.config import Config
from multiwave.scene import simulate
from multiwave.models import SharedAmplitudeModel
from multiwave.probes import PixelProbes, probe_smoothness
from multiwave.losses import poisson_loss
from multiwave.reconstruct import backward_shared_data, reconstruct


class ProbeSmoothTests(unittest.TestCase):
    def test_complex_gradient_and_invariances(self):
        torch.manual_seed(5)
        p = torch.randn(2, 8, 8, dtype=torch.complex128, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(probe_smoothness, (p,)))
        factors = torch.tensor([2j, -3.], dtype=p.dtype)[:, None, None]
        torch.testing.assert_close(probe_smoothness(p*factors), probe_smoothness(p))
        self.assertEqual(float(probe_smoothness(torch.ones_like(p))), 0)
        grad, = torch.autograd.grad(probe_smoothness(p), p)
        self.assertLess(float(probe_smoothness(p-.1*grad)), float(probe_smoothness(p)))

    def test_chunked_regularized_gradient_matches_full(self):
        torch.set_num_threads(2)
        cfg = replace(Config.preset('smoke'), loss='poisson', photons_per_scan=20000,
                      probe_smooth_weight=.01, chunk=3)
        scene = simulate(cfg)
        model = SharedAmplitudeModel(cfg, 'unet_shared_amp', scene.input_stack)
        probe = PixelProbes(cfg)
        with torch.no_grad():
            probe.imag.normal_(0,.05)
        a,_,_ = model(); p=probe()
        objective = poisson_loss(scene.operator(a, scene.train, probes=p),
                                 scene.measured[scene.train], cfg.photons_per_scan)
        (objective+cfg.probe_smooth_weight*probe_smoothness(p)).backward()
        params=list(model.parameters())+list(probe.parameters())
        reference=[p.grad.clone() for p in params]
        model.zero_grad();probe.zero_grad()
        backward_shared_data(model()[0],probe(),scene,cfg)
        for p,g in zip(params,reference):
            torch.testing.assert_close(p.grad,g,rtol=5e-4,atol=3e-7)

    def test_metrics_power_and_invalid_config(self):
        cfg=replace(Config.preset('smoke'),loss='poisson',photons_per_scan=20000,
                    probe_smooth_weight=.01,iterations=2,eval_every=2)
        result=reconstruct(cfg,simulate(cfg),'unet_shared_amp')
        f=result['final']
        self.assertAlmostEqual(f['train_total_objective'],f['train_data_loss']+.01*f['probe_smooth_penalty'])
        for p in f['probes']:
            self.assertAlmostEqual(p['power'],.5,places=5)
        for weight in (-1,float('nan')):
            with self.assertRaises(ValueError): replace(cfg,probe_smooth_weight=weight).validate()
        with self.assertRaises(ValueError): replace(cfg,probe_mode='known').validate()


if __name__ == '__main__': unittest.main()
