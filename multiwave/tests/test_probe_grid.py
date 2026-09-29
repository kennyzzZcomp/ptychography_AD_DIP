import unittest
from dataclasses import replace
import torch
from multiwave.config import Config
from multiwave.probes import PixelProbes, initial_probe_field
from multiwave.scene import simulate
from multiwave.models import SharedAmplitudeModel
from multiwave.losses import poisson_loss
from multiwave.reconstruct import backward_shared_data, reconstruct


class ProbeGridTests(unittest.TestCase):
    def test_default_identity_and_coarse_shape(self):
        cfg = Config.preset('resolved')
        p = PixelProbes(cfg)
        torch.testing.assert_close(p(), initial_probe_field(cfg))
        q = PixelProbes(replace(cfg, probe_grid_size=48))
        self.assertEqual(sum(x.numel() for x in q.parameters()), 9216)
        self.assertEqual(q().shape, (2,192,192))
        self.assertEqual(float(q().imag.abs().max()), 0)
        torch.testing.assert_close(q().abs().square().sum((-1,-2)), torch.tensor([.5,.5]))
        restored = PixelProbes(replace(cfg, probe_grid_size=48))
        restored.load_state_dict(q.state_dict())
        torch.testing.assert_close(q(), restored(), rtol=0, atol=0)
        with torch.no_grad(): q.imag[0,20,20] += .1
        self.assertGreater(float(q()[0].imag.abs().max()), 0)
        self.assertEqual(float(q()[1].imag.abs().max()), 0)

    def test_physics_gradient_and_actual_training(self):
        torch.set_num_threads(2)
        cfg=replace(Config.preset('smoke'),probe_grid_size=8,photons_per_scan=20000,
                    loss='poisson',iterations=3,eval_every=3,chunk=3)
        scene=simulate(cfg)
        model=SharedAmplitudeModel(cfg,'unet_shared_amp',scene.input_stack)
        probe=PixelProbes(cfg)
        a=model()[0];p=probe()
        poisson_loss(scene.operator(a,scene.train,probes=p),scene.measured[scene.train],20000).backward()
        params=list(model.parameters())+list(probe.parameters())
        refs=[p.grad.clone() for p in params]
        model.zero_grad();probe.zero_grad()
        backward_shared_data(model()[0],probe(),scene,cfg)
        for p,g in zip(params,refs): torch.testing.assert_close(p.grad,g,rtol=5e-4,atol=3e-7)
        result=reconstruct(cfg,scene,'unet_shared_amp')
        self.assertEqual(result['probe_parameter_count'],256)
        probe.load_state_dict(result['probe_state_dict'])
        torch.testing.assert_close(probe(),torch.from_numpy(result['probes']))
        self.assertGreater(float(probe.imag.abs().max()),0)

    def test_invalid_config(self):
        cfg=Config.preset('smoke')
        for size in (-1,1,33,3.5,True):
            with self.assertRaises(ValueError): replace(cfg,probe_grid_size=size).validate()
        with self.assertRaises(ValueError): replace(cfg,probe_grid_size=8,probe_mode='known').validate()


if __name__ == '__main__': unittest.main()
