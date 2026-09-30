from dataclasses import replace
import unittest
import torch
from multiwave.config import Config
from multiwave.probes import BasisProbes, initial_probe_field
from multiwave.scene import simulate
from multiwave.reconstruct import reconstruct

class BasisTests(unittest.TestCase):
    def test_initial_power_gradients_and_reload(self):
        cfg=replace(Config.preset("smoke"),probe_mode="basis")
        model=BasisProbes(cfg)
        torch.testing.assert_close(model(),initial_probe_field(cfg))
        self.assertEqual(sum(p.numel() for p in model.parameters()),142)
        with torch.no_grad():
            model.amp_coeff.normal_(0,.1); model.phase_coeff.normal_(0,.1)
        p=model()
        torch.testing.assert_close(p.abs().square().sum((-1,-2)),torch.full((2,),.5))
        (p.real*torch.randn_like(p.real)+p.imag*torch.randn_like(p.imag)).sum().backward()
        for param in model.parameters():
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(float(param.grad.abs().sum()),0)
        restored=BasisProbes(cfg); restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(),model())
        before=model()[1].detach().clone()
        with torch.no_grad(): model.amp_coeff[0,1] += .1
        torch.testing.assert_close(model()[1],before)

    def test_both_methods(self):
        torch.set_num_threads(2)
        cfg=replace(Config.preset("smoke"),probe_mode="basis",iterations=2,eval_every=2,
                    loss="poisson",photons_per_scan=20000,probe_smooth_weight=.2)
        scene=simulate(cfg)
        for method in ("pixel_shared_amp","unet_shared_amp"):
            result=reconstruct(cfg,scene,method)
            self.assertEqual(result["probe_parameter_count"],142)
            self.assertIsNone(result["probe_interpolation"])
            self.assertTrue(torch.isfinite(torch.tensor(result["final"]["train_total_objective"])))

    def test_validation(self):
        base=replace(Config.preset("smoke"),probe_mode="basis")
        base.validate()
        for kw in ({"probe_grid_size":8},{"probe_amp_order":0},{"probe_phase_order":2.5}):
            with self.assertRaises(ValueError): replace(base,**kw).validate()
