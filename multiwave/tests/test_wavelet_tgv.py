import copy
from dataclasses import replace
import unittest
import torch

from multiwave.config import Config
from multiwave.models import SharedAmplitudeModel
from multiwave.scene import simulate
from multiwave.physics import amplitude_loss
from multiwave.reconstruct import backward_shared_data, reconstruct
from multiwave.regularization import AmplitudeTGV
from multiwave.wavelet import DWTConcatSkip
from functions.paperrepro.tgv import tgv2_terms


class WaveletTGVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = replace(Config.preset('smoke'), iterations=3, eval_every=1,
                          unet_activation='softplus', pixel_parameterization='softplus')
        cls.scene = simulate(cls.cfg)

    def test_dwt_identity_and_learning_gradient(self):
        m = DWTConcatSkip(3).double()
        x = torch.randn(1, 3, 7, 10, dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(m(x), x, atol=1e-14, rtol=1e-14)
        self.assertTrue(torch.autograd.gradcheck(m, (x,), fast_mode=True))
        m(x).square().mean().backward()
        self.assertGreater(float(m.mix.weight.grad.abs().sum()), 0)
        with torch.no_grad():
            m.mix.weight -= .1*m.mix.weight.grad
        self.assertFalse(torch.allclose(m(x), x))

    def test_same_backbone_and_initial_field(self):
        torch.manual_seed(31)
        baseline = SharedAmplitudeModel(self.cfg, 'unet_shared_amp', self.scene.input_stack)
        torch.manual_seed(31)
        wavelet = SharedAmplitudeModel(replace(self.cfg, unet_skip='dwt_concat'),
                                       'unet_shared_amp', self.scene.input_stack)
        for k, v in baseline.state_dict().items():
            torch.testing.assert_close(v, wavelet.state_dict()[k], rtol=0, atol=0)
        torch.testing.assert_close(baseline()[0], wavelet()[0], rtol=0, atol=0)
        for b, w in zip(baseline.net.skip_filters, wavelet.net.skip_filters):
            self.assertIsInstance(w, DWTConcatSkip)

    def test_tgv_affine_nullspace_and_gradients(self):
        y, x = torch.meshgrid(torch.arange(6, dtype=torch.float64),
                              torch.arange(7, dtype=torch.float64), indexing='ij')
        mask = torch.ones_like(x, dtype=torch.bool)
        v = torch.stack((torch.full_like(x, .2), torch.full_like(x, -.1)))
        self.assertLess(abs(float(tgv2_terms(1+.2*y-.1*x, v, mask)[0])), 1e-12)
        u = torch.randn_like(x).requires_grad_()
        v = torch.randn_like(v).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda a,b: tgv2_terms(a,b,mask)[0], (u,v), fast_mode=True))

    def test_domain_independent_of_truth_and_roi(self):
        a = AmplitudeTGV(self.cfg, self.scene)
        changed = copy.copy(self.scene)
        changed.roi = torch.zeros_like(self.scene.roi)
        changed.objects = torch.zeros_like(self.scene.objects)
        b = AmplitudeTGV(self.cfg, changed)
        torch.testing.assert_close(a.mask,b.mask)
        u = torch.randn_like(a.v[0], requires_grad=True)
        changed_u = u.detach().clone()
        changed_u[~a.mask] += 100
        torch.testing.assert_close(a.terms(u)[0], a.terms(changed_u)[0])

    def test_chunked_tgv_gradient_matches_full_loss(self):
        cfg = replace(self.cfg, tgv_weight=.01, tgv_inner_steps=2, chunk=3)
        a = self.scene.objects.detach().clone().requires_grad_()
        p = self.scene.operator.probes.detach().clone().requires_grad_()
        full = AmplitudeTGV(cfg,self.scene)
        loss = amplitude_loss(self.scene.operator(a,self.scene.train,probes=p), self.scene.measured[self.scene.train])
        (loss+cfg.tgv_weight*full.penalty(a.real[0])).backward()
        g_a,g_p = a.grad.clone(),p.grad.clone()
        a.grad = None; p.grad = None
        chunked = AmplitudeTGV(cfg,self.scene)
        backward_shared_data(a,p,self.scene,cfg,chunked)
        torch.testing.assert_close(a.grad,g_a,rtol=3e-4,atol=2e-7)
        torch.testing.assert_close(p.grad,g_p,rtol=3e-4,atol=2e-7)
        torch.testing.assert_close(chunked.v,full.v)

    def test_training_switches_and_probe_constraints(self):
        for method,skip in [('pixel_shared_amp','concat'), ('unet_shared_amp','concat'),
                            ('unet_shared_amp','dwt_concat')]:
            cfg = replace(self.cfg,unet_skip=skip,tgv_weight=.001,tgv_inner_steps=2).validate()
            r = reconstruct(cfg,self.scene,method)
            self.assertEqual(r['final']['max_object_imaginary_abs'],0)
            self.assertEqual(r['final']['max_interwavelength_object_difference'],0)
            self.assertGreater(r['final']['tgv_penalty'],0)
            self.assertGreater(float(r['tgv_state_dict']['vector'].abs().sum()),0)
            self.assertTrue(all(abs(float((abs(p)**2).sum())-.5)<1e-6 for p in r['probes']))
            if skip == 'dwt_concat':
                learned = r['state_dict']['net.skip_filters.0.mix.weight']
                self.assertGreater(float(learned.abs().sum()),0)

    def test_invalid_config(self):
        for kw in [dict(tgv_weight=-1),dict(tgv_weight=float('nan')),dict(tgv_inner_steps=0),
                   dict(tgv_eps=0),dict(tgv_weight=.01,tv_weight=.01),dict(unet_skip='wrong')]:
            with self.assertRaises(ValueError):
                replace(self.cfg,**kw).validate()


if __name__ == '__main__':
    unittest.main()
