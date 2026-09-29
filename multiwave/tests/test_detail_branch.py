from dataclasses import replace
import unittest
import torch
from torch import nn
from multiwave.config import Config
from multiwave.models import SharedAmplitudeModel
from multiwave.scene import simulate
from multiwave.probes import PixelProbes
from multiwave.losses import poisson_loss
from multiwave.reconstruct import backward_shared_data, reconstruct


class DetailBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = replace(Config.preset('smoke'), unet_activation='softplus',
                          loss='poisson', photons_per_scan=80000, noise_seed=24,
                          iterations=4, eval_every=4, chunk=3)
        cls.scene = simulate(cls.cfg)

    def make(self, detail):
        torch.manual_seed(31)
        return SharedAmplitudeModel(replace(self.cfg, unet_detail=detail),
                                    'unet_shared_amp', self.scene.input_stack)

    def test_initial_identity_and_unchanged_backbone(self):
        baseline, detail = self.make('none'), self.make('residual')
        for k,v in baseline.state_dict().items():
            torch.testing.assert_close(v,detail.state_dict()[k],rtol=0,atol=0)
        # Nonconstant readout checks more than identical constant initialization.
        with torch.no_grad():
            baseline.net.head_amp.weight.normal_(0,.03)
            detail.net.head_amp.load_state_dict(baseline.net.head_amp.state_dict())
        torch.testing.assert_close(baseline()[0],detail()[0],rtol=0,atol=0)
        self.assertFalse(any(isinstance(m,(nn.BatchNorm2d,nn.MaxPool2d,nn.ConvTranspose2d))
                             for m in detail.net.detail_head.modules()))
        self.assertEqual(int(detail.net.e1.f[1].num_batches_tracked),1)

    def test_chunked_network_and_probe_gradients(self):
        model = self.make('residual')
        probe = PixelProbes(self.cfg)
        # Exercise gradients through inner detail layers as well as final head.
        with torch.no_grad(): model.net.detail_head.out.weight.normal_(0,.02)
        a,_,_ = model(); p=probe()
        poisson_loss(self.scene.operator(a,self.scene.train,probes=p),
                     self.scene.measured[self.scene.train],80000).backward()
        params=list(model.parameters())+list(probe.parameters())
        reference=[p.grad.clone() for p in params]
        model.zero_grad();probe.zero_grad()
        a,_,_=model();p=probe()
        backward_shared_data(a,p,self.scene,self.cfg)
        for p,g in zip(params,reference):
            torch.testing.assert_close(p.grad,g,rtol=5e-4,atol=3e-7)

    def test_learning_and_checkpoint_reload(self):
        cfg=replace(self.cfg,unet_detail='residual').validate()
        r=reconstruct(cfg,self.scene,'unet_shared_amp')
        self.assertEqual(r['unet_detail'],'residual')
        self.assertGreater(r['detail_parameter_count'],0)
        self.assertGreater(float(r['state_dict']['net.detail_head.out.weight'].abs().sum()),0)
        initial=self.make('residual')
        self.assertFalse(torch.equal(initial.net.detail_head.blocks[0].layers[0].weight,
                                    r['state_dict']['net.detail_head.blocks.0.layers.0.weight']))
        initial.load_state_dict(r['state_dict'],strict=False)
        with torch.no_grad(): actual=initial()[0]
        torch.testing.assert_close(actual,torch.from_numpy(r['objects']),rtol=1e-5,atol=1e-6)
        self.assertEqual(r['final']['max_object_imaginary_abs'],0)
        self.assertEqual(r['final']['max_interwavelength_object_difference'],0)
        self.assertLess(r['final']['train_data_loss'],r['initial']['train_data_loss'])

    def test_invalid_config(self):
        for kw in [dict(unet_detail='wrong'),dict(unet_detail='residual',scene='shared_opd')]:
            with self.assertRaises(ValueError):replace(self.cfg,**kw).validate()


if __name__ == '__main__': unittest.main()
