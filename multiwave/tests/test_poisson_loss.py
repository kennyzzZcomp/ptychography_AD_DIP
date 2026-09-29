from dataclasses import replace
import unittest
import torch
from multiwave.config import Config
from multiwave.losses import poisson_loss
from multiwave.scene import simulate
from multiwave.reconstruct import reconstruct, backward_shared_data


class PoissonLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_value_and_gradient_match_count_nll(self):
        f=8000.
        x=torch.tensor([0.,1e-8,.001,.01],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([0.,0.,3.,80.],dtype=torch.float64)
        mu=x*f+1e-8
        nll=(mu-y*mu.log()+torch.lgamma(y+1)).sum()/y.sum()
        actual=poisson_loss(x,y/f,f)
        sat=torch.where(y>0,y-y*torch.log(y.clamp_min(1))+torch.lgamma(y+1),0).sum()/y.sum()
        torch.testing.assert_close(actual,nll-sat)
        torch.testing.assert_close(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(nll,x)[0])
        self.assertTrue(torch.isfinite(actual))

    def test_all_zero_counts_finite_and_penalize_light(self):
        x=torch.tensor([0.,.01],requires_grad=True)
        loss=poisson_loss(x,torch.zeros_like(x),8000.)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(x.grad,torch.full_like(x,8000.))

    def test_chunk_gradient_and_training(self):
        cfg=replace(Config.preset('smoke'),loss='poisson',photons_per_scan=8000.,
                    iterations=3,eval_every=3,chunk=3,unet_skip='dwt_concat',unet_activation='softplus',pixel_parameterization='softplus').validate()
        scene=simulate(cfg)
        a=(scene.objects.detach()*.8).requires_grad_()
        p=scene.operator.probes.detach().clone().requires_grad_()
        loss=poisson_loss(scene.operator(a,scene.train,probes=p),scene.measured[scene.train],cfg.photons_per_scan)
        loss.backward()
        ga,gp=a.grad.clone(),p.grad.clone()
        a.grad=None;p.grad=None
        backward_shared_data(a,p,scene,cfg)
        torch.testing.assert_close(a.grad,ga,rtol=3e-4,atol=2e-7)
        torch.testing.assert_close(p.grad,gp,rtol=3e-4,atol=2e-7)
        for method in ('pixel_shared_amp','unet_shared_amp'):
            result=reconstruct(cfg,scene,method)
            self.assertEqual(result['loss'],'poisson')
            self.assertLess(result['final']['train_data_loss'],result['initial']['train_data_loss'])
            self.assertEqual(result['final']['max_object_imaginary_abs'],0.)

    def test_validation(self):
        with self.assertRaises(ValueError):
            replace(Config.preset('smoke'),loss='poisson',photons_per_scan=0).validate()
        with self.assertRaises(ValueError):
            replace(Config.preset('smoke'),loss='other').validate()


if __name__=='__main__': unittest.main()
