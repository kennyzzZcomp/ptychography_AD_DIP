import copy
from dataclasses import replace
import unittest
import torch
from multiwave.config import Config
from multiwave.scene import simulate
from multiwave.regularization import AmplitudeATV
from multiwave.losses import poisson_loss
from multiwave.reconstruct import backward_shared_data, reconstruct


class ATVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = replace(Config.preset("smoke"), iterations=2, eval_every=2,
                          loss="poisson", photons_per_scan=20000, atv_weight=.01,
                          probe_grid_size=8, probe_smooth_weight=.2)
        cls.scene = simulate(cls.cfg)

    def test_formula_and_domain(self):
        r = AmplitudeATV(self.cfg, self.scene)
        y,x = torch.meshgrid(torch.arange(64.),torch.arange(64.),indexing="ij")
        self.assertAlmostEqual(float(r.penalty(2*x+3*y)),5.)
        self.assertEqual(float(r.penalty(torch.ones_like(x))),0.)
        a = torch.randn_like(x)
        b = a.clone(); b[~r.mask] += 100
        torch.testing.assert_close(r.penalty(a),r.penalty(b))
        altered = copy.copy(self.scene)
        altered.objects = torch.zeros_like(self.scene.objects)
        altered.roi = torch.zeros_like(self.scene.roi)
        torch.testing.assert_close(r.mask,AmplitudeATV(self.cfg,altered).mask)
        a = torch.randn(64,64,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(r.penalty,(a,),fast_mode=True))

    def test_chunked_gradient_and_regularizer_once(self):
        cfg = replace(self.cfg,probe_smooth_weight=0,chunk=3)
        r = AmplitudeATV(cfg,self.scene)
        a = (self.scene.objects + .01*torch.rand_like(self.scene.objects.real)).detach().requires_grad_()
        p = self.scene.operator.probes.detach().clone().requires_grad_()
        pred = self.scene.operator(a,self.scene.train,probes=p)
        loss = poisson_loss(pred,self.scene.measured[self.scene.train],cfg.photons_per_scan)
        (loss+cfg.atv_weight*r.penalty(a.real[0])).backward()
        ga,gp=a.grad.clone(),p.grad.clone()
        a.grad=None; p.grad=None
        backward_shared_data(a,p,self.scene,cfg,atv=r)
        torch.testing.assert_close(a.grad,ga,rtol=5e-4,atol=2e-6)
        torch.testing.assert_close(p.grad,gp,rtol=5e-4,atol=2e-6)

    def test_both_methods_and_objective(self):
        for method in ("pixel_shared_amp","unet_shared_amp"):
            result = reconstruct(self.cfg,self.scene,method)
            f=result["final"]
            self.assertAlmostEqual(f["train_total_objective"],f["train_data_loss"]+
                f["atv_weighted_penalty"]+f["probe_smooth_weighted_penalty"],places=6)
            self.assertEqual(f["max_interwavelength_object_difference"],0)

    def test_validation(self):
        for kw in ({"atv_weight":-1},{"atv_weight":float("nan")},
                   {"tv_weight":.01},{"tgv_weight":.01},{"scene":"shared_complex"}):
            with self.assertRaises(ValueError):
                replace(self.cfg,**kw).validate()


if __name__ == "__main__":
    unittest.main()
