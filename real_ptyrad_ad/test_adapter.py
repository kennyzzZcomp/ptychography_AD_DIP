"""Small GPU tests of the official-step adapter, not long reconstructions."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from run import create_model, _updates
from analyze import dual_fields


class FieldTransformationTests(unittest.TestCase):
    def test_compact_support_dual_preserves_fourier_amplitudes(self):
        # This compact-support example fits both reflected exit waves inside
        # the FFT window at every tested integer scan position. It does not
        # assume that real measured-field tails satisfy this condition.
        rng = np.random.default_rng(13)
        obj = rng.normal(size=(24,24))+1j*rng.normal(size=(24,24))
        probe = np.zeros((16,16),dtype=np.complex128)
        probe[6:10,6:10] = rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
        alternative_obj,alternative_probe = dual_fields(obj,probe,(17,17))
        for y,x in ((0,0),(0,4),(4,0),(4,4),(2,2)):
            original = np.abs(np.fft.fft2(obj[y:y+16,x:x+16]*probe))
            alternative = np.abs(np.fft.fft2(alternative_obj[y:y+16,x:x+16]*alternative_probe))
            np.testing.assert_allclose(alternative,original,rtol=1e-12,atol=1e-12)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA needed for this adapter')
class AdapterTests(unittest.TestCase):
    def setup_model(self,loss='official'):
        torch.manual_seed(2)
        rng = np.random.default_rng(2)
        data = (rng.random((225,16,16))+.1).astype(np.float32)
        obj = np.ones((32,32),np.complex64)
        probe = (rng.random((16,16))+.1).astype(np.complex64)
        positions = rng.integers(0,16,(225,2))
        args = SimpleNamespace(lr_object=.0005,lr_probe=.005,seed=0,loss=loss)
        return create_model(data,obj,probe,positions,dict(object_pixel_um=1.),args)

    def test_only_requested_parameters_and_constraints(self):
        model,opt,loss,constraint,params = self.setup_model()
        self.assertEqual([k for k,v in model.optimizable_tensors.items() if v.requires_grad],
                         ['obja','objp','probe'])
        self.assertTrue(all(v['start_iter'] is None for v in params['constraint'].values()))
        before = {k:v.detach().clone() for k,v in model.optimizable_tensors.items()}
        constraint(model,1)
        for key,value in before.items():
            torch.testing.assert_close(model.optimizable_tensors[key],value,rtol=0,atol=0)

    def test_official_step_equals_manual_same_official_model_loss(self):
        first = self.setup_model()
        second = self.setup_model()
        batch = np.array([0,14,112,224])
        model,opt,loss,constraint,_ = first
        _updates.recon_step([batch],1,model,opt,None,loss,constraint,1,
                            compute_loss_fn=_updates.compute_loss)
        other,other_opt,other_loss,_,_ = second
        other_opt.zero_grad()
        value,_ = _updates.compute_loss(batch,other,other,other.get_measurements(batch),other_loss)
        value.backward()
        other_opt.step()
        for key in ('opt_obja','opt_objp','opt_probe'):
            torch.testing.assert_close(getattr(model,key),getattr(other,key),rtol=0,atol=0)
        torch.testing.assert_close(model.opt_probe_pos_shifts,torch.zeros_like(model.opt_probe_pos_shifts),rtol=0,atol=0)

    def test_explicit_mse_diagnostic_loss(self):
        model,_,criterion,_,_ = self.setup_model(loss='matched-mse')
        batch = np.array([2,3])
        pred = model(batch)
        target = model.get_measurements(batch)
        value,_ = criterion(pred,target)
        expected = (pred.sqrt()-target.sqrt()).square().mean()
        torch.testing.assert_close(value,expected,rtol=0,atol=0)

    def test_unequal_chunks_equal_full_group_mse_gradient(self):
        model,_,criterion,_,_ = self.setup_model(loss='matched-mse')
        indices = np.arange(7)
        full,_ = criterion(model(indices),model.get_measurements(indices))
        params = (model.opt_obja,model.opt_objp,model.opt_probe)
        expected = torch.autograd.grad(full,params)
        total = [torch.zeros_like(v) for v in params]
        criterion.accumulation_chunks = 2
        criterion.group_frames = 7
        for batch in (indices[:4],indices[4:]):
            value,_ = criterion(model(batch),model.get_measurements(batch))
            grad = torch.autograd.grad(value/2,params)
            total = [a+b for a,b in zip(total,grad)]
        for actual,wanted in zip(total,expected):
            torch.testing.assert_close(actual,wanted,rtol=5e-5,atol=2e-7)

    def test_explicit_domain_guard_uses_official_threshold(self):
        model,opt,criterion,constraint,params = self.setup_model()
        # The production adapter only enables this in an explicitly labeled
        # diagnostic. Verify the official constraint alone does not touch P.
        constraint.constraint_params['obja_thresh'].update(start_iter=1,relax=0.,thresh=[1e-6,1e10])
        before_p = model.opt_probe.detach().clone()
        with torch.no_grad():
            model.opt_obja[0,0,2,2] = -.1
        constraint(model,1)
        self.assertGreater(float(model.opt_obja.min()),0.)
        torch.testing.assert_close(model.opt_probe,before_p,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
