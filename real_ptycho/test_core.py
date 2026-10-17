"""Small CPU tests, no experimental reconstruction. python -m unittest discover -s real_ptycho"""
import copy
import contextlib
import io
from pathlib import Path
import uuid
import unittest
import numpy as np
import torch
from core import UNet, NLMPrior, Propagator, chunked_data, geometry, select_center_grid, tgv_terms
from run import arguments, set_optimizer_lr


class LearningRateTests(unittest.TestCase):
    def parse(self, *options):
        # Parsing only: generate a nonexistent path without creating/deleting directories.
        out = Path(__file__).resolve().parent/'results'/f'argument_test_{uuid.uuid4().hex}'
        return arguments(['--data', str(Path(__file__).resolve()), '--prepare-only',
                          '--out', str(out), *options])

    def test_defaults_and_legacy(self):
        a = self.parse()
        self.assertEqual((a.stage1_lr_net,a.stage2_lr_net),(.0005,.0005))
        self.assertEqual((a.stage1_lr_probe,a.stage2_lr_probe),(.01,.01))
        a = self.parse('--lr-net','.002','--lr-probe','.003')
        self.assertEqual((a.stage1_lr_net,a.stage2_lr_net),(.002,.002))
        self.assertEqual((a.stage1_lr_probe,a.stage2_lr_probe),(.003,.003))

    def test_independent_and_override(self):
        a = self.parse('--stage1-lr-net','.001','--stage1-lr-probe','.02',
                       '--stage2-lr-net','.0002','--stage2-lr-probe','.004', '--lr-net','.1')
        self.assertEqual((a.stage1_lr_net,a.stage1_lr_probe,a.stage2_lr_net,a.stage2_lr_probe),
                         (.001,.02,.0002,.004))
        a = self.parse('--stage2-lr-net','.0002','--lr-net','.001')
        self.assertEqual((a.stage1_lr_net,a.stage2_lr_net),(.001,.0002))

    def test_invalid_learning_rates(self):
        for value in ('0','-1','nan','inf'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse('--stage2-lr-probe',value)

    def test_probe_lr_switch_preserves_adam_state(self):
        param = torch.nn.Parameter(torch.tensor([1.]))
        opt = torch.optim.Adam([param],lr=.01)
        param.square().sum().backward(); opt.step()
        state = opt.state[param]
        before = {key:value.clone() for key,value in state.items()}
        set_optimizer_lr(opt,.003)
        self.assertEqual(opt.param_groups[0]['lr'],.003)
        self.assertIs(opt.state[param],state)
        for key,value in state.items():
            torch.testing.assert_close(value,before[key])


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(8)

    def test_chunked_gradient_equals_monolithic(self):
        rng = np.random.default_rng(10)
        prop = Propagator(np.array([[1.2, 2.7], [2.1, 3.4], [3.8, 1.9]]), 8)
        o = torch.randn((16,16), dtype=torch.complex64, requires_grad=True)
        p = torch.randn((8,8), dtype=torch.complex64, requires_grad=True)
        data = rng.uniform(.1, 2, (3,8,8)).astype('float32')
        a, b = prop.amplitude(o,p,[0,1,2]), torch.from_numpy(data).sqrt()
        q = (a*b).sum()/a.square().sum()
        loss = (q*a-b).square().mean()
        go, gp = torch.autograd.grad(loss, (o,p))
        for chunk in (1,2,3):
            co, cp, stats = chunked_data(o,p,[0,1,2],data,prop,chunk)
            torch.testing.assert_close(co, go, rtol=3e-5, atol=2e-7)
            torch.testing.assert_close(cp, gp, rtol=3e-5, atol=2e-7)
            self.assertAlmostEqual(stats['amplitude_mse'], float(loss), places=6)

    def test_center_selection_snake_and_groups(self):
        positions = []
        for y in range(15):
            for x in (range(15) if y%2 == 0 else reversed(range(15))):
                positions.append([y*.03, x*.03])
        ids, coords, groups, meta = select_center_grid(np.array(positions).T)
        self.assertEqual(len(set(ids)), 100)
        self.assertEqual(sorted(np.concatenate(groups).tolist()), list(range(100)))
        self.assertTrue(all(len(g)==25 for g in groups))
        self.assertIn(meta['row_start_zero_based'], (2,3))
        self.assertTrue(np.all(np.diff(coords.reshape(10,10,2)[:,:,1],axis=1)>0))

    def test_geometry_and_subpixel(self):
        pos, side, dx = geometry(np.array([[0.,0.],[.02856,.02856]]),2000,632.8,6,3.76)
        self.assertAlmostEqual(dx*1e6, .504893617, places=8)
        self.assertEqual(side%8,0)
        self.assertGreater(abs(pos[1,0]-round(pos[1,0])), .1)
        prop = Propagator(np.array([[1.25,2.5]]), 8)
        yy, xx = torch.meshgrid(torch.arange(16.),torch.arange(16.),indexing='ij')
        obj = torch.complex(yy+2*xx, torch.zeros_like(xx))
        patch = sum(w*obj[y:y+8,x:x+8] for y,x,w in prop.samples[0])
        self.assertAlmostEqual(float(patch[0,0].real), 6.25)

    def test_checkpoint_bn_and_gradients(self):
        net = UNet(3,2, residual=True, checkpoint_blocks=False)
        # Nonzero heads exercise gradients through all backbone layers.
        with torch.no_grad():
            net.head_amp.weight.normal_(); net.head_phs.weight.normal_()
        check = copy.deepcopy(net); check.checkpoint_blocks = True
        x = torch.randn((1,3,32,32))
        y, z = net(x), check(x)
        torch.testing.assert_close(y,z)
        y.abs().square().mean().backward(); z.abs().square().mean().backward()
        for p,q in zip(net.parameters(),check.parameters()):
            torch.testing.assert_close(p.grad,q.grad)
        for b,c in zip(net.buffers(),check.buffers()):
            torch.testing.assert_close(b,c)

    def test_initial_fields_and_flat_tgv(self):
        x = torch.randn((1,25,32,32))
        out = UNet(25,2,checkpoint_blocks=False)(x)
        torch.testing.assert_close(out,torch.ones_like(out))
        delta = UNet(3,2,residual=True,checkpoint_blocks=False)(x[:,:3])
        self.assertEqual(float(delta.abs().max()),0.)
        self.assertAlmostEqual(float(tgv_terms(torch.ones(16,16),torch.zeros(2,16,16),torch.ones(16,16,dtype=torch.bool))),0.)

    def test_nlm_finite_scale_invariant_and_detached(self):
        base = (1+.05*torch.randn(32,32,dtype=torch.complex64)).requires_grad_(True)
        prior = NLMPrior(base,torch.ones((32,32),dtype=torch.bool),torch.ones((32,32)))
        self.assertFalse(prior.target.requires_grad)
        loss = prior(base)
        torch.testing.assert_close(loss,prior(3*base),rtol=1e-4,atol=1e-7)
        grad, = torch.autograd.grad(loss,base)
        self.assertTrue(torch.isfinite(grad).all())
        prior.refresh(base+.01)
        self.assertFalse(prior.target.requires_grad)


if __name__ == '__main__':
    unittest.main()
