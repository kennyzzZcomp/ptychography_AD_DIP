"""Small CPU tests, no experimental reconstruction. python -m unittest discover -s real_ptycho"""
import copy
import contextlib
import io
from pathlib import Path
import uuid
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from core import (UNet, NLMPrior, Propagator, chunked_data, geometry, select_center_grid, tgv_terms,
                  center_crop_field, center_pad_probe)
from run import arguments, network_input, set_optimizer_lr, train


class LearningRateTests(unittest.TestCase):
    def parse(self, *options):
        # Parsing only: generate a nonexistent path without creating/deleting directories.
        out = Path(__file__).resolve().parent/'results'/f'argument_test_{uuid.uuid4().hex}'
        return arguments(['--data', str(Path(__file__).resolve()), '--prepare-only',
                          '--out', str(out), *options])

    def test_defaults_and_legacy(self):
        a = self.parse()
        self.assertEqual(a.num_frames, 225)
        self.assertEqual(a.probe_size, 512)
        self.assertEqual(self.parse('--num-frames','100').num_frames, 100)
        self.assertEqual(self.parse('--num-frames','200').num_frames, 200)
        self.assertEqual(self.parse('--num-frames','225').num_frames, 225)
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

    def test_probe_size_arguments(self):
        self.assertEqual(self.parse('--probe-size','0').probe_size,0)
        self.assertEqual(self.parse('--smoke').detector_crop,512)
        self.assertEqual(self.parse('--smoke','--probe-size','64').detector_crop,256)
        for size in ('-1','511','2002'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse('--probe-size',size)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse('--detector-crop','256')


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(8)

    def test_probe_padding_native_and_gradient_adjoint(self):
        p = torch.randn(512,512,dtype=torch.complex64)
        full = center_pad_probe(p,2000)
        torch.testing.assert_close(full[744:1256,744:1256],p)
        self.assertEqual(float(full[:744].abs().sum()+full[1256:].abs().sum()),0.)
        self.assertEqual(float(full[:,:744].abs().sum()+full[:,1256:].abs().sum()),0.)
        torch.testing.assert_close(center_crop_field(full,512),p)
        prop = Propagator(np.array([[1.2,2.7],[2.1,1.4]]),8)
        obj = torch.randn(16,16,dtype=torch.complex64,requires_grad=True)
        compact = torch.randn(4,4,dtype=torch.complex64,requires_grad=True)
        data = np.random.default_rng(7).uniform(.1,2,(2,8,8)).astype('float32')
        a = prop.amplitude(obj,center_pad_probe(compact,8),[0,1])
        b = torch.from_numpy(data).sqrt()
        gain = (a*b).sum()/a.square().sum()
        direct_go,direct_gp = torch.autograd.grad((gain*a-b).square().mean(),(obj,compact))
        go,gp,_ = chunked_data(obj,center_pad_probe(compact,8),[0,1],data,prop,chunk=1)
        torch.testing.assert_close(go,direct_go,rtol=3e-5,atol=2e-7)
        torch.testing.assert_close(center_crop_field(gp,4),direct_gp,rtol=3e-5,atol=2e-7)

    def test_compact_probe_two_stage_synthetic_updates(self):
        # Tiny synthetic 4+2 updates only, not reconstruction of the experimental data.
        with tempfile.TemporaryDirectory(prefix='real_ptycho_test_', dir=Path(__file__).resolve().parent) as temp:
            dest = Path(temp)/'result'
            a = arguments(['--data',str(Path(__file__).resolve()),'--out',str(dest),
                           '--device','cpu','--probe-size','8','--stage1-steps','4',
                           '--stage2-steps','2','--eval-every','2','--chunk','25',
                           '--nlm-refresh','1','--num-frames','100'])
            dest.mkdir()
            data = np.random.default_rng(3).uniform(.1,2,(100,16,16)).astype('float32')
            pos = np.array([[2+.1*(j//10),2+.1*(j%10)] for j in range(100)])
            groups = [np.arange(k,100,4) for k in range(4)]
            config = {'selected_frame_ids_zero_based':list(range(100)), 'object_pixel_um':1.}
            with patch('run.plot_results'), contextlib.redirect_stdout(io.StringIO()):
                train(a,data,pos,32,groups,config)
            for name in ('stage1.npz','reconstruction.npz'):
                with np.load(dest/name) as result:
                    self.assertEqual(result['probe_compact'].shape,(8,8))
                    self.assertEqual(result['probe_rec'].shape,(16,16))
                    full = result['probe_rec'].copy()
                    np.testing.assert_array_equal(full[4:12,4:12],result['probe_compact'])
                    full[4:12,4:12] = 0
                    self.assertEqual(float(np.abs(full).sum()),0.)
                    self.assertTrue(np.isfinite(result['obj_rec']).all())
            for stage in (1,2):
                saved = torch.load(dest/f'stage{stage}_weights.pt',weights_only=False)
                self.assertEqual(tuple(saved['probe_real'].shape),(8,8))
                self.assertEqual(tuple(saved['probe_imag'].shape),(8,8))
                for state in saved['probe_optimizer']['state'].values():
                    self.assertEqual(tuple(state['exp_avg'].shape),(8,8))
                    self.assertEqual(int(state['step']),4 if stage == 1 else 6)

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
        ids, coords, groups, meta = select_center_grid(np.array(positions).T, 100)
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

    def test_200_selection_mapping_parity_and_distance(self):
        rng = np.random.default_rng(9)
        positions = np.array([[y*.03, x*.03] for y in range(15)
                              for x in (range(15) if y%2 == 0 else reversed(range(15)))])
        positions += rng.uniform(-1e-5, 1e-5, positions.shape)
        ids, coords, groups, meta = select_center_grid(positions.T, 200)
        self.assertEqual(len(set(ids)), 200)
        np.testing.assert_array_equal(coords, positions[ids])
        self.assertEqual(sorted(np.concatenate(groups).tolist()), list(range(200)))
        rr, cc = np.array(meta['grid_rows_zero_based']), np.array(meta['grid_cols_zero_based'])
        self.assertTrue(np.all(np.diff(15*rr+cc)>0))
        for k, g in enumerate(groups):
            self.assertTrue(np.all(2*(rr[g]%2)+cc[g]%2 == k))
        distance = np.linalg.norm(positions-(positions.min(0)+positions.max(0))/2, axis=1)
        excluded = np.setdiff1d(np.arange(225), ids)
        self.assertLessEqual(distance[ids].max(), distance[excluded].min())
        np.testing.assert_array_equal(ids, select_center_grid(positions.T, 200)[0])
        with self.assertRaises(ValueError):
            select_center_grid(positions.T, 201)

    def test_unequal_groups_network_channel_padding(self):
        positions = np.array([[y*.03,x*.03] for y in range(15) for x in range(15)]).T
        _, _, groups, _ = select_center_grid(positions, 200)
        channels = max(map(len, groups))
        net = UNet(channels, 2, checkpoint_blocks=False)
        for g in groups:
            frames = np.ones((len(g),16,16), dtype=np.float32)*2
            x = network_input(frames,32,channels,'cpu')
            self.assertEqual(tuple(x.shape),(1,channels,32,32))
            self.assertTrue(torch.all(x[0,:len(g),8:24,8:24] == 1))
            self.assertEqual(float(x[0,len(g):].abs().sum()),0.)
            self.assertEqual(float(x[:,:,:8].abs().sum()),0.)
            out = net(x)
            torch.testing.assert_close(out,torch.ones_like(out))

    def test_full_225_default_selection_and_inputs(self):
        positions = np.array([[y*.03,x*.03] for y in range(15)
                              for x in (range(15) if y%2 == 0 else reversed(range(15)))])
        ids, coords, groups, meta = select_center_grid(positions.T)
        self.assertEqual(sorted(ids.tolist()),list(range(225)))
        self.assertEqual([len(g) for g in groups],[64,56,56,49])
        self.assertEqual(sorted(np.concatenate(groups).tolist()),list(range(225)))
        self.assertEqual(meta['selection_rule'],'full_15x15')
        np.testing.assert_array_equal(coords,positions[ids])
        rr, cc = np.indices((15,15))
        np.testing.assert_array_equal(meta['grid_rows_zero_based'],rr.ravel())
        np.testing.assert_array_equal(meta['grid_cols_zero_based'],cc.ravel())
        for k,g in enumerate(groups):
            self.assertTrue(np.all(2*(rr.ravel()[g]%2)+cc.ravel()[g]%2 == k))
        net = UNet(64,2,checkpoint_blocks=False)
        for g in groups:
            frames = np.ones((len(g),16,16),dtype=np.float32)
            x = network_input(frames,32,64,'cpu')
            self.assertEqual(tuple(x.shape),(1,64,32,32))
            self.assertEqual(float(x[:,len(g):].abs().sum()),0.)
            out = net(x)
            torch.testing.assert_close(out,torch.ones_like(out))

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
