"""Shared-probe control: paired init, common physics, gradients and solver wiring."""
import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import torch

from functions.addip.model import ProPtyUNet, SharedProbeUNet, make_field, shared_probe_field
from functions.paperrepro.optics import forward_field
from functions.paperrepro.solvers_addip import _initialize_object_heads, _net_optimizers, run_net
from simulations.ProPtyNet_paper import Cfg, PRESETS


class SharedProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def pair(self):
        torch.manual_seed(8)
        pixel = ProPtyUNet(4, base=2)
        torch.manual_seed(8)
        shared = SharedProbeUNet(4, base=2)
        return pixel, shared

    def test_paired_seed_preserves_all_existing_weights_and_outputs(self):
        pixel, shared = self.pair()
        for key, value in pixel.state_dict().items():
            self.assertTrue(torch.equal(value, shared.state_dict()[key]), key)
        x = torch.rand(1, 4, 24, 24)
        for a, b in zip(pixel(x), shared(x)[:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_original_forward_matches_pre_refactor_formula(self):
        net = ProPtyUNet(4, base=2)
        x = torch.rand(1, 4, 24, 24)
        a = net.e1(x); b = net.e2(net.pool(a)); c = net.e3(net.pool(b))
        z = net.d3(torch.cat([net.u3(net.bot(net.pool(c))), c], 1))
        z = net.d2(torch.cat([net.u2(z), b], 1))
        z = net.d1(torch.cat([net.u1(z), a], 1))
        expected = net.head_amp(z)[0], net.head_phs(z)[0]
        for a, b in zip(net(x), expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_neutral_fields_and_identical_initial_physics(self):
        pixel, shared = self.pair()
        for net in (pixel, shared):
            _initialize_object_heads(net, 0.)
        x = torch.rand(1, 4, 24, 24)
        a, p = pixel(x); sa, sp, raw = shared(x)
        O, SO = make_field(a[0], p), make_field(sa[0], sp)
        P = shared_probe_field(raw, 24, 16)
        torch.testing.assert_close(O, torch.ones_like(O))
        torch.testing.assert_close(SO, O, rtol=0, atol=0)
        torch.testing.assert_close(P, torch.ones_like(P), rtol=0, atol=0)
        pos = torch.tensor([[0, 0], [0, 4], [4, 0], [4, 4]])
        Q = torch.exp(1j * torch.rand(16, 16))
        u = forward_field(O, torch.ones_like(P), pos, Q, 16)
        su = forward_field(SO, P, pos, Q, 16)
        torch.testing.assert_close(u, su, rtol=0, atol=0)

    def test_center_crop_coordinates_and_gradient(self):
        for ns, m, n in [(624, 624, 512), (616, 612, 512), (32, 27, 16)]:
            raw = torch.arange(2*ns*ns, dtype=torch.float32).reshape(2, ns, ns).requires_grad_()
            P = shared_probe_field(raw, m, n)
            pn, po = (ns-m)//2, (m-n)//2
            expected = raw[:, pn:pn+m, pn:pn+m][:, po:po+n, po:po+n]
            torch.testing.assert_close(P, torch.complex(expected[0], expected[1]))
            (P.real.sum()+P.imag.sum()).backward()
            self.assertEqual(raw.grad.sum().item(), 2*n*n)
            self.assertEqual(P.shape, (n, n))

    def test_optimizer_partition_and_both_paths_reach_backbone(self):
        net = SharedProbeUNet(4, base=2)
        cfg = Cfg(probe_mode='shared', lr_net=.001, lr_probe=.002)
        on, op = _net_optimizers(cfg, net, 'shared', None, None)
        ids = lambda opt: {id(p) for g in opt.param_groups for p in g['params']}
        self.assertFalse(ids(on) & ids(op))
        self.assertEqual(ids(on) | ids(op), {id(p) for p in net.parameters()})
        self.assertEqual(ids(op), {id(p) for p in net.head_probe.parameters()})
        self.assertEqual(on.param_groups[0]['lr'], .001)
        self.assertEqual(op.param_groups[0]['lr'], .002)
        # Nonzero readout weights model the state after the first optimizer update.
        with torch.no_grad():
            net.head_probe.weight.normal_(std=.02)
        x = torch.rand(1, 4, 24, 24)
        a, p, raw = net(x)
        weight = next(net.e1.parameters())
        go = torch.autograd.grad(make_field(a[0], p).real.square().mean(), weight, retain_graph=True)[0]
        gp = torch.autograd.grad(shared_probe_field(raw, 24, 16).abs().square().mean(), weight)[0]
        self.assertGreater(go.abs().sum().item(), 0)
        self.assertGreater(gp.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(go).all() and torch.isfinite(gp).all())

    def test_state_round_trip(self):
        net = SharedProbeUNet(4, base=2).eval()
        clone = SharedProbeUNet(4, base=2).eval()
        clone.load_state_dict(copy.deepcopy(net.state_dict()))
        x = torch.rand(1, 4, 24, 24)
        for a, b in zip(net(x), clone(x)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_configuration_guards(self):
        for changes in [dict(probe_init='disk'), dict(obj_init_alpha=.1),
                        dict(network_type='complex'), dict(skip_mode='wavelet'),
                        dict(measurement_schedule='0:1'), dict(input_policy='follow_measurements')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Cfg(probe_mode='shared', **changes)
        self.assertEqual(Cfg().probe_mode, 'pixel')
        Cfg(probe_mode='shared', lr_cosine=True)

    def test_solver_smoke_and_metadata(self):
        first = {}
        for mode, cosine in [('pixel', True), ('shared', True), ('shared', False)]:
            cfg = Cfg(**PRESETS['smoke'], preset='smoke', probe_mode=mode, base_ch=2,
                      iters=3, eval_every=1, eval_size=16, device='cpu', lr_cosine=cosine,
                      lr_schedule='' if cosine else '1:0.0005:0.002')
            with patch('functions.paperrepro.solvers_addip._save') as save, \
                 patch('pathlib.Path.write_text') as write, redirect_stdout(io.StringIO()):
                hist = run_net(cfg)
            self.assertEqual(len(hist), 3)
            for row in hist:
                self.assertTrue(all(np.isfinite(row[k]) for k in ['loss', 'relerr', 'relerr_p']))
            self.assertEqual(hist[-1]['lr_net'], 0. if cosine else .0005)
            self.assertEqual(hist[-1]['lr_probe'], 0. if cosine else .002)
            metadata = json.loads(write.call_args_list[0].args[0])
            self.assertEqual(metadata['probe_mode'], mode)
            for stats in metadata['initial_fields'].values():
                self.assertLess(stats['max_abs_difference_from_one'], 1e-6)
            if cosine:
                first[mode] = hist[0]['loss']
            self.assertEqual(save.call_args.args[1].shape, (cfg.obj_size, cfg.obj_size))
            self.assertEqual(save.call_args.args[2].shape, (cfg.N, cfg.N))
            # Saved training field at iteration 3 must have moved away from ones.
            self.assertGreater(np.abs(save.call_args.args[2]-1).max(), 1e-7)
            self.assertGreater(metadata['total_trainable_parameters'], 0)
        self.assertEqual(first['pixel'], first['shared'])


if __name__ == '__main__':
    unittest.main()
