import copy
import math
import unittest
import torch
from torch.nn import functional as F
from functions.common.complex_layers import (
    ComplexConv2d, ComplexConvTranspose2d, ComplexBatchNorm2d,
    ComplexModReLU, ComplexMaxPool2d)
from functions.addip.complex_model import ComplexProPtyUNet
from functions.addip.model import make_field
from simulations.ProPtyNet_paper import Cfg


class ComplexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_convolution_matches_native_values_and_gradients(self):
        for transpose in [False, True]:
            with self.subTest(transpose=transpose):
                layer = (ComplexConvTranspose2d(2,3) if transpose else
                         ComplexConv2d(2,3,3,padding=1)).double()
                r, i = [torch.randn(1,2,4,4,dtype=torch.double,requires_grad=True) for _ in range(2)]
                actual = torch.complex(*layer((r,i)))
                weight = torch.complex(layer.weight_real,layer.weight_imag)
                bias = torch.complex(layer.bias_real,layer.bias_imag)
                expected = (F.conv_transpose2d(torch.complex(r,i),weight,bias,stride=2,padding=1,output_padding=1)
                            if transpose else F.conv2d(torch.complex(r,i),weight,bias,padding=1))
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
                inputs = (r,i,*layer.parameters())
                ga = torch.autograd.grad(actual.abs().square().sum(),inputs,retain_graph=True)
                gb = torch.autograd.grad(expected.abs().square().sum(),inputs)
                for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-10)

    def test_complex_glorot_total_variance(self):
        layer = ComplexConv2d(64,64,3)
        total = (layer.weight_real.square()+layer.weight_imag.square()).mean().item()
        self.assertAlmostEqual(total/(2/(128*9)),1,delta=.03)

    def test_modrelu_zero_tiny_and_gradcheck(self):
        for bias in [-.01,0.,.1]:
            layer = ComplexModReLU(2,bias_init=bias).double()
            r = torch.zeros(1,2,2,2,dtype=torch.double,requires_grad=True)
            i = torch.full_like(r,1e-15,requires_grad=True)
            y = layer((r,i))
            sum(v.sum() for v in y).backward()
            for v in (*y,r.grad,i.grad,layer.bias.grad):self.assertTrue(torch.isfinite(v).all())
        layer = ComplexModReLU(2).double()
        r,i = [torch.randn(1,2,2,2,dtype=torch.double,requires_grad=True) for _ in range(2)]
        self.assertTrue(torch.autograd.gradcheck(lambda r,i:layer((r,i)),(r,i)))

    def test_bn_whitens_correlated_data_and_gradcheck(self):
        torch.manual_seed(7)
        r = torch.randn(2,3,32,32,dtype=torch.double)
        i = .8*r+.2*torch.randn_like(r)
        bn = ComplexBatchNorm2d(3,eps=1e-9,affine=False).double()
        a,b = bn((r,i))
        torch.testing.assert_close(a.mean((0,2,3)),torch.zeros(3,dtype=torch.double),atol=1e-12,rtol=0)
        torch.testing.assert_close(a.square().mean((0,2,3)),torch.ones(3,dtype=torch.double),atol=1e-6,rtol=0)
        torch.testing.assert_close(b.square().mean((0,2,3)),torch.ones(3,dtype=torch.double),atol=1e-6,rtol=0)
        torch.testing.assert_close((a*b).mean((0,2,3)),torch.zeros(3,dtype=torch.double),atol=1e-6,rtol=0)
        small = ComplexBatchNorm2d(1).double()
        r,i = [torch.randn(1,1,2,3,dtype=torch.double,requires_grad=True) for _ in range(2)]
        self.assertTrue(torch.autograd.gradcheck(lambda r,i:small((r,i)),(r,i)))

    def test_bn_degenerate_inputs_and_running_statistics(self):
        bn = ComplexBatchNorm2d(2)
        for kind in ['zero','constant','correlated']:
            r = torch.randn(1,2,4,4) if kind=='correlated' else torch.full((1,2,4,4),float(kind=='constant'))
            r.requires_grad_();i=r.detach().clone().requires_grad_()
            a,b=bn((r,i));(a.square().mean()+b.square().mean()).backward()
            for v in (a,b,r.grad,i.grad):self.assertTrue(torch.isfinite(v).all())
        bn.eval();counter=bn.num_batches_tracked.clone()
        restored=ComplexBatchNorm2d(2);restored.load_state_dict(bn.state_dict());restored.eval()
        z=(torch.randn(1,2,4,4),torch.randn(1,2,4,4))
        for a,b in zip(bn(z),restored(z)):torch.testing.assert_close(a,b)
        self.assertTrue(torch.equal(counter,bn.num_batches_tracked))

    def test_pool_uses_same_location(self):
        r=torch.tensor([[[[3.,0.],[1.,2.]]]],requires_grad=True)
        i=torch.tensor([[[[0.,4.],[0.,0.]]]],requires_grad=True)
        a,b=ComplexMaxPool2d()((r,i))
        self.assertEqual(a.item(),0);self.assertEqual(b.item(),4)
        (a+b).sum().backward()
        torch.testing.assert_close(r.grad,torch.tensor([[[[0.,1.],[0.,0.]]]]))
        torch.testing.assert_close(r.grad,i.grad)

    def test_network_neutral_init_physics_and_optimizer(self):
        for activation in ['modrelu','crelu']:
            net=ComplexProPtyUNet(3,base=2,activation=activation)
            with torch.no_grad():
                net.head_amp.weight.zero_();net.head_amp.bias.fill_(math.log(math.e-1))
                net.head_phs.weight.zero_();net.head_phs.bias.copy_(torch.tensor([1.,0.]))
            opt=torch.optim.Adam(net.parameters(),lr=.001)
            x=torch.rand(1,3,24,24)
            initial=net.e1.layers[0].weight_real.detach().clone()
            for step in range(3):
                amp,phs=net(x);obj=make_field(amp[0],phs)
                if step==0:torch.testing.assert_close(obj,torch.ones_like(obj))
                predicted=torch.fft.fft2(obj,norm='ortho').abs()
                loss=(predicted-torch.ones_like(predicted)).square().mean()
                opt.zero_grad();loss.backward()
                for p in net.parameters():self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all())
                opt.step()
            self.assertFalse(torch.equal(initial,net.e1.layers[0].weight_real))
            restored=ComplexProPtyUNet(3,base=2,activation=activation)
            restored.load_state_dict(copy.deepcopy(net.state_dict()))
            net.eval();restored.eval()
            for a,b in zip(net(x),restored(x)):torch.testing.assert_close(a,b)

    def test_config_rejects_unsupported_combinations(self):
        for settings in [dict(skip_mode='wavelet'),dict(skip_mode='wavelet-identity'),
                         dict(measurement_schedule='0:3,1000:1'),dict(complex_base_ch=0),
                         dict(complex_activation='relu')]:
            with self.assertRaises(ValueError):Cfg(network_type='complex',**settings)
        self.assertEqual(Cfg().network_type,'real')


if __name__=='__main__':unittest.main()
