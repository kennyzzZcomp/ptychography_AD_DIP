import unittest
import torch
from functions.common.wavelet_skip import haar_dwt, haar_idwt, WaveletSkip
from functions.addip.model import ProPtyUNet


class WaveletTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_roundtrip_and_gradient_even_odd(self):
        for shape in [(2, 3, 16, 18), (1, 2, 7, 9), (1, 1, 1, 1)]:
            x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
            bands, size = haar_dwt(x)
            y = haar_idwt(bands, size)
            torch.testing.assert_close(y, x, atol=1e-14, rtol=1e-14)
            weight = torch.randn_like(x)
            (y*weight).sum().backward()
            torch.testing.assert_close(x.grad, weight, atol=1e-14, rtol=1e-14)

    def test_shrinkage_ll_and_trainable_threshold(self):
        x = torch.randn(2, 4, 16, 16, requires_grad=True)
        layer = WaveletSkip(.1)
        layer.capture_stats = True
        y = layer(x)
        b, _ = haar_dwt(x)
        by, _ = haar_dwt(y)
        torch.testing.assert_close(by[:, :, 0], b[:, :, 0])
        self.assertTrue(torch.all(by[:, :, 1:].abs() <= b[:, :, 1:].abs()+1e-6))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(layer.raw_threshold.grad).all())
        self.assertGreater(layer.raw_threshold.grad.abs().sum().item(), 0)
        self.assertEqual(len(layer.last_stats['zero_fraction']), 3)

    def test_identity_network_matches_baseline_gradients(self):
        torch.manual_seed(1)
        a = ProPtyUNet(3, base=2).double().eval()
        torch.manual_seed(1)
        b = ProPtyUNet(3, base=2, skip_mode='wavelet-identity').double().eval()
        b.load_state_dict(a.state_dict(), strict=True)
        x = torch.randn(1, 3, 24, 24, dtype=torch.float64)
        xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
        ya, yb = a(xa), b(xb)
        for u, v in zip(ya, yb):
            torch.testing.assert_close(u, v, atol=1e-12, rtol=1e-10)
        sum(u.square().sum() for u in ya).backward()
        sum(u.square().sum() for u in yb).backward()
        torch.testing.assert_close(xa.grad, xb.grad, atol=1e-12, rtol=1e-10)
        for u, v in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(u.grad, v.grad, atol=1e-12, rtol=1e-10)

    def test_wavelet_network_optimizer_updates_thresholds(self):
        net = ProPtyUNet(3, base=2, skip_mode='wavelet', wavelet_threshold=.05)
        opt = torch.optim.Adam(net.parameters(), lr=.001)
        before = [s.raw_threshold.detach().clone() for s in net.skip_filters]
        out = net(torch.randn(1, 3, 24, 24))
        sum(y.square().mean() for y in out).backward()
        opt.step()
        for previous, s in zip(before, net.skip_filters):
            self.assertFalse(torch.equal(previous, s.raw_threshold))

    def test_invalid_settings(self):
        for v in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                WaveletSkip(v)


if __name__ == '__main__':
    unittest.main()
