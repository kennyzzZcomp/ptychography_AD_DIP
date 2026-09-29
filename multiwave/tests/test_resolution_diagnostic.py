import unittest
from dataclasses import replace
import torch
from multiwave.config import Config
from multiwave.scene import simulate
from multiwave.diagnose_resolution import DiagnosticOperator, MixedDiagnosticOperator


class DiagnosticTests(unittest.TestCase):
    def test_two_wavelength_sum_matches_production(self):
        torch.set_num_threads(2)
        cfg = replace(Config.preset('smoke'), probe_mode='known')
        s = simulate(cfg)
        op = MixedDiagnosticOperator(s, cfg, fft_size=64)
        torch.testing.assert_close(op(s.objects[0].real, s.train[:2], 32),
                                   s.clean[s.train[:2]], rtol=0, atol=0)

    def test_original_path_and_nested_detector_crops(self):
        torch.set_num_threads(2)
        cfg = replace(Config.preset('smoke'), wavelengths_nm=(515.,), probe_mode='known')
        s = simulate(cfg)
        op = DiagnosticOperator(s, cfg, fft_size=64)
        ids = s.train[:2]
        a = s.objects[0].real.detach().requires_grad_()
        original = s.operator(torch.complex(a, torch.zeros_like(a))[None], ids)
        small = op(a, ids, 32)
        large = op(a, ids, 64)
        torch.testing.assert_close(small, original, rtol=0, atol=0)
        torch.testing.assert_close(small, large[:,16:48,16:48], rtol=0, atol=0)
        g1 = torch.autograd.grad(original.square().sum(), a, retain_graph=True)[0]
        g2 = torch.autograd.grad(small.square().sum(), a)[0]
        torch.testing.assert_close(g1, g2, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
