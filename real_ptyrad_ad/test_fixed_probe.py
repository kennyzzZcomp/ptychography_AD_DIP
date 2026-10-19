"""Bounded CPU checks of the process-local frozen-probe adapter."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from network_fixed_probe import FixedProbeAdam,RealAdam


class FrozenProbeTests(unittest.TestCase):
    def test_reference_probe_never_moves(self):
        real = torch.nn.Parameter(torch.ones(8,8))
        imag = torch.nn.Parameter(torch.zeros(8,8))
        opt = FixedProbeAdam([real,imag],lr=.005)
        before = (real.detach().clone(),imag.detach().clone())
        for _ in range(5):
            real.grad = torch.randn_like(real)
            imag.grad = torch.randn_like(imag)
            opt.step()
        self.assertEqual(opt.suppressed_steps,5)
        torch.testing.assert_close(real,before[0],rtol=0,atol=0)
        torch.testing.assert_close(imag,before[1],rtol=0,atol=0)
        self.assertEqual(len(opt.state),0)

    def test_network_adam_class_is_original(self):
        self.assertIs(RealAdam,torch.optim.Adam)
        network = torch.nn.Linear(3,2)
        opt = RealAdam(network.parameters(),lr=.0005)
        before = network.weight.detach().clone()
        network(torch.ones(1,3)).square().sum().backward()
        opt.step()
        self.assertFalse(torch.equal(before,network.weight))


if __name__=='__main__':
    unittest.main()
