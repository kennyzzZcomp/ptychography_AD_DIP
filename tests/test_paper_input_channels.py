import copy
import unittest
from unittest.mock import patch

import torch
from functions.paperrepro.input_channels import SelectedInputConv, install_selected_input
from functions.addip.model import ProPtyUNet


class InputChannelTests(unittest.TestCase):
    def test_gather_matches_zero_mask_output_and_gradients(self):
        torch.manual_seed(3)
        conv = torch.nn.Conv2d(4, 3, 3, padding=1).double()
        dense = copy.deepcopy(conv)
        selected = SelectedInputConv(conv)
        indices = torch.tensor([0, 3])
        selected.select(indices)
        x = torch.randn(1, 4, 8, 8, dtype=torch.float64)
        mask = torch.tensor([1, 0, 0, 1], dtype=torch.float64)[None, :, None, None]
        a, b = selected(x), dense(x * mask)
        torch.testing.assert_close(a, b)
        a.square().sum().backward()
        b.square().sum().backward()
        torch.testing.assert_close(conv.weight.grad, dense.weight.grad)
        self.assertEqual(conv.weight.grad[:, [1, 2]].abs().sum().item(), 0)

    def test_convolution_really_has_fewer_channels(self):
        conv = torch.nn.Conv2d(4, 3, 3, padding=1)
        selected = SelectedInputConv(conv)
        selected.select(torch.tensor([0, 2]))
        with patch("functions.paperrepro.input_channels.F.conv2d",
                   wraps=torch.nn.functional.conv2d) as call:
            selected(torch.ones(1, 4, 8, 8))
        self.assertEqual(call.call_args.args[0].shape[1], 2)
        self.assertEqual(call.call_args.args[1].shape[1], 2)

    def test_parameter_identity_optimizer_state_and_full_path(self):
        conv = torch.nn.Conv2d(4, 3, 3, padding=1)
        optimizer = torch.optim.Adam(conv.parameters(), lr=.001)
        selected = SelectedInputConv(conv)
        x = torch.randn(1, 4, 8, 8)
        selected.select(torch.tensor([0, 2]))
        selected(x).square().mean().backward()
        optimizer.step()
        momentum = optimizer.state[conv.weight]["exp_avg"]
        selected.select(None)
        self.assertIs(selected.conv.weight, optimizer.param_groups[0]["params"][0])
        self.assertIs(optimizer.state[conv.weight]["exp_avg"], momentum)
        torch.testing.assert_close(selected(x), conv(x), rtol=0, atol=0)

    def test_full_unet_gather_matches_masked_input(self):
        # Tiny forward-only architecture check, no simulation or training loop.
        torch.manual_seed(4)
        net = ProPtyUNet(4, base=1).eval()
        dense = copy.deepcopy(net)
        layer = install_selected_input(net)
        x = torch.randn(1, 4, 16, 16)
        layer.select(torch.tensor([0, 3]))
        mask = torch.tensor([1, 0, 0, 1])[None, :, None, None]
        with torch.no_grad():
            for a, b in zip(net(x), dense(x * mask)):
                torch.testing.assert_close(a, b)
            layer.select(None)
            for a, b in zip(net(x), dense(x)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
