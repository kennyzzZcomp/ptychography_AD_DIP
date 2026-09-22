"""Gathered first-convolution channels for progressive diffraction conditioning.

Keeps original Parameter objects: optimizer state and channel identity survive
schedule changes. Only first-convolution work shrinks, not the whole U-Net.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectedInputConv(nn.Module):
    def __init__(self, conv):
        super().__init__()
        if not isinstance(conv, nn.Conv2d) or conv.groups != 1 or conv.padding_mode != "zeros":
            raise ValueError("SelectedInputConv requires ungrouped zero-padded Conv2d")
        self.conv = conv
        self.register_buffer("indices", None, persistent=False)

    def select(self, indices):
        # None is the exact original full-channel path. Caller provides unique,
        # in-range indices from MeasurementSchedule (no per-step GPU validation).
        self.indices = indices

    def forward(self, x):
        if self.indices is None:
            return self.conv(x)
        return F.conv2d(x.index_select(1, self.indices),
                        self.conv.weight.index_select(1, self.indices), self.conv.bias,
                        self.conv.stride, self.conv.padding, self.conv.dilation, 1)


def install_selected_input(net):
    layer = SelectedInputConv(net.e1.f[0])
    net.e1.f[0] = layer
    return layer
