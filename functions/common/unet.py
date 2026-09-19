# -*- coding: utf-8 -*-
"""两条实验线共用的 U-Net 基本块。"""

from __future__ import annotations

import torch.nn as nn


class DoubleConv(nn.Module):
    """Conv-BN-LeakyReLU ×2，尺寸不变。对应 PhysenNet 的 layer_0x。"""

    def __init__(self, cin, cout):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True),
            nn.Conv2d(cout, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.f(x)
