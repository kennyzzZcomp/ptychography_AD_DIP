# -*- coding: utf-8 -*-
"""论文 Fig.1(b) 的 U-Net：4 个独立输出头。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from functions.common.unet import DoubleConv

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_paper import Cfg

class ProPtyUNet(nn.Module):
    """Fig.1(b): J·M·N 输入 -> 32 -> 64 -> 128 -> 256(瓶颈) -> 解码 -> 4 个单通道。

    输出头严格按 Fig.1(b) 用 4 个【独立的】单通道 Conv2d:
        amp_s, amp_p : Conv2d + LeakyReLU
        phs_s, phs_p : Conv2d + Tanh      -> 复场 = amp · exp(jπ·phs)
    输出层没有 BatchNorm —— 在输出层做 BN 会把振幅强制成零均值单位方差。

    Section 2.2 的每级解码顺序为:
        ConvTranspose2d -> BatchNorm2d -> LeakyReLU -> concat -> DoubleConv.
    这里仅用于 paper run；addip/shared/windows 的网络是另一个类。
    三处上采样采用 4×4、stride=2、padding=1；这是本实现选定的核大小，
    论文没有明确核尺寸。J=100、base=32 时为 2,492,516 参数（约 2.5M）。
    按图补入入口 J->32 和分头前 32->32 单卷积；两者采用 3×3。
    这是图示的显式实现选择，不代表作者未公开的逐层源码。
    """

    def __init__(self, in_ch, base=32, up_kernel=4, obj_amp_activation="leakyrelu"):
        super().__init__()
        if obj_amp_activation not in ("leakyrelu", "softplus"):
            raise ValueError("obj_amp_activation must be leakyrelu or softplus")
        self.obj_amp_activation = obj_amp_activation
        if up_kernel not in (3, 4):
            raise ValueError("up_kernel must be 3 or 4")
        self.up_kernel = up_kernel
        output_padding = 4 - up_kernel
        c = [base, base * 2, base * 4, base * 8]
        self.input_conv = nn.Conv2d(in_ch, c[0], 3, 1, 1)
        self.e1, self.e2, self.e3 = DoubleConv(c[0], c[0]), DoubleConv(c[0], c[1]), DoubleConv(c[1], c[2])
        self.bot = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2, 2)
        self.u3 = nn.ConvTranspose2d(c[3], c[2], up_kernel, 2, 1, output_padding=output_padding)
        self.up_norm3 = nn.BatchNorm2d(c[2])
        self.up_act3 = nn.LeakyReLU(0.2, inplace=True)
        self.d3 = DoubleConv(c[3], c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], up_kernel, 2, 1, output_padding=output_padding)
        self.up_norm2 = nn.BatchNorm2d(c[1])
        self.up_act2 = nn.LeakyReLU(0.2, inplace=True)
        self.d2 = DoubleConv(c[2], c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], up_kernel, 2, 1, output_padding=output_padding)
        self.up_norm1 = nn.BatchNorm2d(c[0])
        self.up_act1 = nn.LeakyReLU(0.2, inplace=True)
        self.d1 = DoubleConv(c[1], c[0])
        self.output_conv = nn.Conv2d(c[0], c[0], 3, 1, 1)
        self.amp_s = nn.Conv2d(c[0], 1, 3, 1, 1)
        self.phs_s = nn.Conv2d(c[0], 1, 3, 1, 1)
        self.amp_p = nn.Conv2d(c[0], 1, 3, 1, 1)
        self.phs_p = nn.Conv2d(c[0], 1, 3, 1, 1)

    def forward(self, x):
        x1 = self.e1(self.input_conv(x))
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        y = self.up_act3(self.up_norm3(self.u3(self.bot(self.pool(x3)))))
        y = self.d3(torch.cat([y, x3], 1))
        y = self.up_act2(self.up_norm2(self.u2(y)))
        y = self.d2(torch.cat([y, x2], 1))
        y = self.up_act1(self.up_norm1(self.u1(y)))
        y = self.d1(torch.cat([y, x1], 1))
        y = self.output_conv(y)
        lr = lambda t: F.leaky_relu(t, 0.2)
        # Optional object-only diagnostic; probe and phase heads stay unchanged.
        obj_amp = F.softplus(self.amp_s(y)) if self.obj_amp_activation == "softplus" else lr(self.amp_s(y))
        return (obj_amp[0, 0], torch.tanh(self.phs_s(y))[0, 0],
                lr(self.amp_p(y))[0, 0], torch.tanh(self.phs_p(y))[0, 0])
