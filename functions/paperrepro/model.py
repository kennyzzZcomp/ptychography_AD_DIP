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
    论文没有明确核尺寸。J=100、base=32 时为 2,474,020 参数（约 2.5M）。
    Fig.1(b) 的入口/末端单卷积符号是否表示额外投影层，正文未明确，
    本实现保留现有编码/解码通道拓扑，不把这些符号擅自解释为额外层。
    """

    def __init__(self, in_ch, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3 = DoubleConv(in_ch, c[0]), DoubleConv(c[0], c[1]), DoubleConv(c[1], c[2])
        self.bot = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2, 2)
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 4, 2, 1, output_padding=0)
        self.up_norm3 = nn.BatchNorm2d(c[2])
        self.up_act3 = nn.LeakyReLU(0.2, inplace=True)
        self.d3 = DoubleConv(c[3], c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 4, 2, 1, output_padding=0)
        self.up_norm2 = nn.BatchNorm2d(c[1])
        self.up_act2 = nn.LeakyReLU(0.2, inplace=True)
        self.d2 = DoubleConv(c[2], c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 4, 2, 1, output_padding=0)
        self.up_norm1 = nn.BatchNorm2d(c[0])
        self.up_act1 = nn.LeakyReLU(0.2, inplace=True)
        self.d1 = DoubleConv(c[1], c[0])
        self.amp_s = nn.Conv2d(c[0], 1, 3, 1, 1)
        self.phs_s = nn.Conv2d(c[0], 1, 3, 1, 1)
        self.amp_p = nn.Conv2d(c[0], 1, 3, 1, 1)
        self.phs_p = nn.Conv2d(c[0], 1, 3, 1, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        y = self.up_act3(self.up_norm3(self.u3(self.bot(self.pool(x3)))))
        y = self.d3(torch.cat([y, x3], 1))
        y = self.up_act2(self.up_norm2(self.u2(y)))
        y = self.d2(torch.cat([y, x2], 1))
        y = self.up_act1(self.up_norm1(self.u1(y)))
        y = self.d1(torch.cat([y, x1], 1))
        lr = lambda t: F.leaky_relu(t, 0.2)
        return (lr(self.amp_s(y))[0, 0], torch.tanh(self.phs_s(y))[0, 0],
                lr(self.amp_p(y))[0, 0], torch.tanh(self.phs_p(y))[0, 0])
