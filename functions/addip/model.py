# -*- coding: utf-8 -*-
"""物体网络与输出头的复场合成。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from functions.common.unet import DoubleConv
from functions.common.wavelet_skip import WaveletSkip

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_torch import Cfg

class ProPtyUNet(nn.Module):
    """论文 Fig.1(b)。相对 PhysenNet_torch.py 的 net_model 有三处必改:

      1. 首层输入通道 1 -> J（= 衍射图张数），输入是零填充后的实测衍射图堆栈，全程不变；
      2. 末层【去掉 BatchNorm2d(1)】。原代码在输出层做 BN 会把输出强制成零均值单位方差，
         直接毁掉振幅的绝对尺度 —— 这是从 PhysenNet 迁移过来时最致命的一处；
      3. 原 forward() 把 encoder 重复算了 4 遍（x6_1/x7_1/x8_1/x9_1），这里只算一次。

    【本版改动】输出头返回【未激活】的原始张量，激活与复数合成统一放在 make_field()，
    激活与复数合成统一放在 make_field()，网络本身不需要知道用哪种参数化。

    n_fields=1 -> 只出物体（探针另有其人）；n_fields=2 -> 论文的物体+探针共享写法。
    """

    def __init__(self, in_ch, base=32, n_fields=1, ph_ch=2,
                 skip_mode="concat", wavelet_threshold=.01):
        super().__init__()
        if skip_mode not in ("concat", "wavelet", "wavelet-identity"):
            raise ValueError(f"Unknown skip_mode: {skip_mode}")
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3 = DoubleConv(in_ch, c[0]), DoubleConv(c[0], c[1]), DoubleConv(c[1], c[2])
        self.bot = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2, 2)
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 3, 2, 1, output_padding=1)
        self.d3 = DoubleConv(c[3], c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 3, 2, 1, output_padding=1)
        self.d2 = DoubleConv(c[2], c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 3, 2, 1, output_padding=1)
        self.d1 = DoubleConv(c[1], c[0])
        self.n_fields, self.ph_ch = n_fields, ph_ch
        self.head_amp = nn.Conv2d(c[0], n_fields, 3, 1, 1)
        self.head_phs = nn.Conv2d(c[0], n_fields * ph_ch, 3, 1, 1)
        self.skip_filters = nn.ModuleList([
            nn.Identity() if skip_mode == "concat" else
            WaveletSkip(wavelet_threshold, identity=skip_mode == "wavelet-identity")
            for _ in range(3)])

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        xb = self.bot(self.pool(x3))
        y = self.d3(torch.cat([self.u3(xb), self.skip_filters[2](x3)], 1))
        y = self.d2(torch.cat([self.u2(y), self.skip_filters[1](x2)], 1))
        y = self.d1(torch.cat([self.u1(y), self.skip_filters[0](x1)], 1))
        return self.head_amp(y)[0], self.head_phs(y)[0]     # (F,H,W), (F*ph,H,W)

def make_field(amp_raw, phs_raw):
    """原始头 -> 复数场。振幅 softplus，相位 cos/sin 单位圆。

    amp_raw: (H,W) ; phs_raw: (2,H,W)

    cos/sin 而不是 span*tanh：没有 ±pi 边界、没有饱和、没有 2pi 缠绕的等价解 ——
    这三样正是让优化在等价解之间漂移（loss 不变而指标变差）的来源。
    softplus 而不是 leaky_relu：后者允许负振幅 = 隐藏的 pi 相位翻转。
    """
    amp = F.softplus(amp_raw)
    c, sn = phs_raw[0], phs_raw[1]
    n = torch.sqrt(c * c + sn * sn + 1e-8)
    return torch.complex(amp * c / n, amp * sn / n)
