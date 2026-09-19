# -*- coding: utf-8 -*-
"""Fresnel 单次 FFT 传播与 ptychography 前向（与 addip 的角谱是两套物理，不可互换）。"""

from __future__ import annotations

import math
import torch


PI = math.pi

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_paper import Cfg

def make_quad_phase(cfg: Cfg, device, dtype=torch.complex64):
    """exp(-j·k/(2z)·((uΔx1)² + (vΔy1)²))  —— Eq.(2) 花括号里的二次相位因子。

    符号照论文取负。教科书标准 Fresnel 单 FFT 是 +j，两者差一个共轭（等价 z→-z）。
    若重建出来是镜像/共轭，把 --quad-sign 翻过来。

    这个因子固定在实验室系（以光轴为中心），对所有扫描位置是同一个常数张量。
    Eq.(2) 写成"探针平移 P(uΔx1-p_i)、物体不动"，本实现写成"物体 patch 滑动、
    探针不动"——两者是同一件事换了参考系，且后者只需在 M×M 窗口上做 FFT。
    """
    n = cfg.N
    x = (torch.arange(n, device=device, dtype=torch.float64) - n // 2) * cfg.dx1
    X, Y = torch.meshgrid(x, x, indexing="ij")
    k = 2 * PI / cfg.wlength
    ph = cfg.quad_sign * k / (2 * cfg.z) * (X ** 2 + Y ** 2)
    return torch.exp(1j * ph).to(dtype)

def forward_ptycho(obj, probe, positions, Q, n, chunk=0):
    """I_i = |DFT{ P·S_patch·Q }|²  —— Eq.(2)

    obj (M,M)c ; probe (n,n)c ; positions (I,2)long ; Q (n,n)c  ->  (I,n,n) float
    绝对不做逐张归一化：不同扫描点的相对强度本身就是信息。
    """
    idx = torch.arange(n, device=obj.device)

    def _one(pos):
        rows = pos[:, 0:1] + idx[None, :]
        cols = pos[:, 1:2] + idx[None, :]
        psi = obj[rows[:, :, None], cols[:, None, :]] * probe[None] * Q[None]
        far = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(psi, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1))
        return far.real ** 2 + far.imag ** 2

    if chunk <= 0 or chunk >= positions.shape[0]:
        return _one(positions)
    return torch.cat([_one(positions[s:s + chunk])
                      for s in range(0, positions.shape[0], chunk)], 0)
