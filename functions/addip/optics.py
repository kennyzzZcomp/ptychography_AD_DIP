# -*- coding: utf-8 -*-
"""角谱传播、亚像素裁剪与前向模型。"""

from __future__ import annotations

import math
import numpy as np
import torch


PI = math.pi

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_torch import Cfg

def make_H(cfg: Cfg, dz, band_limit=True):
    """H(fx,fy,dz) = exp(i·sqrt(k²-fx²-fy²)·dz)，建在未 shift 的 FFT 排布上。

    倏逝波区必须写成衰减；band_limit 是 Matsushima 带限。
    """
    N, dx, k0 = cfg.N, cfg.dx, cfg.k0
    f = 2 * PI * np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(f, f, indexing="ij")
    arg = k0 ** 2 - FX ** 2 - FY ** 2
    kz = np.sqrt(np.abs(arg))
    H = np.where(arg >= 0, np.exp(1j * kz * dz), np.exp(-kz * abs(dz)))
    if band_limit and dz != 0:
        f_lim = 2 * PI / (cfg.wlength * np.sqrt((2 * abs(dz) / (N * dx)) ** 2 + 1))
        H = np.where(FX ** 2 + FY ** 2 <= f_lim ** 2, H, 0)
    return H.astype(np.complex64)

def propagate_np(field, H):
    return np.fft.ifft2(np.fft.fft2(field) * H)

def fourier_shift_np(f, rx, ry):
    n0, n1 = f.shape[:2]
    fx = np.fft.fftfreq(n0).reshape(n0, 1)
    fy = np.fft.fftfreq(n1).reshape(1, n1)
    return np.fft.ifft2(np.fft.fft2(f) * np.exp(-2j * PI * (fx * rx + fy * ry)))

def crop_patch_np(canvas, corner, n):
    c = np.asarray(corner, float)
    i = np.floor(c).astype(int)
    fr = c - i
    assert i.min() >= 0 and i[0] + n <= canvas.shape[0] and i[1] + n <= canvas.shape[1], \
        f"patch 越界: corner={c}, n={n}, canvas={canvas.shape}"
    w = canvas[i[0]:i[0] + n, i[1]:i[1] + n]
    return w if not fr.any() else fourier_shift_np(w, -fr[0], -fr[1])

def crop_patch_torch(canvas: torch.Tensor, corners: torch.Tensor, n: int):
    """canvas: (M,M) complex ; corners: (I,2) float（左上角，可小数）-> (I,n,n) complex

    整数部分用 gather（反向 = scatter-add 回画布，等价于 ePIE 手写的贴回更新），
    小数部分只在裁出的 patch 上做相位斜坡（tike / PtyRAD 的做法）。
    """
    ij = torch.floor(corners)
    fr = corners - ij
    ij = ij.long()
    ar = torch.arange(n, device=canvas.device)
    rows = ij[:, 0:1] + ar[None, :]                       # (I,n)
    cols = ij[:, 1:2] + ar[None, :]
    w = canvas[rows[:, :, None], cols[:, None, :]]        # (I,n,n)

    if torch.count_nonzero(fr) == 0:
        return w
    ff = torch.fft.fftfreq(n, device=canvas.device, dtype=torch.float32)
    # 位移量取负号，与 crop_patch_np 一致
    ph = -2 * PI * (ff[None, :, None] * (-fr[:, 0, None, None])
                    + ff[None, None, :] * (-fr[:, 1, None, None]))
    ramp = torch.polar(torch.ones_like(ph), ph).to(w.dtype)
    return torch.fft.ifft2(torch.fft.fft2(w) * ramp)

def forward_np(cfg, O, P, pos, H):
    psi = crop_patch_np(O, cfg.PATCH_C0 - np.asarray(pos, float), cfg.N) * P
    return np.fft.ifft2(np.fft.fft2(psi) * H)

def forward_torch(cfg, O, P, corners, H):
    """O:(M,M)c ; P:(n,n)c ; corners:(I,2)f ; H:(n,n)c -> U:(I,n,n) complex"""
    psi = crop_patch_torch(O, corners, cfg.N) * P[None]
    return torch.fft.ifft2(torch.fft.fft2(psi) * H[None])
