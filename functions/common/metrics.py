# -*- coding: utf-8 -*-
"""两条实验线共用的图像质量指标。"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def ssim(x, y, data_range=None, win_size=7):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if data_range is None:
        data_range = max(x.max(), y.max()) - min(x.min(), y.min())
    if data_range == 0:
        return 1.0
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    NP = win_size ** x.ndim
    cn = NP / (NP - 1)
    f = lambda a: uniform_filter(a, size=win_size)
    ux, uy = f(x), f(y)
    vx = cn * (f(x * x) - ux * ux); vy = cn * (f(y * y) - uy * uy)
    vxy = cn * (f(x * y) - ux * uy)
    S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux ** 2 + uy ** 2 + C1) * (vx + vy + C2))
    pad = (win_size - 1) // 2
    return float(S[pad:-pad, pad:-pad].mean())

def psnr(rec, gt, data_range=None):
    rec = np.asarray(rec, np.float64); gt = np.asarray(gt, np.float64)
    if data_range is None:
        data_range = gt.max() - gt.min()
    mse = np.mean((rec - gt) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(data_range ** 2 / mse))

def align_global_factor(rec, gt):
    """消去全局复因子: c = <rec,gt>/<rec,rec>。相位恢复天然有这个自由度。"""
    den = np.vdot(rec, rec)
    if den == 0:
        return rec, 0j
    c = np.vdot(rec, gt) / den
    return rec * c, c
