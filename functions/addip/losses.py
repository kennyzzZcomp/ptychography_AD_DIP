# -*- coding: utf-8 -*-
"""数据项、正则项与数值安全的小工具。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_torch import Cfg

def cabs(z, eps=1e-12):
    """torch.abs 对复数在 z=0 处的反向是 grad*z/|z| -> NaN。加 eps 走 sqrt 更安全。"""
    return torch.sqrt(z.real ** 2 + z.imag ** 2 + eps)

def safe_angle(z, eps=1e-20):
    """angle 在 0 处梯度是 -inf。两处 where 缺一不可（0*inf 仍然 nan）。"""
    mag2 = z.real ** 2 + z.imag ** 2
    ok = mag2 > eps
    z_safe = torch.where(ok, z, torch.ones_like(z))
    return torch.where(ok, torch.angle(z_safe), torch.zeros_like(mag2))

def masked_mse(pred, target, M=None):
    """M: 1=进 loss，0=留出。None 表示全用。"""
    if M is None:
        return F.mse_loss(pred, target)
    d = (pred - target) ** 2 * M
    return d.sum() / M.sum().clamp_min(1.0)

def held_out_mse(pred, target, M):
    d = (pred - target) ** 2 * (1.0 - M)
    return (d.sum() / (1.0 - M).sum().clamp_min(1.0)).item()

def data_loss_direct(U, sqrtI):
    """notebook 的写法: Keras mse(|U|, sqrt(I))，逐元素均值。"""
    return F.mse_loss(cabs(U), sqrtI)

def _grad_xy(u):
    """u: (H,W) 实数"""
    return u[1:, :] - u[:-1, :], u[:, 1:] - u[:, :-1]

def tgv_loss(u, w1=1.0, w2=2.0, beta=1e-2):
    dx, dy = _grad_xy(u)
    tv1 = dx.abs().mean() + dy.abs().mean()
    dxx = dx[1:, :] - dx[:-1, :]
    dyy = dy[:, 1:] - dy[:, :-1]
    tv2 = dxx.abs().mean() + dyy.abs().mean()
    return beta * (w1 * tv1 + w2 * tv2)

def tv_loss(u, beta=1e-2, eps=1e-8):
    dx, dy = _grad_xy(u)
    return beta * torch.sqrt(dx[:, :-1] ** 2 + dy[:-1, :] ** 2 + eps).mean()
