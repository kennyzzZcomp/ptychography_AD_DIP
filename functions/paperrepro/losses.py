# -*- coding: utf-8 -*-
"""论文 Eq.(4)/(5) 的损失。"""

from __future__ import annotations

import torch


from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_paper import Cfg

def overexposure_mask(Im):
    """Eq.(6) for measurements clipped to [0,1]: only saturated pixels are downweighted.

    Values strictly below 1 remain valid; no near-saturation tolerance is added.
    """
    return (Im < 1.0).to(dtype=Im.dtype)


def pinhole_mask(n, radius_px, device, dtype=torch.float32):
    """Eq.(6): r < pinhole radius, without imposing a hard constraint on P."""
    coordinates = torch.arange(n, device=device, dtype=torch.float64) - n / 2
    rr_squared = coordinates[:, None].square() + coordinates[None, :].square()
    return (rr_squared < float(radius_px) ** 2).to(dtype=dtype)


def paper_loss(Ic, Im, S2, gamma, amp_p, S1, beta):
    """Loss  = β·Loss1 + (1-β)·Loss2                                    Eq.(4)
       Loss1 = L2{ (Ic-Im)·S2 + γ·(Ic-Im)·(1-S2) }                      Eq.(5)
       Loss2 = L2{ amp_p·(1-S1) }
    用未平方的整体 L2 范数（不是 MSE、平方和或逐帧范数的平均）。
    Ic 是前向模型的衍射强度，不是传播场振幅；此函数不估计标定系数。
    S1 只作用于探针软惩罚，不能用它把前向中的探针硬截断。
    """
    r = Ic - Im
    loss1 = torch.linalg.vector_norm(r * S2 + gamma * r * (1.0 - S2))
    loss2 = torch.linalg.vector_norm(amp_p * (1.0 - S1))
    return beta * loss1 + (1.0 - beta) * loss2, loss1.detach(), loss2.detach()
