# -*- coding: utf-8 -*-
"""论文 Eq.(4)/(5) 的损失。"""

from __future__ import annotations

import torch


from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_paper import Cfg

def paper_loss(Ic, Im, S2, gamma, amp_p, S1, beta):
    """Loss  = β·Loss1 + (1-β)·Loss2                                    Eq.(4)
       Loss1 = L2{ (Ic-Im)·S2 + γ·(Ic-Im)·(1-S2) }                      Eq.(5)
       Loss2 = L2{ amp_p·(1-S1) }
    用 L2 范数（不是均值）——论文写的是 L2。
    """
    r = Ic - Im
    loss1 = torch.linalg.vector_norm(r * S2 + gamma * r * (1.0 - S2))
    loss2 = torch.linalg.vector_norm(amp_p * (1.0 - S1))
    return beta * loss1 + (1.0 - beta) * loss2, loss1.detach(), loss2.detach()
