# -*- coding: utf-8 -*-
"""物理伴随实空间融合 —— 把测量数据反投影到全局物体坐标，做成固定 4 通道网络输入。

动机：raw 模式把整摞衍射图当通道喂进 U-Net，输入通道数 = 扫描点数，而且网络要
自己学会「探测器频域坐标 -> 物体实空间坐标」这个映射。这里改成先用已知的前向模型
把数据搬到物体坐标系，网络只需要在实空间做去噪/补全。

四个通道：
  0,1  伴随融合复场 B 的实部 / 虚部
  2    照明覆盖强度 Γ（归一化）
  3    数据一致性残差强度 ρ（归一化并截到 [0,1]）

【不读真值】O_ref 恒为全 1 的中性物体；P_ref 是 make_probe_init 给的 P0
（只有 probe_mode='truth' 这个上界对照才允许传真值探针进来）。
"""

from __future__ import annotations

import math

import numpy as np
import torch

from functions.addip.losses import cabs
from functions.addip.optics import crop_patch_torch, forward_torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_torch import Cfg

PI = math.pi


def _coverage(cfg: Cfg, P_ref, corners_act, canvas_shape, dev):
    """Γ(r) = Σ_{i∈A} S_iᵀ |P_ref|²，用 crop_patch_torch 的 VJP 算。

    【不能手工 scatter】整数坐标下 scatter 和 VJP 等价，但亚像素坐标（scan_step
    为小数）下 crop_patch_torch 会在 patch 上做相位斜坡，它的伴随不是"取整后散射"。用 VJP
    才能保证覆盖图与前向严格同一个算子。
    """
    G = torch.zeros(canvas_shape, dtype=torch.complex64, device=dev, requires_grad=True)
    W = crop_patch_torch(G, corners_act, cfg.N)                      # (J,N,N)
    w = (P_ref.abs() ** 2).to(W.dtype).expand_as(W)
    g = torch.autograd.grad(W, G, grad_outputs=w)[0]
    return g.real.clamp_min(0.0)                                     # 覆盖是实的非负量


def build_adjoint_features(cfg: Cfg, sqrtI, corners, H, P_ref, active_idx, O_ref):
    """返回 (1, 4, N_OBJ, N_OBJ) 的实数特征，已 detach。

    sqrtI    : (J,N,N) 实测衍射振幅
    corners  : (J,2)   全部扫描坐标
    H        : (N,N)   传播算子
    P_ref    : (N,N)   参考探针（pixel 模式 = P0，禁止真值）
    active_idx: 参与融合的位置编号
    O_ref    : (M,M)   中性参考物体（全 1），禁止真值
    """
    eps = float(cfg.adjoint_eps)
    dev = O_ref.device
    ca = corners[active_idx]

    # ---- 1) 在中性物体上做一次前向，再做测量振幅替换 ----
    O_var = O_ref.detach().clone().requires_grad_(True)
    U_pred = forward_torch(cfg, O_var, P_ref, ca, H)
    phase = U_pred / cabs(U_pred).clamp_min(eps)
    U_proj = sqrtI[active_idx] * phase
    delta_U = (U_proj - U_pred).detach()

    # ---- 2) 复数 VJP = 物理伴随。autograd 自动兼容整数与亚像素坐标 ----
    adj = torch.autograd.grad(U_pred, O_var, grad_outputs=delta_U,
                              retain_graph=False, create_graph=False)[0]

    # ---- 3) 覆盖归一化，未照明区保持中性物体 ----
    gamma = _coverage(cfg, P_ref, ca, O_ref.shape, dev)
    delta_O = adj / gamma.clamp_min(eps)
    B = O_ref + delta_O
    thr = 1e-3 * gamma.max().clamp_min(eps)          # 低于峰值千分之一视为未照明
    lit = gamma > thr
    B = torch.where(lit, B, O_ref)

    # 用照明区内 |B| 的【中位数】定标，使典型振幅 ≈ 1。
    # 不对实部/虚部分别 min-max —— 那会破坏复场的相位关系。
    if lit.any():
        med = B[lit].abs().median().clamp_min(eps)
        B = B / med

    # ---- 4) 四通道 ----
    rho = adj.abs() / gamma.clamp_min(eps)
    rho_q = rho.flatten().quantile(0.99).clamp_min(eps)
    feat = torch.stack([
        B.real,
        B.imag,
        gamma / gamma.max().clamp_min(eps),
        (rho / rho_q).clamp(0.0, 1.0),
    ], dim=0)[None].contiguous()
    return feat.detach().to(torch.float32)


def save_adjoint_features(cfg, feat, gamma_ch_thr, path):
    """诊断图：B 振幅 / B 相位 / 覆盖 / 残差。相位只在覆盖有效区显示。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    f = feat[0].detach().cpu().numpy()
    B = f[0] + 1j * f[1]
    cov, res = f[2], f[3]
    m = cov > gamma_ch_thr                      # 未照明区的 angle() 是纯噪声，掩掉
    panels = [
        (np.abs(B), "B amplitude", (0.0, 2.0), "gray"),
        (np.where(m, np.angle(B), 0.0), "B phase (lit only)", (-PI, PI), "twilight"),
        (cov, "normalized coverage", (0.0, 1.0), "viridis"),
        (res, "normalized residual", (0.0, 1.0), "magma"),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    for a, (im, t, (lo, hi), cm) in zip(ax.ravel(), panels):
        h = a.imshow(im, cmap=cm, vmin=lo, vmax=hi)
        a.set_title(f"{t}  [{im.min():.3f}, {im.max():.3f}]", fontsize=9)
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(h, ax=a, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[net-input] 诊断图 -> {path}")
