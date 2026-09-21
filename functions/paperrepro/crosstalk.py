# -*- coding: utf-8 -*-
"""object–probe 串扰的定量指标。

【动机】物体误差和探针误差各自的大小说明不了问题：blind ptychography 的观测量是
乘积 O·P，所以两个场的误差可以沿着耦合方向相互抵消，数据拟合得很好、两个场却都错。
这才是"串扰"。下面把它写成可测量的数。

一阶展开（对齐规范之后，O = O_gt + ΔO，P = P_gt + ΔP）：

    Δψ_i ≈ S_i(ΔO)·P_gt  +  S_i(O_gt)·ΔP  +  O(Δ²)
           \____a_i____/     \____b_i____/
            物体误差经真值探针     探针误差经真值物体
            投到出射波上         投到出射波上

定义

    κ (kappa) = ‖a + b‖ / (‖a‖ + ‖b‖)        串扰抵消比
    ρ (rho)   = ‖b‖ / (‖a‖ + ‖b‖)            误差分配：多少比例挂在探针上
    ν (nu)    = 探针能量落在"物体≈0"区域的比例   零空间占用

κ → 0 : a 与 b 几乎完全抵消 —— 误差整个躺在 O–P 耦合的零空间里 = 纯串扰
κ → 1 : a 与 b 正交/独立叠加 —— 两个场各错各的，不是串扰
ν     : 自由像素探针有 2·N² 个未知量，其中只有针孔内那 ~2·πR² 个被数据定住；
        剩下的是零空间。ν 直接量化算法有多少能量跑进了这个零空间。

【2026-09-21 实测：κ 目前不能用，ν 可以用】
在 paper preset 的三次 2000 步运行上测下来：

    指标        ProPtyNet     纯 AD        DIP
    κ(probe)    0.9880       0.9753      0.9265      <- 三者挤在一起，不区分
    ρ(probe)    0.0124       0.0249      0.0767      <- 随规范翻转到 0.98，不稳
    ν           0.0049       0.0003      0.8902      <- 差三个数量级，强区分

κ 退化的原因：在 probe 锚定规范下 ΔP 本来就被对齐到很小，‖b‖ << ‖a‖，
于是 κ ≈ ‖a‖/(‖a‖+‖b‖) ≈ 1，恒等于 1 而不是在测抵消。换 object 锚定则反过来。
要救它得把 κ 定义成【沿规范轨道取最小】κ* = min_a κ(a)，即"存不存在一个规范
使两个误差抵消"。那才是串扰的正确提法。κ* 尚未实现。

另外物体内部的振幅↔相位线性相关也测了，三者都在 0.05~0.12，同样不区分 ——
AD 图上肉眼可见的放射条纹是 ± 对称结构，线性相关会抵消掉，要换成相位梯度
或频谱重叠才测得出来。

所以现阶段只用 ν 和 probe_relerr_pin 下结论。

【规范选择会影响 κ 和 ρ】(O,P) → (aO, P/a) 是精确的规范自由度，沿着这条轨道走
a 和 b 会反向缩放。所以必须说清楚锚在哪一端：
    gauge="probe"  : 用针孔支撑内的最小二乘把 P 对齐到真值，剩下的由 ΔO 承担（默认）
    gauge="object" : 用照明区内的最小二乘把 O 对齐到真值，剩下的由 ΔP 承担
两个都算出来一起报，结论才不依赖这个选择。
"""

from __future__ import annotations

import numpy as np
import torch

from functions.paperrepro.optics import forward_field

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ProPtyNet_paper import Cfg


def _ls_scalar(x, y, w=None):
    """min_a ‖a·x − y‖ 的闭式解。w 是布尔掩膜。"""
    if w is not None:
        x, y = x[w], y[w]
    d = np.vdot(x, x).real
    return (np.vdot(x, y) / d) if d > 1e-30 else 1.0 + 0j


def pinhole_mask(cfg: Cfg, margin=1.0):
    n = cfg.N
    yy, xx = np.mgrid[0:n, 0:n] - n / 2
    return np.sqrt(xx ** 2 + yy ** 2) <= margin * cfg.probe_diam_px / 2


def lit_mask(cfg: Cfg, probe_gt, positions, thr=1e-3):
    """真值探针实际照到的画布像素。"""
    M = cfg.obj_size
    cov = np.zeros((M, M))
    w = np.abs(probe_gt) ** 2
    for (py, px) in positions:
        cov[py:py + cfg.N, px:px + cfg.N] += w
    return cov > thr * cov.max()


def crosstalk(cfg: Cfg, O_rec, P_rec, O_gt, P_gt, positions, Q, device,
              gauge="probe", chunk=8):
    """返回 {kappa, rho, nu, norm_a, norm_b} 。O/P 都是 numpy 复数。"""
    pin = pinhole_mask(cfg)
    lit = lit_mask(cfg, P_gt, positions)

    # ---- 规范对齐：沿 (aO, P/a) 轨道选一个点 ----
    if gauge == "probe":
        a = _ls_scalar(P_rec, P_gt, pin)          # 把 P 对到真值
        P = P_rec * a
        O = O_rec / a
    else:
        a = _ls_scalar(O_rec, O_gt, lit)          # 把 O 对到真值
        O = O_rec * a
        P = P_rec / a

    dO = (O - O_gt).astype(np.complex64)
    dP = (P - P_gt).astype(np.complex64)

    # ---- 一阶投影到出射波 ----
    t = lambda z: torch.from_numpy(np.ascontiguousarray(z)).to(device)
    pos_t, Qi = t(positions.astype(np.int64)), Q
    idx = torch.arange(cfg.N, device=device)
    O_gt_t, P_gt_t, dO_t, dP_t = t(O_gt), t(P_gt), t(dO), t(dP)

    na = nb = nab = 0.0
    with torch.no_grad():
        for s in range(0, len(positions), chunk):
            p = pos_t[s:s + chunk]
            rows = p[:, 0:1] + idx[None, :]
            cols = p[:, 1:2] + idx[None, :]
            sel = (rows[:, :, None], cols[:, None, :])
            A = dO_t[sel] * P_gt_t[None]                 # a_i
            B = O_gt_t[sel] * dP_t[None]                 # b_i
            na += (A.abs() ** 2).sum().item()
            nb += (B.abs() ** 2).sum().item()
            nab += ((A + B).abs() ** 2).sum().item()
    na, nb, nab = np.sqrt(na), np.sqrt(nb), np.sqrt(nab)

    # ---- 零空间占用：探针能量落在"物体几乎为 0"的位置 ----
    # 用重建物体本身判断哪里是 0（这是算法自己的选择，不能用真值）
    o_amp = np.abs(O)
    ref = np.median(o_amp[lit]) if lit.any() else 1.0
    dark = o_amp < 0.05 * max(ref, 1e-12)
    M_, n = cfg.obj_size, cfg.N
    hit = np.zeros((n, n))                              # 探针每个像素落在暗区的次数
    for (py, px) in positions:
        hit += dark[py:py + n, px:px + n]
    hit /= max(len(positions), 1)
    e = np.abs(P) ** 2
    nu = float((e * hit).sum() / max(e.sum(), 1e-30))

    return {"kappa": float(nab / max(na + nb, 1e-30)),
            "rho": float(nb / max(na + nb, 1e-30)),
            "nu": nu, "norm_a": float(na), "norm_b": float(nb),
            "gauge": gauge,
            "probe_relerr_pin": float(np.linalg.norm((P - P_gt)[pin])
                                      / max(np.linalg.norm(P_gt[pin]), 1e-30)),
            "probe_energy_in_pin": float((np.abs(P) ** 2)[pin].sum()
                                         / max((np.abs(P) ** 2).sum(), 1e-30))}


def report(cfg, O_rec, P_rec, O_gt, P_gt, positions, Q, device, tag=""):
    """两种规范都算，打印并返回 dict。"""
    out = {}
    for g in ("probe", "object"):
        m = crosstalk(cfg, O_rec, P_rec, O_gt, P_gt, positions, Q, device, gauge=g)
        out[g] = m
        print(f"[crosstalk{tag}] gauge={g:6s} κ={m['kappa']:.4f} ρ={m['rho']:.4f} "
              f"ν={m['nu']:.4f} | ‖a‖={m['norm_a']:.3e} ‖b‖={m['norm_b']:.3e} "
              f"| 针孔内探针误差 {m['probe_relerr_pin']:.4f} "
              f"针孔内能量 {100*m['probe_energy_in_pin']:.1f}%")
    return out
