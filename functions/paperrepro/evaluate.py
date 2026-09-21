# -*- coding: utf-8 -*-
"""照明 ROI 与评估指标。"""

from __future__ import annotations

import math
import numpy as np

from functions.common.metrics import align_global_factor, psnr, ssim

PI = math.pi

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_paper import Cfg

def illum_roi(cfg, probe, positions):
    """照明覆盖区的外接框。

    !! 这一步不能省。物体画布远大于被照明的区域（paper preset: 612² 画布里只有
       约 150² 被照到），在没有数据约束的区域上算 SSIM/PSNR 会把指标整个压死 ——
       看着像"重建不出来"，其实是评价区选错了。论文取"中心 200×200"，对它的几何
       恰好约等于照明区；换任何其它参数都必须按覆盖图重新算。
    """
    M = cfg.obj_size
    cov = np.zeros((M, M), np.float64)
    w = np.abs(probe) ** 2
    for (py, px) in positions:
        cov[py:py + cfg.N, px:px + cfg.N] += w
    m = cov > 0.5 * cov.max()
    r = np.where(m.any(1))[0]; c = np.where(m.any(0))[0]
    return slice(r[0], r[-1] + 1), slice(c[0], c[-1] + 1)

def evaluate(rec, gt):
    """论文判据 (3)(4)。传进来的 rec/gt 应当【已经裁到照明 ROI】。先消全局复因子。"""
    rec = align_global_factor(rec, gt)[0]
    ra, ga = np.abs(rec), np.abs(gt)
    dra = ga.max() - ga.min()
    pr, pg = np.angle(rec), np.angle(gt)
    pr = pr - pr.mean() + pg.mean()
    drp = max(pg.max() - pg.min(), 1e-12)
    h, w = ga.shape
    k = min(200, h, w)
    s = slice((h - k) // 2, (h - k) // 2 + k)
    return {
        "ssim_amp": ssim(ra, ga, dra), "ssim_phs": ssim(pr, pg, drp),
        "psnr_amp": psnr(ra[s, s], ga[s, s], dra),
        "psnr_phs": psnr(pr[s, s], pg[s, s], drp),
        "relerr": float(np.linalg.norm(rec - gt) / np.linalg.norm(gt)),
    }

def probe_relerr(rec_p, gt_p):
    """探针复相对误差，消去全局复因子。三种算法同一口径。"""
    al = align_global_factor(rec_p, gt_p)[0]
    den = np.linalg.norm(gt_p)
    return float(np.linalg.norm(al - gt_p) / den) if den else float("nan")

def seam_diag(rec, gt, amp_s):
    """参数化接缝诊断。

    论文的输出头是 amp=LeakyReLU（可为负）+ phase=π·tanh。二者组合表达能力没有缺口
    （amp<0 等价于相位 +π），但要从 φ 走到 φ+π，优化器要么把 tanh 横跨整个值域，
    要么让振幅穿过 0 —— 两条路都是高成本区，走过去就会在相位图上留下人为的 π 跳变。

    wrap_frac : 对齐后相位残差 |Δφ| > 0.9π 的像素占比（π 跳变的直接证据）
    signmix   : 振幅符号的【少数派】占比。整幅统一取负是规范选择（等价于全局相位 π，
                对齐时就消掉了），不是问题；只有符号在空间上混着才会在相位图上留缝。
                0 = 全图同号（干净），0.5 = 一半一半（最糟）。
    """
    r = align_global_factor(rec, gt)[0]
    d = np.angle(np.exp(1j * (np.angle(r) - np.angle(gt))))
    neg = float((amp_s < 0).mean())
    return float((np.abs(d) > 0.9 * PI).mean()), min(neg, 1.0 - neg)
