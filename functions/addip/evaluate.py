# -*- coding: utf-8 -*-
"""物体与探针的评估指标。"""

from __future__ import annotations

import numpy as np

from functions.common.metrics import align_global_factor, psnr, ssim

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_torch import Cfg

def evaluate_object(rec, gt, phase_support=0.05):
    rec, _ = align_global_factor(rec, gt)
    ra, ga = np.abs(rec), np.abs(gt)
    dr = ga.max() - ga.min()
    mask = ga > phase_support * ga.max()
    pr = np.where(mask, np.angle(rec), 0.0)
    pg = np.where(mask, np.angle(gt), 0.0)
    dphi = np.angle(np.exp(1j * (np.angle(rec) - np.angle(gt))))[mask]
    pdr = max(pg.max() - pg.min(), 1e-12)
    return {
        "psnr_o_amp": psnr(ra, ga, dr), "ssim_o_amp": ssim(ra, ga, dr),
        "psnr_o_phi": psnr(pr, pg, pdr), "ssim_o_phi": ssim(pr, pg, pdr),
        "rmse_o_phi_rad": float(np.sqrt(np.mean(dphi ** 2))) if mask.any() else float("nan"),
        "relerr_o_complex": float(np.linalg.norm(rec - gt) / np.linalg.norm(gt)),
    }

def evaluate_object_roi(cfg, rec, gt):
    c = cfg.EVAL_CROP
    return evaluate_object(rec[c:-c, c:-c], gt[c:-c, c:-c], cfg.phase_support)

def evaluate_probe(rec_c, gt_field, mask, r_far=None):
    """r_far: 远端能量诊断的参照半径（px），纯几何量。

    模型里没有探针支撑，上面那些指标覆盖整个阵列，看不出"探针在远端长垃圾"。
    这一项 = 重建探针落在 r>r_far 之外的能量占比。真值探针的参照值（r_far=probe_dia=48）
    是 0.0015%；明显高于它 = 数据约束不住远端，需要正则（论文 Eq.5 的 beta 软惩罚），
    接近它 = 不需要任何支撑。
    """
    m = mask > 0
    gt_amp, gt_phi = np.abs(gt_field), np.angle(gt_field) * m
    rec_amp, rec_phi = np.abs(rec_c), np.angle(rec_c) * m
    dr = max(gt_amp.max() - gt_amp.min(), 1e-12)
    pdr = max(gt_phi.max() - gt_phi.min(), 1e-12)
    al, _ = align_global_factor(rec_c * m, gt_field * m)
    den = np.linalg.norm(gt_field * m)
    p_rec = rec_phi[m] - rec_phi[m].mean(); p_gt = gt_phi[m] - gt_phi[m].mean()
    d = np.angle(np.exp(1j * (p_rec - p_gt)))
    return {
        "psnr_p_amp": psnr(rec_amp, gt_amp, dr), "ssim_p_amp": ssim(rec_amp, gt_amp, dr),
        "psnr_p_phi": psnr(rec_phi, gt_phi, pdr), "ssim_p_phi": ssim(rec_phi, gt_phi, pdr),
        "rms_p_phi_rad": float(np.sqrt(np.mean(d ** 2))),
        "relerr_p_complex": float(np.linalg.norm(al - gt_field * m) / den) if den else float("nan"),
        **({} if r_far is None else {"p_far_frac": _far_energy_frac(rec_c, r_far)}),
    }

def _far_energy_frac(field, r_far):
    n0, n1 = field.shape
    _y, _x = np.mgrid[0:n0, 0:n1]
    _rr = np.sqrt((_x - n1 / 2.0) ** 2 + (_y - n0 / 2.0) ** 2)
    e = np.abs(field) ** 2
    return float((e * (_rr > r_far)).sum() / max(e.sum(), 1e-30))
