# -*- coding: utf-8 -*-
"""真值物体/探针、探针初值、扫描位置与衍射数据仿真。"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from functions.addip.optics import forward_np, make_H, propagate_np
from functions.common.assets import asset_dir
from functions.common.metrics import align_global_factor

PI = math.pi

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_torch import Cfg

def _center_crop_square(a):
    """居中裁成正方形。原来直接 resize 到 (n,n) 不保持宽高比，非正方形图会被拉扁 ——
    对 USAF 这种分辨率靶尤其致命（线对不再是方的，横竖分辨率不一致）。
    对已有的 256x256 素材这是 no-op，不影响历史结果。"""
    h, w = a.shape[:2]
    if h == w:
        return a
    m = min(h, w)
    return a[(h - m) // 2:(h - m) // 2 + m, (w - m) // 2:(w - m) // 2 + m]

def _imread_resize(path: Path, n: int):
    """读灰度 -> 居中裁方 -> 缩放到 n x n。

    下采样超过 2 倍时改用 INTER_AREA（面积平均）。INTER_CUBIC 在大倍率下采样时
    不做抗混叠预滤波，会把高频折叠成摩尔纹 —— 例如 2498px 的 USAF 靶直接 cubic
    降到 224px，细线对会变成一团假条纹，等于给了个假的样品。
    已有素材 256->224 只有 1.14 倍，仍走 cubic，历史结果逐位不变。
    """
    try:
        import cv2
        a = cv2.imread(str(path), 0)
        if a is None:
            raise IOError(path)
        a = _center_crop_square(a)
        interp = cv2.INTER_AREA if a.shape[0] > 2 * n else cv2.INTER_CUBIC
        return cv2.resize(a, (n, n), interpolation=interp).astype(float)
    except ImportError:
        from PIL import Image
        a = np.asarray(Image.open(path).convert("L"), dtype=float)
        a = _center_crop_square(a)
        rs = Image.BOX if a.shape[0] > 2 * n else Image.BICUBIC
        return np.asarray(Image.fromarray(a).resize((n, n), rs), dtype=float)

def make_truth(cfg: Cfg):
    """返回 obj(复), probe(复, 峰值归一), rr(到中心的像素距离)。"""
    d = asset_dir(cfg)

    def _resolve(name):
        p = Path(name)
        return p if (p.is_absolute() or p.exists()) else (d / name)

    _a = _imread_resize(_resolve(cfg.obj_amp_img), cfg.N_OBJ)
    _a = cfg.obj_amp_min + (1.0 - cfg.obj_amp_min) * _a / _a.max()
    _p = _imread_resize(_resolve(cfg.obj_phase_img), cfg.N_OBJ)
    _p = -1.0 + 2.0 * (_p - _p.min()) / max(_p.max() - _p.min(), 1e-12)
    obj = (_a * np.exp(1j * cfg.obj_phase_span * _p)).astype(np.complex64)

    N = cfg.N
    yy, xx = np.mgrid[0:N, 0:N] - N / 2
    rr = np.sqrt(xx ** 2 + yy ** 2)

    pupil_mask = rr <= cfg.R_AP
    if cfg.probe_aberr > 0:
        _r = gaussian_filter(np.random.default_rng(cfg.aberr_seed).normal(size=(N, N)), 1.5)
        _phi_ap = _r * (cfg.probe_aberr / np.sqrt(np.mean(_r[pupil_mask] ** 2)))
    else:
        _phi_ap = np.zeros((N, N))
    pupil_true = (pupil_mask * np.exp(1j * _phi_ap)).astype(np.complex64)

    probe = propagate_np(pupil_true, make_H(cfg, cfg.z_probe))
    probe = (probe / np.abs(probe).max()).astype(np.complex64)
    return obj, probe, rr

def make_probe_init(cfg: Cfg, rr):
    """探针初值 P0 = 平滑圆盘 + 零相位，不传播。

    只编码"光阑大概多大"。不给波前相位，也不给传播产生的 Fresnel 环纹 ——
    本几何（R_AP=18px, z=3mm）实测 err_P0 ≈ 0.27，拆解为只丢相位 0.28 /
    只丢振幅 0.30，两者相当。

    【ad / net 两条路径必须都调这个函数】。历史教训：AD 分支曾经自己复制了一份
    初值构造代码，结果换初值的开关只对 net 生效，两个方法在不同起跑线上比。
    """
    amp = (rr <= cfg.R_AP).astype(np.float32)
    s_px = float(cfg.probe_init_sigma) * float(cfg.R_AP)
    if s_px > 0:
        amp = gaussian_filter(amp, sigma=s_px)
    P0 = amp.astype(np.complex64)                      # 零相位
    return (P0 / np.abs(P0).max()).astype(np.complex64)

def probe_init_err(cfg: Cfg, rr, probe):
    """P0 相对真值探针的复相对误差（消去全局复因子，与 evaluate_probe 同口径）。

    err_P0 是刻画"探针先验强度"的唯一诚实数字，ad 与 net 同口径同函数。
    """
    P0 = make_probe_init(cfg, rr)
    Pt = probe.astype(np.complex64)
    return float(np.linalg.norm(align_global_factor(P0, Pt)[0] - Pt)
                 / max(np.linalg.norm(Pt), 1e-12))

def make_scan_positions(cfg: Cfg):
    """确定性扫描位置。raster = 规则栅格；fermat = 费马螺旋。"""
    pat, n_pos, step = cfg.scan_pattern, cfg.scan_npos, cfg.scan_step
    if pat == "raster":
        k = int(round(np.sqrt(n_pos)))
        off = (k - 1) * step / 2
        p = np.array([[i * step - off, j * step - off]
                      for i in range(k) for j in range(k)], float)
    elif pat == "fermat":
        R = step * np.sqrt(n_pos) / 2
        n = np.arange(1, n_pos + 1)
        r = (R / np.sqrt(n_pos)) * np.sqrt(n)
        th = n * np.deg2rad(137.508)
        p = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    else:
        raise ValueError(f"scan_pattern={pat}，只能是 raster | fermat")
    return p.astype(np.float32)

def check_scan_fits(cfg, positions, verbose=True):
    """位置越界必须在跑之前拦住。

    crop_patch_torch 用 tensor 高级索引取 patch。corner 过【大】会 IndexError，
    但 corner 为【负】时 torch 跟 python 一样静默回绕到画布另一侧 —— 不报错，
    重建看起来正常但完全是错的。所以这里显式检查。
    """
    P = np.asarray(positions, float)
    mx = float(np.abs(P).max())
    if mx > cfg.SCAN_LIMIT:
        raise ValueError(
            f"扫描位置极值 {mx:.1f} px 超过画布余量 {cfg.SCAN_LIMIT:.0f} px，patch 会越出画布。\n"
            f"  修法: --n-obj 提到 >= {int(np.ceil(cfg.N + 2 * mx))}，"
            f"或调小 --scan-step / --scan-npos。")
    dd = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)) + np.eye(len(P)) * 1e9
    nn_ = dd.min(1)
    lin = 1 - np.median(nn_) / cfg.probe_dia
    ar = float(np.median([overlap_areal(d, cfg.probe_dia) for d in nn_]))
    if verbose:
        print(f"[scan] {len(P)} 点 step {cfg.scan_step:g} {cfg.scan_pattern}  "
              f"线性重叠 {lin:.1%}  面积重叠 {ar:.1%}  "
              f"位置极值 {mx:.1f}/{cfg.SCAN_LIMIT:.0f}px  "
              f"每像素被照亮 {len(P)*np.pi*(cfg.probe_dia/2)**2/(2*mx+cfg.probe_dia)**2:.2f} 次")
    return {"n": len(P), "linear_overlap": lin, "areal_overlap": ar, "pos_max": mx}

def overlap_areal(d, D):
    R = D / 2.0
    if d >= 2 * R:
        return 0.0
    if d <= 0:
        return 1.0
    a = 2 * R ** 2 * np.arccos(d / (2 * R)) - (d / 2) * np.sqrt(max(4 * R ** 2 - d ** 2, 0))
    return float(a / (PI * R ** 2))

def simulate(cfg: Cfg, obj, probe, positions, H_true):
    """生成衍射强度。噪声 = 可选的散粒噪声，别的都没有。"""
    I = np.empty((len(positions), cfg.N, cfg.N), np.float32)
    for i, p in enumerate(positions):
        I[i] = np.abs(forward_np(cfg, obj, probe, p, H_true)) ** 2
    I_clean = I.copy()
    if cfg.poisson:
        rng = np.random.default_rng(cfg.noise_seed)
        gmax = float(I.max())
        for i in range(len(I)):
            imax = gmax if cfg.noise_global_norm else float(I[i].max())
            s = cfg.peak_photons / (imax + 1e-12)
            I[i] = rng.poisson(np.maximum(I[i], 0) * s).astype(np.float32) / s
    return I, I_clean
