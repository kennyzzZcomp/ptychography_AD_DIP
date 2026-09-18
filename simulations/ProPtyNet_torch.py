#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ProPtyNet (PyTorch) —— 未训练网络先验 vs 纯 AD 的 ptychography 对照。

固定下来的约定（不再是开关，改了就是换了一个实验）：
  * 扫描        raster 或 fermat，位置严格确定性
  * 探针支撑    无。模型能表示完整探针，数据/模型零失配
  * 探针初值    平滑圆盘 + 零相位、不传播（err_P0 ≈ 0.27）。探针从第 0 步就参与优化
  * 物体初值    相位中性：第 0 步 O ≡ 1·exp(i0)，与 AD 完全相同
  * 参数化      振幅 softplus、相位 cos/sin 单位圆、标度每步闭式最小二乘(VarPro)
  * 数据项      振幅域 ‖|U|-√I‖²
  * AD          标准 AD ptychography：联合更新 + 单个 Adam + 全批量 + 无 lr 调度

两个方法在【同一份数据、同一个初值、同一套协议】下跑，唯一变量是物体的参数化：
  ad  -> 物体 = 自由复数像素
  net -> 物体 = U-Net 的输出（DIP 先验）
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import argparse
from dataclasses import dataclass, asdict, fields as dataclass_fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter, gaussian_filter

PI = math.pi


# ============================================================================ #
# 配置
# ============================================================================ #

@dataclass
class Cfg:
    # ---- 光学 ----
    wlength: float = 632.8e-9
    N: int = 128                 # 探测器 / patch 边长
    N_OBJ: int = 224             # 物体画布
    dx: float = 10e-6            # 样品面 = 探测器面像素（角谱保持像素尺寸）
    dz_true: float = 8e-3        # 样品-探测器
    probe_dia: int = 48          # R_AP = probe_dia/2 - 6 = 18 px
    z_probe: float = 3e-3        # 光阑 -> 样品，合成真值探针
    probe_aberr: float = 0.0     # >0 时给真值光阑加随机波前（真值探针不再理想）
    aberr_seed: int = 7

    # ---- 探针初值 P0：平滑圆盘 + 零相位、不传播 ----
    # 只编码"光阑大概多大"，不给波前、也不给传播产生的 Fresnel 环纹
    # （真值探针被传播摊开到约 33px = 光阑的 1.8 倍）。
    # sigma 单位 = R_AP 的倍数，物理含义 = "知道光阑大小、不知道边缘锐度"。
    # ⚠ 不要按 err_P0 最小去调它 —— 那等于偷用真值信息。定死 0.15。
    probe_init_sigma: float = 0.15

    # ---- 扫描 ----
    scan_pattern: str = "raster"     # raster | fermat
    scan_npos: int = 25
    scan_step: float = 20.0          # 重叠轴有 26.7 / 13.3 这些小数档，必须是 float

    # ---- 区域 ----
    eval_size: int = 96          # -> EVAL_CROP = (224-96)/2 = 64
    reg_size: int = 128          # -> REG_CROP  = (224-128)/2 = 48
    phase_support: float = 0.05  # 相位指标只在振幅 > 该比例×峰值 的像素上统计

    # ---- 噪声 ----
    poisson: bool = False
    peak_photons: float = 5000.0
    noise_global_norm: bool = True
    noise_seed: int = 42

    # ---- AD（标准 AD ptychography）----
    ad_iters: int = 2000         # 全批量梯度步数（与 net 的 iters 同刻度）
    lr_obj: float = 1e-2
    lr_prb: float = 1e-2
    tv1: float = 0.0             # 物体振幅 TGV，0 = 关
    tv2: float = 0.0             # 物体相位 TGV，0 = 关

    # ---- net（DIP）----
    iters: int = 2000
    lr_net: float = 1e-3
    lr_probe: float = 1e-2       # 自由像素探针，与 AD 的 lr_prb 同量级
    lr_cosine: bool = False      # 两个 lr 一起余弦退火到 0，治后期 loss 尖峰
    base_ch: int = 32            # -> 约 2.2 M 参数
    weight_decay: float = 0.0    # DIP 不该有权重衰减
    eval_every: int = 25
    # 物体输出头权重的缩放：0 = 严格中性（第 0 步 O ≡ 1，与 AD 初值相同），
    # 1 = 保持 PyTorch 默认随机。中间值 = 一条"初始相位粗糙度"的连续轴。
    obj_init_alpha: float = 0.0
    # 探针参数化：pixel = 自由复数像素；truth = 冻结在真值上（上界对照，不参与优化）
    probe_mode: str = "pixel"    # pixel | truth

    # ---- 无 GT 早停：留出探测器像素 ----
    holdout_frac: float = 0.0    # >0 时每张图随机留出这么多像素，永不进 loss
    holdout_seed: int = 1234

    # ---- 真值物体 ----
    obj_amp_min: float = 0.4     # 振幅 = [obj_amp_min, 1.0]
    obj_phase_span: float = 0.8  # 相位 = ±该值 (rad)。0 = 纯振幅物体
    obj_amp_img: str = "Siemens.jpg"
    obj_phase_img: str = "Peppers.jpg"

    # ---- 结果 PNG 的显示范围（只影响出图，不影响任何指标）----
    disp_amp_lo: float = 0.15
    disp_amp_hi: float = 1.15
    disp_phase_lim: float = 1.0
    disp_err_amp: float = 0.3
    disp_err_phase: float = 0.3

    # ---- 其它 ----
    seed: int = 0                # 只影响 net。AD 完全确定性，--seed 对它无效
    device: str = "auto"
    assets: str = ""
    outdir: str = "results_proptynet"

    def __post_init__(self):
        self.k0 = 2 * PI / self.wlength
        self.R_AP = self.probe_dia / 2 - 6
        self.EVAL_CROP = (self.N_OBJ - self.eval_size) // 2
        self.REG_CROP = (self.N_OBJ - self.reg_size) // 2
        self.PATCH_C0 = (self.N_OBJ - self.N) / 2.0
        self.SCAN_LIMIT = self.PATCH_C0
        self.dz_max = self.N * self.dx ** 2 / self.wlength

    def dev(self):
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ============================================================================ #
# 素材 / 传播 / 亚像素裁剪  —— 以下为原实现逐字保留
# ============================================================================ #


def asset_dir(cfg: Cfg) -> Path:
    if cfg.assets:
        return Path(cfg.assets)
    here = Path(__file__).resolve().parent
    for c in (here, here.parent, here.parent.parent):
        if (c / "cameraman.bmp").is_file():
            return c
    return here

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



# ============================================================================ #
# 真值 / 探针初值 / 扫描 / 仿真
# ============================================================================ #


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


def make_field(amp_raw, phs_raw):
    """原始头 -> 复数场。振幅 softplus，相位 cos/sin 单位圆。

    amp_raw: (H,W) ; phs_raw: (2,H,W)

    cos/sin 而不是 span*tanh：没有 ±pi 边界、没有饱和、没有 2pi 缠绕的等价解 ——
    这三样正是让优化在等价解之间漂移（loss 不变而指标变差）的来源。
    softplus 而不是 leaky_relu：后者允许负振幅 = 隐藏的 pi 相位翻转。
    """
    amp = F.softplus(amp_raw)
    c, sn = phs_raw[0], phs_raw[1]
    n = torch.sqrt(c * c + sn * sn + 1e-8)
    return torch.complex(amp * c / n, amp * sn / n)

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



# ============================================================================ #
# 前向模型
# ============================================================================ #


def forward_np(cfg, O, P, pos, H):
    psi = crop_patch_np(O, cfg.PATCH_C0 - np.asarray(pos, float), cfg.N) * P
    return np.fft.ifft2(np.fft.fft2(psi) * H)

def forward_torch(cfg, O, P, corners, H):
    """O:(M,M)c ; P:(n,n)c ; corners:(I,2)f ; H:(n,n)c -> U:(I,n,n) complex"""
    psi = crop_patch_torch(O, corners, cfg.N) * P[None]
    return torch.fft.ifft2(torch.fft.fft2(psi) * H[None])



# ============================================================================ #
# 评估指标
# ============================================================================ #


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



# ============================================================================ #
# 正则项与数值安全的小工具
# ============================================================================ #


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

def safe_angle(z, eps=1e-20):
    """angle 在 0 处梯度是 -inf。两处 where 缺一不可（0*inf 仍然 nan）。"""
    mag2 = z.real ** 2 + z.imag ** 2
    ok = mag2 > eps
    z_safe = torch.where(ok, z, torch.ones_like(z))
    return torch.where(ok, torch.angle(z_safe), torch.zeros_like(mag2))

def cabs(z, eps=1e-12):
    """torch.abs 对复数在 z=0 处的反向是 grad*z/|z| -> NaN。加 eps 走 sqrt 更安全。"""
    return torch.sqrt(z.real ** 2 + z.imag ** 2 + eps)

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



# ============================================================================ #
# U-Net
# ============================================================================ #


class DoubleConv(nn.Module):
    """Conv-BN-LeakyReLU ×2，尺寸不变。对应 PhysenNet 的 layer_0x。"""

    def __init__(self, cin, cout):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True),
            nn.Conv2d(cout, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.f(x)

class ProPtyUNet(nn.Module):
    """论文 Fig.1(b)。相对 PhysenNet_torch.py 的 net_model 有三处必改:

      1. 首层输入通道 1 -> J（= 衍射图张数），输入是零填充后的实测衍射图堆栈，全程不变；
      2. 末层【去掉 BatchNorm2d(1)】。原代码在输出层做 BN 会把输出强制成零均值单位方差，
         直接毁掉振幅的绝对尺度 —— 这是从 PhysenNet 迁移过来时最致命的一处；
      3. 原 forward() 把 encoder 重复算了 4 遍（x6_1/x7_1/x8_1/x9_1），这里只算一次。

    【本版改动】输出头返回【未激活】的原始张量，激活与复数合成统一放在 make_field()，
    激活与复数合成统一放在 make_field()，网络本身不需要知道用哪种参数化。

    n_fields=1 -> 只出物体（探针另有其人）；n_fields=2 -> 论文的物体+探针共享写法。
    """

    def __init__(self, in_ch, base=32, n_fields=1, ph_ch=2):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3 = DoubleConv(in_ch, c[0]), DoubleConv(c[0], c[1]), DoubleConv(c[1], c[2])
        self.bot = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2, 2)
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 3, 2, 1, output_padding=1)
        self.d3 = DoubleConv(c[3], c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 3, 2, 1, output_padding=1)
        self.d2 = DoubleConv(c[2], c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 3, 2, 1, output_padding=1)
        self.d1 = DoubleConv(c[1], c[0])
        self.n_fields, self.ph_ch = n_fields, ph_ch
        self.head_amp = nn.Conv2d(c[0], n_fields, 3, 1, 1)
        self.head_phs = nn.Conv2d(c[0], n_fields * ph_ch, 3, 1, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        xb = self.bot(self.pool(x3))
        y = self.d3(torch.cat([self.u3(xb), x3], 1))
        y = self.d2(torch.cat([self.u2(y), x2], 1))
        y = self.d1(torch.cat([self.u1(y), x1], 1))
        return self.head_amp(y)[0], self.head_phs(y)[0]     # (F,H,W), (F*ph,H,W)

# ============================================================================ #
# 自检
# ============================================================================ #

def _banner(cfg: Cfg, err_P0: float, tag: str):
    print(f"[{tag}] 探针初值 P0 = 平滑圆盘(sigma={cfg.probe_init_sigma:g}R) + 零相位，"
          f"相对真值复误差 err_P0 = {err_P0:.4f}")


def run_check(cfg: Cfg):
    dev = cfg.dev()
    obj, probe, rr = make_truth(cfg)
    positions = make_scan_positions(cfg)
    H_true = make_H(cfg, cfg.dz_true)

    print("=" * 74)
    print(f"角谱采样上限 dz_max = {cfg.dz_max*1e3:.2f} mm   当前 dz = {cfg.dz_true*1e3:.2f} mm"
          f"   -> {'OK' if abs(cfg.dz_true) <= cfg.dz_max else '超限！会混叠'}")
    a = cfg.probe_dia * cfg.dx / 2
    print(f"菲涅耳数 a²/(λ·dz) = {a**2/(cfg.wlength*cfg.dz_true):.1f}  (深近场 -> 角谱是正确选择)")
    print("-" * 74)
    check_scan_fits(cfg, positions)
    print(f"  评估区 {cfg.eval_size}²(crop {cfg.EVAL_CROP})   "
          f"正则区 {cfg.reg_size}²(crop {cfg.REG_CROP})")
    _banner(cfg, probe_init_err(cfg, rr, probe), "check")
    print("-" * 74)

    # A) 角谱往返
    rngt = (np.random.default_rng(0).random((cfg.N, cfg.N))
            + 1j * np.random.default_rng(1).random((cfg.N, cfg.N)))
    back = np.fft.ifft2(np.fft.fft2(propagate_np(rngt, H_true)) * np.conj(H_true))
    print(f"[A] 角谱往返相对误差            {np.linalg.norm(back-rngt)/np.linalg.norm(rngt):.3e}  (应 ~1e-8)")

    # B) torch 前向 vs numpy 前向
    Ot = torch.from_numpy(obj).to(dev)
    Pt = torch.from_numpy(probe).to(dev)
    Ht = torch.from_numpy(H_true).to(dev)
    corners = torch.from_numpy(cfg.PATCH_C0 - positions).to(dev)
    with torch.no_grad():
        Ut = forward_torch(cfg, Ot, Pt, corners, Ht).cpu().numpy()
    Un = np.stack([forward_np(cfg, obj, probe, p, H_true) for p in positions])
    print(f"[B] torch 前向 vs numpy 前向     {np.linalg.norm(Ut-Un)/np.linalg.norm(Un):.3e}  (应 <1e-6)")

    # C) 整数位置下 crop_patch == 直接切片
    ci = np.array([cfg.PATCH_C0, cfg.PATCH_C0])
    w1 = crop_patch_np(obj, ci, cfg.N)
    i0 = int(cfg.PATCH_C0)
    w2 = obj[i0:i0 + cfg.N, i0:i0 + cfg.N]
    print(f"[C] 整数位置 crop == 直接切片     {np.abs(w1-w2).max():.3e}  (应 0)")

    # D) 梯度回流窗口
    Op = torch.nn.Parameter(torch.ones((cfg.N_OBJ, cfg.N_OBJ), device=dev))
    Oc = (Op * torch.exp(torch.zeros_like(Op) * 1j)).to(torch.complex64)
    U = forward_torch(cfg, Oc, Pt, corners, Ht)
    U.abs().pow(2).sum().backward()
    g = (Op.grad.abs() > 0).cpu().numpy()
    ys, xs = np.where(g)
    print(f"[D] 物体梯度回流窗口             行 {ys.min()}..{ys.max()}  列 {xs.min()}..{xs.max()}"
          f"  ({ys.max()-ys.min()+1}×{xs.max()-xs.min()+1})，画布 {cfg.N_OBJ}")

    # E) 真值代入的数据残差 —— 无探针支撑，所以应当是 0（除噪声外）
    I, _ = simulate(cfg, obj, probe, positions, H_true)
    y0 = torch.from_numpy(np.sqrt(np.maximum(I, 0))).to(dev)
    with torch.no_grad():
        Um = forward_torch(cfg, Ot, Pt, corners, Ht)
        r = ((Um.abs() - y0).norm() / y0.norm()).item()
    print(f"[E] 真值代入的数据残差           {r:.3e}  "
          + ("  <- 无探针支撑，数据/模型零失配" if not cfg.poisson else "  (泊松噪声下非 0 属正常)"))
    print("=" * 74)


# ============================================================================ #
# 模式 1: AD —— 标准 AD ptychography（Kandel et al. 2019）
# ============================================================================ #

def run_ad(cfg: Cfg):
    """物体 = 自由复数像素。联合更新 + 全程单个 Adam + 全批量 + 无 lr 调度。

    与 run_net 共用：同一份数据、同一个探针初值、探针从第 0 步就更新。
    唯一的差别是物体的参数化。
    """
    dev = cfg.dev()
    obj, probe, rr = make_truth(cfg)
    positions = make_scan_positions(cfg)
    check_scan_fits(cfg, positions)
    H_true = make_H(cfg, cfg.dz_true)
    I, _ = simulate(cfg, obj, probe, positions, H_true)

    Ht = torch.from_numpy(H_true).to(dev)
    corners = torch.from_numpy(cfg.PATCH_C0 - positions).to(dev)
    y0 = torch.from_numpy(np.sqrt(np.maximum(I, 0))).to(dev)
    ones = np.ones_like(rr, dtype=np.float32)

    # 初值：物体 O ≡ 1（与 net 的 obj_init_alpha=0 相同）；探针 = make_probe_init
    Or = nn.Parameter(torch.ones((cfg.N_OBJ, cfg.N_OBJ), device=dev))
    Oi = nn.Parameter(torch.zeros((cfg.N_OBJ, cfg.N_OBJ), device=dev))
    P0 = make_probe_init(cfg, rr)
    _banner(cfg, probe_init_err(cfg, rr, probe), "ad")
    Pr = nn.Parameter(torch.from_numpy(P0.real.astype(np.float32)).to(dev))
    Pi = nn.Parameter(torch.from_numpy(P0.imag.astype(np.float32)).to(dev))

    c = cfg.REG_CROP
    opt = torch.optim.Adam([{"params": [Or, Oi], "lr": cfg.lr_obj},
                            {"params": [Pr, Pi], "lr": cfg.lr_prb}])
    print(f"[ad] 标准 AD: 全批量 {len(positions)} 位置/步 × {cfg.ad_iters} 步  "
          f"lr_obj={cfg.lr_obj:g} lr_prb={cfg.lr_prb:g}（不衰减）  "
          f"正则 tv1={cfg.tv1:g} tv2={cfg.tv2:g}")

    hist, t0, rec, pc = [], time.time(), None, None
    for it in range(cfg.ad_iters):
        O = torch.complex(Or, Oi)
        P = torch.complex(Pr, Pi)
        U = forward_torch(cfg, O, P, corners, Ht)
        loss = data_loss_direct(U, y0)
        if cfg.tv1 > 0 or cfg.tv2 > 0:
            Oreg = O[c:-c, c:-c] if c > 0 else O
            if cfg.tv1 > 0:
                loss = loss + cfg.tv1 * tgv_loss(Oreg.abs(), beta=1.0) * 1e-2
            if cfg.tv2 > 0:
                loss = loss + cfg.tv2 * tgv_loss(safe_angle(Oreg), beta=1.0) * 1e-2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % cfg.eval_every == 0 or it == cfg.ad_iters - 1:
            rec = torch.complex(Or, Oi).detach().cpu().numpy()
            pc = torch.complex(Pr, Pi).detach().cpu().numpy()
            mo = evaluate_object_roi(cfg, rec, obj)
            mp = evaluate_probe(pc, probe, ones, r_far=cfg.probe_dia)
            hist.append({"it": it + 1, "loss": loss.item(), **mo, **mp})
            if (it + 1) % (cfg.eval_every * 4) == 0 or it == cfg.ad_iters - 1:
                print(f"  it {it+1:5d} | loss {loss.item():.4e} | "
                      f"amp PSNR {mo['psnr_o_amp']:6.2f} dB | SSIM {mo['ssim_o_amp']:.4f} | "
                      f"phase RMSE {mo['rmse_o_phi_rad']:.4f} rad | complex err "
                      f"{mo['relerr_o_complex']:.4f} | probe err {mp['relerr_p_complex']:.4f}",
                      flush=True)
    print(f"[ad] 用时 {time.time()-t0:.1f}s / {cfg.ad_iters} it")
    _save(cfg, "ad", rec, pc, obj, probe, ones, hist)
    return hist


# ============================================================================ #
# 模式 2: net —— 物体 = U-Net 输出（DIP 先验）
# ============================================================================ #

def run_net(cfg: Cfg):
    dev = cfg.dev()
    torch.manual_seed(cfg.seed)
    obj, probe, rr = make_truth(cfg)
    positions = make_scan_positions(cfg)
    check_scan_fits(cfg, positions)
    H_true = make_H(cfg, cfg.dz_true)
    I, I_clean = simulate(cfg, obj, probe, positions, H_true)

    Ht = torch.from_numpy(H_true).to(dev)
    corners = torch.from_numpy(cfg.PATCH_C0 - positions).to(dev)
    Im = torch.from_numpy(I).to(dev)
    Iclean = torch.from_numpy(I_clean).to(dev)
    sqrtIm = torch.sqrt(Im.clamp_min(0))
    pad = (cfg.N_OBJ - cfg.N) // 2
    ones = np.ones_like(rr, dtype=np.float32)

    # 留出探测器像素：无 GT 的早停判据。不破坏扫描几何（重叠率不变）。
    if cfg.holdout_frac > 0:
        g = torch.Generator().manual_seed(cfg.holdout_seed)
        Mtr = (torch.rand(Im.shape, generator=g) >= cfg.holdout_frac).float().to(dev)
        print(f"[net] 留出探测器像素 {100*(1-Mtr.mean().item()):.2f}% 作验证集（不进 loss）")
    else:
        Mtr = None

    # 网络输入：零填充到画布尺寸的实测衍射图堆栈，全程固定
    x_in = F.pad(Im[None], (pad, pad, pad, pad))
    x_in = x_in / x_in.amax(dim=(2, 3), keepdim=True).clamp_min(1e-12)

    net = ProPtyUNet(len(positions), cfg.base_ch, n_fields=1, ph_ch=2).to(dev)
    print(f"[net] U-Net {sum(p.numel() for p in net.parameters())/1e6:.2f} M 参数  "
          f"输入 {tuple(x_in.shape)}")

    # 物体输出头的相位中性初始化。alpha=0 时第 0 步 O ≡ 1·exp(i0)，与 AD 初值完全相同。
    # alpha=0 不会死梯度：cossin 在 (c,sn)=(1,0) 处 d(phi)/d(sn)=1，softplus 在
    # b=ln(e-1) 处导数 0.632，且权重梯度随像素变化，第一步就破对称。
    _a = float(cfg.obj_init_alpha)
    with torch.no_grad():
        net.head_amp.weight[0].mul_(_a)
        net.head_phs.weight[0:2].mul_(_a)
        net.head_amp.bias[0] = math.log(math.e - 1.0)   # softplus(b) = 1
        net.head_phs.bias[0:2] = 0.0
        net.head_phs.bias[0] = 1.0                      # (c, sn) = (1, 0) -> phi = 0
    print(f"[net] 物体输出头 = 相位中性初始化 alpha={_a:g}"
          + ("（第 0 步 O ≡ 1·exp(i0)，与 AD 初值相同）" if _a == 0 else ""))

    # 探针
    prb_params, Ptruth = [], None
    if cfg.probe_mode == "truth":
        Ptruth = torch.from_numpy(probe.astype(np.complex64)).to(dev)
        print("[net] 探针 = 真值且冻结（上界对照：只考察物体表示本身的能力）")
    else:
        P0 = make_probe_init(cfg, rr)
        _banner(cfg, probe_init_err(cfg, rr, probe), "net")
        Pr = nn.Parameter(torch.from_numpy(P0.real.astype(np.float32)).to(dev))
        Pi = nn.Parameter(torch.from_numpy(P0.imag.astype(np.float32)).to(dev))
        prb_params = [Pr, Pi]
        print(f"[net] 探针 = 自由复数像素（{2*rr.size} 个未知量），从第 0 步起与物体联合更新")

    def decode():
        a_raw, p_raw = net(x_in)
        O = make_field(a_raw[0], p_raw[0:2])
        Pc = Ptruth if Ptruth is not None else torch.complex(Pr, Pi)
        return O, Pc

    opt_net = torch.optim.Adam(net.parameters(), lr=cfg.lr_net,
                               weight_decay=cfg.weight_decay)
    opt_prb = torch.optim.Adam(prb_params, lr=cfg.lr_probe) if prb_params else None

    hist, t0 = [], time.time()
    best_val = {"val": float("inf"), "it": -1, "ssim": float("nan")}
    best_gt = {"ssim": -1.0, "it": -1}
    rec = pc = None

    for it in range(cfg.iters):
        if cfg.lr_cosine:
            f = 0.5 * (1 + math.cos(PI * it / max(cfg.iters - 1, 1)))
            for g in opt_net.param_groups:
                g["lr"] = cfg.lr_net * f
            if opt_prb is not None:
                for g in opt_prb.param_groups:
                    g["lr"] = cfg.lr_probe * f
        O, Pc = decode()
        Ua = cabs(forward_torch(cfg, O, Pc, corners, Ht))
        # 全局幅度标度：物体与探针之间有 O->aO, P->P/a 的规范自由度。每步解析地求最优
        # 标量（VarPro），不 detach —— 这样标度方向上的梯度恒为 0，那个自由度被消掉。
        Ua = Ua * ((Ua * sqrtIm).sum() / (Ua * Ua).sum().clamp_min(1e-20))
        loss = masked_mse(Ua, sqrtIm, Mtr)

        opt_net.zero_grad(set_to_none=True)
        if opt_prb is not None:
            opt_prb.zero_grad(set_to_none=True)
        loss.backward()
        opt_net.step()
        if opt_prb is not None:
            opt_prb.step()

        if (it + 1) % cfg.eval_every == 0 or it == cfg.iters - 1:
            with torch.no_grad():
                real = torch.linalg.vector_norm(Ua ** 2 - Iclean).item()
                val = held_out_mse(Ua, sqrtIm, Mtr) if Mtr is not None else float("nan")
                rec = O.detach().cpu().numpy(); pc = Pc.detach().cpu().numpy()
            mo = evaluate_object_roi(cfg, rec, obj)
            mp = evaluate_probe(pc, probe, ones, r_far=cfg.probe_dia)
            hist.append({"it": it + 1, "loss": loss.item(), "val": val, "real": real,
                         **mo, **mp})
            if mo["ssim_o_amp"] > best_gt["ssim"]:
                best_gt = {"ssim": mo["ssim_o_amp"], "it": it + 1}
            if Mtr is not None and val < best_val["val"]:
                best_val = {"val": val, "it": it + 1, "ssim": mo["ssim_o_amp"]}
            if (it + 1) % (cfg.eval_every * 4) == 0 or it == cfg.iters - 1:
                v = f" | val {val:.4e}" if Mtr is not None else ""
                print(f"  it {it+1:5d} | loss {loss.item():.4e}{v} | real {real:.4e} | "
                      f"amp PSNR {mo['psnr_o_amp']:6.2f} dB | SSIM {mo['ssim_o_amp']:.4f} | "
                      f"phase RMSE {mo['rmse_o_phi_rad']:.4f} rad | complex err "
                      f"{mo['relerr_o_complex']:.4f} | probe err {mp['relerr_p_complex']:.4f}",
                      flush=True)

    print(f"[net] 用时 {time.time()-t0:.1f}s / {cfg.iters} it")
    if len(hist) > 8:
        tail = [h["real"] for h in hist[-len(hist)//4:]]
        k = np.polyfit(np.arange(len(tail)), np.array(tail), 1)[0]
        print(f"[net] 尾段 real error 斜率 {k:+.3e}  "
              f"({'仍在下降' if k < 0 else '已回升 -> 开始拟合噪声'})")
    if Mtr is not None:
        print(f"[net] 留出验证最优步 = {best_val['it']} (val {best_val['val']:.4e}, "
              f"该步 SSIM {best_val['ssim']:.4f})")
        print(f"[net] 真值 SSIM 最优步 = {best_gt['it']} (SSIM {best_gt['ssim']:.4f})"
              f"   <- 两者越接近，留出验证越可以替代 GT 做早停")

    _save(cfg, "net", rec, pc, obj, probe, ones, hist)
    return hist



# ============================================================================ #
# 保存 / 绘图
# ============================================================================ #


def _save(cfg, tag, rec, pc, obj, probe, support, hist):
    os.makedirs(cfg.outdir, exist_ok=True)
    cfg_out = asdict(cfg)
    np.savez_compressed(os.path.join(cfg.outdir, f"{tag}_result.npz"),
                        obj_rec=rec, probe_rec=pc, hist=json.dumps(hist),
                        cfg=json.dumps(cfg_out, default=str))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    c = cfg.EVAL_CROP
    ra, _ = align_global_factor(rec[c:-c, c:-c], obj[c:-c, c:-c])
    ga = obj[c:-c, c:-c]
    _m = (support > 0)
    pa, _ = align_global_factor(pc * _m, probe * _m)
    pg = probe * _m                 # GT 探针也用同一块区域，两张探针图口径一致

    def _dphi(a, b):                # 缠绕相位差，不会在 ±pi 边界炸掉
        return np.angle(np.exp(1j * (np.angle(a) - np.angle(b))))

    # 探针相位只在【振幅够大】的地方有意义：支撑外振幅 ~0，angle() 是纯噪声，
    # 会把三张探针相位图全糊成 ±pi 的椒盐。用真值探针的 5% 峰值做阈值掩掉。
    _pm = (np.abs(pg) > 0.05 * max(np.abs(pg).max(), 1e-30)).astype(np.float32)
    AMP = (cfg.disp_amp_lo, cfg.disp_amp_hi)
    PHS = (-cfg.disp_phase_lim, cfg.disp_phase_lim)
    EA = (-cfg.disp_err_amp, cfg.disp_err_amp)
    EP = (-cfg.disp_err_phase, cfg.disp_err_phase)
    PAMP, PPHS = (0.0, 1.1), (-PI, PI)   # 真值探针恒归一化到峰值 1，这两个不用调
    panels = [
        (np.abs(ra), "rec amp", AMP, "gray"),
        (np.angle(ra), "rec phase", PHS, "gray"),
        (np.abs(pa), "rec probe amp", PAMP, "gray"),
        (np.angle(pa) * _pm, "rec probe phase", PPHS, "gray"),
        (np.abs(ga), "GT amp", AMP, "gray"),
        (np.angle(ga), "GT phase", PHS, "gray"),
        (np.abs(pg), "GT probe amp", PAMP, "gray"),
        (np.angle(pg) * _pm, "GT probe phase", PPHS, "gray"),
        (np.abs(ra) - np.abs(ga), "err amp (rec-GT)", EA, "RdBu_r"),
        (_dphi(ra, ga), "err phase (rad)", EP, "RdBu_r"),
        (np.abs(pa) - np.abs(pg), "err probe amp", EA, "RdBu_r"),
        (_dphi(pa, pg) * _pm, "err probe phase (rad)", EP, "RdBu_r"),
    ]
    fig, ax = plt.subplots(3, 4, figsize=(15, 11.4))
    over = []
    for a, (im, t, (vmin, vmax), cm) in zip(ax.ravel(), panels):
        a.imshow(im, cmap=cm, vmin=vmin, vmax=vmax)
        lo, hi = float(np.min(im)), float(np.max(im))
        clip = lo < vmin - 1e-9 or hi > vmax + 1e-9
        if clip:
            over.append(f"{t}: 实际 [{lo:.3f}, {hi:.3f}] 超出显示范围 [{vmin:.3f}, {vmax:.3f}]")
        a.set_title(f"{t}  [{lo:.3f}, {hi:.3f}]" + ("  ⚠clip" if clip else ""), fontsize=8)
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    f = os.path.join(cfg.outdir, f"{tag}_result.png")
    fig.savefig(f, dpi=140); plt.close(fig)
    print(f"[{tag}] 结果 -> {f}")
    if over:
        print(f"[{tag}] ⚠ 显示范围没覆盖住（只影响出图，指标用的是未截断的原始数组）:")
        for o in over:
            print(f"        {o}")
        print(f"        -> 把 --disp-amp-lo/-hi、--disp-phase-lim、--disp-err-amp/"
              f"--disp-err-phase 调宽后重出图，不要去截断数据。")
    return

# ============================================================================ #
# CLI —— 直接从 Cfg 的字段生成，不再手工维护一份平行列表
#   （旧版就是因为两份列表会漂移，加了 Cfg 字段却忘了加 --flag）
# ============================================================================ #

_CHOICES = {
    "scan_pattern": ["raster", "fermat"],
    "probe_mode": ["pixel", "truth"],
    "device": None,
}


def build_parser():
    ap = argparse.ArgumentParser(
        description="ProPtyNet —— 未训练网络先验 vs 纯 AD 的 ptychography 对照")
    ap.add_argument("mode", choices=["check", "ad", "net"],
                    help="check=自检 | ad=纯 AD 基线 | net=DIP")
    for f in dataclass_fields(Cfg):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            ap.add_argument(flag, dest=f.name, action="store_true", default=None)
        elif f.name in _CHOICES and _CHOICES[f.name]:
            ap.add_argument(flag, dest=f.name, choices=_CHOICES[f.name], default=None)
        else:
            ap.add_argument(flag, dest=f.name, type=type(f.default), default=None)
    return ap


def main():
    a = build_parser().parse_args()
    kw = {k: v for k, v in vars(a).items() if k != "mode" and v is not None}
    cfg = Cfg(**kw)
    {"check": run_check, "ad": run_ad, "net": run_net}[a.mode](cfg)


if __name__ == "__main__":
    main()
