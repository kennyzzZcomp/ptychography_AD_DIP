# -*- coding: utf-8 -*-
"""
ProPtyNet_torch.py —— PyTorch 版无训练神经网络 ptychography

对应论文: Z. Liu, Y. Chen, N. Lin, "Noise-robust ptychography using unsupervised
neural network", Optics and Lasers in Engineering 186 (2025) 108791.

【定位】本文件是 INNM_Ptycho.ipynb 的 PyTorch 侧兄弟。物理前向、扫描位置、探针真值
与初值、噪声模型、评估裁剪，全部逐条对齐 INNM_Ptycho.ipynb / functions/innm_common.py，
以满足 simulations/README.md 里"可比测试"的规则。唯一改变的因子是【未知量的参数化】:

    AD   : 物体/探针 = 自由可训练张量                    (INNM_Ptycho.ipynb, TF)
    本文件: 物体/探针 = 一个未训练 U-Net 的 4 通道输出     (ProPtyNet, 本文件)

两种模式都在这里实现，`--mode ad` 是 TF 版的 PyTorch 移植，用来验证移植没走样
（对得上 notebook 里记录的 stage 数字才能开始比 net）。

用法:
    python ProPtyNet_torch.py check              # 采样自检 + 前向与 numpy 对拍
    python ProPtyNet_torch.py ad   --stages 8    # AD 基线（移植校验用）
    python ProPtyNet_torch.py net  --iters 2000  # ProPtyNet
    python ProPtyNet_torch.py net  --iters 2000 --data-loss paper --noise-clip

依赖: torch, numpy, scipy, matplotlib, (cv2 或 Pillow)
资源: 与 notebook 共用 ../cameraman.bmp 与 ../westconcordorthophoto.bmp
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter

PI = math.pi


# ============================================================================ #
# 参数 —— 逐条对应 INNM_Ptycho.ipynb 的 CELL 2
# ============================================================================ #

@dataclass
class Cfg:
    # ---- 光学 ----
    wlength: float = 632.8e-9
    N: int = 128                 # 探测器 / patch 边长
    N_OBJ: int = 224             # 物体画布
    dx: float = 10e-6            # 样品面 = 探测器面像素（角谱保持像素尺寸）
    dz_true: float = 8e-3        # 样品-探测器
    probe_dia: int = 48
    z_probe: float = 3e-3        # 孔径->样品，合成真值探针
    z_probe_init: float = 3e-3 * 1.3   # 故意差 30%
    probe_aberr: float = 0.0
    aberr_seed: int = 7

    # ---- 扫描 ----
    scan_pattern: str = "raster_jitter"
    scan_npos: int = 25
    scan_step: int = 20
    scan_jitter: float = 0.2
    scan_seed: int = 0

    # ---- 区域 ----
    eval_size: int = 96          # -> EVAL_CROP = (224-96)/2 = 64
    reg_size: int = 128          # -> REG_CROP  = (224-128)/2 = 48
    phase_support: float = 0.05

    # ---- 噪声 ----
    poisson: bool = False
    peak_photons: float = 5000.0
    noise_global_norm: bool = True
    noise_seed: int = 42
    # 论文 Table 1 的额外一步: 归一化到 [0,1] 后 clip，制造过曝像素。
    # 只有开了它，论文 Eq.(5) 的过曝掩膜 S2 和 γ 才有东西可作用。
    noise_clip: bool = False
    gauss_snr_db: float = 0.0    # >0 时按论文 Table 1 追加高斯噪声

    # ---- AD 模式（移植校验）----
    opt_mode: str = "alternating"
    stages: int = 8
    obj_epoch: int = 8
    prb_epoch: int = 8
    lr_obj: float = 3e-2
    lr_prb: float = 3e-2
    decay: float = 0.75
    tv1: float = 0.0
    tv2: float = 0.0

    # ---- ProPtyNet 模式 ----
    iters: int = 2000
    lr_net: float = 1e-3         # 论文: 5e-4 ~ 5e-3
    base_ch: int = 32            # 32/64/128/256 -> 约 2.2 M 参数（论文称 2.5 M）
    phase_span_obj: float = 2 * PI   # 论文建议样品放宽到 2π
    phase_span_prb: float = PI       # 论文: amp_p × exp(jπ·phs_p)
    amp_act: str = "leaky"       # leaky(论文) | softplus | relu
    data_loss: str = "direct"    # direct(=notebook 的 ‖|U|-√I‖²) | paper(Eq.4/5 强度域)
    beta: float = 0.90           # 论文 Eq.4
    gamma0: float = 1.0          # 论文 Eq.5，随迭代衰减
    gamma_end: float = 0.05
    eval_every: int = 25

    # ---- 其它 ----
    seed: int = 0
    device: str = "auto"
    assets: str = ""             # 空 = 自动找 ../cameraman.bmp
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


def asset_dir(cfg: Cfg) -> Path:
    if cfg.assets:
        return Path(cfg.assets)
    here = Path(__file__).resolve().parent
    for c in (here, here.parent, here.parent.parent):
        if (c / "cameraman.bmp").is_file():
            return c
    return here


# ============================================================================ #
# 角谱传播 —— 移植自 notebook CELL 3 的 make_H / propagate_np
# ============================================================================ #

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


# ============================================================================ #
# 亚像素平移 + 局部裁剪 —— 移植自 innm_common.fourier_shift_* / crop_patch_*
# ============================================================================ #

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


# ============================================================================ #
# 真值物体 / 探针 / 扫描 —— 移植自 notebook CELL 4
# ============================================================================ #

def _imread_resize(path: Path, n: int):
    try:
        import cv2
        a = cv2.imread(str(path), 0)
        if a is None:
            raise IOError(path)
        return cv2.resize(a, (n, n), interpolation=cv2.INTER_CUBIC).astype(float)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(path).convert("L").resize((n, n), Image.BICUBIC),
                          dtype=float)


def make_truth(cfg: Cfg):
    """返回 obj(复), probe(复), support(float), pupil_true, H_true —— 与 notebook 逐位一致。"""
    d = asset_dir(cfg)
    _a = _imread_resize(d / "cameraman.bmp", cfg.N_OBJ)
    _a = 0.4 + 0.6 * _a / _a.max()
    _p = _imread_resize(d / "westconcordorthophoto.bmp", cfg.N_OBJ)
    _p = -1.0 + 2.0 * (_p - _p.min()) / max(_p.max() - _p.min(), 1e-12)   # = cv2.normalize(-1,1)
    obj = (_a * np.exp(1j * 0.8 * _p)).astype(np.complex64)

    N = cfg.N
    yy, xx = np.mgrid[0:N, 0:N] - N / 2
    rr = np.sqrt(xx ** 2 + yy ** 2)

    pupil_mask = rr <= cfg.R_AP
    if cfg.probe_aberr > 0:
        try:
            import cv2
            _r = cv2.GaussianBlur(np.random.default_rng(cfg.aberr_seed)
                                  .normal(size=(N, N)).astype(np.float64), (0, 0), 1.5)
        except ImportError:
            from scipy.ndimage import gaussian_filter
            _r = gaussian_filter(np.random.default_rng(cfg.aberr_seed)
                                 .normal(size=(N, N)), 1.5)
        _phi_ap = _r * (cfg.probe_aberr / np.sqrt(np.mean(_r[pupil_mask] ** 2)))
    else:
        _phi_ap = np.zeros((N, N))
    pupil_true = (pupil_mask * np.exp(1j * _phi_ap)).astype(np.complex64)

    H_probe = make_H(cfg, cfg.z_probe)
    probe = propagate_np(pupil_true, H_probe)
    probe = (probe / np.abs(probe).max()).astype(np.complex64)

    # 支撑域：二值，取到包住 99.5% 能量。必须二值（模型里有 Pr*support）
    e = np.abs(probe) ** 2
    for _R in range(int(cfg.probe_dia / 2), N // 2):
        if (e * (rr <= _R)).sum() / e.sum() > 0.995:
            break
    support = (rr <= _R).astype(np.float32)
    return obj, probe, support, pupil_true, rr, _R


def make_scan_positions(cfg: Cfg):
    """与 notebook 的 make_scan_positions 使用同一 RNG 调用序列 -> 位置逐位一致。"""
    rng = np.random.default_rng(cfg.scan_seed)
    pat, n_pos, step = cfg.scan_pattern, cfg.scan_npos, cfg.scan_step
    if pat in ("raster", "raster_jitter"):
        k = int(round(np.sqrt(n_pos)))
        off = (k - 1) * step / 2
        p = np.array([[i * step - off, j * step - off]
                      for i in range(k) for j in range(k)], float)
        if pat == "raster_jitter":
            p += rng.uniform(-cfg.scan_jitter * step, cfg.scan_jitter * step, p.shape)
    elif pat == "fermat":
        R = step * np.sqrt(n_pos) / 2
        n = np.arange(1, n_pos + 1)
        r = (R / np.sqrt(n_pos)) * np.sqrt(n)
        th = n * np.deg2rad(137.508)
        p = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    else:
        raise ValueError(pat)
    return p.astype(np.float32)


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


def simulate(cfg: Cfg, obj, probe, positions, H_true):
    """生成衍射强度。噪声流程与 notebook CELL 5 一致，再可选加论文 Table 1 的 clip。"""
    I = np.empty((len(positions), cfg.N, cfg.N), np.float32)
    for i, p in enumerate(positions):
        I[i] = np.abs(forward_np(cfg, obj, probe, p, H_true)) ** 2
    I_clean = I.copy()

    rng = np.random.default_rng(cfg.noise_seed)
    if cfg.poisson:
        gmax = float(I.max())
        for i in range(len(I)):
            imax = gmax if cfg.noise_global_norm else float(I[i].max())
            s = cfg.peak_photons / (imax + 1e-12)
            I[i] = rng.poisson(np.maximum(I[i], 0) * s).astype(np.float32) / s

    if cfg.gauss_snr_db > 0:
        # 论文 Table 1 的高斯列: σ = mean(signal)/sqrt(10^(SNR/10))，逐图
        g = float(I_clean.max())
        for i in range(len(I)):
            s = I_clean[i] / g
            sigma = s.mean() / math.sqrt(10 ** (cfg.gauss_snr_db / 10))
            I[i] = I[i] + rng.normal(0, sigma, s.shape).astype(np.float32) * g

    if cfg.noise_clip:
        # 论文: 全局归一化到 [0,1] 后 clip -> 负值归零、>1 饱和（过曝的来源）
        g = float(I.max())
        I = np.clip(I / g, 0.0, 1.0).astype(np.float32) * g
        I_clean = np.clip(I_clean / g, 0.0, 1.0).astype(np.float32) * g
    return I, I_clean


# ============================================================================ #
# 评估指标 —— 移植自 innm_common（数值与 TF 版一致）
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


def evaluate_probe(rec_c, gt_field, mask):
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
    }


# ============================================================================ #
# 正则项（TGV / TV）—— 移植自 innm_common，reduce='mean'
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


# ============================================================================ #
# 数据项
# ============================================================================ #

def data_loss_direct(U, sqrtI):
    """notebook 的写法: Keras mse(|U|, sqrt(I))，逐元素均值。"""
    return F.mse_loss(U.abs(), sqrtI)


def data_loss_paper(Ic, Im, S2, gamma, probe_amp=None, S1=None, beta=0.9):
    """论文 Eq.(4)/(5)。强度域 L2 范数（不是均值），带过曝掩膜 S2 与衰减的 γ。

    注意: 本项目的探针已被二值 support 硬约束，Loss2 恒为 0 —— 只有在把 support
    去掉、改用软约束时 Loss2 才有意义。同理 S2 只有在 noise_clip=True 时才非平凡。
    """
    r = Ic - Im
    loss1 = torch.linalg.vector_norm(r * S2 + gamma * r * (1.0 - S2))
    if probe_amp is None or S1 is None:
        return loss1
    loss2 = torch.linalg.vector_norm(probe_amp * (1.0 - S1))
    return beta * loss1 + (1.0 - beta) * loss2


# ============================================================================ #
# ProPtyNet 的 U-Net —— 论文 Fig.1(b)
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
      3. 输出拆成 4 张单通道图: 振幅走 Conv+LeakyReLU，相位走 Conv+tanh。

      另外原 forward() 把 encoder 重复算了 4 遍（x6_1/x7_1/x8_1/x9_1），这里改成算一次。
    """

    def __init__(self, in_ch, base=32, amp_act="leaky"):
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
        self.head_amp = nn.Conv2d(c[0], 2, 3, 1, 1)     # amp_s, amp_p
        self.head_phs = nn.Conv2d(c[0], 2, 3, 1, 1)     # phs_s, phs_p
        self.amp_act = amp_act

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        xb = self.bot(self.pool(x3))
        y = self.d3(torch.cat([self.u3(xb), x3], 1))
        y = self.d2(torch.cat([self.u2(y), x2], 1))
        y = self.d1(torch.cat([self.u1(y), x1], 1))
        a = self.head_amp(y)
        if self.amp_act == "softplus":
            a = F.softplus(a)
        elif self.amp_act == "relu":
            a = F.relu(a)
        else:
            a = F.leaky_relu(a, 0.2)          # 论文写法
        p = torch.tanh(self.head_phs(y))
        return a, p


# ============================================================================ #
# 自检
# ============================================================================ #

def run_check(cfg: Cfg):
    dev = cfg.dev()
    obj, probe, support, pupil_true, rr, R_sup = make_truth(cfg)
    positions = make_scan_positions(cfg)
    H_true = make_H(cfg, cfg.dz_true)

    print("=" * 74)
    print(f"角谱采样上限 dz_max = {cfg.dz_max*1e3:.2f} mm   当前 dz = {cfg.dz_true*1e3:.2f} mm"
          f"   -> {'OK' if abs(cfg.dz_true) <= cfg.dz_max else '超限！会混叠'}")
    a = cfg.probe_dia * cfg.dx / 2
    print(f"菲涅耳数 a²/(λ·dz) = {a**2/(cfg.wlength*cfg.dz_true):.1f}  (深近场 -> 角谱是正确选择)")
    dx1_fresnel = cfg.wlength * cfg.dz_true / (cfg.N * cfg.dx)
    print(f"[对照] 若改用 Fresnel 单次 FFT，样品面像素会变成 {dx1_fresnel*1e6:.2f} µm 而不是 "
          f"{cfg.dx*1e6:.1f} µm；且 chirp 采样要求 dx ≤ sqrt(λz/N) = "
          f"{math.sqrt(cfg.wlength*cfg.dz_true/cfg.N)*1e6:.2f} µm，当前 {cfg.dx*1e6:.1f} µm 已超 -> 会混叠")
    print("-" * 74)
    m = float(np.abs(positions).max())
    print(f"扫描 {cfg.scan_pattern} {len(positions)} 点  步长 {cfg.scan_step}px  "
          f"位置极值 {m:.1f} / 余量 {cfg.SCAN_LIMIT:.0f}px  {'OK' if m <= cfg.SCAN_LIMIT else '越界!'}")
    dd = np.sqrt(((positions[:, None] - positions[None]) ** 2).sum(-1)) + np.eye(len(positions)) * 1e9
    nn_ = dd.min(1)
    print(f"  最近邻中位数 {np.median(nn_):.1f}px  线性重叠 {1-np.median(nn_)/cfg.probe_dia:.1%}"
          f"  面积重叠 {np.median([overlap_areal(d, cfg.probe_dia) for d in nn_]):.1%}")
    print(f"  支撑半径 {R_sup}px   评估区 {cfg.eval_size}²(crop {cfg.EVAL_CROP})   "
          f"正则区 {cfg.reg_size}²(crop {cfg.REG_CROP})")
    print("-" * 74)

    # A) 角谱往返
    rngt = np.random.default_rng(0).random((cfg.N, cfg.N)) + 1j * np.random.default_rng(1).random((cfg.N, cfg.N))
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
          f"  ({ys.max()-ys.min()+1}×{xs.max()-xs.min()+1})，画布 {cfg.N_OBJ}\n"
          f"    单点应为 {cfg.N}×{cfg.N}；{len(positions)} 点的并集 = N + 扫描跨度 "
          f"= {cfg.N + int(np.ceil(2*np.abs(positions).max()))}，边缘个别像素梯度恰为 0 属正常")

    # E) 真值代入的数据残差（模型探针 = probe×support，支撑域外还剩能量）
    I, _ = simulate(cfg, obj, probe, positions, H_true)
    Ps = torch.from_numpy(probe * support).to(dev)
    with torch.no_grad():
        Um = forward_torch(cfg, Ot, Ps, corners, Ht)
        r = (Um.abs() - torch.from_numpy(np.sqrt(np.maximum(I, 0))).to(dev)).norm() \
            / torch.from_numpy(np.sqrt(np.maximum(I, 0))).to(dev).norm()
    print(f"[E] 真值代入的数据残差           {r.item():.3e}  "
          f"(notebook 记录约 5e-2，来自 support 外残余能量，不是 bug)")
    print("=" * 74)


# ============================================================================ #
# 模式 1: AD 基线（TF 版的 PyTorch 移植，用于校验）
# ============================================================================ #

def run_ad(cfg: Cfg):
    dev = cfg.dev()
    torch.manual_seed(cfg.seed)
    obj, probe, support, _, rr, _R = make_truth(cfg)
    positions = make_scan_positions(cfg)
    H_true = make_H(cfg, cfg.dz_true)
    I, _ = simulate(cfg, obj, probe, positions, H_true)

    Ht = torch.from_numpy(H_true).to(dev)
    corners = torch.from_numpy(cfg.PATCH_C0 - positions).to(dev)
    sup = torch.from_numpy(support).to(dev)
    y0 = torch.from_numpy(np.sqrt(np.maximum(I, 0))).to(dev)

    # 初值: 物体全 1；探针 = 针孔场传播（估计距离，非真值）
    Or = nn.Parameter(torch.ones((cfg.N_OBJ, cfg.N_OBJ), device=dev))
    Oi = nn.Parameter(torch.zeros((cfg.N_OBJ, cfg.N_OBJ), device=dev))
    P0 = propagate_np((rr <= cfg.probe_dia / 2 - 6).astype(np.complex64),
                      make_H(cfg, cfg.z_probe_init))
    P0 = P0 / np.abs(P0).max() * support
    Pr = nn.Parameter(torch.from_numpy(P0.real.astype(np.float32)).to(dev))
    Pi = nn.Parameter(torch.from_numpy(P0.imag.astype(np.float32)).to(dev))

    c = cfg.REG_CROP
    lo, lp = cfg.lr_obj, cfg.lr_prb
    hist = []
    t0 = time.time()

    for st in range(cfg.stages):
        for which in (("obj", "prb") if cfg.opt_mode != "joint" else ("both",)):
            if which == "obj":
                params, lr = [Or, Oi], lo
            elif which == "prb":
                params, lr = [Pr, Pi], lp
            else:
                params, lr = [Or, Oi, Pr, Pi], lo
            # Keras 每个 stage 重新 compile -> 优化器状态清零，这里照做
            opt = torch.optim.Adam(params, lr=lr, amsgrad=True)
            n_ep = cfg.obj_epoch if which != "prb" else cfg.prb_epoch
            for _ in range(n_ep):
                for i in range(len(positions)):        # batch_size=1, shuffle=False
                    O = torch.complex(Or, Oi)
                    P = torch.complex(Pr * sup, Pi * sup)
                    U = forward_torch(cfg, O, P, corners[i:i + 1], Ht)
                    loss = data_loss_direct(U, y0[i:i + 1])
                    if which != "prb" and (cfg.tv1 > 0 or cfg.tv2 > 0):
                        Oreg = O[c:-c, c:-c] if c > 0 else O
                        if cfg.tv1 > 0:
                            loss = loss + cfg.tv1 * tgv_loss(Oreg.abs(), beta=1.0) * 1e-2
                        if cfg.tv2 > 0:
                            loss = loss + cfg.tv2 * tgv_loss(safe_angle(Oreg), beta=1.0) * 1e-2
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            if which == "obj":
                lo *= cfg.decay
            elif which == "prb":
                lp *= cfg.decay
            else:
                lo *= cfg.decay

        rec = (Or + 1j * Oi).detach().cpu().numpy()
        pc = ((Pr * sup) + 1j * (Pi * sup)).detach().cpu().numpy()
        mo = evaluate_object_roi(cfg, rec, obj)
        mp = evaluate_probe(pc, probe, support)
        hist.append({**mo, **mp})
        print(f"  stage {st+1}/{cfg.stages} | amplitude PSNR {mo['psnr_o_amp']:6.2f} dB | "
              f"SSIM {mo['ssim_o_amp']:.4f} | phase RMSE {mo['rmse_o_phi_rad']:.4f} rad | "
              f"complex error {mo['relerr_o_complex']:.4f} | probe error {mp['relerr_p_complex']:.4f}",
              flush=True)
    print(f"[ad] 用时 {time.time()-t0:.1f}s")
    _save(cfg, "ad", rec, pc, obj, probe, support, hist)
    return hist


# ============================================================================ #
# 模式 2: ProPtyNet
# ============================================================================ #

def run_net(cfg: Cfg):
    dev = cfg.dev()
    torch.manual_seed(cfg.seed)
    obj, probe, support, _, rr, _R = make_truth(cfg)
    positions = make_scan_positions(cfg)
    H_true = make_H(cfg, cfg.dz_true)
    I, I_clean = simulate(cfg, obj, probe, positions, H_true)

    Ht = torch.from_numpy(H_true).to(dev)
    corners = torch.from_numpy(cfg.PATCH_C0 - positions).to(dev)
    sup = torch.from_numpy(support).to(dev)
    Im = torch.from_numpy(I).to(dev)
    Iclean = torch.from_numpy(I_clean).to(dev)
    sqrtIm = torch.sqrt(Im.clamp_min(0))

    # ---- 网络输入: 零填充到画布尺寸的实测衍射图堆栈，全程固定 (论文 Fig.1c) ----
    pad = (cfg.N_OBJ - cfg.N) // 2
    x_in = F.pad(Im[None], (pad, pad, pad, pad))            # (1, J, N_OBJ, N_OBJ)
    x_in = x_in / x_in.amax(dim=(2, 3), keepdim=True).clamp_min(1e-12)
    net = ProPtyUNet(len(positions), cfg.base_ch, cfg.amp_act).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    print(f"[net] U-Net 参数量 {nparam/1e6:.2f} M  (论文 2.5 M)  输入 {tuple(x_in.shape)}")

    # 论文 Eq.5 的掩膜。本项目探针被二值 support 硬约束 -> Loss2 恒 0；
    # S2 只有在 noise_clip=True 时才非平凡。
    S2 = (Im < Im.max() - 1e-12).float() if cfg.noise_clip else torch.ones_like(Im)
    S1 = sup
    if cfg.data_loss == "paper":
        print(f"[net] 论文损失: 过曝像素占比 {float((1-S2).mean())*100:.4f} %"
              + ("" if cfg.noise_clip else"   <- noise_clip=False，S2 全 1，Eq.5 退化成普通 L2"))

    def decode(a, p):
        amp_s, amp_p = a[0, 0], a[0, 1]
        phs_s, phs_p = p[0, 0], p[0, 1]
        O = (amp_s * torch.exp(1j * cfg.phase_span_obj * phs_s)).to(torch.complex64)
        Pfull = amp_p * torch.exp(1j * cfg.phase_span_prb * phs_p)
        P = Pfull[pad:pad + cfg.N, pad:pad + cfg.N].to(torch.complex64) * sup   # Fig.1(c) 裁剪
        return O, P, amp_p[pad:pad + cfg.N, pad:pad + cfg.N]

    # 尺度标定: 网络初始输出幅度是任意的，先算一次前向把整体尺度对上（之后冻结）。
    with torch.no_grad():
        O, P, _ = decode(*net(x_in))
        U0 = forward_torch(cfg, O, P, corners, Ht)
        scale = (sqrtIm.mean() / U0.abs().mean().clamp_min(1e-12)).item()
    print(f"[net] 冻结的幅度标定系数 = {scale:.4g}")

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr_net)
    hist, t0 = [], time.time()
    for it in range(cfg.iters):
        gamma = cfg.gamma0 * (cfg.gamma_end / cfg.gamma0) ** (it / max(cfg.iters - 1, 1))
        O, P, amp_p = decode(*net(x_in))
        U = forward_torch(cfg, O, P, corners, Ht) * scale
        if cfg.data_loss == "paper":
            loss = data_loss_paper(U.abs() ** 2, Im, S2, gamma, amp_p, S1, cfg.beta)
        else:
            loss = data_loss_direct(U, sqrtIm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (it + 1) % cfg.eval_every == 0 or it == cfg.iters - 1:
            with torch.no_grad():
                real = torch.linalg.vector_norm(U.abs() ** 2 - Iclean).item()
                rec = O.cpu().numpy(); pc = P.cpu().numpy()
            mo = evaluate_object_roi(cfg, rec, obj)
            mp = evaluate_probe(pc, probe, support)
            hist.append({"it": it + 1, "loss": loss.item(), "real": real, **mo, **mp})
            if (it + 1) % (cfg.eval_every * 4) == 0 or it == cfg.iters - 1:
                print(f"  it {it+1:5d} | loss {loss.item():.4e} | real {real:.4e} | "
                      f"amplitude PSNR {mo['psnr_o_amp']:6.2f} dB | SSIM {mo['ssim_o_amp']:.4f} | "
                      f"phase RMSE {mo['rmse_o_phi_rad']:.4f} rad | complex error "
                      f"{mo['relerr_o_complex']:.4f} | probe error {mp['relerr_p_complex']:.4f}",
                      flush=True)
    print(f"[net] 用时 {time.time()-t0:.1f}s / {cfg.iters} it")

    # 论文判据(2): real error 与 loss 同趋势且更低 = 网络在屏蔽噪声
    if len(hist) > 8:
        tail = [h["real"] for h in hist[-len(hist)//4:]]
        k = np.polyfit(np.arange(len(tail)), np.array(tail), 1)[0]
        print(f"[net] 尾段 real error 斜率 {k:+.3e}  "
              f"({'仍在下降' if k < 0 else '已回升 -> 开始拟合噪声'})")
    _save(cfg, "net", rec, pc, obj, probe, support, hist)
    return hist


# ============================================================================ #
# 保存 / 绘图
# ============================================================================ #

def _save(cfg, tag, rec, pc, obj, probe, support, hist):
    os.makedirs(cfg.outdir, exist_ok=True)
    np.savez_compressed(os.path.join(cfg.outdir, f"{tag}_result.npz"),
                        obj_rec=rec, probe_rec=pc, hist=json.dumps(hist),
                        cfg=json.dumps(asdict(cfg), default=str))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    c = cfg.EVAL_CROP
    ra, _ = align_global_factor(rec[c:-c, c:-c], obj[c:-c, c:-c])
    ga = obj[c:-c, c:-c]
    pa, _ = align_global_factor(pc * (support > 0), probe * (support > 0))
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.5))
    ims = [(np.abs(ra), "rec amp"), (np.angle(ra), "rec phase"),
           (np.abs(pa), "rec probe amp"), (np.angle(pa), "rec probe phase"),
           (np.abs(ga), "GT amp"), (np.angle(ga), "GT phase"),
           (np.abs(probe), "GT probe amp"), (np.angle(probe), "GT probe phase")]
    for a, (im, t) in zip(ax.ravel(), ims):
        a.imshow(im, cmap="gray"); a.set_title(t, fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    f = os.path.join(cfg.outdir, f"{tag}_result.png")
    fig.savefig(f, dpi=140); plt.close(fig)
    print(f"[{tag}] 结果 -> {f}")


# ============================================================================ #
# CLI
# ============================================================================ #

def main():
    ap = argparse.ArgumentParser(description="ProPtyNet (PyTorch) —— 对齐 INNM_Ptycho 的约定")
    ap.add_argument("mode", choices=["check", "ad", "net"])
    for k, t in [("N", int), ("N_OBJ", int), ("dz_true", float), ("probe_dia", int),
                 ("scan_pattern", str), ("scan_npos", int), ("scan_step", int),
                 ("stages", int), ("iters", int), ("lr_net", float), ("lr_obj", float),
                 ("lr_prb", float), ("base_ch", int), ("beta", float), ("seed", int),
                 ("peak_photons", float), ("gauss_snr_db", float), ("eval_every", int),
                 ("device", str), ("outdir", str), ("assets", str), ("amp_act", str),
                 ("tv1", float), ("tv2", float), ("opt_mode", str),
                 ("phase_span_obj", float), ("phase_span_prb", float)]:
        ap.add_argument("--" + k.replace("_", "-"), dest=k, type=t)
    ap.add_argument("--data-loss", dest="data_loss", choices=["direct", "paper"])
    ap.add_argument("--poisson", dest="poisson", action="store_true", default=None)
    ap.add_argument("--noise-clip", dest="noise_clip", action="store_true", default=None)
    a = ap.parse_args()

    kw = {k: v for k, v in vars(a).items() if k != "mode" and v is not None}
    cfg = Cfg(**kw)
    {"check": run_check, "ad": run_ad, "net": run_net}[a.mode](cfg)


if __name__ == "__main__":
    main()
