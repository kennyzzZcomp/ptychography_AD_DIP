# -*- coding: utf-8 -*-
"""
ProPtyNet_paper.py —— 严格按论文复现

Z. Liu, Y. Chen, N. Lin, "Noise-robust ptychography using unsupervised neural
network", Optics and Lasers in Engineering 186 (2025) 108791.

【与 ProPtyNet_torch.py 的关系】那一份是为了跟 INNM_Ptycho.ipynb 的 AD 基线做
可比对照，沿用了 INNM 的角谱前向和 INNM 的振幅域损失；本文件不管可比性，只按论文:

    前向    Eq.(2)  Fresnel 单次 FFT,  Δx1·Δx2 = λz/M       (那份是角谱 Δx1=Δx2)
    输出    Fig.1   S = amp_s·exp(jπ·phs_s), P = amp_p·exp(jπ·phs_p)   相位跨度 ±π
    损失    Eq.(4)(5)  β·Loss1 + (1-β)·Loss2, 双掩膜 S1/S2 + γ 衰减
    探针    只有 Loss2 的软约束, 【没有】二值 support 硬掩膜
    噪声    Table 1  全局归一化 -> 加噪 -> clip[0,1] (clip 产生过曝, 才有 S2)

用法:
    python ProPtyNet_paper.py check                       # 采样/几何自检
    python ProPtyNet_paper.py check  --preset smoke
    python ProPtyNet_paper.py run    --preset smoke --iters 400        # CPU 冒烟
    python ProPtyNet_paper.py run    --preset paper --iters 2000       # 需要 GPU
    python ProPtyNet_paper.py run    --preset paper --noise mixed --snr 30

资源: cameraman.bmp / westconcordorthophoto.bmp (自动在上级目录找)
"""

from __future__ import annotations

import os
import json
import time
import math
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter

PI = math.pi


# ============================================================================ #
# 配置
# ============================================================================ #

PRESETS = {
    # 论文第 3 节的仿真参数，原样照抄
    "paper": dict(wlength=632e-9, N=512, det_pixel=15.04e-6, z=0.165,
                  grid=10, step_px=10, probe_diam_um=800.0, obj_size=612),
    # 只为在 CPU 上验证代码通路，物理上不等价于论文，别拿它的数字下结论
    "smoke": dict(wlength=632e-9, N=128, det_pixel=30.0e-6, z=0.050,
                  grid=5, step_px=8, probe_diam_um=400.0, obj_size=168),
}


@dataclass
class Cfg:
    preset: str = "paper"
    wlength: float = 632e-9
    N: int = 512                 # 探测器像素数 M = FFT 尺寸
    det_pixel: float = 15.04e-6  # Δx2
    z: float = 0.165             # 样品 -> 探测器
    grid: int = 10               # grid×grid 个扫描点
    step_px: int = 10            # 步长（重建面像素 Δx1）
    probe_diam_um: float = 800.0 # 针孔直径
    obj_size: int = 612          # 论文写的物体画布

    # ---- 噪声 (Table 1) ----
    noise: str = "none"          # none | gaussian | poisson | mixed
    snr_db: float = 30.0
    noise_seed: int = 42

    # ---- 损失 (Eq.4/5) ----
    beta: float = 0.90
    gamma0: float = 1.0
    gamma_end: float = 0.02
    s1_margin: float = 1.0       # S1 半径 = margin × 针孔半径

    # ---- 网络 ----
    base_ch: int = 32            # 32/64/128/256, 3 次池化 (Fig.1b)
    phase_span: float = PI       # 论文: exp(jπ·phs)

    # ---- 优化 ----
    iters: int = 2000
    lr: float = 1e-3             # 论文: 5e-4 ~ 5e-3
    lr_final_frac: float = 0.1   # cosine 衰减到 lr 的这个比例; 1.0 = 不衰减
    pos_batch: int = 0           # 0 = 全 batch（论文写法）; >0 = 每步随机取这么多位置

    # ---- 其它 ----
    scale_cal: bool = True       # 冻结的幅度标定（论文没写，见下方说明）
    seed: int = 0
    device: str = "auto"
    assets: str = ""
    outdir: str = "results_paper"
    eval_every: int = 25

    def __post_init__(self):
        # 采样关系 Eq.(2) 下面那条: Δx1·Δx2 = λz/M
        self.dx1 = self.wlength * self.z / (self.N * self.det_pixel)
        self.probe_diam_px = self.probe_diam_um * 1e-6 / self.dx1
        self.scan_span = self.N + (self.grid - 1) * self.step_px
        self.n_pat = self.grid * self.grid
        self.scan_offset = (self.obj_size - self.scan_span) // 2
        # 3 次池化要求边长能被 8 整除；612 不行（612/8=76.5），必须再 pad
        self.net_size = int(math.ceil(self.obj_size / 8) * 8)
        self.chirp_limit = math.sqrt(self.wlength * self.z / self.N)

    def dev(self):
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_cfg(args) -> Cfg:
    kw = dict(PRESETS[args.preset or "paper"])
    kw["preset"] = args.preset or "paper"
    for k, v in vars(args).items():
        if k in ("mode", "preset") or v is None:
            continue
        kw[k] = v
    return Cfg(**kw)


def asset_dir(cfg: Cfg) -> Path:
    if cfg.assets:
        return Path(cfg.assets)
    here = Path(__file__).resolve().parent
    for c in (here, here.parent, here.parent.parent):
        if (c / "cameraman.bmp").is_file():
            return c
    return here


# ============================================================================ #
# 前向模型 —— 论文 Eq.(2)
# ============================================================================ #

def make_quad_phase(cfg: Cfg, device, dtype=torch.complex64):
    """exp(-j·k/(2z)·((uΔx1)² + (vΔy1)²))  —— Eq.(2) 花括号里的二次相位因子。

    符号照论文取负。教科书标准 Fresnel 单 FFT 是 +j，两者差一个共轭（等价 z→-z）。
    若重建出来是镜像/共轭，把 --quad-sign 翻过来。

    这个因子固定在实验室系（以光轴为中心），对所有扫描位置是同一个常数张量。
    Eq.(2) 写成"探针平移 P(uΔx1-p_i)、物体不动"，本实现写成"物体 patch 滑动、
    探针不动"——两者是同一件事换了参考系，且后者只需在 M×M 窗口上做 FFT。
    """
    n = cfg.N
    x = (torch.arange(n, device=device, dtype=torch.float64) - n // 2) * cfg.dx1
    X, Y = torch.meshgrid(x, x, indexing="ij")
    k = 2 * PI / cfg.wlength
    ph = cfg.quad_sign * k / (2 * cfg.z) * (X ** 2 + Y ** 2)
    return torch.exp(1j * ph).to(dtype)


def make_positions(cfg: Cfg):
    """整数像素的方形光栅，左上角坐标 (I,2)。步长以 Δx1 为单位。"""
    p = [(cfg.scan_offset + r * cfg.step_px, cfg.scan_offset + c * cfg.step_px)
         for r in range(cfg.grid) for c in range(cfg.grid)]
    return np.asarray(p, dtype=np.int64)


def forward_ptycho(obj, probe, positions, Q, n, chunk=0):
    """I_i = |DFT{ P·S_patch·Q }|²  —— Eq.(2)

    obj (M,M)c ; probe (n,n)c ; positions (I,2)long ; Q (n,n)c  ->  (I,n,n) float
    绝对不做逐张归一化：不同扫描点的相对强度本身就是信息。
    """
    idx = torch.arange(n, device=obj.device)

    def _one(pos):
        rows = pos[:, 0:1] + idx[None, :]
        cols = pos[:, 1:2] + idx[None, :]
        psi = obj[rows[:, :, None], cols[:, None, :]] * probe[None] * Q[None]
        far = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(psi, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1))
        return far.real ** 2 + far.imag ** 2

    if chunk <= 0 or chunk >= positions.shape[0]:
        return _one(positions)
    return torch.cat([_one(positions[s:s + chunk])
                      for s in range(0, positions.shape[0], chunk)], 0)


# ============================================================================ #
# 真值物体 / 探针 —— 论文第 3 节
# ============================================================================ #

def _imread(path: Path, n: int):
    try:
        import cv2
        a = cv2.imread(str(path), 0)
        if a is None:
            raise FileNotFoundError(path)
        return cv2.resize(a, (n, n), interpolation=cv2.INTER_CUBIC).astype(np.float64)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(path).convert("L").resize((n, n), Image.BICUBIC),
                          dtype=np.float64)


def _lowpass_noise(n, sigma_px, rng):
    f = rng.standard_normal((n, n))
    fx = np.fft.fftfreq(n)[:, None]; fy = np.fft.fftfreq(n)[None, :]
    g = np.exp(-2 * (PI * sigma_px) ** 2 * (fx ** 2 + fy ** 2))
    f = np.real(np.fft.ifft2(np.fft.fft2(f) * g))
    f -= f.min()
    return f / max(f.max(), 1e-12)


def make_truth(cfg: Cfg):
    """物体: 两张分辨率靶（论文用 resolution test targets，这里沿用你手上的两张图）。
    探针: 800 µm 圆孔（论文的 "circular part with an 800 µm diameter"）+
          "mandrill" 振幅纹理。手上没有 mandrill，用低通随机场代替 —— 论文的重点是
          探针振幅不是平的，纹理来源不影响结论，但这是一处替代，报数据时要说明。
    """
    d = asset_dir(cfg)
    M = cfg.obj_size
    a = _imread(d / "cameraman.bmp", M)
    p = _imread(d / "westconcordorthophoto.bmp", M)
    amp = 0.2 + 0.8 * a / a.max()
    phs = (-1 + 2 * (p - p.min()) / max(p.max() - p.min(), 1e-12)) * (0.8 * PI)
    obj = (amp * np.exp(1j * phs)).astype(np.complex64)

    n = cfg.N
    yy, xx = np.mgrid[0:n, 0:n] - n / 2
    rr = np.sqrt(xx ** 2 + yy ** 2)
    R = cfg.probe_diam_px / 2
    rng = np.random.default_rng(cfg.seed)
    aperture = (rr <= R).astype(np.float64)
    tex = 0.4 + 0.6 * _lowpass_noise(n, max(R / 8, 2.0), rng)
    p_amp = aperture * tex
    # 针孔面到样品面的一点离焦，让探针相位不是恒 0（论文的探针相位来自实际光路）
    p_phs = aperture * (2.0 * (rr / max(R, 1)) ** 2)
    probe = (p_amp * np.exp(1j * p_phs)).astype(np.complex64)
    probe = probe / np.abs(probe).max()

    S1 = (rr <= cfg.s1_margin * R).astype(np.float32)     # Eq.(6) 的针孔掩膜
    return obj, probe, S1, rr


# ============================================================================ #
# 噪声 —— 论文 Table 1
# ============================================================================ #

def add_noise(I, kind, snr_db, rng):
    """输入 I 已按【全局】最大值归一到 [0,1]（Table 1 第 3 步）。

    !! Table 1 的 Poisson 列第 5 行写的是概率质量函数 λ^r·e^(-λ)/255^r，第 6 行又
       写 Signal = Pois + Signal（把 pmf 的值当噪声加上去），物理上讲不通，几乎肯定
       是转写错误。这里实现标准散粒噪声 I' = Poisson(I·P)/P，P 由目标 SNR 定出。
       Gaussian 列严格照论文: σ = mean(signal)/sqrt(10^(SNR/10))。
    """
    out = I.astype(np.float64).copy()
    lin = 10.0 ** (snr_db / 10.0)
    if kind in ("poisson", "mixed"):
        for i in range(len(out)):
            m = max(I[i].mean(), 1e-12)
            P = lin / m
            out[i] = rng.poisson(np.clip(I[i], 0, None) * P) / P
    if kind in ("gaussian", "mixed"):
        for i in range(len(out)):
            sigma = I[i].mean() / math.sqrt(lin)
            out[i] = out[i] + rng.normal(0.0, sigma, I[i].shape)
    # Table 1 最后一步: clip。负值归零，>1 饱和 —— 过曝就是这么来的，S2 靠它才非平凡
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def simulate(cfg: Cfg, obj, probe, positions, Q, device):
    with torch.no_grad():
        I = forward_ptycho(torch.from_numpy(obj).to(device),
                           torch.from_numpy(probe).to(device),
                           torch.from_numpy(positions).to(device),
                           Q, cfg.N, chunk=16).cpu().numpy()
    I = I / I.max()                                   # 全局归一化（Table 1 第 3 步）
    I_clean = np.clip(I, 0.0, 1.0).astype(np.float32)
    if cfg.noise == "none":
        return I_clean.copy(), I_clean
    rng = np.random.default_rng(cfg.noise_seed)
    return add_noise(I, cfg.noise, cfg.snr_db, rng), I_clean


# ============================================================================ #
# 损失 —— 论文 Eq.(4)(5)
# ============================================================================ #

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


# ============================================================================ #
# U-Net —— 论文 Fig.1(b)
# ============================================================================ #

class DoubleConv(nn.Module):
    """Conv2D-BN-LeakyReLU-Conv2D-BN-LeakyReLU，尺寸不变。"""

    def __init__(self, cin, cout):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True),
            nn.Conv2d(cout, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True))

    def forward(self, x):
        return self.f(x)


class ProPtyUNet(nn.Module):
    """Fig.1(b): J·M·N 输入 -> 32 -> 64 -> 128 -> 256(瓶颈) -> 解码 -> 4 个单通道。

    输出头严格按 Fig.1(b) 用 4 个【独立的】单通道 Conv2d:
        amp_s, amp_p : Conv2d + LeakyReLU
        phs_s, phs_p : Conv2d + Tanh      -> 复场 = amp · exp(jπ·phs)
    输出层没有 BatchNorm —— 在输出层做 BN 会把振幅强制成零均值单位方差。
    """

    def __init__(self, in_ch, base=32):
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
        self.amp_s = nn.Conv2d(c[0], 1, 3, 1, 1)
        self.phs_s = nn.Conv2d(c[0], 1, 3, 1, 1)
        self.amp_p = nn.Conv2d(c[0], 1, 3, 1, 1)
        self.phs_p = nn.Conv2d(c[0], 1, 3, 1, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        y = self.d3(torch.cat([self.u3(self.bot(self.pool(x3))), x3], 1))
        y = self.d2(torch.cat([self.u2(y), x2], 1))
        y = self.d1(torch.cat([self.u1(y), x1], 1))
        lr = lambda t: F.leaky_relu(t, 0.2)
        return (lr(self.amp_s(y))[0, 0], torch.tanh(self.phs_s(y))[0, 0],
                lr(self.amp_p(y))[0, 0], torch.tanh(self.phs_p(y))[0, 0])


# ============================================================================ #
# 指标
# ============================================================================ #

def ssim(x, y, dr, win=7):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if dr == 0:
        return 1.0
    C1, C2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    NP = win ** 2; cn = NP / (NP - 1)
    f = lambda a: uniform_filter(a, size=win)
    ux, uy = f(x), f(y)
    vx = cn * (f(x * x) - ux * ux); vy = cn * (f(y * y) - uy * uy)
    vxy = cn * (f(x * y) - ux * uy)
    S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux ** 2 + uy ** 2 + C1) * (vx + vy + C2))
    p = (win - 1) // 2
    return float(S[p:-p, p:-p].mean())


def psnr(a, b, dr):
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(dr ** 2 / max(mse, 1e-30)))


def align(rec, gt):
    den = np.vdot(rec, rec).real
    return rec * (np.vdot(rec, gt) / max(den, 1e-30))


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
    rec = align(rec, gt)
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


# ============================================================================ #
# 自检
# ============================================================================ #

def run_check(cfg: Cfg):
    ok = lambda c: "OK " if c else "!! "
    print("=" * 78)
    print(f"preset = {cfg.preset}   （论文第 3 节参数：512 探测器 / 15.04µm / 16.5cm / 10×10）")
    print("=" * 78)
    print(f"  λ={cfg.wlength*1e9:.1f}nm  z={cfg.z*100:.2f}cm  Δx2={cfg.det_pixel*1e6:.2f}µm  M={cfg.N}")
    print(f"  -> Δx1 = λz/(M·Δx2) = {cfg.dx1*1e6:.3f} µm            [Eq.(2) 下的采样关系]")
    print(f"  {ok(cfg.dx1 <= cfg.chirp_limit)}chirp 采样: Δx1 ≤ √(λz/M) = {cfg.chirp_limit*1e6:.3f} µm"
          f"   (用了 {100*cfg.dx1/cfg.chirp_limit:.0f}%)")
    print(f"  针孔 {cfg.probe_diam_um:.0f} µm = {cfg.probe_diam_px:.1f} px"
          f"  (占窗口 {100*cfg.probe_diam_px/cfg.N:.1f}%)")
    print(f"  扫描 {cfg.grid}×{cfg.grid}={cfg.n_pat} 点  步长 {cfg.step_px} px = "
          f"{cfg.step_px*cfg.dx1*1e6:.1f} µm")
    print(f"  线性重叠 = 1 - step/D = {100*(1-cfg.step_px/cfg.probe_diam_px):.1f} %")
    print(f"  扫描跨度 {cfg.scan_span}  物体画布 {cfg.obj_size}  "
          f"{ok(cfg.scan_offset >= 0)}余量 {cfg.scan_offset} px")
    print(f"  {ok(cfg.obj_size % 8 == 0)}画布能被 8 整除（Fig.1b 三次池化要求）"
          f"  -> 网络内部再 pad 到 {cfg.net_size}")
    if cfg.preset == "paper":
        print("-" * 78)
        print("论文自身的三处不自洽（不是本代码的 bug，报数据时要说明取了哪一种）:")
        print(f"  1) 正文说重叠 0.7，但 step={cfg.step_px}px / D={cfg.probe_diam_px:.0f}px 给出 "
              f"{100*(1-cfg.step_px/cfg.probe_diam_px):.0f}%")
        print(f"  2) 物体写 612，而 {cfg.N}+({cfg.grid}-1)×{cfg.step_px} = "
              f"{cfg.N+(cfg.grid-1)*cfg.step_px}")
        print("  3) 正文说 zero-pad 到 612 是「网络结构要求」，但 612/8 = 76.5 不是整数，"
              "612 恰恰不满足三次池化")
    print("-" * 78)

    device = cfg.dev()
    cfg.quad_sign = getattr(cfg, "quad_sign", -1.0)
    obj, probe, S1, rr = make_truth(cfg)
    pos = make_positions(cfg)
    Q = make_quad_phase(cfg, device)
    Ot = torch.from_numpy(obj).to(device); Pt = torch.from_numpy(probe).to(device)
    post = torch.from_numpy(pos).to(device)

    with torch.no_grad():
        I = forward_ptycho(Ot, Pt, post, Q, cfg.N, chunk=8)
    Im = (I / I.max()).clamp(0, 1)
    rs, cs = illum_roi(cfg, probe, pos)
    print(f"[F] 照明覆盖 ROI = 行 {rs.start}..{rs.stop-1} 列 {cs.start}..{cs.stop-1}"
          f"  ({rs.stop-rs.start}×{cs.stop-cs.start})，画布 {cfg.obj_size}"
          f"  -> 只占 {100*(rs.stop-rs.start)**2/cfg.obj_size**2:.0f}% 面积，指标必须只在这里算")
    print(f"[A] 衍射图 {tuple(I.shape)}  动态范围 [{Im.min():.2e}, {Im.max():.2e}]")

    with torch.no_grad():
        I2 = forward_ptycho(Ot, Pt, post, Q, cfg.N, chunk=8)
        rel = ((I2 / I2.max()).clamp(0, 1) - Im).norm() / Im.norm()
    print(f"[B] 前向可重复性                 {rel.item():.3e}   (应 0)")

    with torch.no_grad():
        I3 = forward_ptycho(Ot, Pt, post + 1, Q, cfg.N, chunk=8)
        rel1 = ((I3 / I3.max()).clamp(0, 1) - Im).norm() / Im.norm()
    print(f"[C] 位置整体偏 1 px 的残差       {rel1.item():.3e}   (应远大于 B)")

    # 梯度校验（float64，在偏离真值的点上做有限差分）
    Qd = make_quad_phase(cfg, device, torch.complex128)
    Od, Pd = Ot.to(torch.complex128), Pt.to(torch.complex128)
    Imd = Im.to(torch.float64); sc = I.max().to(torch.float64)
    post_g = post[:min(8, len(post))]   # complex128 在 GPU 上极慢且吃显存，只取 8 个位置
    Imd_g = Imd[:len(post_g)]
    torch.manual_seed(0)
    base = (Od.abs() * (1 + 0.15 * torch.randn(Od.shape, device=device,
                                               dtype=torch.float64)))
    ang = torch.angle(Od)

    def L(a):
        return ((forward_ptycho((a * torch.exp(1j * ang)), Pd, post_g, Qd, cfg.N, chunk=8)
                 / sc - Imd_g) ** 2).sum()

    p_ = nn.Parameter(base.clone()); L(p_).backward()
    # 取样点必须落在【被这 8 个位置照明的】区域内，否则梯度本来就该是 0（假通过）
    i = int(pos[0][0]) + cfg.N // 2
    j = int(pos[0][1]) + cfg.N // 2
    with torch.no_grad():
        e = 1e-6; b = p_.detach().clone()

        def f(v):
            a = b.clone(); a[i, j] = v
            return L(a).item()

        num = (f(b[i, j] + e) - f(b[i, j] - e)) / (2 * e)
    ana = p_.grad[i, j].item()
    r = abs(num - ana) / max(abs(num), abs(ana), 1e-30)
    good = ana != 0.0 and r < 1e-4
    print(f"[D] 梯度 @({i},{j}) 数值 {num:+.4e} vs 解析 {ana:+.4e}  相对差 {r:.2e}  "
          + ("OK " if good else ("!! 解析梯度为 0，取样点没被照到，换点" if ana == 0.0 else "!! ")))

    _report_device(cfg, device)
    Imn, Icl = simulate(cfg, obj, probe, pos, Q, device)
    S2 = (Imn < 1.0 - 1e-6).mean()
    print(f"[E] 噪声 {cfg.noise}@{cfg.snr_db}dB  ->  过曝(S2=0)像素占比 "
          f"{100*(1-S2):.4f} %"
          + ("   <- noise=none 时几乎为 0，Eq.5 的 γ 无事可做" if cfg.noise == "none" else ""))
    print("=" * 78)


def _report_device(cfg: Cfg, device):
    """设备 + 显存估算。代码本身与设备无关：cfg.dev() 见到 CUDA 就用 CUDA。"""
    NS, J, n, b = cfg.net_size, cfg.n_pat, cfg.N, cfg.base_ch
    mb = lambda x: x / 2 ** 20
    unet = mb(J * NS * NS * 4)
    for i, c in enumerate([b, b * 2, b * 4, b * 8]):
        unet += mb(c * (NS // 2 ** i) ** 2 * 4) * 6
    for i, c in enumerate([b * 4, b * 2, b]):
        unet += mb(c * (NS // 2 ** (2 - i)) ** 2 * 4) * 8
    fwd = mb(J * n * n * 36)                      # psi/shift/fft/shift(complex64) + |·|²
    peak = (unet + fwd) * 1.4 / 1024
    print(f"[G] 设备 {device}"
          + (f" ({torch.cuda.get_device_name(0)}, "
             f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB)"
             if device.type == "cuda" else "  <- 无 GPU 时自动退回 CPU，代码路径相同"))
    print(f"    显存估算: U-Net 激活 {unet:.0f} MB + 前向 {fwd:.0f} MB "
          f"-> 峰值约 {peak:.1f} GB")
    if peak > 8:
        print(f"    偏大，可用 --pos-batch 50 或 --base-ch 16 降下来"
              f"（注意 base-ch 改了就不是论文的 2.5 M 参数了）")


# ============================================================================ #
# 主流程
# ============================================================================ #

def run(cfg: Cfg):
    device = cfg.dev()
    torch.manual_seed(cfg.seed)
    cfg.quad_sign = getattr(cfg, "quad_sign", -1.0)
    if device.type == "cuda":
        # 网络输入尺寸全程固定，benchmark 能稳定选到最快 conv 算法
        torch.backends.cudnn.benchmark = True
    _report_device(cfg, device)

    obj, probe, S1_np, rr = make_truth(cfg)
    pos = make_positions(cfg)
    rs, cs = illum_roi(cfg, probe, pos)
    Q = make_quad_phase(cfg, device)
    Im_np, Icl_np = simulate(cfg, obj, probe, pos, Q, device)

    post = torch.from_numpy(pos).to(device)
    Im = torch.from_numpy(Im_np).to(device)
    Icl = torch.from_numpy(Icl_np).to(device)
    S1 = torch.from_numpy(S1_np).to(device)
    S2 = (Im < 1.0 - 1e-6).float()                       # Eq.(6)

    # ---- 网络输入: 零填充后的实测衍射图堆栈，全程固定 (Fig.1c) ----
    M, n, NS = cfg.obj_size, cfg.N, cfg.net_size
    pad_o = (M - n) // 2                                  # 512 -> 612
    pad_n = (NS - M) // 2                                 # 612 -> 616 (被 8 整除)
    x = F.pad(Im[None], (pad_o,) * 4)
    x = F.pad(x, (pad_n, NS - M - pad_n, pad_n, NS - M - pad_n))
    x = x / x.amax(dim=(2, 3), keepdim=True).clamp_min(1e-12)

    net = ProPtyUNet(cfg.n_pat, cfg.base_ch).to(device)
    npar = sum(p.numel() for p in net.parameters())
    print(f"[net] 参数 {npar/1e6:.2f} M (论文 2.5 M) | 输入 {tuple(x.shape)} | "
          f"过曝像素 {100*float(1-S2.mean()):.4f}% | 设备 {device}")
    print(f"[net] 评价 ROI = 照明覆盖区 {rs.stop-rs.start}×{cs.stop-cs.start} "
          f"(画布 {cfg.obj_size}²)")

    def decode():
        a_s, p_s, a_p, p_p = net(x)
        c = slice(pad_n, pad_n + M)
        a_s, p_s, a_p, p_p = a_s[c, c], p_s[c, c], a_p[c, c], p_p[c, c]
        O = (a_s * torch.exp(1j * cfg.phase_span * p_s)).to(torch.complex64)
        # Fig.1(c): 612 的探针裁到中心 512 再进前向；不加任何硬 support，只靠 Loss2
        d = slice(pad_o, pad_o + n)
        P = (a_p[d, d] * torch.exp(1j * cfg.phase_span * p_p[d, d])).to(torch.complex64)
        return O, P, a_p[d, d]

    scale = 1.0
    if cfg.scale_cal:
        # 论文没写网络输出的绝对幅度怎么锚定。这里在第一次前向后算一个常数并冻结，
        # 纯数值辅助，不改物理；--no-scale-cal 可关掉看差别。
        with torch.no_grad():
            O, P, _ = decode()
            I0 = forward_ptycho(O, P, post, Q, n, chunk=8)
            scale = (Im.mean() / I0.mean().clamp_min(1e-20)).item()
        print(f"[net] 冻结幅度标定 = {scale:.4g}")

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.iters, eta_min=cfg.lr * cfg.lr_final_frac)
    rng = np.random.default_rng(cfg.seed)
    hist, t0 = [], time.time()

    for it in range(cfg.iters):
        gamma = cfg.gamma0 * (cfg.gamma_end / cfg.gamma0) ** (it / max(cfg.iters - 1, 1))
        if cfg.pos_batch > 0:
            sel = torch.from_numpy(rng.choice(cfg.n_pat, cfg.pos_batch, replace=False)).to(device)
        else:
            sel = torch.arange(cfg.n_pat, device=device)

        O, P, amp_p = decode()
        Ic = forward_ptycho(O, P, post[sel], Q, n, chunk=0) * scale
        loss, l1, l2 = paper_loss(Ic, Im[sel], S2[sel], gamma, amp_p, S1, cfg.beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if (it + 1) % cfg.eval_every == 0 or it == cfg.iters - 1:
            with torch.no_grad():
                real = torch.linalg.vector_norm(Ic - Icl[sel]).item()   # 论文判据(2)
                rec = O.cpu().numpy()
            m = evaluate(rec[rs, cs], obj[rs, cs])
            hist.append({"it": it + 1, "loss": loss.item(), "real": real, **m})
            if (it + 1) % (cfg.eval_every * 4) == 0 or it == cfg.iters - 1:
                print(f"  it {it+1:5d} | loss {loss.item():.4e} (L1 {l1:.3e} L2 {l2:.3e}) | "
                      f"real {real:.4e} | γ {gamma:.3f} | amp SSIM {m['ssim_amp']:.4f} "
                      f"PSNR {m['psnr_amp']:5.2f} | phs SSIM {m['ssim_phs']:.4f} | "
                      f"relerr {m['relerr']:.4f}", flush=True)

    print(f"[net] 用时 {time.time()-t0:.1f}s / {cfg.iters} it")
    if len(hist) > 8:
        tail = np.array([h["real"] for h in hist[-len(hist)//4:]])
        k = np.polyfit(np.arange(len(tail)), tail, 1)[0]
        print(f"[net] 尾段 real error 斜率 {k:+.3e} "
              f"({'仍在下降' if k < 0 else '已回升 → 开始拟合噪声'})"
              + ("   （noise=none 时这一项没有意义）" if cfg.noise == "none" else ""))
    _save(cfg, rec[rs, cs], P.detach().cpu().numpy(), obj[rs, cs], probe, hist)


def _save(cfg, rec, pc, obj, probe, hist):
    os.makedirs(cfg.outdir, exist_ok=True)
    np.savez_compressed(os.path.join(cfg.outdir, "paper_result.npz"),
                        obj_rec=rec, probe_rec=pc, hist=json.dumps(hist),
                        cfg=json.dumps(asdict(cfg), default=str))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ra = align(rec, obj)
    pa = align(pc, probe)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.5))
    for a, (im, t) in zip(ax.ravel(), [
            (np.abs(ra), "rec amp"), (np.angle(ra), "rec phase"),
            (np.abs(pa), "rec probe amp"), (np.angle(pa), "rec probe phase"),
            (np.abs(obj), "GT amp"), (np.angle(obj), "GT phase"),
            (np.abs(probe), "GT probe amp"), (np.angle(probe), "GT probe phase")]):
        a.imshow(im, cmap="gray"); a.set_title(t, fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    f = os.path.join(cfg.outdir, "paper_result.png")
    fig.savefig(f, dpi=140); plt.close(fig)
    print(f"[net] 结果 -> {f}")


def main():
    ap = argparse.ArgumentParser(description="ProPtyNet 严格复现 (Opt. Lasers Eng. 186 (2025) 108791)")
    ap.add_argument("mode", choices=["check", "run"])
    ap.add_argument("--preset", choices=list(PRESETS))
    for k, t in [("N", int), ("obj_size", int), ("z", float), ("det_pixel", float),
                 ("grid", int), ("step_px", int), ("probe_diam_um", float),
                 ("iters", int), ("lr", float), ("lr_final_frac", float),
                 ("base_ch", int), ("beta", float), ("gamma0", float),
                 ("gamma_end", float), ("s1_margin", float), ("phase_span", float),
                 ("snr_db", float), ("pos_batch", int), ("eval_every", int),
                 ("seed", int), ("device", str), ("outdir", str), ("assets", str)]:
        ap.add_argument("--" + k.replace("_", "-"), dest=k, type=t)
    ap.add_argument("--noise", choices=["none", "gaussian", "poisson", "mixed"])
    ap.add_argument("--snr", dest="snr_db", type=float)
    ap.add_argument("--quad-sign", dest="quad_sign", type=float, choices=[-1.0, 1.0])
    ap.add_argument("--no-scale-cal", dest="scale_cal", action="store_false", default=None)
    a = ap.parse_args()

    cfg = build_cfg(a)
    cfg.quad_sign = a.quad_sign if a.quad_sign is not None else -1.0
    os.makedirs(cfg.outdir, exist_ok=True)
    (run_check if a.mode == "check" else run)(cfg)


if __name__ == "__main__":
    main()
