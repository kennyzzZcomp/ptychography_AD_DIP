# -*- coding: utf-8 -*-
"""真值物体/探针、扫描位置、噪声与衍射数据仿真。"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch

from functions.common.assets import asset_dir
from functions.paperrepro.optics import forward_ptycho

PI = math.pi

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_paper import Cfg

def _resolve(d: Path, name: str) -> Path:
    """大小写兜底：Windows 上 usaf.jpg == USAF.jpg，Linux 服务器上不是。"""
    p = d / name
    if p.is_file():
        return p
    for q in d.iterdir():
        if q.is_file() and q.name.lower() == name.lower():
            return q
    raise FileNotFoundError(f"{name} 不在 {d}（注意大小写；实际文件名可能是 USAF.jpg）")

def _siemens_star(n: int, spokes: int = 16, r_out: float = 0.46, min_bar_px: float = 2.0):
    """合成辐条靶，对应论文 Fig.2(a) 的 Object Phase。返回 0..255。

    中心的平坦盘半径【自动】定在"最细辐条恰好 min_bar_px 像素宽"处:
        辐条宽(r) = π·r / spokes  =>  r_in = spokes·min_bar_px/π
    再往里就会混叠。默认 16 条而不是论文那种密辐条，是因为评价区只有画布中心
    约 100×100 —— 辐条太密的话中心平坦盘会吃掉评价区一大块，指标就虚了。
    """
    y, x = np.mgrid[0:n, 0:n] - n / 2.0
    rr = np.sqrt(x * x + y * y)
    r_in = spokes * min_bar_px / PI
    v = (np.cos(spokes * np.arctan2(y, x)) > 0).astype(np.float64)
    v[(rr < r_in) | (rr / n > r_out)] = 0.5
    return v * 255.0

def _imread(path: Path, n: int):
    """读灰度图 -> (n,n) float64 0..255。非方图先【中心裁成方形】再缩放，
    否则 USAF.jpg (1931x2498) 会被横向压扁，靶条的线宽比就不对了。"""
    try:
        import cv2
        a = cv2.imread(str(path), 0)
        if a is None:
            raise FileNotFoundError(path)
        h, w = a.shape
        k = min(h, w)
        a = a[(h - k) // 2:(h - k) // 2 + k, (w - k) // 2:(w - k) // 2 + k]
        return cv2.resize(a, (n, n), interpolation=cv2.INTER_CUBIC).astype(np.float64)
    except ImportError:
        from PIL import Image
        im = Image.open(path).convert("L")
        w, h = im.size
        k = min(h, w)
        im = im.crop(((w - k) // 2, (h - k) // 2, (w - k) // 2 + k, (h - k) // 2 + k))
        return np.asarray(im.resize((n, n), Image.BICUBIC), dtype=np.float64)

def _object_map(cfg: Cfg, spec: str, n: int):
    if spec.lower() in ("siemens", "star", "spoke"):
        return _siemens_star(n)
    return _imread(_resolve(asset_dir(cfg), spec), n)

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
    M = cfg.obj_size
    a = _object_map(cfg, cfg.amp_image, M)
    p = _object_map(cfg, cfg.phs_image, M)
    amp = 0.2 + 0.8 * a / a.max()
    phs = (-1 + 2 * (p - p.min()) / max(p.max() - p.min(), 1e-12)) * cfg.obj_phase_rad
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

def make_positions(cfg: Cfg):
    """整数像素的方形光栅，左上角坐标 (I,2)。步长以 Δx1 为单位。"""
    p = [(cfg.scan_offset + r * cfg.step_px, cfg.scan_offset + c * cfg.step_px)
         for r in range(cfg.grid) for c in range(cfg.grid)]
    return np.asarray(p, dtype=np.int64)

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
