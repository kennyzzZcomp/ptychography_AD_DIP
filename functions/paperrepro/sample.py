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
    if spec.lower() == "siemens-sharp":
        # Binary straight-sided radial bars: no artificial gray central disk.
        y, x = np.mgrid[0:n, 0:n] - (n - 1) / 2.0
        theta = np.arctan2(y, x)
        spacing = 2 * PI / 16
        delta = (theta + spacing / 2) % spacing - spacing / 2
        along = np.hypot(x, y) * np.cos(delta)
        return ((np.abs(delta) < spacing / 4) & (along <= .46 * n)).astype(float) * 255
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

def _probe_image_texture(cfg, n, radius):
    """Fit the complete grayscale image across the GT aperture, not across N.

    Map intensities to 0.4..1, retaining the historical texture contrast range.
    The caller applies the same aperture and final peak-amplitude normalization.
    This changes simulation truth only, never reconstruction initialization.
    """
    half = int(math.ceil(radius))
    image = _imread(_resolve(asset_dir(cfg), cfg.probe_amp_image), 2 * half + 1)
    image = 0.4 + 0.6 * np.clip(image / 255.0, 0.0, 1.0)
    texture = np.zeros((n, n), dtype=np.float64)
    origin = n // 2 - half
    begin, end = max(origin, 0), min(origin + image.shape[0], n)
    if end > begin:
        source = slice(begin - origin, end - origin)
        texture[begin:end, begin:end] = image[source, source]
    return texture


def make_truth(cfg: Cfg):
    """物体: 两张分辨率靶（论文用 resolution test targets，这里沿用你手上的两张图）。
    探针: 800 µm 圆孔内的低通随机振幅（或 probe_amp_image 灰度纹理）+ 二次相位。
    论文 Section 3 写探针振幅和相位来自 mandrill 图及圆形区域；这里是替代场景，
    不能认定这种替代不影响结论，也不能用于逐数值复现论文 Fig.2。
    """
    M = cfg.obj_size
    a = _object_map(cfg, cfg.amp_image, M)
    p = _object_map(cfg, cfg.phs_image, M)
    amp = 0.2 + 0.8 * a / a.max()
    if getattr(cfg, 'obj_amp_binary_invert', False):
        # White bars on black background, with an explicit physical amplitude floor.
        a01 = (a - a.min()) / max(a.max() - a.min(), 1e-12)
        amp = cfg.obj_amp_floor + (1 - cfg.obj_amp_floor) * (a01 < .5)
    phs = (-1 + 2 * (p - p.min()) / max(p.max() - p.min(), 1e-12)) * cfg.obj_phase_rad
    obj = (amp * np.exp(1j * phs)).astype(np.complex64)

    n = cfg.N
    yy, xx = np.mgrid[0:n, 0:n] - n / 2
    rr = np.sqrt(xx ** 2 + yy ** 2)
    R = cfg.probe_diam_px / 2
    rng = np.random.default_rng(cfg.seed)
    aperture = (rr <= R).astype(np.float64)
    if getattr(cfg, 'probe_amp_image', '').lower() == 'simple-stripes':
        # Two smooth vertical bright bands across the aperture diameter.
        # Generated at the physical probe grid; no photograph or added noise.
        tex = 0.4 + 0.6 * (0.5 - 0.5 * np.cos(2 * PI * xx / max(R, 1)))
    elif getattr(cfg, 'probe_amp_image', ''):
        tex = _probe_image_texture(cfg, n, R)
    else:
        tex = 0.4 + 0.6 * _lowpass_noise(n, max(R / 8, 2.0), rng)
    p_amp = aperture * tex
    # 本仿真的二次波前（0..2 rad）；不是论文 mandrill 相位图的复刻。
    p_phs = aperture * (2.0 * (rr / max(R, 1)) ** 2)
    if getattr(cfg, 'probe_phase_mode', 'quadratic') == 'same-texture':
        inside = aperture > 0
        t = (tex - tex[inside].min()) / max(np.ptp(tex[inside]), 1e-12)
        p_phs = aperture * (2 * t - 1) * cfg.probe_phase_rad
    probe = (p_amp * np.exp(1j * p_phs)).astype(np.complex64)
    probe = probe / np.abs(probe).max()

    S1 = (rr <= cfg.s1_margin * R).astype(np.float32)     # Eq.(6) 的针孔掩膜
    return obj, probe, S1, rr

def make_probe_init(cfg: Cfg, rr):
    """探针初值 P0。三种算法必须共用这一个函数（addip 那条线的历史教训：
    AD 分支曾自己复制过一份初值构造，导致换初值的开关只对 net 生效）。

    ones : P0 ≡ 1，与论文 run 的中性输出头一致（不给任何光阑先验）
    disk : 平滑圆盘 + 零相位，只编码"针孔大概多大"
    """
    if cfg.probe_init == "ones":
        return np.ones_like(rr, dtype=np.complex64)
    from scipy.ndimage import gaussian_filter
    R = cfg.probe_diam_px / 2.0
    amp = (rr <= R).astype(np.float32)
    s_px = float(cfg.probe_init_sigma) * R
    if s_px > 0:
        amp = gaussian_filter(amp, sigma=s_px)
    P0 = amp.astype(np.complex64)
    return (P0 / max(np.abs(P0).max(), 1e-12)).astype(np.complex64)

def probe_init_err(cfg: Cfg, rr, probe):
    """P0 相对真值探针的复相对误差，消去全局复因子。刻画探针先验强度的唯一数字。"""
    from functions.common.metrics import align_global_factor
    P0 = make_probe_init(cfg, rr)
    Pt = probe.astype(np.complex64)
    return float(np.linalg.norm(align_global_factor(P0, Pt)[0] - Pt)
                 / max(np.linalg.norm(Pt), 1e-12))

def make_positions(cfg: Cfg):
    """整数像素的方形光栅，左上角坐标 (I,2)。步长以 Δx1 为单位。"""
    p = [(cfg.scan_offset + r * cfg.step_px, cfg.scan_offset + c * cfg.step_px)
         for r in range(cfg.grid) for c in range(cfg.grid)]
    return np.asarray(p, dtype=np.int64)

def add_noise(I, kind, snr_db, rng):
    """输入 I 已按【全局】最大值归一到 [0,1]（Table 1 第 3 步）。

    Table 1 的 Poisson 写法含概率质量函数，采样步骤不明确；本实现采用
    I' = Poisson(I·P)/P，P 由 snr_db 控制，并非逐式复刻表中的 λ=Signal*255。
    Gaussian 保留历史设置 σ = mean(signal)/sqrt(10^(SNR/10))，而论文 Table 1
    写的是 sqrt(mean(signal)/10^(SNR/10))；二者不同，snr_db 不应视作经实测
    校准的信噪比。此处保留数值以免改变现有各方法共享的测量数据。
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
