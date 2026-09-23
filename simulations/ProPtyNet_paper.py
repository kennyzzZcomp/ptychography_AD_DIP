#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ProPtyNet_paper.py —— 严格按论文复现

Z. Liu, Y. Chen, N. Lin, "Noise-robust ptychography using unsupervised neural
network", Optics and Lasers in Engineering 186 (2025) 108791.

【与 ProPtyNet_torch.py 的关系】那一份是为了跟 INNM_Ptycho.ipynb 的 AD 基线做
可比对照，沿用了 INNM 的角谱前向和 INNM 的振幅域损失；本文件不管可比性，只按论文:

    前向    Eq.(2)  Fresnel 单次 FFT,  Δx1·Δx2 = λz/M       (那份是角谱 Δx1=Δx2)
    输出    Fig.1   S = amp_s·exp(jπ·phs_s), P = amp_p·exp(jπ·phs_p)   相位跨度 ±π
            振幅 Conv2d+LeakyReLU / 相位 Conv2d+tanh —— 这是原版基线，不要改
    损失    Eq.(4)(5)  β·Loss1 + (1-β)·Loss2, 双掩膜 S1/S2 + γ 衰减
    探针    只有 Loss2 的软约束, 【没有】二值 support 硬掩膜
    噪声    Table 1  全局归一化 -> 加噪 -> clip[0,1] (clip 产生过曝, 才有 S2)

用法（四种模式共用同一份仿真数据，见 functions/paperrepro/scene.py）:
    python ProPtyNet_paper.py check                       # 采样/几何自检
    python ProPtyNet_paper.py ad   --preset paper --iters 2000   # 纯 AD 对照
    python ProPtyNet_paper.py net  --preset paper --iters 2000   # DIP 对照
    python ProPtyNet_paper.py check  --preset smoke
    python ProPtyNet_paper.py run    --preset smoke --iters 400        # CPU 冒烟
    python ProPtyNet_paper.py run    --preset paper --iters 2000       # 需要 GPU
    python ProPtyNet_paper.py run    --preset paper --noise mixed --snr 30

资源: 默认 USAF.jpg (振幅) + 内置合成辐条靶 (相位)，对应论文 Fig.2(a)。
      换回自然图像做串扰诊断: --amp-image cameraman.bmp --phs-image westconcordorthophoto.bmp
"""

from __future__ import annotations

import os
import sys
import math
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 让 functions/ 可被 import

from functions.paperrepro.solvers import run_check, run
from functions.paperrepro.solvers_addip import run_ad, run_net

PI = math.pi


# ============================================================================ #
# 配置 —— 本文件【只放配置与 CLI】，所有实现在 functions/paperrepro/ 与 functions/common/
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
    # 物体的振幅图 / 相位图。论文: "synthesized from two kinds of resolution test targets"
    #   USAF.jpg  -> 与论文 Fig.2(a) 的 Object Amplitude 同款 USAF 1951 靶
    #   "siemens" -> 内置合成的辐条靶，对应论文 Fig.2(a) 的 Object Phase（不需要额外素材）
    #   也可以填任意文件名，例如换回自然图像做串扰诊断：
    #     --amp-image cameraman.bmp --phs-image westconcordorthophoto.bmp
    amp_image: str = "USAF.jpg"
    phs_image: str = "siemens"
    obj_phase_rad: float = 0.8   # 物体相位幅度 (±rad)。论文没规定样品相位多大，
                                 # 取 0.8 rad 与 INNM_Ptycho.ipynb / ProPtyNet_torch.py 一致，
                                 # 这样两条线的物体是同一个，指标才可比。
                                 # (曾误写成 0.8*PI = 2.513 rad，把全局相位歧义的余量压到 20%)

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
    # 论文 Fig.1: S = amp_s·exp(jπ·phs_s), P = amp_p·exp(jπ·phs_p)
    # 两个都默认 π = 论文原版基线（振幅 Conv2d+LeakyReLU，相位 Conv2d+tanh，输出头不动）。
    # 拆成两个只是为了做消融时能单独放宽样品那一路（论文正文建议样品放到 2π）。
    phase_span_obj: float = PI
    phase_span_prb: float = PI

    # ---- 优化 ----
    iters: int = 2000
    lr: float = 5e-4             # 论文: 5e-4 ~ 5e-3
    lr_final_frac: float = 0.1   # cosine 衰减到 lr 的这个比例; 1.0 = 不衰减
    pos_batch: int = 0           # 0 = 全 batch（论文写法）; >0 = 每步随机取这么多位置

    # ---- ad / net 两条对照算法（与 run 共用同一份仿真数据，见 paperrepro/scene.py）----
    #   ad  : 物体 = 自由复数像素，振幅域损失
    #   net : 物体 = 未训练 U-Net（softplus 振幅 + cos/sin 相位），振幅域损失
    # 三者的迭代数都用上面的 iters，探针初值都用下面这两项。
    # 【默认 disk】ones 时 err_P0 ≈ 0.996（与真值几乎不相关），而论文那一路另有
    # Eq.(5) Loss2 把探针按回针孔里 —— 给 ad/net 用 ones 等于让它们既没初值也没约束，
    # 实测直接跑不出来。disk 只编码"针孔多大"，与 Loss2 的先验强度大致对等。
    probe_init: str = "ones"     # disk = 平滑圆盘 + 零相位 | ones = P0 ≡ 1
    probe_init_sigma: float = 0.15   # 仅 disk 用，单位 = 针孔半径的倍数
    # 探针参数化（只对 ad / net 生效；run 的探针是网络输出的，改不了）
    #   pixel   自由复数像素，2·N² = 524k 个未知量，其中只有针孔内那 ~5.5k 被数据定住
    #   support 只在针孔内参数化，外面【恒等于 0】。未知量 524k -> 5.5k，直接消掉零空间
    #   truth   冻结在真值上（非盲上界诊断：它也崩 = 数据本身不够，与探针无关）
    probe_mode: str = "pixel"
    # support 档的掩膜半径 = margin × 针孔半径。1.2 允许一圈衍射光晕，更接近真实光路。
    # 与论文 Loss2 的 s1_margin 是两回事，互不影响。
    probe_support_margin: float = 1.0
    obj_init_alpha: float = 0.0  # net: 0 = 严格相位中性，第 0 步 O ≡ 1，与 ad 同初值
    lr_obj: float = 3e-2         # ad  物体自由像素（1e-2 在本几何下明显偏小）
    lr_prb: float = 3e-2         # ad  探针自由像素
    lr_net: float = 1e-3         # net U-Net
    lr_probe: float = 1e-2       # net 探针自由像素
    lr_cosine: bool = False      # net 两个 lr 一起余弦退火到 0
    weight_decay: float = 0.0    # net DIP 不该有权重衰减
    fwd_chunk: int = 0           # 前向分块，0 = 全批量；显存不够时设 10 / 20

    # ad/net amplitude TGV; phase TGV remains net only.
    # L = amplitude-MSE + mean(I) * (tgv_amp * R_amp + tgv_phase * R_phase).
    # The regularization domain is the nominal illuminated disk union, not eval ROI.
    tgv_amp: float = 0.0
    tgv_amp_schedule: str = ""  # 1000:0.01 => update 1001 begins with lambda=.01
    tgv_phase: float = 0.0       # independent wrapped-gradient phase TGV (radians)
    tgv_alpha0: float = 2.0      # symmetric-gradient weight
    tgv_alpha1: float = 1.0      # gradient-minus-vector weight
    tgv_eps: float = 1e-3        # smoothed vector norm
    tgv_inner_steps: int = 5    # warm-started auxiliary-vector updates per net step
    tgv_lr: float = 1e-2
    timing_warmup: int = -1     # net only: -1 off; >=0 synchronized stage timings
    measurement_schedule: str = ""     # empty: legacy; explicit 0:1: timed full baseline
    measurement_policy: str = "fixed"  # fixed | random | rotate; input stays full
    measurement_seed: int = 0
    input_policy: str = "full"  # full | follow_measurements (gathered first conv)

    # ---- 其它 ----
    scale_cal: bool = True       # 冻结的幅度标定（论文没写，见下方说明）
    seed: int = 0
    device: str = "auto"
    assets: str = ""
    outdir: str = "results_paper"
    eval_every: int = 25
    # 0 = 根据每次照明覆盖自适应；>0 = 固定画布中心方形评价区。
    # overlap sweep 要横向比较 SSIM/PSNR 时应给所有 run 传同一个值。
    eval_size: int = 0
    def __post_init__(self):
        from functions.paperrepro.tgv_schedule import parse_tgv_schedule
        parse_tgv_schedule(self.tgv_amp_schedule, self.tgv_amp, self.iters)
        if self.input_policy not in ("full", "follow_measurements"):
            raise ValueError("input_policy must be full or follow_measurements")
        if self.input_policy == "follow_measurements" and not self.measurement_schedule:
            raise ValueError("follow_measurements requires explicit measurement_schedule")
        from functions.paperrepro.sampling import MeasurementSchedule
        MeasurementSchedule(self.measurement_schedule, self.grid, self.iters,
                            self.measurement_policy, self.measurement_seed)
        if self.timing_warmup < -1:
            raise ValueError("timing_warmup must be -1 (disabled) or >= 0")
        if not math.isfinite(self.tgv_amp) or self.tgv_amp < 0:
            raise ValueError("tgv_amp must be finite and >= 0")
        if not math.isfinite(self.tgv_phase) or self.tgv_phase < 0:
            raise ValueError("tgv_phase must be finite and >= 0")
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.tgv_alpha0, self.tgv_alpha1, self.tgv_eps, self.tgv_lr)):
            raise ValueError("TGV alpha0, alpha1, eps and lr must be finite and > 0")
        if self.tgv_inner_steps < 1:
            raise ValueError("tgv_inner_steps must be >= 1")
        # 采样关系 Eq.(2) 下面那条: Δx1·Δx2 = λz/M
        self.dx1 = self.wlength * self.z / (self.N * self.det_pixel)
        self.probe_diam_px = self.probe_diam_um * 1e-6 / self.dx1
        self.scan_span = self.N + (self.grid - 1) * self.step_px
        if self.grid < 1 or self.step_px < 0:
            raise ValueError(f"grid and step_px must be non-negative/positive: {self.grid=}, {self.step_px=}")
        if self.scan_span > self.obj_size:
            raise ValueError(
                f"scan does not fit object canvas: N + (grid-1)*step_px = "
                f"{self.scan_span} > obj_size={self.obj_size}. "
                "Lower grid/step_px or increase --obj-size for every compared run."
            )
        if self.eval_size and not 7 <= self.eval_size <= self.obj_size:
            raise ValueError(
                f"eval_size must be 0 or within [7, obj_size]; got {self.eval_size}"
            )
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


# ============================================================================ #
# CLI
# ============================================================================ #

def main():
    ap = argparse.ArgumentParser(description="ProPtyNet 严格复现 (Opt. Lasers Eng. 186 (2025) 108791)")
    ap.add_argument("mode", choices=["check", "run", "ad", "net"],
                    help="check=自检 | run=论文原版 ProPtyNet | ad=纯 AD | net=DIP")
    ap.add_argument("--preset", choices=list(PRESETS))
    for k, t in [("N", int), ("obj_size", int), ("z", float), ("det_pixel", float),
                 ("grid", int), ("step_px", int), ("probe_diam_um", float),
                 ("iters", int), ("lr", float), ("lr_final_frac", float),
                 ("base_ch", int), ("beta", float), ("gamma0", float),
                 ("gamma_end", float), ("s1_margin", float), ("obj_phase_rad", float),
                 ("amp_image", str), ("phs_image", str),
                 ("phase_span_obj", float), ("phase_span_prb", float),
                 ("snr_db", float), ("pos_batch", int), ("eval_every", int),
                 ("eval_size", int),
                 ("seed", int), ("device", str), ("outdir", str), ("assets", str),
                 ("probe_init", str), ("probe_init_sigma", float), ("probe_mode", str), ("probe_support_margin", float),
                 ("obj_init_alpha", float), ("lr_obj", float), ("lr_prb", float),
                 ("lr_net", float), ("lr_probe", float), ("weight_decay", float),
                 ("tgv_amp", float), ("tgv_phase", float), ("tgv_alpha0", float), ("tgv_alpha1", float),
                 ("tgv_amp_schedule", str),
                 ("tgv_eps", float), ("tgv_inner_steps", int), ("tgv_lr", float),
                 ("timing_warmup", int),
                 ("measurement_schedule", str), ("measurement_policy", str),
                 ("measurement_seed", int),
                 ("input_policy", str),
                 ("fwd_chunk", int), ("noise_seed", int)]:
        ap.add_argument("--" + k.replace("_", "-"), dest=k, type=t)
    ap.add_argument("--noise", choices=["none", "gaussian", "poisson", "mixed"])
    ap.add_argument("--snr", dest="snr_db", type=float)
    ap.add_argument("--quad-sign", dest="quad_sign", type=float, choices=[-1.0, 1.0])
    ap.add_argument("--no-scale-cal", dest="scale_cal", action="store_false", default=None)
    ap.add_argument("--lr-cosine", dest="lr_cosine", action="store_true", default=None)
    a = ap.parse_args()

    cfg = build_cfg(a)
    if cfg.tgv_amp_schedule and a.mode != "net":
        ap.error("--tgv-amp-schedule is implemented only for mode net")
    if cfg.measurement_schedule and a.mode != "net":
        ap.error("--measurement-schedule is implemented only for mode net")
    if cfg.timing_warmup >= 0 and a.mode != "net":
        ap.error("--timing-warmup is implemented only for mode net")
    if cfg.tgv_amp > 0 and a.mode not in ("ad", "net"):
        ap.error("--tgv-amp is implemented only for modes ad and net")
    if cfg.tgv_phase > 0 and a.mode != "net":
        ap.error("--tgv-phase is implemented only for mode net")
    cfg.quad_sign = a.quad_sign if a.quad_sign is not None else -1.0
    os.makedirs(cfg.outdir, exist_ok=True)
    {"check": run_check, "run": run, "ad": run_ad, "net": run_net}[a.mode](cfg)


if __name__ == "__main__":
    main()
