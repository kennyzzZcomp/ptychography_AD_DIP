# -*- coding: utf-8 -*-
"""共用仿真场景 —— paper / ad / net 三种算法【唯一】的数据来源。

存在的理由：把"物体真值、探针真值、扫描位置、前向算子、噪声"关在一个函数里，
任何一种算法都只能通过 build_scene() 拿数据，从结构上杜绝"两条线各自生成、
参数悄悄漂开"这件事。每次运行都打印指纹，三次运行的指纹必须一模一样 ——
这是可验证的，不是靠约定。
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import torch

from functions.paperrepro.evaluate import evaluation_roi, illum_roi
from functions.paperrepro.optics import make_quad_phase
from functions.paperrepro.sample import (make_positions, make_probe_init,
                                         make_truth, probe_init_err)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ProPtyNet_paper import Cfg


def build_scene(cfg: Cfg, device, verbose=True):
    """返回三种算法共用的全部输入。完全由 cfg 决定，与算法无关。"""
    from functions.paperrepro.sample import simulate     # 延迟导入，避免循环

    cfg.quad_sign = getattr(cfg, "quad_sign", -1.0)
    obj, probe, S1, rr = make_truth(cfg)
    pos = make_positions(cfg)
    illum_rs, illum_cs = illum_roi(cfg, probe, pos)
    rs, cs = evaluation_roi(cfg, probe, pos)
    Q = make_quad_phase(cfg, device)
    Im, Icl = simulate(cfg, obj, probe, pos, Q, device)
    P0 = make_probe_init(cfg, rr)

    sc = SimpleNamespace(
        obj=obj, probe=probe, S1=S1, rr=rr, pos=pos, Q=Q,
        Im=Im, Icl=Icl, P0=P0, roi=(rs, cs),
        illum_roi=(illum_rs, illum_cs),
        err_P0=probe_init_err(cfg, rr, probe),
        # torch 侧
        post=torch.from_numpy(pos).to(device),
        Imt=torch.from_numpy(Im).to(device),
        Iclt=torch.from_numpy(Icl).to(device),
        S1t=torch.from_numpy(S1).to(device),
        sqrtIm=torch.from_numpy(np.sqrt(np.maximum(Im, 0.0))).to(device),
    )
    sc.fp = scene_fingerprint(sc)
    if verbose:
        banner(cfg, sc)
    return sc


def scene_fingerprint(sc) -> str:
    """物体/探针/位置/含噪数据/干净数据的联合 SHA1 前 12 位。"""
    h = hashlib.sha1()
    for a in (sc.obj, sc.probe, sc.pos, sc.Im, sc.Icl, sc.P0):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:12]


def banner(cfg: Cfg, sc):
    rs, cs = sc.roi
    irs, ics = sc.illum_roi
    step, dia = cfg.step_px, cfg.probe_diam_px
    illum = (cfg.grid - 1) * step + dia
    meas = cfg.n_pat * cfg.N * cfg.N
    unk = illum ** 2 * 2 + cfg.N * cfg.N * 2
    print("-" * 78)
    print(f"[scene] 指纹 {sc.fp}   <- paper / ad / net 三次运行必须完全相同")
    print(f"[scene] 前向 = Fresnel 单次 FFT (Eq.2)  Δx1 = {cfg.dx1*1e6:.2f} µm  "
          f"quad_sign = {cfg.quad_sign:+.0f}")
    print(f"[scene] 物体 {cfg.obj_size}²  振幅 {cfg.amp_image}  相位 {cfg.phs_image} "
          f"±{cfg.obj_phase_rad:g} rad")
    print(f"[scene] 探针 {dia:.1f} px 针孔 / 探测器 {cfg.N} px  "
          f"-> 过采样 {cfg.N/max(dia,1e-9):.1f}×")
    print(f"[scene] 扫描 {cfg.grid}×{cfg.grid} = {cfg.n_pat} 点  step {step} px  "
          f"-> 线性重叠 {1-step/max(dia,1e-9):.1%}  照明区 ≈ {illum:.0f}²")
    print(f"[scene] 测量数 {meas/1e6:.1f}M / 未知量 {unk/1e3:.0f}k "
          f"-> 冗余度 {meas/max(unk,1):.0f}×")
    print(f"[scene] 噪声 {cfg.noise}"
          + (f" SNR {cfg.snr_db:g} dB seed {cfg.noise_seed}" if cfg.noise != "none" else "")
          + f"   过曝像素 {100*float((sc.Im >= 1.0-1e-6).mean()):.4f}%")
    if getattr(cfg, "eval_size", 0):
        print(f"[scene] 评价 ROI = 固定中心区 {rs.stop-rs.start}×{cs.stop-cs.start} "
              f"[{rs.start}:{rs.stop}, {cs.start}:{cs.stop}]"
              f"  | 本次自适应照明框 {irs.stop-irs.start}×{ics.stop-ics.start}")
    else:
        print(f"[scene] 评价 ROI = 自适应照明覆盖区 {rs.stop-rs.start}×{cs.stop-cs.start}"
              f"（设 --eval-size 可固定）")
    print("[scene] 评价函数 paperrepro.evaluate.evaluate，paper / AD / DIP / ePIE 共用同一口径")
    print(f"[scene] 探针初值 {cfg.probe_init}"
          + (f" sigma={cfg.probe_init_sigma:g}R" if cfg.probe_init == "disk" else "")
          + f"   err_P0 = {sc.err_P0:.4f}")
    print("-" * 78)
