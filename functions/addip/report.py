# -*- coding: utf-8 -*-
"""结果保存与出图。"""

from __future__ import annotations

import os
import json
import math
from dataclasses import asdict
import numpy as np

from functions.common.metrics import align_global_factor

PI = math.pi

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_torch import Cfg

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

def _banner(cfg: Cfg, err_P0: float, tag: str):
    print(f"[{tag}] 探针初值 P0 = 平滑圆盘(sigma={cfg.probe_init_sigma:g}R) + 零相位，"
          f"相对真值复误差 err_P0 = {err_P0:.4f}")
