# -*- coding: utf-8 -*-
"""结果保存、出图与设备播报。"""

from __future__ import annotations

import os
import json
from dataclasses import asdict
import numpy as np
import torch

from functions.common.metrics import align_global_factor

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_paper import Cfg

def _save(cfg, rec, pc, obj, probe, hist, roi, positions, tag="paper"):
    """全量存盘 + 三个尺度的对照图。

    rec / obj 是【完整 612² 画布】，roi 是照明覆盖区。npz 存全量，图上三行分别是
    全画布 / 照明 ROI / 探针放大 —— 只看 ROI 会漏掉画布外围的发散和扫描 footprint。
    """
    rs, cs = roi
    os.makedirs(cfg.outdir, exist_ok=True)
    np.savez_compressed(os.path.join(cfg.outdir, f"{tag}_result.npz"),
                        obj_rec=rec, obj_gt=obj, probe_rec=pc, probe_gt=probe,
                        roi=np.array([rs.start, rs.stop, cs.start, cs.stop]),
                        positions=positions, hist=json.dumps(hist),
                        cfg=json.dumps(asdict(cfg), default=str))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # 全局复因子必须在 ROI 上定：画布外围没有数据约束，拿它定因子会把结果整个带偏
    c = np.vdot(rec[rs, cs], obj[rs, cs]) / max(np.vdot(rec[rs, cs], rec[rs, cs]).real, 1e-30)
    ra = rec * c
    pa = align_global_factor(pc, probe)[0]
    # 探针放大框
    n = cfg.N
    h = int(max(cfg.probe_diam_px, 20))
    ps = slice(max(n // 2 - h, 0), min(n // 2 + h, n))

    fig, ax = plt.subplots(4, 4, figsize=(16, 16.5))
    rows = [
        ("full canvas %d²" % cfg.obj_size,
         [(np.abs(ra), "rec amp"), (np.angle(ra), "rec phase"),
          (np.abs(obj), "GT amp"), (np.angle(obj), "GT phase")]),
        ("illuminated ROI %d×%d" % (rs.stop - rs.start, cs.stop - cs.start),
         [(np.abs(ra[rs, cs]), "rec amp"), (np.angle(ra[rs, cs]), "rec phase"),
          (np.abs(obj[rs, cs]), "GT amp"), (np.angle(obj[rs, cs]), "GT phase")]),
        ("probe (zoom)",
         [(np.abs(pa[ps, ps]), "rec probe amp"), (np.angle(pa[ps, ps]), "rec probe phase"),
          (np.abs(probe[ps, ps]), "GT probe amp"), (np.angle(probe[ps, ps]), "GT probe phase")]),
    ]
    for r, (tag, items) in enumerate(rows):
        for k, (im, t) in enumerate(items):
            a = ax[r, k]
            a.imshow(im, cmap="gray")
            a.set_title(f"{t}\n[{tag}]" if k == 0 else t, fontsize=9)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:      # 在全画布上标出 ROI
                a.add_patch(plt.Rectangle((cs.start, rs.start), cs.stop - cs.start,
                                          rs.stop - rs.start, fill=False,
                                          ec="red", lw=1.2))

    # 第 4 行: 照明覆盖 / 振幅残差 / 两条曲线
    cov = np.zeros_like(obj, dtype=np.float64)
    w = np.abs(probe) ** 2
    for (py, px) in positions:
        cov[py:py + cfg.N, px:px + cfg.N] += w
    ax[3, 0].imshow(cov, cmap="magma"); ax[3, 0].set_title("illumination coverage", fontsize=9)
    ax[3, 0].add_patch(plt.Rectangle((cs.start, rs.start), cs.stop - cs.start,
                                     rs.stop - rs.start, fill=False, ec="cyan", lw=1.2))
    d = np.abs(np.abs(ra[rs, cs]) - np.abs(obj[rs, cs]))
    ax[3, 1].imshow(d, cmap="inferno")
    ax[3, 1].set_title(f"|amp error| (ROI), max {d.max():.2f}", fontsize=9)
    for a in ax[3, :2]:
        a.set_xticks([]); a.set_yticks([])
    if hist:
        it = [h["it"] for h in hist]
        ax[3, 2].plot(it, [h["ssim_amp"] for h in hist], label="amp SSIM")
        ax[3, 2].plot(it, [h["ssim_phs"] for h in hist], label="phase SSIM")
        ax[3, 2].plot(it, [h["relerr"] for h in hist], label="relerr")
        ax[3, 2].set_xlabel("iteration"); ax[3, 2].legend(fontsize=8)
        ax[3, 2].set_title("object metrics", fontsize=9); ax[3, 2].grid(alpha=.3)
        ax[3, 3].semilogy(it, [h["loss"] for h in hist], label="loss")
        ax[3, 3].semilogy(it, [h["real"] for h in hist], "--", label="real error")
        ax[3, 3].set_xlabel("iteration"); ax[3, 3].legend(fontsize=8)
        ax[3, 3].set_title("convergence", fontsize=9); ax[3, 3].grid(alpha=.3)
    else:
        ax[3, 2].axis("off"); ax[3, 3].axis("off")

    fig.tight_layout()
    f = os.path.join(cfg.outdir, f"{tag}_result.png")
    fig.savefig(f, dpi=130); plt.close(fig)
    print(f"[{tag}] 结果 -> {f}   (npz 里存的是【全画布】未裁剪的 obj_rec/obj_gt)")

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
