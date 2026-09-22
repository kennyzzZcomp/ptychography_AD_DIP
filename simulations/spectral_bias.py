#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""谱偏置判据：未训练 U-Net 拟合"真值物体" vs "栅格规范场调制后的假解"。

【为什么要做这个】
规则栅格扫描（步长 p）下存在一个精确的规范自由度：对任意以 p 为周期的复函数 g，
    O -> O·g ,  P -> P/g
出射波逐点不变，任何数据项都定不住它。纯 AD 会掉进去（实测 loss 2.16e-16 而
relerr 0.2403），未训练网络先验不会（同数据同探针参数化 relerr 0.0144）。

本脚本直接检验那个"不会"的机制：**U-Net 拟合规范场 g 比拟合真值物体困难得多。**
同一个网络、同一个输入、同一套优化器、同样的中性初始化，只换拟合目标：

    object   O_gt                     真值物体（自然图像）
    false    O_gt · g                 假解 = 真值 × 栅格规范场
    gauge    g                        纯规范场
    smooth   O_gt · g_smooth          对照：同样 RMS 相位、但是低频平滑的调制
                                      （用来排除"只是因为多了扰动"）

g 直接从跑出来的 AD 结果里反解：g = O_rec/O_gt，再按格点折叠成一个 p×p 元胞后平铺
（实测折叠后与原 g 的复相关 0.9996，也就是说整个假解就是 p² 个复数）。

用法:
    python spectral_bias.py --from-npz ov60_ad_sup/ad_result.npz --iters 1500
"""

from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ProPtyNet_paper import Cfg, PRESETS
from functions.addip.model import ProPtyUNet as AddipUNet, make_field
from functions.paperrepro.scene import build_scene

TARGETS = ["object", "false", "gauge", "smooth"]


def _agf(rec, gt):
    c = np.vdot(rec, gt) / max(np.vdot(rec, rec).real, 1e-30)
    return rec * c


def gauge_cell_from_npz(path):
    """从 AD 的重建结果反解规范场，折叠成 p×p 元胞。返回 (cell, p, 折叠相关系数)。"""
    d = np.load(path, allow_pickle=True)
    cfg = json.loads(str(d["cfg"]))
    p = int(cfg["step_px"])
    r = d["roi"]
    rs, cs = slice(r[0], r[1]), slice(r[2], r[3])
    gt = d["obj_gt"][rs, cs]
    g = _agf(d["obj_rec"][rs, cs], gt) / gt
    g = g / np.median(np.abs(g))
    H, W = g.shape
    cell = np.zeros((p, p), complex); cnt = np.zeros((p, p))
    for i in range(H):
        for j in range(W):
            cell[(i + r[0]) % p, (j + r[2]) % p] += g[i, j]
            cnt[(i + r[0]) % p, (j + r[2]) % p] += 1
    cell /= np.maximum(cnt, 1)
    tile = cell[(np.arange(H)[:, None] + r[0]) % p, (np.arange(W)[None, :] + r[2]) % p]
    corr = abs(np.vdot(g, tile)) / math.sqrt(np.vdot(g, g).real * np.vdot(tile, tile).real)
    return cell, p, float(corr)


def tile_cell(cell, M, p):
    idx = np.arange(M) % p
    return cell[idx[:, None], idx[None, :]].astype(np.complex64)


def smooth_control(M, rms_phase, seed=0):
    """低频平滑的复调制，相位 RMS 与规范场对齐。用来排除'任何扰动都难拟合'。"""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((M, M))
    fx = np.fft.fftfreq(M)[:, None]; fy = np.fft.fftfreq(M)[None, :]
    k = np.exp(-2 * (math.pi * (M / 24.0)) ** 2 * (fx ** 2 + fy ** 2))  # 截止远低于格点频率
    f = np.real(np.fft.ifft2(np.fft.fft2(f) * k))
    f = f / max(f.std(), 1e-12) * rms_phase
    return np.exp(1j * f).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser(description="U-Net 谱偏置判据")
    ap.add_argument("--from-npz", required=True, help="AD 的重建结果 npz，用来反解规范场")
    ap.add_argument("--preset", default="paper", choices=list(PRESETS))
    ap.add_argument("--step-px", type=int, default=None, help="默认沿用 npz 里的值")
    ap.add_argument("--grid", type=int, default=None)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--outdir", default="spectral_bias")
    ap.add_argument("--targets", default=",".join(TARGETS))
    a = ap.parse_args()

    cell, p, corr = gauge_cell_from_npz(a.from_npz)
    src = json.loads(str(np.load(a.from_npz, allow_pickle=True)["cfg"]))

    kw = dict(PRESETS[a.preset]); kw["preset"] = a.preset
    kw["step_px"] = a.step_px if a.step_px is not None else int(src["step_px"])
    kw["grid"] = a.grid if a.grid is not None else int(src["grid"])
    cfg = Cfg(**kw); cfg.quad_sign = -1.0; cfg.seed = a.seed
    cfg.outdir = a.outdir
    dev = cfg.dev()
    M, NS = cfg.obj_size, cfg.net_size
    os.makedirs(a.outdir, exist_ok=True)

    print("=" * 78)
    print(f"规范场来自 {a.from_npz}  步长 p = {p} px")
    print(f"  折叠成 {p}×{p} 元胞后平铺，与原 g 的复相关 = {corr:.6f}"
          f"  -> 整个假解就是 {p*p} 个复数")
    ph = np.angle(cell * np.conj(cell.mean()))
    print(f"  元胞相位 RMS {ph.std():.4f} rad   振幅范围 "
          f"[{np.abs(cell).min():.4f}, {np.abs(cell).max():.4f}]")
    print("=" * 78)

    sc = build_scene(cfg, dev, verbose=False)
    Ogt = torch.from_numpy(sc.obj.astype(np.complex64)).to(dev)
    G = torch.from_numpy(tile_cell(cell, M, p)).to(dev)
    S = torch.from_numpy(smooth_control(M, float(ph.std()), a.seed)).to(dev)
    tgts = {"object": Ogt, "false": Ogt * G, "gauge": G, "smooth": Ogt * S}

    # 网络输入：与真实重建完全一致的衍射图堆栈
    pad_o, pad_n = (M - cfg.N) // 2, (NS - M) // 2
    x = F.pad(sc.Imt[None], (pad_o,) * 4)
    x = F.pad(x, (pad_n, NS - M - pad_n, pad_n, NS - M - pad_n))
    x = x / x.amax(dim=(2, 3), keepdim=True).clamp_min(1e-12)
    c_ns = slice(pad_n, pad_n + M)

    hist = {}
    for name in a.targets.split(","):
        name = name.strip()
        if name not in tgts:
            raise ValueError(f"未知目标 {name}，可选 {TARGETS}")
        T = tgts[name]; Tn = torch.linalg.vector_norm(T)
        torch.manual_seed(a.seed)                      # 每个目标同一个网络初值
        net = AddipUNet(cfg.n_pat, cfg.base_ch, n_fields=1, ph_ch=2).to(dev)
        with torch.no_grad():                          # 与重建同一套中性初始化
            net.head_amp.weight[0].zero_(); net.head_phs.weight[0:2].zero_()
            net.head_amp.bias[0] = math.log(math.e - 1.0)
            net.head_phs.bias[0:2] = 0.0; net.head_phs.bias[0] = 1.0
        opt = torch.optim.Adam(net.parameters(), lr=a.lr)
        with torch.no_grad():          # 起点误差：中性初值 O ≡ 1 到该目标的距离
            ar, pr = net(x)
            e0 = (torch.linalg.vector_norm(make_field(ar[0], pr[0:2])[c_ns, c_ns] - T)
                  / Tn).item()
        print(f"  [{name:7s}] 起点相对误差 {e0:.5f}（中性初值 O ≡ 1 到该目标的距离）")
        rec, t0 = [], time.time()
        for it in range(a.iters):
            ar, pr = net(x)
            O = make_field(ar[0], pr[0:2])[c_ns, c_ns]
            loss = ((O - T).abs() ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
                with torch.no_grad():
                    e = (torch.linalg.vector_norm(O - T) / Tn).item()
                rec.append({"it": it + 1, "relerr": e, "relerr_norm": e / max(e0, 1e-12),
                            "loss": loss.item()})
                if (it + 1) % (a.eval_every * 8) == 0 or it == a.iters - 1:
                    print(f"  [{name:7s}] it {it+1:5d} | 相对误差 {e:.5f} "
                          f"| 相对起点 {e/max(e0,1e-12):.4f}", flush=True)
        hist[name] = rec
        hist[name + "_e0"] = e0
        print(f"  [{name:7s}] 末值 {rec[-1]['relerr']:.5f}  "
              f"降到起点的 {rec[-1]['relerr']/max(e0,1e-12):.4f}   用时 {time.time()-t0:.1f}s")

    out = os.path.join(a.outdir, "spectral_bias.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"cell_real": cell.real.tolist(), "cell_imag": cell.imag.tolist(),
                   "p": p, "fold_corr": corr, "hist": hist,
                   "cfg": {k: str(v) for k, v in vars(a).items()}}, fh)
    print(f"\n结果 -> {out}")

    print("\n" + "=" * 78)
    print(f"{'目标':10s} {'起点误差':>10s} {'末步误差':>10s} {'降到起点的':>11s} "
          f"{'vs object':>11s}")
    print("-" * 78)
    ks = [t for t in TARGETS if t in hist]
    norm = {k: hist[k][-1]["relerr"] / max(hist[k + "_e0"], 1e-12) for k in ks}
    base = norm.get("object", float("nan"))
    for k in ks:
        print(f"{k:10s} {hist[k+'_e0']:10.5f} {hist[k][-1]['relerr']:10.5f} "
              f"{norm[k]:11.4f} {norm[k]/max(base,1e-12):10.2f}×")
    print("=" * 78)
    print("判据（看最后一列，已按各自起点归一）：")
    print("  false / gauge 的倍数 >> 1 而 smooth ≈ 1")
    print("  -> 未训练网络对【栅格周期调制】有选择性的排斥，不是对任何扰动都排斥。")


if __name__ == "__main__":
    main()
