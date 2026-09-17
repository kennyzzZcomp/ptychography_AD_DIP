#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""量化重建误差里的【格点周期伪影】—— raster grid pathology 的判据。

    python diag_periodic.py runs0.1/overlap/adplain_n16_c0/ad_result.npz
    python diag_periodic.py runs0.1/overlap/*/ *_result.npz        # 批量

原理: 对周期扫描, 任意"扫描格点周期"的调制 g 满足 g(r+r_j)=g(r), 于是
      O->O·g, P->P/g 让每一个出射波完全不变 —— 这是前向模型的【严格零空间】。
      表现为误差图里以【扫描步长】为周期的花纹。

判据: 误差图的二维功率谱在频率 k/step (k=1,2,3...) 处应出现峰。
      lattice_ratio = 格点谐波处的功率 / 同半径背景功率。
      >> 1 说明伪影确实锁在扫描步长上 = 零空间, 不是欠训练;
      ≈ 1 说明误差是宽谱的, 那才是优化/噪声问题。
"""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ProPtyNet_torch import Cfg, make_truth, align_global_factor


def lattice_ratio(err, step, n_harm=3, halfwidth=1.0):
    """误差图功率谱里, 格点谐波处的功率 / 同半径圆环背景。"""
    M = err.shape[0]
    e = err - err.mean()
    e = e * np.outer(np.hanning(M), np.hanning(M))       # 去掉边界泄漏
    P = np.abs(np.fft.fftshift(np.fft.fft2(e))) ** 2
    fy, fx = np.mgrid[0:M, 0:M] - M // 2
    rad = np.sqrt(fx ** 2 + fy ** 2)
    out = []
    for k in range(1, n_harm + 1):
        f = k * M / step                                  # 频率 k/step 对应的 bin 半径
        if f > M // 2 - 2:
            break
        # 格点点: (±f,0) (0,±f) —— 方形光栅的一阶方向
        peak = max(P[M // 2 + int(round(dy)), M // 2 + int(round(dx))]
                   for dx, dy in ((f, 0), (-f, 0), (0, f), (0, -f)))
        ring = P[(rad > f - 3) & (rad < f + 3)]
        bg = np.median(ring) if ring.size else np.nan
        out.append((k, f, peak / bg if bg and np.isfinite(bg) else np.nan))
    return out


def one(path):
    d = np.load(path, allow_pickle=True)
    cfg_d = json.loads(str(d["cfg"]))
    fields = {f for f in Cfg.__dataclass_fields__}
    cfg = Cfg(**{k: v for k, v in cfg_d.items() if k in fields})
    obj, probe, support, _, rr, _R = make_truth(cfg)
    rec = d["obj_rec"]
    c = cfg.EVAL_CROP
    ra, _ = align_global_factor(rec[c:-c, c:-c], obj[c:-c, c:-c])
    ga = obj[c:-c, c:-c]
    dphi = np.angle(np.exp(1j * (np.angle(ra) - np.angle(ga))))
    damp = np.abs(ra) - np.abs(ga)
    step = float(cfg.scan_step)
    print(f"\n{path}")
    print(f"  n={cfg.scan_npos}  step={step:g}px  jitter={cfg.scan_jitter:g}  "
          f"phase_span={cfg.obj_phase_span:g}  probe_support={getattr(cfg,'probe_support','hard')}")
    print(f"  相位误差 RMS {np.sqrt((dphi**2).mean()):.4f} rad "
          f"(= 量程 ±{cfg.obj_phase_span:g} 的 {np.sqrt((dphi**2).mean())/cfg.obj_phase_span*100:.0f}%)"
          f"   振幅误差 RMS {np.sqrt((damp**2).mean()):.4f}")
    for name, err in (("相位", dphi), ("振幅", damp)):
        r = lattice_ratio(err, step)
        s = "  ".join(f"k={k}(f={f:.1f}px⁻¹): {v:6.1f}×" for k, f, v in r)
        print(f"  {name}误差在格点谐波处的功率/背景:  {s}")
    return dphi, damp


if __name__ == "__main__":
    args = sys.argv[1:]
    paths = []
    for a in args:
        paths += sorted(glob.glob(a)) if any(ch in a for ch in "*?[") else [a]
    if not paths:
        print(__doc__); sys.exit(1)
    for p in paths:
        if p.endswith(".npz"):
            try:
                one(p)
            except Exception as e:
                print(f"  跳过 {p}: {type(e).__name__} {e}")
    print("\n判读: lattice_ratio >> 1（比如 >5×）= 伪影锁在扫描步长上 -> 零空间/病态性，")
    print("      调 lr 或加步数都没用；≈1 = 宽谱误差 -> 才是优化问题。")
