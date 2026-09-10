# -*- coding: utf-8 -*-
"""
sweep_noise.py —— 噪声扫描对照驱动脚本

不改 ProPtyNet_torch.py，只是反复调用它，然后把结果汇总成表和图。

对照的两条腿:
    AD        : 标准 AD ptychography  = ad --opt-mode plain
                （一个 Adam 全程不重建 / 全批量 / 物体探针联合更新 / 无 lr 衰减 /
                  无正则 / 无 stage）—— 这是教科书基线，也是本脚本【唯一】会跑的 AD。
    ProPtyNet : net --probe-mode pixel
                （物体走未训练 U-Net，探针走自由像素）

【本脚本永远不会跑 alternating / joint】。那两个是 INNM 的 stage 结构，不是
传统 AD ptychography；而且它们的参数更新次数是 stages×(obj+prb)epoch×位置数，
跟 net 的全批量步数根本不在一个刻度上，拿来当基线是错的。要跑请自己手动调
ProPtyNet_torch.py。

步数对齐: AD 拿 --ad-iters，net 拿 --iters，两个都等于本脚本的 --iters，
          所以两边的梯度步数与数据遍历次数一致。

用法:
    # 先看计划，不真跑
    python sweep_noise.py --dry-run

    # 泊松剂量轴（默认）: 无噪 / 5000 / 1000 / 300 / 100 光子
    python sweep_noise.py --iters 2000

    # 高斯 SNR 轴
    python sweep_noise.py --axis gauss --levels 40,30,25,20 --iters 2000

    # 加上论文的 clip 过曝
    python sweep_noise.py --noise-clip

    # 跑完只重新汇总（不重跑）
    python sweep_noise.py --collect-only

断点续跑: 某个组合的 npz 已经存在就跳过，加 --force 才重跑。
"""

from __future__ import annotations

import os
import sys
import csv
import json
import time
import argparse
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# 从 npz 的 hist 最后一条里取这些指标。键名来自 ProPtyNet_torch.py 的
# evaluate_object_roi() / evaluate_probe()，别在这里瞎改。
# (npz 里的键, 表格用的中文名, 越大越好还是越小越好, 图里用的英文名)
# 图用英文是因为 matplotlib 默认字体没有中文，会变成一堆方块。
METRICS = [
    ("ssim_o_amp",       "物体振幅 SSIM", "high", "object amp SSIM"),
    ("psnr_o_amp",       "物体振幅 PSNR", "high", "object amp PSNR (dB)"),
    ("ssim_o_phi",       "物体相位 SSIM", "high", "object phase SSIM"),
    ("rmse_o_phi_rad",   "物体相位 RMSE", "low",  "object phase RMSE (rad)"),
    ("relerr_o_complex", "物体复场误差",  "low",  "object complex rel. err"),
    ("relerr_p_complex", "探针复场误差",  "low",  "probe complex rel. err"),
]

METHODS = {
    # 名字 -> (mode, 该方法专属参数)   mode 决定 npz 叫 ad_result.npz 还是 net_result.npz
    # --opt-mode plain 写死在这里，不做成开关，就是为了防止手滑跑成 alternating。
    "AD":         ("ad",  ["--opt-mode", "plain"]),
    "ProPtyNet":  ("net", ["--probe-mode", "pixel"]),
}


# --------------------------------------------------------------------------- #
# 单次运行
# --------------------------------------------------------------------------- #

def noise_args(axis: str, level, clip: bool):
    """把一个噪声档位翻译成 ProPtyNet_torch.py 的命令行参数。"""
    a = []
    if level is not None:                     # None = 无噪声基准
        if axis == "poisson":
            a += ["--poisson", "--peak-photons", str(level)]
        elif axis == "gauss":
            a += ["--gauss-snr-db", str(level)]
        else:
            raise ValueError(f"未知噪声轴 {axis}")
    if clip:
        a += ["--noise-clip"]
    return a


def level_tag(axis: str, level):
    if level is None:
        return "clean"
    return f"{axis}{level:g}"


def run_one(script, method, axis, level, root, common, iters, clip, force, dry):
    mode, extra = METHODS[method]
    # 步数旗标两边不同名: AD 用 --ad-iters（plain 分支读它），net 用 --iters
    extra = extra + (["--ad-iters", str(iters)] if mode == "ad" else ["--iters", str(iters)])
    outdir = Path(root) / f"{method}_{level_tag(axis, level)}"
    npz = outdir / f"{mode}_result.npz"

    if npz.is_file() and not force:
        print(f"  [跳过] {npz} 已存在（--force 可重跑）")
        return npz

    cmd = ([sys.executable, str(script), mode]
           + noise_args(axis, level, clip)
           + ["--outdir", str(outdir)] + extra + common)

    if dry:
        print("  " + " ".join(cmd))
        return None

    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"  [运行] {method} / {level_tag(axis, level)}")
    with open(outdir / "log.txt", "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    if p.returncode != 0:
        print(f"  [失败] 返回码 {p.returncode}，看 {outdir/'log.txt'}")
        return None
    print(f"  [完成] {dt:.0f}s -> {npz}")
    return npz


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #

def read_result(npz: Path):
    """读 npz 的 hist 最后一条。ad 的 alternating 分支没有 'it' 键，所以用 .get()。"""
    if not npz.is_file():
        return None
    d = np.load(npz, allow_pickle=True)
    try:
        hist = json.loads(str(d["hist"]))
    except Exception:
        return None
    if not hist:
        return None
    last = hist[-1]
    row = {k: last.get(k, float("nan")) for k, _, _, _ in METRICS}
    row["it"] = last.get("it", "")
    row["loss"] = last.get("loss", float("nan"))
    return row


def collect(root, axis, levels, methods):
    rows = []
    for lv in levels:
        for m in methods:
            mode, _ = METHODS[m]
            npz = Path(root) / f"{m}_{level_tag(axis, lv)}" / f"{mode}_result.npz"
            r = read_result(npz)
            if r is None:
                continue
            r["method"] = m
            r["axis"] = axis
            r["level"] = "clean" if lv is None else lv
            rows.append(r)
    return rows


def write_csv(rows, path):
    if not rows:
        return
    cols = ["method", "axis", "level", "it", "loss"] + [k for k, _, _, _ in METRICS]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[汇总] CSV -> {path}")


def print_table(rows, axis, levels, methods):
    if not rows:
        print("[汇总] 没有可读的结果")
        return
    idx = {(r["method"], r["level"]): r for r in rows}
    for key, label, better, _ in METRICS:
        print("\n" + "=" * 74)
        print(f"{label}   ({'越大越好' if better == 'high' else '越小越好'})")
        print("=" * 74)
        head = f"  {'噪声档':<14}" + "".join(f"{m:>14}" for m in methods) + f"{'差值':>14}"
        print(head)
        for lv in levels:
            tag = "clean" if lv is None else lv
            vals = []
            for m in methods:
                r = idx.get((m, tag))
                vals.append(float("nan") if r is None else float(r[key]))
            cell = "".join("        --    " if np.isnan(v) else f"{v:14.4f}" for v in vals)
            if len(vals) == 2 and not any(np.isnan(vals)):
                dv = vals[1] - vals[0]
                win = (dv > 0) if better == "high" else (dv < 0)
                diff = f"{dv:+14.4f}" + ("  <-" if win else "")
            else:
                diff = ""
            print(f"  {str(tag):<14}{cell}{diff}")
    print("\n  （差值 = ProPtyNet − AD；箭头标出 ProPtyNet 占优的档位）")


def plot(rows, axis, levels, methods, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[汇总] 没有 matplotlib，跳过画图")
        return
    idx = {(r["method"], r["level"]): r for r in rows}
    # x 轴: 泊松剂量取 log 且越左越噪；无噪声档单独放最右
    fin = [lv for lv in levels if lv is not None]
    if not fin:
        return
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for k, (key, _cn, better, label) in enumerate(METRICS):
        a = ax[k // 3, k % 3]
        for m in methods:
            xs, ys = [], []
            for lv in fin:
                r = idx.get((m, lv))
                if r is not None and not np.isnan(float(r[key])):
                    xs.append(lv); ys.append(float(r[key]))
            if xs:
                a.plot(xs, ys, "o-", label=m)
            rc = idx.get((m, "clean"))
            if rc is not None and not np.isnan(float(rc[key])):
                a.axhline(float(rc[key]), ls=":", lw=1,
                          color=a.lines[-1].get_color() if a.lines else None)
        if axis == "poisson":
            a.set_xscale("log"); a.set_xlabel("peak photons  (left = noisier)")
        else:
            a.set_xlabel("Gaussian SNR (dB)  (left = noisier)")
        a.set_title(f"{label}  ({'higher better' if better=='high' else 'lower better'})",
                    fontsize=9)
        a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle("ProPtyNet (pixel probe) vs AD  —  noise robustness"
                 "   (dotted = each method's noise-free baseline)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"[汇总] 图 -> {path}")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="ProPtyNet(pixel probe) vs AD 的噪声扫描")
    ap.add_argument("--script", default=str(HERE / "ProPtyNet_torch.py"),
                    help="被调用的主程序（默认同目录的 ProPtyNet_torch.py）")
    ap.add_argument("--root", default="sweep_out", help="所有结果的根目录")
    ap.add_argument("--axis", choices=["poisson", "gauss"], default="poisson")
    ap.add_argument("--levels", default="",
                    help="逗号分隔。泊松轴给峰值光子数(如 5000,1000,300,100)；"
                         "高斯轴给 SNR dB(如 40,30,25,20)。留空用该轴的默认档位")
    ap.add_argument("--no-clean", action="store_true", help="不跑无噪声基准档")
    ap.add_argument("--noise-clip", action="store_true", help="加论文的 clip 过曝")
    ap.add_argument("--methods", default="AD,ProPtyNet",
                    help=f"逗号分隔，可选 {list(METHODS)}")
    ap.add_argument("--iters", type=int, default=2000,
                    help="两边共用的梯度步数（AD 走 --ad-iters，net 走 --iters）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--extra", default="",
                    help="原样透传给主程序的额外参数，用空格分隔，例如 "
                         "\"--scan-pattern raster_jitter --lr-cosine\"")
    ap.add_argument("--force", action="store_true", help="已有结果也重跑")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令")
    ap.add_argument("--collect-only", action="store_true", help="不跑，只汇总已有结果")
    a = ap.parse_args()

    script = Path(a.script)
    if not script.is_file():
        sys.exit(f"找不到主程序: {script}")

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            sys.exit(f"未知方法 {m}，可选 {list(METHODS)}")

    if a.levels.strip():
        levels = [float(x) for x in a.levels.split(",")]
    else:
        levels = [5000, 1000, 300, 100] if a.axis == "poisson" else [40, 30, 25, 20]
    if not a.no_clean:
        levels = [None] + levels          # None = 无噪声基准，放在最前

    # 两边共用的参数。net 用 --iters；ad 在 plain 模式下用 --ad-iters 对齐步数。
    common = ["--seed", str(a.seed), "--noise-seed", str(a.noise_seed)]
    if a.device:
        common += ["--device", a.device]
    if a.extra.strip():
        common += a.extra.split()

    root = Path(a.root); root.mkdir(parents=True, exist_ok=True)
    print(f"主程序   {script}")
    print(f"输出根   {root.resolve()}")
    print(f"噪声轴   {a.axis}   档位 {[('clean' if l is None else l) for l in levels]}")
    print(f"方法     {methods}")
    print(f"AD 基线  --opt-mode plain = 标准 AD ptychography"
          f"（单 Adam / 全批量 / 联合更新 / 无 stage 无衰减无正则）")
    print(f"步数     两边都 {a.iters} 步（AD:--ad-iters, net:--iters）")
    print(f"共用参数 {' '.join(common)}")
    print()

    if not a.collect_only:
        total = len(levels) * len(methods)
        k = 0
        for lv in levels:
            for m in methods:
                k += 1
                print(f"[{k}/{total}] {m} @ {level_tag(a.axis, lv)}")
                run_one(script, m, a.axis, lv, root, common, a.iters,
                        a.noise_clip, a.force, a.dry_run)
        if a.dry_run:
            print("\n（--dry-run，什么都没跑）")
            return

    rows = collect(root, a.axis, levels, methods)
    write_csv(rows, root / "summary.csv")
    print_table(rows, a.axis, levels, methods)
    plot(rows, a.axis, levels, methods, root / "summary.png")


if __name__ == "__main__":
    main()
