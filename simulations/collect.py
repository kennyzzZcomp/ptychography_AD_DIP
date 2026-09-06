# -*- coding: utf-8 -*-
"""把 runs/ 底下所有 *_result.npz 汇总成一张 CSV。

    python collect.py runs                 -> runs/summary.csv
    python collect.py runs -o out.csv

每行一次跑，列 = 条件 + 关键指标（终值 / 最优值 / 最优步）。
把这个 CSV 发出来就够了，不用发几十个 npz。
"""
import sys, os, json, glob, argparse
import numpy as np

CFG_KEYS = ["probe_mode", "phase_repr", "amp_act", "scale_mode", "weight_decay",
            "lr_cosine", "support_energy", "iters", "stages", "lr_net", "lr_probe",
            "probe_warmup", "holdout_frac", "poisson", "peak_photons", "noise_seed",
            "scan_npos", "scan_step", "scan_seed", "base_ch", "data_loss", "seed",
            "N", "N_OBJ", "eval_size", "reg_size"]

MET = ["ssim_o_amp", "psnr_o_amp", "relerr_o_complex", "rmse_o_phi_rad",
       "ssim_o_phi", "relerr_p_complex"]


def one(path):
    d = np.load(path, allow_pickle=True)
    cfg = json.loads(str(d["cfg"]))
    hist = json.loads(str(d["hist"]))
    tag = os.path.basename(path).replace("_result.npz", "")
    row = {"run": os.path.relpath(os.path.dirname(path)), "mode": tag,
           "n_records": len(hist)}
    for k in CFG_KEYS:
        if k in cfg:
            row[k] = cfg[k]
    if not hist:
        return row
    last = hist[-1]
    for m in MET:
        if m in last:
            row[m + "_final"] = last[m]
    # 最优（SSIM 取最大，复场误差取最小）及其位置
    ss = [h.get("ssim_o_amp", np.nan) for h in hist]
    ce = [h.get("relerr_o_complex", np.nan) for h in hist]
    bi, ci = int(np.nanargmax(ss)), int(np.nanargmin(ce))
    row["ssim_o_amp_best"] = ss[bi]
    row["relerr_o_complex_best"] = ce[ci]
    row["best_at"] = hist[bi].get("it", bi + 1)          # net 有 it，ad 用 stage 序号
    row["last_at"] = hist[-1].get("it", len(hist))
    row["loss_final"] = last.get("loss", np.nan)
    row["loss_min"] = float(np.nanmin([h.get("loss", np.nan) for h in hist])) if "loss" in last else np.nan
    # 后半程 loss 降而 SSIM 也降的比例 —— 等价解漂移的指标
    if len(hist) > 8 and "loss" in last:
        h2 = hist[len(hist) // 2:]
        dl = np.diff([h["loss"] for h in h2]); ds = np.diff([h["ssim_o_amp"] for h in h2])
        row["drift_frac"] = float(((dl < 0) & (ds < 0)).mean())
    # 留出验证挑的停止点 vs 真值挑的停止点
    if "val" in last and np.isfinite(last.get("val", np.nan)):
        vi = int(np.nanargmin([h["val"] for h in hist]))
        row["val_best_at"] = hist[vi].get("it", vi + 1)
        row["ssim_at_val_best"] = hist[vi]["ssim_o_amp"]
        row["earlystop_gap"] = row["ssim_o_amp_best"] - row["ssim_at_val_best"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="runs")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.root, "**", "*_result.npz"), recursive=True))
    if not files:
        print(f"在 {a.root} 底下没找到 *_result.npz"); return
    rows = []
    for f in files:
        try:
            rows.append(one(f))
        except Exception as e:
            print(f"  跳过 {f}: {type(e).__name__} {e}")
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    out = a.out or os.path.join(a.root, "summary.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("" if k not in r else str(r[k]) for k in cols) + "\n")
    print(f"{len(rows)} 次跑 -> {out}")
    key = [c for c in ["run", "mode", "peak_photons", "scan_npos", "noise_seed", "scan_seed",
                       "ssim_o_amp_final", "ssim_o_amp_best", "best_at",
                       "relerr_o_complex_final", "relerr_p_complex_final", "drift_frac"] if c in seen]
    w = [max(len(k), *(len(f'{r.get(k,"")}'[:12]) for r in rows)) for k in key]
    print("  ".join(k.ljust(x) for k, x in zip(key, w)))
    for r in rows:
        print("  ".join(f'{r.get(k,"")}'[:12].ljust(x) for k, x in zip(key, w)))


if __name__ == "__main__":
    main()
