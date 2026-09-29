"""Save numerical evidence and fixed-scale figures (no independent min/max scaling)."""
import csv
import json
from pathlib import Path
import numpy as np


def plot_scene(scene, cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arrays = lambda t: t.detach().cpu().numpy()
    obj, roi = arrays(scene.objects), arrays(scene.roi)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels = [(np.abs(obj[0]), "True amplitude, first wavelength", "gray", 0, 1),
              (arrays(scene.opd_um[0])*1000, "True OPD, first wavelength (nm)", "viridis", None, None),
              (roi, "Evaluation ROI (training illumination)", "gray", 0, 1),
              (arrays(scene.operator.probes[0].abs()), "True probe amplitude (simulation)", "viridis", 0, None),
              (arrays(scene.clean[0]), "Mixed intensity, scan 0", "magma", 0, None),
              (arrays(scene.measured[0]), "Observed intensity, scan 0", "magma", 0, None)]
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title); ax.set_axis_off(); fig.colorbar(im, ax=ax, shrink=.7)
    fig.savefig(out/"scene.png", dpi=140)
    plt.close(fig)


def plot_result(result, scene, cfg, out):
    import matplotlib.pyplot as plt
    truth = scene.objects.cpu().numpy()
    roi = scene.roi.cpu().numpy()
    rec = result["objects"]
    if cfg.scene == "usaf_zero_phase" and result["method"].endswith("_shared_amp"):
        gt, estimate = np.abs(truth[0]), np.abs(rec[0])
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
        panels = [(gt, "Shared USAF amplitude (truth)", "gray", 0, 1),
                  (estimate, "Shared amplitude (reconstruction)", "gray", 0, 1),
                  (estimate-gt, "Amplitude error (recon - truth)", "coolwarm", -1, 1)]
        for ax, (arr, title, cmap, vmin, vmax) in zip(axes, panels):
            im = ax.imshow(np.ma.masked_where(~roi, arr), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9); ax.set_axis_off(); fig.colorbar(im, ax=ax, shrink=.7)
        row = cfg.object_size//2
        axes[3].plot(np.where(roi[row], gt[row], np.nan), label="Truth")
        axes[3].plot(np.where(roi[row], estimate[row], np.nan), label="Reconstruction")
        axes[3].set_title(f"Center-row amplitude (row {row})", fontsize=9)
        axes[3].set_ylim(-.05, 1.05); axes[3].set_xlabel("Object pixel")
        axes[3].legend(fontsize=8); axes[3].grid(alpha=.3)
        fig.suptitle(f"{result['method']} | one object, phase fixed to zero")
        fig.savefig(out/f"{result['method']}_reconstruction.png", dpi=150)
        plt.close(fig)
        return
    lcount = len(cfg.wavelengths_nm)
    fig, axes = plt.subplots(lcount, 4, figsize=(13, 3*lcount), squeeze=False,
                             constrained_layout=True)
    for l in range(lcount):
        piston = np.angle(np.vdot(rec[l][roi], truth[l][roi]))
        phase_diff = np.angle(rec[l]*np.exp(1j*piston)*truth[l].conj())
        panels = [(np.abs(truth[l]), "True A", "gray", 0, 1),
                  (np.abs(rec[l]), "Reconstructed A", "gray", 0, 1),
                  (np.abs(rec[l])-np.abs(truth[l]), "Amplitude error", "coolwarm", -.25, .25),
                  (phase_diff, "Phase error (rad, piston removed)", "coolwarm", -1, 1)]
        for ax, (arr, title, cmap, vmin, vmax) in zip(axes[l], panels):
            im = ax.imshow(np.ma.masked_where(~roi, arr), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{cfg.wavelengths_nm[l]:g} nm | {title}")
            ax.set_axis_off(); fig.colorbar(im, ax=ax, shrink=.7)
    fig.suptitle(result["method"])
    fig.savefig(out/f"{result['method']}_reconstruction.png", dpi=140)
    plt.close(fig)


def plot_probes(result, scene, cfg, out):
    import matplotlib.pyplot as plt
    truth = scene.operator.probes.cpu().numpy()
    estimated, initial = result["probes"], result["initial_probes"]
    count = len(truth)
    fig, axes = plt.subplots(count, 5, figsize=(15, 3*count), squeeze=False, constrained_layout=True)
    for l in range(count):
        piston = np.angle(np.vdot(estimated[l], truth[l]))
        aligned = estimated[l]*np.exp(1j*piston)
        vmax = max(np.abs(truth[l]).max(), np.abs(estimated[l]).max(), np.abs(initial[l]).max())
        mask = np.abs(truth[l]) < .05*np.abs(truth[l]).max()
        panels = [(np.abs(truth[l]), 'True amplitude', 'viridis', 0, vmax),
                  (np.abs(initial[l]), 'Initial amplitude (zero phase)', 'viridis', 0, vmax),
                  (np.abs(estimated[l]), 'Recovered amplitude', 'viridis', 0, vmax),
                  (np.ma.masked_where(mask, np.angle(truth[l])), 'True phase', 'twilight', -np.pi, np.pi),
                  (np.ma.masked_where(mask, np.angle(aligned)), 'Recovered phase (piston aligned)', 'twilight', -np.pi, np.pi)]
        for ax, (arr, title, cmap, vmin, high) in zip(axes[l], panels):
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=high)
            ax.set_title(f'{cfg.wavelengths_nm[l]:g} nm | {title}', fontsize=8)
            ax.set_axis_off(); fig.colorbar(im, ax=ax, shrink=.6)
    fig.suptitle(f"{result['method']} | probe mode: {cfg.probe_mode}")
    fig.savefig(out/f"{result['method']}_probes.png", dpi=140)
    plt.close(fig)


def save_summary(results, cfg, audit, out):
    import matplotlib.pyplot as plt
    summary = {}
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    is_usaf = cfg.scene == "usaf_zero_phase"
    keys = ("train_observed_amplitude_nrmse", "holdout_clean_amplitude_nrmse",
            "shared_amplitude_relative_error" if is_usaf else "mean_complex_relative_error")
    for result in results:
        method = result["method"]
        serial = {k: v for k, v in result.items() if k not in
                  ("objects", "optical_depth", "opd_um", "predicted_intensity", "state_dict", "probes", "initial_probes", "probe_state_dict")}
        summary[method] = serial
        (out/f"{method}_metrics.json").write_text(json.dumps(serial, indent=2, allow_nan=False), encoding="utf-8")
        for channel in result["final"]["channels"]:
            rows.append({"method": method, **channel})
        for ax, key in zip(axes, keys):
            ax.semilogy([r["iteration"] for r in result["history"]],
                        [max(r[key], 1e-12) for r in result["history"]], label=method)
    for ax, key in zip(axes, keys):
        ax.set_title(key.replace("_", " "), fontsize=9)
        ax.set_xlabel("Updates"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.savefig(out/"comparison.png", dpi=150)
    plt.close(fig)
    with (out/"channel_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out/"summary.json").write_text(json.dumps({"audit": audit, "methods": summary},
                                             indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 仿真运行记录", "", f"场景：`{cfg.scene}`；波长（nm）：`{cfg.wavelengths_nm}`；探针：`{cfg.probe_mode}`；光谱模式：`{cfg.spectral_mode}`。",
             f"每位置入射光子数：`{cfg.photons_per_scan}`（0 表示无噪声）；迭代数：`{cfg.iterations}`。", "",
             "pixel 模式联合优化独立复数像素探针；known 模式固定真值探针。equal_power 模式直接相加强度，每探针功率固定为 1/L，不估计光谱权重。使用最终迭代，不根据真值或留出误差选取最优迭代。", "",
             "| 方法 | 训练振幅 NRMSE | 留出干净数据 NRMSE | 物体相对误差 | 耗时 s（含评价） |",
             "|---|---:|---:|---:|---:|"]
    for r in results:
        f = r["final"]
        lines.append(f"| {r['method']} | {f[keys[0]]:.5f} | {f[keys[1]]:.5f} | {f[keys[2]]:.5f} | {r['elapsed_s_including_evaluation']:.2f} |")
    lines += ["", f"传播 padding 检查（当前 vs 额外一倍窗口）：强度相对差 `{audit['padding_relative_intensity_difference']:.3g}`。",
              "", ("USAF 主线：各波长共享一个实数振幅，物体相位固定为零；不进行光谱物体分离。振幅区域对比度不是 USAF 线组分辨率。"
                     if is_usaf else "串扰矩阵以真实吸收标记为评价基底：理想对角线为 1、非对角线为 0；对角线接近 0 不能解释成成功抑制串扰。"),
              "", "这是同一离散传播器生成和拟合数据的模型匹配仿真。单次运行不能证明物理唯一性、泛化、盲重建能力或论文创新性。",
              "等迭代数不等于等计算量；此处报告参数量和耗时，尚未按相同时间预算调优各方法。"]
    (out/"run_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary
