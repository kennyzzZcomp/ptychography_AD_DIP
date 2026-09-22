#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 Colab 上批跑并汇总 ProPtyNet_paper.py 的 overlap 对照实验。

实验矩阵（默认共 20 次运行）：

    overlap = 80%, 70%, 60%, 50%, 40%
    method  = AD-known, DIP-known, AD-blind, DIP-blind

其中 known 使用 ``--probe-mode truth``（真值探针冻结），blind 默认使用
``--probe-mode support``（探针未知，但已知针孔支撑）。若要完全自由的盲探针，传
``--blind-probe-mode pixel``。不同 overlap 的 grid 随 step 改变，以便扫描中心跨度
保持在约 70--90 px；因此这是一条“overlap / 扫描密度轴”，测量张数也会变化。

Colab 示例（在仓库根目录执行）：

    # 先只打印将要执行的命令
    !python simulations/paper_overlap_colab.py run \
        --root /content/drive/MyDrive/ProPtyNet/overlap_runs --dry-run

    # 正式运行；Colab 中断后重复同一命令会跳过已经完成的结果
    !python simulations/paper_overlap_colab.py run \
        --root /content/drive/MyDrive/ProPtyNet/overlap_runs \
        --iters 2000 --fwd-chunk 8

    # 随时重新汇总（不运行重建）
    !python simulations/paper_overlap_colab.py collect \
        --root /content/drive/MyDrive/ProPtyNet/overlap_runs

汇总会在 ROOT 下生成：

    summary_long.csv          每次运行一行（含最终值、最优值和最优迭代）
    summary_mean.csv          多 seed 的均值/标准差
    summary_wide.csv          适合直接看或导入论文表格的宽表
    overlap_metrics.png       四项指标随 overlap 的变化
    reconstruction_amplitude.png / reconstruction_phase.png
                              5×4 重建结果拼图
    overlap_report.zip        CSV、图、日志和单次结果图的轻量打包（默认不含 NPZ）

本脚本只负责任务编排和结果汇总，不改变 ProPtyNet_paper.py 的算法实现。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_SCRIPT = REPO_ROOT / "simulations" / "ProPtyNet_paper.py"
sys.path.insert(0, str(REPO_ROOT))

# step/grid 与用户给出的实验设计一致。实际 overlap 由 paper preset 的物理探针直径计算。
OVERLAP_CASES: dict[int, tuple[int, int]] = {
    80: (12, 8),
    70: (18, 6),
    60: (24, 4),
    50: (29, 4),
    40: (35, 3),
}

CONDITION_ORDER = ("ad_known", "dip_known", "ad_blind", "dip_blind")
TGV_DEFAULTS = {
    "tgv_amp": 0.0, "tgv_alpha0": 2.0, "tgv_alpha1": 1.0,
    "tgv_eps": 1e-3, "tgv_inner_steps": 5, "tgv_lr": 1e-2,
}
DISPLAY_CONDITION_ORDER = (*CONDITION_ORDER, "ptylab_epie")
CONDITION_INFO = {
    "ad_known": ("AD-known", "ad", "truth"),
    "dip_known": ("DIP-known", "net", "truth"),
    "ad_blind": ("AD-blind", "ad", None),
    "dip_blind": ("DIP-blind", "net", None),
}

HIST_ALIASES = {
    "ssim_amp": ("ssim_amp", "ssim_o_amp"),
    "ssim_phs": ("ssim_phs", "ssim_o_phi"),
    "psnr_amp": ("psnr_amp", "psnr_o_amp"),
    "psnr_phs": ("psnr_phs",),
    "relerr": ("relerr", "relerr_o_complex"),
    "relerr_p": ("relerr_p", "relerr_p_complex"),
    "loss": ("loss",),
    "real": ("real",),
}

FINAL_METRICS = (
    "ssim_amp",
    "ssim_phs",
    "psnr_amp",
    "psnr_phs",
    "relerr",
    "relerr_p",
    "loss",
    "real",
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    tmp.replace(path)


def _npz_json(data: Any) -> Any:
    """读取 np.savez 保存的 0-D JSON 字符串。"""
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(str(data))


def _parse_csv_ints(raw: str, *, valid: Iterable[int] | None = None) -> list[int]:
    try:
        values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"必须是逗号分隔的整数：{raw!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("列表不能为空")
    if valid is not None:
        allowed = set(valid)
        bad = [x for x in values if x not in allowed]
        if bad:
            raise argparse.ArgumentTypeError(f"不支持 {bad}；可选值为 {sorted(allowed)}")
    return values


def _parse_conditions(raw: str) -> list[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in values if x not in CONDITION_INFO]
    if not values or bad:
        raise argparse.ArgumentTypeError(
            f"--only 只能使用逗号分隔的 {', '.join(CONDITION_ORDER)}；收到 {raw!r}"
        )
    return values


def _cfg_actual_overlap(cfg: dict[str, Any]) -> float:
    dx1 = float(cfg["wlength"]) * float(cfg["z"]) / (
        int(cfg["N"]) * float(cfg["det_pixel"])
    )
    probe_diam_px = float(cfg["probe_diam_um"]) * 1e-6 / dx1
    return 100.0 * (1.0 - float(cfg["step_px"]) / probe_diam_px)


def _paper_actual_overlap(step_px: int) -> float:
    # PRESETS["paper"] 的值；独立写在这里，dry-run 时无需导入 torch/主实验脚本。
    dx1 = 632e-9 * 0.165 / (512 * 15.04e-6)
    probe_diam_px = 800e-6 / dx1
    return 100.0 * (1.0 - step_px / probe_diam_px)


def _expected_cfg(args: argparse.Namespace, step: int, grid: int, mode: str,
                  probe_mode: str, seed: int) -> dict[str, Any]:
    expected = {
        "preset": "paper",
        "step_px": step,
        "grid": grid,
        "obj_size": args.obj_size,
        "iters": args.iters,
        "probe_mode": probe_mode,
        "probe_init": args.probe_init,
        "probe_init_sigma": args.probe_init_sigma,
        "seed": seed,
        "eval_every": args.eval_every,
        "eval_size": args.eval_size,
        "fwd_chunk": args.fwd_chunk,
        "noise": args.noise,
        "noise_seed": args.noise_seed,
        "mode": mode,
    }
    if mode == "net":
        expected.update({key: getattr(args, key) for key in TGV_DEFAULTS})
    return expected


def _existing_result_matches(path: Path, expected: dict[str, Any]) -> tuple[bool, str]:
    try:
        with np.load(path, allow_pickle=True) as d:
            cfg = _npz_json(d["cfg"])
    except Exception as exc:  # 损坏/未写完的 NPZ 不能当作已完成
        return False, f"无法读取现有结果：{type(exc).__name__}: {exc}"
    mismatches = []
    for key, want in expected.items():
        if key == "mode":
            continue
        got = cfg.get(key, TGV_DEFAULTS.get(key))
        if isinstance(want, float):
            same = got is not None and math.isclose(float(got), want, rel_tol=1e-9, abs_tol=1e-12)
        else:
            same = got == want
        if not same:
            mismatches.append(f"{key}: existing={got!r}, requested={want!r}")
    return not mismatches, "; ".join(mismatches)


def _build_command(args: argparse.Namespace, outdir: Path, mode: str,
                   probe_mode: str, step: int, grid: int, seed: int) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(SIM_SCRIPT),
        mode,
        "--preset", "paper",
        "--iters", str(args.iters),
        "--step-px", str(step),
        "--grid", str(grid),
        "--obj-size", str(args.obj_size),
        "--probe-mode", probe_mode,
        "--probe-init", args.probe_init,
        "--probe-init-sigma", str(args.probe_init_sigma),
        "--eval-every", str(args.eval_every),
        "--eval-size", str(args.eval_size),
        "--fwd-chunk", str(args.fwd_chunk),
        "--device", args.device,
        "--seed", str(seed),
        "--noise", args.noise,
        "--noise-seed", str(args.noise_seed),
        "--outdir", str(outdir),
    ]
    if args.noise != "none":
        cmd.extend(["--snr", str(args.snr)])
    if mode == "net" and args.dip_lr_cosine:
        cmd.append("--lr-cosine")
    if mode == "net":
        for key in TGV_DEFAULTS:
            cmd.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
    if args.base_ch is not None:
        cmd.extend(["--base-ch", str(args.base_ch)])
    for name in ("lr_obj", "lr_prb", "lr_net", "lr_probe"):
        value = getattr(args, name)
        if value is not None:
            cmd.extend(["--" + name.replace("_", "-"), str(value)])
    return cmd


def _display_command(cmd: list[str]) -> str:
    # 路径可能含空格；这是日志展示，不交回 shell 执行。
    return " ".join(f'"{part}"' if any(c.isspace() for c in part) else part for part in cmd)


def _run_one(cmd: list[str], outdir: Path, verbose: bool) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "log.txt"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    interesting = re.compile(
        r"^(\[G\]|\[scene\]|\[ad\]|\[net\]|\[paper\]|\s+it\s+|[-]{10,})"
    )
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if verbose or interesting.search(line):
                    print("    " + line.rstrip(), flush=True)
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait(timeout=30)
            raise
        return proc.wait()


def run_matrix(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    overlaps = _parse_csv_ints(args.overlaps, valid=OVERLAP_CASES)
    seeds = _parse_csv_ints(args.seeds)
    conditions = _parse_conditions(args.only)

    jobs: list[dict[str, Any]] = []
    for target in overlaps:
        step, default_grid = OVERLAP_CASES[target]
        grid = args.fixed_grid or default_grid
        scan_span = 512 + (grid - 1) * step
        if scan_span > args.obj_size:
            raise SystemExit(
                f"ov{target}: N + (grid-1)*step = {scan_span} exceeds "
                f"obj_size={args.obj_size}; enlarge --obj-size or lower --fixed-grid"
            )
        for condition in conditions:
            label, mode, fixed_probe_mode = CONDITION_INFO[condition]
            probe_mode = fixed_probe_mode or args.blind_probe_mode
            run_seeds = seeds if mode == "net" else [0]
            for seed in run_seeds:
                suffix = "" if fixed_probe_mode == "truth" else f"_{probe_mode}"
                condition_dir = condition + suffix
                outdir = root / f"ov{target:02d}" / condition_dir
                if mode == "net":
                    outdir = outdir / f"seed_{seed}"
                cmd = _build_command(args, outdir, mode, probe_mode, step, grid, seed)
                jobs.append({
                    "target_overlap_pct": target,
                    "actual_overlap_pct": _paper_actual_overlap(step),
                    "step_px": step,
                    "grid": grid,
                    "n_patterns": grid * grid,
                    "condition_key": condition,
                    "condition_label": label,
                    "mode": mode,
                    "probe_knowledge": "known" if probe_mode == "truth" else "blind",
                    "probe_mode": probe_mode,
                    "seed": seed,
                    "outdir": outdir,
                    "command": cmd,
                })

    print("=" * 78)
    print(f"ProPtyNet paper overlap matrix: {len(jobs)} runs")
    print(f"root: {root}")
    print(f"overlaps: {overlaps}; conditions: {conditions}; DIP seeds: {seeds}")
    print(f"blind probe mode: {args.blind_probe_mode}; iterations: {args.iters}")
    if args.fixed_grid:
        print(f"固定扫描数：所有 overlap 均为 {args.fixed_grid}×{args.fixed_grid}; "
              f"固定中心评价区 {args.eval_size}×{args.eval_size}"
              if args.eval_size else
              f"固定扫描数：所有 overlap 均为 {args.fixed_grid}×{args.fixed_grid}; "
              "评价区仍按照明范围自适应")
    else:
        print("注意：grid 随 overlap 改变，所以这是 overlap / 扫描密度联合轴。")
    print("=" * 78)

    if args.dry_run:
        for i, job in enumerate(jobs, 1):
            print(
                f"[{i:02d}/{len(jobs):02d}] ov target={job['target_overlap_pct']}% "
                f"actual={job['actual_overlap_pct']:.2f}%  {job['condition_label']} "
                f"({job['probe_mode']}) seed={job['seed']}"
            )
            print("    " + _display_command(job["command"]))
        return 0

    failures: list[tuple[Path, int]] = []
    completed = 0
    skipped = 0
    for i, job in enumerate(jobs, 1):
        outdir: Path = job["outdir"]
        result = outdir / f"{job['mode']}_result.npz"
        expected = _expected_cfg(
            args, job["step_px"], job["grid"], job["mode"], job["probe_mode"], job["seed"]
        )
        title = (
            f"[{i:02d}/{len(jobs):02d}] ov{job['target_overlap_pct']} "
            f"({job['actual_overlap_pct']:.2f}%) {job['condition_label']} "
            f"probe={job['probe_mode']} seed={job['seed']}"
        )
        if result.exists() and not args.force:
            matches, reason = _existing_result_matches(result, expected)
            if matches:
                print(title + " -> 已完成，跳过")
                skipped += 1
                continue
            print(title + " -> 现有结果与本次参数冲突")
            print("    " + reason)
            print("    请换 --root，或确认后使用 --force 覆盖该运行。")
            failures.append((outdir, 2))
            if not args.keep_going:
                break
            continue

        if result.exists() and args.force:
            # --force 是用户对这一条精确结果的显式覆盖授权。先移走旧产物，避免新运行
            # 失败后 collector 把旧 NPZ/PNG 误当成本次成功结果。
            result.unlink()
            old_figure = outdir / f"{job['mode']}_result.png"
            if old_figure.exists():
                old_figure.unlink()

        manifest = {k: v for k, v in job.items() if k not in ("outdir", "command")}
        manifest.update({
            "schema_version": 1,
            "status": "running",
            "started_at": _now(),
            "outdir": str(outdir),
            "command": job["command"],
        })
        _json_dump(outdir / "run_spec.json", manifest)
        print(title + " -> 开始", flush=True)
        print("    log: " + str(outdir / "log.txt"), flush=True)
        try:
            code = _run_one(job["command"], outdir, args.verbose_logs)
        except KeyboardInterrupt:
            manifest.update({"status": "interrupted", "finished_at": _now()})
            _json_dump(outdir / "run_spec.json", manifest)
            print("\n收到中断；已经完成的 NPZ 会在下次执行时自动跳过。")
            return 130
        manifest.update({
            "status": "complete" if code == 0 and result.exists() else "failed",
            "returncode": code,
            "finished_at": _now(),
        })
        _json_dump(outdir / "run_spec.json", manifest)
        if code != 0 or not result.exists():
            print(f"    失败 (return code {code})；查看 {outdir / 'log.txt'}")
            failures.append((outdir, code or 1))
            if not args.keep_going:
                break
        else:
            completed += 1
            print("    完成: " + str(result), flush=True)

    print(f"运行结束：新完成 {completed}，跳过 {skipped}，失败 {len(failures)}。")
    if not args.no_collect and root.exists():
        collect_results(root, include_npz=args.include_npz, make_zip=True)
    if failures:
        print("失败目录：")
        for path, code in failures:
            print(f"  [{code}] {path}")
        return 1
    return 0


def _hist_value(record: dict[str, Any], metric: str) -> float:
    for key in HIST_ALIASES[metric]:
        if key in record:
            try:
                return float(record[key])
            except (TypeError, ValueError):
                pass
    return float("nan")


def _finite_arg(values: list[float], fn: Any) -> int | None:
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).any():
        return None
    return int(fn(a))


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.parent / "run_spec.json"
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _target_from_step(step: int) -> int | None:
    for target, (known_step, _grid) in OVERLAP_CASES.items():
        if step == known_step:
            return target
    return None


def _parse_log_metadata(npz_path: Path) -> tuple[str, float]:
    log_path = npz_path.parent / "log.txt"
    if not log_path.exists():
        return "", float("nan")
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", float("nan")
    fp = re.search(r"\[scene\]\s+指纹\s+([0-9a-f]+)", text)
    elapsed = re.search(r"\[(?:ad|net|paper)\]\s+用时\s+([0-9.]+)s", text)
    return (fp.group(1) if fp else "", float(elapsed.group(1)) if elapsed else float("nan"))


def _row_from_npz(path: Path, root: Path, recompute_eval_size: int = 0) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as d:
        cfg = _npz_json(d["cfg"])
        hist = _npz_json(d["hist"])
        roi = np.asarray(d["roi"], dtype=int).tolist() if "roi" in d else None
        if recompute_eval_size:
            obj_rec = np.asarray(d["obj_rec"])
            obj_gt = np.asarray(d["obj_gt"])
        else:
            obj_rec = obj_gt = None
    if not isinstance(hist, list):
        raise ValueError("hist 不是 list")
    recomputed = None
    if recompute_eval_size:
        if obj_rec is None or obj_gt is None or obj_rec.shape != obj_gt.shape:
            raise ValueError("cannot recompute fixed ROI: obj_rec/obj_gt are missing or shape-mismatched")
        size = int(recompute_eval_size)
        if size < 7 or size > obj_rec.shape[-1]:
            raise ValueError(f"recompute eval size must be within [7, {obj_rec.shape[-1]}]")
        start = (obj_rec.shape[-1] - size) // 2
        roi = [start, start + size, start, start + size]
        from functions.paperrepro.evaluate import evaluate
        rs, cs = slice(roi[0], roi[1]), slice(roi[2], roi[3])
        recomputed = evaluate(obj_rec[rs, cs], obj_gt[rs, cs])

    manifest = _read_manifest(path)
    mode = path.name.removesuffix("_result.npz")
    probe_mode = str(cfg.get("probe_mode", ""))
    known = probe_mode == "truth"
    condition_key = manifest.get("condition_key") or f"{'dip' if mode == 'net' else mode}_{'known' if known else 'blind'}"
    base_label = "DIP" if mode == "net" else mode.upper()
    label = manifest.get("condition_label") or f"{base_label}-{'known' if known else 'blind'}"
    if not known and mode != "epie":
        label = f"{label} ({probe_mode})"
    if mode == "net" and float(cfg.get("tgv_amp", 0)) > 0:
        label += f" + TGV(lambda={float(cfg['tgv_amp']):g})"
    step = int(cfg["step_px"])
    grid = int(cfg["grid"])
    target = manifest.get("target_overlap_pct")
    if target is None:
        target = _target_from_step(step)
    fingerprint, elapsed = _parse_log_metadata(path)

    row: dict[str, Any] = {
        "run": str(path.parent.relative_to(root)),
        "result_npz": str(path),
        "target_overlap_pct": target,
        "actual_overlap_pct": _cfg_actual_overlap(cfg),
        "step_px": step,
        "grid": grid,
        "obj_size": cfg.get("obj_size"),
        "n_patterns": grid * grid,
        "scan_center_span_px": (grid - 1) * step,
        "condition_key": condition_key,
        "condition_label": label,
        "mode": mode,
        "probe_knowledge": "known" if known else "blind",
        "probe_mode": probe_mode,
        "probe_init": cfg.get("probe_init"),
        "probe_init_sigma": cfg.get("probe_init_sigma"),
        "seed": cfg.get("seed"),
        "noise": cfg.get("noise"),
        "noise_seed": cfg.get("noise_seed"),
        "snr_db": cfg.get("snr_db"),
        "iters": cfg.get("iters"),
        "eval_every": cfg.get("eval_every"),
        "eval_size": recompute_eval_size or cfg.get("eval_size", 0),
        "eval_roi_height": (roi[1] - roi[0]) if roi else None,
        "eval_roi_width": (roi[3] - roi[2]) if roi else None,
        "eval_r0": roi[0] if roi else None,
        "eval_r1": roi[1] if roi else None,
        "eval_c0": roi[2] if roi else None,
        "eval_c1": roi[3] if roi else None,
        "metrics_recomputed_from_final_npz": bool(recomputed),
        "fwd_chunk": cfg.get("fwd_chunk"),
        "lr_obj": cfg.get("lr_obj"),
        "lr_prb": cfg.get("lr_prb"),
        "lr_net": cfg.get("lr_net"),
        "lr_probe": cfg.get("lr_probe"),
        "lr_cosine": cfg.get("lr_cosine"),
        "n_records": len(hist),
        "scene_fingerprint": fingerprint,
        "elapsed_sec": elapsed,
    }
    if not hist:
        return row

    final = hist[-1]
    row.update({key: cfg.get(key, default) for key, default in TGV_DEFAULTS.items()})
    for key in ("data_loss", "relative_data_loss", "tgv_amp", "tgv_first",
                "tgv_second", "tgv_weighted"):
        # tgv_amp in cfg is lambda; hist.tgv_amp is the unweighted regularizer.
        row[key + "_final"] = final.get(key)
    row["final_at"] = final.get("it", len(hist))
    for metric in FINAL_METRICS:
        row[f"{metric}_final"] = _hist_value(final, metric)
    if recomputed:
        for metric, value in recomputed.items():
            row[f"{metric}_final"] = value

    for metric, reducer in (
        ("ssim_amp", np.nanargmax),
        ("ssim_phs", np.nanargmax),
        ("psnr_amp", np.nanargmax),
        ("psnr_phs", np.nanargmax),
        ("relerr", np.nanargmin),
        ("relerr_p", np.nanargmin),
        ("loss", np.nanargmin),
        ("real", np.nanargmin),
    ):
        values = [_hist_value(h, metric) for h in hist]
        idx = _finite_arg(values, reducer)
        row[f"{metric}_best"] = values[idx] if idx is not None else float("nan")
        row[f"{metric}_best_at"] = hist[idx].get("it", idx + 1) if idx is not None else ""
    if recomputed:
        # Only the final reconstruction is stored. Historical checkpoints used
        # the old ROI, so reporting their old "best" value beside a new ROI
        # would be misleading.
        for metric in ("ssim_amp", "ssim_phs", "psnr_amp", "psnr_phs", "relerr"):
            row[f"{metric}_best"] = float("nan")
            row[f"{metric}_best_at"] = ""
    return row


def _condition_rank(row: dict[str, Any]) -> tuple[int, str]:
    key = str(row.get("condition_key", ""))
    try:
        rank = DISPLAY_CONDITION_ORDER.index(key)
    except ValueError:
        rank = len(DISPLAY_CONDITION_ORDER)
    return rank, str(row.get("condition_label", ""))


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    target = row.get("target_overlap_pct")
    target_sort = -float(target) if target is not None else float("inf")
    return (
        target_sort,
        *_condition_rank(row),
        int(row.get("seed") or 0),
        str(row.get("run", "")),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields, seen = [], set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k, "")) for k in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return value


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("target_overlap_pct"),
            row.get("condition_key"),
            row.get("condition_label"),
            row.get("probe_mode"),
        )
        groups[key].append(row)

    metrics = [f"{m}_final" for m in FINAL_METRICS] + [
        "ssim_amp_best", "ssim_phs_best", "relerr_best", "relerr_p_best"
    ]
    out = []
    for (target, key, label, probe_mode), items in groups.items():
        first = items[0]
        agg: dict[str, Any] = {
            "target_overlap_pct": target,
            "actual_overlap_pct": float(np.mean([float(x["actual_overlap_pct"]) for x in items])),
            "step_px": first["step_px"],
            "grid": first["grid"],
            "obj_size": first.get("obj_size"),
            "n_patterns": first["n_patterns"],
            "eval_size": first.get("eval_size", 0),
            "eval_roi_height": first.get("eval_roi_height"),
            "eval_roi_width": first.get("eval_roi_width"),
            "condition_key": key,
            "condition_label": label,
            "probe_mode": probe_mode,
            "n_runs": len(items),
        }
        for metric in metrics:
            values = np.asarray([x.get(metric, np.nan) for x in items], dtype=float)
            values = values[np.isfinite(values)]
            agg[metric + "_mean"] = float(values.mean()) if values.size else float("nan")
            agg[metric + "_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        out.append(agg)
    return sorted(out, key=_row_sort_key)


def _write_wide(path: Path, aggregated: list[dict[str, Any]]) -> None:
    by_overlap: dict[Any, dict[str, Any]] = defaultdict(dict)
    metrics = ("ssim_amp_final", "ssim_phs_final", "psnr_amp_final", "relerr_final", "relerr_p_final")
    for row in aggregated:
        target = row["target_overlap_pct"]
        dest = by_overlap[target]
        dest.update({
            "target_overlap_pct": target,
            "actual_overlap_pct": row["actual_overlap_pct"],
            "step_px": row["step_px"],
            "grid": row["grid"],
            "obj_size": row.get("obj_size"),
            "n_patterns": row["n_patterns"],
            "eval_size": row.get("eval_size", 0),
            "eval_roi_height": row.get("eval_roi_height"),
            "eval_roi_width": row.get("eval_roi_width"),
        })
        prefix = re.sub(r"[^a-z0-9]+", "_", str(row["condition_label"]).lower()).strip("_")
        for metric in metrics:
            dest[f"{prefix}_{metric}_mean"] = row.get(metric + "_mean")
            dest[f"{prefix}_{metric}_std"] = row.get(metric + "_std")
    wide = [by_overlap[k] for k in sorted(by_overlap, reverse=True)]
    _write_csv(path, wide)


def _plot_metrics(root: Path, aggregated: list[dict[str, Any]]) -> Path | None:
    if not aggregated:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregated:
        groups[str(row["condition_label"])].append(row)

    specs = (
        ("ssim_amp_final", "Object amplitude SSIM", "higher is better"),
        ("ssim_phs_final", "Object phase SSIM", "higher is better"),
        ("relerr_final", "Object complex relative error", "lower is better"),
        ("relerr_p_final", "Probe complex relative error", "lower is better"),
    )
    colors = ("#2b6cb0", "#dd6b20", "#38a169", "#805ad5", "#d53f8c", "#4a5568")
    markers = ("o", "s", "^", "D", "v", "P")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (metric, title, direction) in zip(axes.flat, specs):
        for i, (label, items) in enumerate(sorted(groups.items(), key=lambda kv: _condition_rank(kv[1][0]))):
            items = sorted(items, key=lambda x: float(x["actual_overlap_pct"]))
            x = np.asarray([r["actual_overlap_pct"] for r in items], dtype=float)
            y = np.asarray([r.get(metric + "_mean", np.nan) for r in items], dtype=float)
            e = np.asarray([r.get(metric + "_std", 0.0) for r in items], dtype=float)
            good = np.isfinite(x) & np.isfinite(y)
            if not good.any():
                continue
            ax.errorbar(
                x[good], y[good], yerr=e[good], label=label,
                color=colors[i % len(colors)], marker=markers[i % len(markers)],
                linewidth=1.8, markersize=6, capsize=3,
            )
        ax.set_title(f"{title}\n({direction})")
        ax.set_xlabel("actual linear overlap (%)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("ProPtyNet paper simulation: overlap / scan-density sweep", fontsize=14)
    path = root / "overlap_metrics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _representatives(rows: list[dict[str, Any]]) -> dict[tuple[Any, str], dict[str, Any]]:
    groups: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("target_overlap_pct"), str(row.get("condition_label")))].append(row)
    chosen = {}
    for key, items in groups.items():
        # 多 seed 时取最终 amplitude SSIM 的中位代表，而不是偷偷挑最好的一次。
        finite = [r for r in items if np.isfinite(float(r.get("ssim_amp_final", np.nan)))]
        if finite:
            finite.sort(key=lambda r: float(r["ssim_amp_final"]))
            chosen[key] = finite[(len(finite) - 1) // 2]
        else:
            chosen[key] = items[0]
    return chosen


def _plot_recon_grid(root: Path, rows: list[dict[str, Any]], component: str) -> Path | None:
    if not rows:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = sorted({r.get("target_overlap_pct") for r in rows if r.get("target_overlap_pct") is not None}, reverse=True)
    label_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        label_rows.setdefault(str(row["condition_label"]), row)
    labels = sorted(label_rows, key=lambda label: _condition_rank(label_rows[label]))
    reps = _representatives(rows)
    if not targets or not labels:
        return None

    fig, axes = plt.subplots(
        len(targets), len(labels),
        figsize=(3.4 * len(labels), 3.1 * len(targets)),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for ri, target in enumerate(targets):
        for ci, label in enumerate(labels):
            ax = axes[ri, ci]
            row = reps.get((target, label))
            if row is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.axis("off")
                continue
            try:
                with np.load(row["result_npz"], allow_pickle=True) as d:
                    rec = np.asarray(d["obj_rec"])
                    gt = np.asarray(d["obj_gt"])
                    roi = np.asarray(d["roi"], dtype=int).tolist()
                if row.get("eval_r0") is not None:
                    roi = [int(row[k]) for k in ("eval_r0", "eval_r1", "eval_c0", "eval_c1")]
                rs, cs = slice(roi[0], roi[1]), slice(roi[2], roi[3])
                den = max(float(np.vdot(rec[rs, cs], rec[rs, cs]).real), 1e-30)
                aligned = rec * (np.vdot(rec[rs, cs], gt[rs, cs]) / den)
                if component == "amplitude":
                    arr = np.abs(aligned[rs, cs])
                    vmax = float(np.abs(gt[rs, cs]).max())
                    image = ax.imshow(arr, cmap="gray", vmin=0, vmax=max(vmax, 1e-12))
                    score = f"SSIM={float(row.get('ssim_amp_final', np.nan)):.3f}"
                else:
                    arr = np.angle(aligned[rs, cs])
                    phase_max = max(float(np.abs(np.angle(gt[rs, cs])).max()), 1e-6)
                    image = ax.imshow(arr, cmap="twilight", vmin=-phase_max, vmax=phase_max)
                    score = f"SSIM={float(row.get('ssim_phs_final', np.nan)):.3f}"
                ax.set_title((label + "\n" if ri == 0 else "") + score, fontsize=9)
                if ci == 0:
                    ax.set_ylabel(f"target {target}%\nactual {float(row['actual_overlap_pct']):.2f}%")
                ax.set_xticks([])
                ax.set_yticks([])
            except Exception as exc:
                ax.text(0.5, 0.5, f"load failed\n{type(exc).__name__}", ha="center", va="center")
                ax.axis("off")
    fig.suptitle(f"Final object {component} (median-seed representative)", fontsize=14)
    path = root / f"reconstruction_{component}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _zip_report(root: Path, include_npz: bool) -> Path:
    out = root / "overlap_report.zip"
    fixed = (
        "summary_long.csv",
        "summary_mean.csv",
        "summary_wide.csv",
        "overlap_metrics.png",
        "reconstruction_amplitude.png",
        "reconstruction_phase.png",
    )
    candidates = [root / name for name in fixed]
    candidates.extend(root.rglob("run_spec.json"))
    candidates.extend(root.rglob("log.txt"))
    candidates.extend(root.rglob("*_result.png"))
    if include_npz:
        candidates.extend(root.rglob("*_result.npz"))
    unique = sorted({p.resolve() for p in candidates if p.is_file() and p.resolve() != out.resolve()})
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in unique:
            zf.write(path, path.relative_to(root))
    return out


def collect_results(root: Path, *, include_npz: bool, make_zip: bool,
                    recompute_eval_size: int = 0) -> int:
    root = root.expanduser().resolve()
    files = sorted(root.rglob("*_result.npz")) if root.exists() else []
    if not files:
        print(f"在 {root} 下没有找到 *_result.npz；没有可汇总的结果。")
        return 1

    rows, errors = [], []
    for path in files:
        try:
            rows.append(_row_from_npz(path, root, recompute_eval_size))
        except Exception as exc:
            errors.append((path, f"{type(exc).__name__}: {exc}"))
    rows.sort(key=_row_sort_key)
    if not rows:
        print("所有 NPZ 都读取失败。")
        for path, error in errors:
            print(f"  {path}: {error}")
        return 1

    geometries = sorted({
        (r.get("obj_size"), r.get("eval_roi_height"), r.get("eval_roi_width"))
        for r in rows
    }, key=str)
    if len(geometries) == 1:
        print(f"评价几何一致：obj_size / ROI(H,W) = {geometries[0]}")
    else:
        print("警告：这个 root 中混有不同的画布或评价区域，指标不应放在同一条曲线上：")
        for geometry in geometries:
            print(f"  obj_size / ROI(H,W) = {geometry}")
        print("请用新的 --root 重跑，并给所有 run 使用同一个 --obj-size 和 --eval-size。")

    long_path = root / "summary_long.csv"
    mean_path = root / "summary_mean.csv"
    wide_path = root / "summary_wide.csv"
    _write_csv(long_path, rows)
    aggregated = _aggregate(rows)
    _write_csv(mean_path, aggregated)
    _write_wide(wide_path, aggregated)
    metric_plot = _plot_metrics(root, aggregated)
    amp_plot = _plot_recon_grid(root, rows, "amplitude")
    phase_plot = _plot_recon_grid(root, rows, "phase")
    bundle = _zip_report(root, include_npz) if make_zip else None

    print("=" * 78)
    print(f"已汇总 {len(rows)} 次运行（{len(aggregated)} 个 overlap/condition 组合）")
    print(f"  {long_path}")
    print(f"  {mean_path}")
    print(f"  {wide_path}")
    for path in (metric_plot, amp_plot, phase_plot, bundle):
        if path is not None:
            print(f"  {path}")
    if errors:
        print(f"有 {len(errors)} 个 NPZ 读取失败：")
        for path, error in errors:
            print(f"  {path}: {error}")

    existing = {(r.get("target_overlap_pct"), r.get("condition_key")) for r in rows}
    # AD/DIP 和外部 PtyLab-ePIE 可以分别运行；只报告当前 root 中已经出现的矩阵，
    # 避免单独跑 ePIE 时提示 20 个无关的 AD/DIP 缺项。
    if any(r.get("condition_key") in CONDITION_ORDER for r in rows):
        missing = [(ov, condition) for ov in OVERLAP_CASES for condition in CONDITION_ORDER
                   if (ov, condition) not in existing]
        if missing:
            print(f"AD/DIP 5×4 矩阵仍缺 {len(missing)} 格（中途汇总时正常）：")
            print("  " + ", ".join(f"ov{ov}/{condition}" for ov, condition in missing))
        else:
            print("AD/DIP 5×4 实验矩阵完整。")
    if any(r.get("condition_key") == "ptylab_epie" for r in rows):
        missing_epie = [ov for ov in OVERLAP_CASES if (ov, "ptylab_epie") not in existing]
        if missing_epie:
            print("PtyLab-ePIE overlap 轴仍缺：" + ", ".join(f"ov{ov}" for ov in missing_epie))
        else:
            print("PtyLab-ePIE 五档 overlap 完整。")
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Colab batch runner and collector for ProPtyNet_paper overlap experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="运行 5 个 overlap × 4 个条件，并在结束时自动汇总",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run.add_argument("--root", default="runs_paper_overlap", help="结果根目录；Colab 推荐放在 Google Drive")
    run.add_argument("--overlaps", default="80,70,60,50,40", help="要跑的目标 overlap，逗号分隔")
    run.add_argument("--only", default=",".join(CONDITION_ORDER), help="要跑的条件，逗号分隔")
    run.add_argument("--iters", type=int, default=2000)
    run.add_argument("--obj-size", type=int, default=612,
                     help="object canvas size; fixed grid=4 needs at least 617 for the 40% case")
    run.add_argument("--fixed-grid", type=int, default=0,
                     help="use one grid×grid scan count for every overlap; 0 keeps the original per-overlap mapping")
    run.add_argument("--seeds", default="0", help="DIP seed，逗号分隔；AD 每格只跑一次")
    run.add_argument("--blind-probe-mode", choices=("support", "pixel"), default="support",
                     help="blind 的探针参数化；support 仍是盲重建，但已知针孔支撑")
    run.add_argument("--probe-init", choices=("ones", "disk"), default="ones")
    run.add_argument("--probe-init-sigma", type=float, default=0.15)
    run.add_argument("--fwd-chunk", type=int, default=8, help="前向位置分块；8 对 Colab 显存更稳")
    run.add_argument("--eval-every", type=int, default=25)
    run.add_argument("--eval-size", type=int, default=0,
                     help="fixed centered metric ROI for every run; 0 keeps adaptive illumination ROI")
    run.add_argument("--device", default="auto")
    run.add_argument("--noise", choices=("none", "gaussian", "poisson", "mixed"), default="none")
    run.add_argument("--snr", type=float, default=30.0)
    run.add_argument("--noise-seed", type=int, default=42)
    run.add_argument("--base-ch", type=int, default=None)
    run.add_argument("--lr-obj", type=float, default=None)
    run.add_argument("--lr-prb", type=float, default=None)
    run.add_argument("--lr-net", type=float, default=None)
    run.add_argument("--lr-probe", type=float, default=None)
    run.add_argument("--dip-lr-cosine", action="store_true", help="仅给 DIP 开启 cosine LR；默认关闭以保持基础对照")
    for key, default in TGV_DEFAULTS.items():
        run.add_argument("--" + key.replace("_", "-"), type=type(default), default=default,
                         help="net-only object amplitude TGV2 setting")
    run.add_argument("--dry-run", action="store_true", help="只打印命令，不创建结果或运行实验")
    run.add_argument("--force", action="store_true", help="覆盖同目录的现有单次结果；默认安全地跳过匹配结果")
    run.add_argument("--keep-going", action="store_true", help="某次失败后继续后续任务")
    run.add_argument("--verbose-logs", action="store_true", help="在 notebook 中显示完整日志；完整日志始终保存到文件")
    run.add_argument("--no-collect", action="store_true", help="运行结束后不自动汇总")
    run.add_argument("--include-npz", action="store_true", help="把体积较大的 NPZ 也放入 overlap_report.zip")

    collect = sub.add_parser(
        "collect",
        help="只读取已有 NPZ 并重新生成 CSV/图/ZIP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    collect.add_argument("--root", default="runs_paper_overlap")
    collect.add_argument("--include-npz", action="store_true", help="把 NPZ 也加入 ZIP")
    collect.add_argument("--no-zip", action="store_true")
    collect.add_argument("--recompute-eval-size", type=int, default=0,
                         help="recompute final object metrics/plots on one centered square without modifying NPZ")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        return run_matrix(args)
    return collect_results(
        Path(args.root),
        include_npz=args.include_npz,
        make_zip=not args.no_zip,
        recompute_eval_size=args.recompute_eval_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
