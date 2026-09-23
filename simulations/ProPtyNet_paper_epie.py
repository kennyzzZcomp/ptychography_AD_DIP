#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the official PtyLab ePIE engine on ProPtyNet_paper simulation data.

This is an external-baseline adapter, not an ePIE reimplementation.  The script:

1. calls ``functions.paperrepro.scene.build_scene`` so that the object, physical
   probe, scan positions, diffraction intensities, noise, and geometry are the
   same as ProPtyNet_paper.py;
2. converts those arrays to PtyLab's documented CPM HDF5 input format;
3. runs the unmodified ``PtyLab.Engines.ePIE`` with its stock blind settings;
4. converts the effective probe back to the physical probe convention and saves
   an ``epie_result.npz`` for subsequent analysis.

Why PtyLab uses Fraunhofer here
-------------------------------
ProPtyNet_paper's one-step Fresnel field is

    U = FFT(object_patch * probe * Q),

where Q is a fixed quadratic phase.  Defining ``effective_probe = probe * Q``
makes this exactly PtyLab's centered unitary Fraunhofer operator.  This avoids a
quadratic-phase sign/grid convention mismatch without changing ePIE itself.  On
output, ``physical_probe = effective_probe * conj(Q)``.

Examples (run from the repository root):

    # One overlap, useful as a quick first check
    python simulations/ProPtyNet_paper_epie.py run \
        --step-px 24 --grid 4 --epochs 50 --outdir runs_paper_overlap/ov60/ptylab_epie/seed_0

    # Five overlap cases; completed NPZ files
    # are skipped, so the same command safely resumes after a Colab disconnect.
    python simulations/ProPtyNet_paper_epie.py sweep \
        --root runs_paper_overlap --epochs 50
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SIM_DIR))

from ProPtyNet_paper import Cfg, PRESETS  # noqa: E402
from functions.paperrepro.evaluate import evaluate, probe_relerr  # noqa: E402
from functions.paperrepro.optics import forward_ptycho, make_quad_phase  # noqa: E402
from functions.paperrepro.report import _save  # noqa: E402
from functions.paperrepro.scene import build_scene  # noqa: E402
OVERLAP_CASES = {
    80: (12, 8), 70: (18, 6), 60: (24, 4), 50: (29, 4), 40: (35, 3),
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _parse_int_csv(raw: str) -> list[int]:
    try:
        values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"必须是逗号分隔的整数：{raw!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("列表不能为空")
    return values


def _parse_overlaps(raw: str) -> list[int]:
    values = _parse_int_csv(raw)
    bad = [x for x in values if x not in OVERLAP_CASES]
    if bad:
        raise argparse.ArgumentTypeError(
            f"不支持 overlap {bad}；可选 {sorted(OVERLAP_CASES, reverse=True)}"
        )
    return values


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    tmp.replace(path)


def _require_ptylab():
    try:
        import PtyLab
        from PtyLab import Engines
        from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
        from PtyLab.Monitor.Monitor import DummyMonitor
        from PtyLab.Params.Params import Params
        from PtyLab.Reconstruction.Reconstruction import Reconstruction
    except ImportError as exc:
        raise SystemExit(
            "找不到 PtyLab。Colab 中请保留它自带的 IPython，只安装 GPU/读数据依赖和 PtyLab 本体：\n"
            "  pip install 'cupy-cuda12x[ctk]' 'tables>=3.8,<4'\n"
            "  pip install --no-deps 'ptylab @ git+https://github.com/PtyLab/PtyLab.py.git@1c7a0f52c55839603a2bcb23204d1672379cd8c3'\n"
            f"原始错误：{exc}"
        ) from exc
    return PtyLab, Engines, ExperimentalData, DummyMonitor, Params, Reconstruction


def _make_cfg(args: argparse.Namespace) -> Cfg:
    values = dict(PRESETS["paper"])
    values.update(
        preset="paper",
        obj_size=args.obj_size,
        step_px=args.step_px,
        grid=args.grid,
        noise=args.noise,
        snr_db=args.snr,
        noise_seed=args.noise_seed,
        seed=args.seed,
        device=args.torch_device,
        outdir=str(Path(args.outdir).expanduser().resolve()),
        iters=args.epochs,
        eval_every=args.epochs,
        eval_size=args.eval_size,
        amp_image=args.amp_image,
        phs_image=args.phs_image,
        obj_phase_rad=args.obj_phase_rad,
    )
    cfg = Cfg(**values)
    cfg.quad_sign = args.quad_sign
    return cfg


def _write_ptylab_input(path: Path, cfg: Cfg, intensities: np.ndarray,
                        positions: np.ndarray) -> np.ndarray:
    """Write the official PtyLab CPM fields and return encoder coordinates."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("缺少 h5py；安装 PtyLab 时会自动安装该依赖") from exc
    # PtyLab: pixel_pos = round(encoder/dxo) + No//2 - Np//2.
    # Solve this equation for encoder so its top-left patch indices equal cfg.pos.
    origin = np.array(
        [cfg.obj_size // 2 - cfg.N // 2, cfg.obj_size // 2 - cfg.N // 2],
        dtype=np.float64,
    )
    encoder = (positions.astype(np.float64) - origin[None]) * cfg.dx1
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hf:
        # Keep this uncompressed: diffraction stacks are short-lived by default,
        # and compression noticeably slows Colab/Drive I/O without affecting ePIE.
        hf.create_dataset("ptychogram", data=intensities.astype(np.float32))
        hf.create_dataset("encoder", data=encoder.astype(np.float64))
        hf.create_dataset("wavelength", data=np.float64(cfg.wlength))
        hf.create_dataset("dxd", data=np.float64(cfg.det_pixel))
        hf.create_dataset("zo", data=np.float64(cfg.z))
        hf.create_dataset("entrancePupilDiameter", data=np.float64(cfg.probe_diam_um * 1e-6))
        hf.create_dataset("orientation", data=np.int32(0))
        # Extra audit fields are ignored by PtyLab's loader.
        hf.create_dataset("object_size", data=np.int32(cfg.obj_size))
        hf.create_dataset("step_px", data=np.int32(cfg.step_px))
        hf.create_dataset("grid", data=np.int32(cfg.grid))
    return encoder


def _set_ptylab_device(params: Any, choice: str) -> str:
    if choice == "cpu":
        params.gpuSwitch = False
        return "cpu"
    if choice == "cuda":
        try:
            params.gpuSwitch = True
        except (AttributeError, ImportError) as exc:
            raise RuntimeError(
                "--ptylab-device cuda 需要可用的 CUDA 和 CuPy；Colab 请安装 ptylab[gpu]"
            ) from exc
        return "cuda"
    return "cuda" if bool(params.gpuSwitch) else "cpu"


def _final_forward_error(cfg: Cfg, rec: np.ndarray, probe: np.ndarray,
                         positions: np.ndarray, target: np.ndarray) -> float:
    device = cfg.dev()
    q = make_quad_phase(cfg, device)
    with torch.no_grad():
        pred = forward_ptycho(
            torch.from_numpy(rec.astype(np.complex64)).to(device),
            torch.from_numpy(probe.astype(np.complex64)).to(device),
            torch.from_numpy(positions.astype(np.int64)).to(device),
            q,
            cfg.N,
            chunk=8,
        ).cpu().numpy()
    # Same "real" convention as solvers_addip.py: unnormalised L2 distance to
    # the clean simulated intensity, even when the fitted input contains noise.
    return float(np.linalg.norm(pred - target))


def _run_with_dataset(args: argparse.Namespace, cfg: Cfg, dataset_path: Path) -> int:
    PtyLab, Engines, ExperimentalData, DummyMonitor, Params, Reconstruction = _require_ptylab()

    device = cfg.dev()
    sc = build_scene(cfg, device)
    _write_ptylab_input(dataset_path, cfg, sc.Im, sc.pos)
    # The scene object also keeps Torch copies of the full diffraction stack.
    # PtyLab/CuPy needs the same GPU next, so release those duplicate tensors.
    for name in ("Q", "post", "Imt", "Iclt", "S1t", "sqrtIm"):
        if hasattr(sc, name):
            delattr(sc, name)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    data = ExperimentalData(dataset_path, operationMode="CPM")
    params = Params()
    ptylab_device = _set_ptylab_device(params, args.ptylab_device)
    params.propagatorType = "Fraunhofer"
    params.intensityConstraint = "standard"
    params.positionOrder = "random"
    # Explicitly retain the stock ePIE baseline: no support/TV/position correction.
    params.probeBoundary = False
    params.absorbingProbeBoundary = False
    params.probeSmoothenessSwitch = False
    params.objectSmoothenessSwitch = False
    params.objectTVregSwitch = False
    params.positionCorrectionSwitch = False

    monitor = DummyMonitor()
    # PtyLab's stock "ones" and "circ" initializers add 0.001 random noise.
    # Seed before initialization so both that noise and random scan order reproduce.
    np.random.seed(args.seed)
    reconstruction = Reconstruction(data, params)
    # PtyLab intentionally chooses a generous automatic canvas.  The benchmark
    # must instead use the same 612x612 canvas as ProPtyNet_paper.
    reconstruction.No = cfg.obj_size
    reconstruction.encoder_corrected = data.encoder.copy()
    reconstruction.positions0 = reconstruction.positions.copy()
    reconstruction.initialObject = "ones"
    reconstruction.initialProbe = "circ"
    reconstruction.initializeObjectProbe()

    got_positions = np.asarray(reconstruction.positions, dtype=np.int64)
    if not np.array_equal(got_positions, sc.pos):
        max_err = int(np.abs(got_positions - sc.pos).max())
        raise RuntimeError(f"PtyLab 扫描坐标转换失败，最大偏差 {max_err} px")

    engine = Engines.ePIE(reconstruction, data, params, monitor)
    engine.numIterations = args.epochs
    engine.betaObject = args.beta_object
    engine.betaProbe = args.beta_probe

    try:
        version = importlib.metadata.version("ptylab")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(PtyLab, "__version__", "unknown")
    actual_overlap = 100.0 * (1.0 - cfg.step_px / cfg.probe_diam_px)
    print("=" * 78)
    print(f"PtyLab {version} official Engines.ePIE | device={ptylab_device}")
    print(f"target geometry: step={cfg.step_px}px grid={cfg.grid}x{cfg.grid} "
          f"actual overlap={actual_overlap:.2f}%")
    print(f"epochs={args.epochs} betaObject={args.beta_object:g} betaProbe={args.beta_probe:g}")
    print("initialObject=ones initialProbe=circ constraint=standard propagator=Fraunhofer")
    print("probe/object support, TV, position correction: OFF")
    print("=" * 78, flush=True)

    t0 = time.time()
    engine.reconstruct()
    elapsed = time.time() - t0

    rec = np.squeeze(np.asarray(reconstruction.object)).astype(np.complex64)
    effective_probe = np.squeeze(np.asarray(reconstruction.probe)).astype(np.complex64)
    if ptylab_device == "cuda":
        # ePIE has copied its outputs back to NumPy; release CuPy's cached blocks
        # before the Torch-based common forward-error calculation below.
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except ImportError:
            pass
    q_np = make_quad_phase(cfg, torch.device("cpu")).cpu().numpy()
    physical_probe = (effective_probe * np.conj(q_np)).astype(np.complex64)
    rs, cs = sc.roi
    metrics = evaluate(rec[rs, cs], sc.obj[rs, cs])
    p_error = probe_relerr(physical_probe, sc.probe)
    data_error = _final_forward_error(cfg, rec, physical_probe, sc.pos, sc.Icl)
    engine_error = float(np.asarray(reconstruction.error)[-1]) if len(reconstruction.error) else float("nan")
    hist = [{
        "it": args.epochs,
        "loss": engine_error,
        "real": data_error,
        "ptylab_error": engine_error,
        "relerr_p": p_error,
        **metrics,
    }]

    # Reuse the repository's exact report layout and then enrich cfg in the NPZ
    # with external-baseline provenance that Cfg.asdict cannot carry dynamically.
    _save(cfg, rec, physical_probe, sc.obj, sc.probe, hist, sc.roi, sc.pos, tag="epie")
    cfg_out = asdict(cfg)
    cfg_out.update({
        "solver": "PtyLab.Engines.ePIE",
        "ptylab_version": version,
        "epie_epochs": args.epochs,
        "beta_object": args.beta_object,
        "beta_probe": args.beta_probe,
        "probe_mode": "ptylab_blind",
        "initial_object": "ones",
        "initial_probe": "circ",
        "position_order": "random",
        "intensity_constraint": "standard",
        "ptylab_propagator": "Fraunhofer",
        "fresnel_bridge": "effective_probe=physical_probe*Q",
        "ptylab_device": ptylab_device,
        "elapsed_sec": elapsed,
        "scene_fingerprint": sc.fp,
        "quad_sign": cfg.quad_sign,
    })
    out_npz = Path(cfg.outdir) / "epie_result.npz"
    np.savez_compressed(
        out_npz,
        obj_rec=rec,
        obj_gt=sc.obj,
        probe_rec=physical_probe,
        probe_gt=sc.probe,
        probe_effective_rec=effective_probe,
        roi=np.array([rs.start, rs.stop, cs.start, cs.stop]),
        positions=sc.pos,
        hist=json.dumps(hist),
        cfg=json.dumps(cfg_out, default=str),
    )
    print(f"[epie] 用时 {elapsed:.1f}s / {args.epochs} epochs")
    print(f"[epie] amp SSIM {metrics['ssim_amp']:.4f} | phase SSIM {metrics['ssim_phs']:.4f} "
          f"| relerr {metrics['relerr']:.4f} | probe err {p_error:.4f}")
    print(f"[epie] result -> {out_npz}")
    return 0


def run_one(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = _make_cfg(args)
    if args.save_dataset:
        dataset = outdir / "ptylab_input.hdf5"
        return _run_with_dataset(args, cfg, dataset)
    with tempfile.TemporaryDirectory(prefix="proptynet_ptylab_") as temp_dir:
        return _run_with_dataset(args, cfg, Path(temp_dir) / "ptylab_input.hdf5")


def _stream_subprocess(command: list[str], log_path: Path, verbose: bool) -> int:
    interesting = re.compile(
        r"^(PtyLab |\[scene\]|\[epie\]|target geometry|epochs=|initialObject=|probe/object|ePIE:)"
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
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
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if verbose or interesting.search(line):
                print("    " + line.rstrip(), flush=True)
        return proc.wait()


def _single_command(args: argparse.Namespace, target: int, step: int, grid: int,
                    seed: int, outdir: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "run",
        "--step-px", str(step),
        "--grid", str(grid),
        "--obj-size", str(args.obj_size),
        "--epochs", str(args.epochs),
        "--eval-size", str(args.eval_size),
        "--beta-object", str(args.beta_object),
        "--beta-probe", str(args.beta_probe),
        "--seed", str(seed),
        "--torch-device", args.torch_device,
        "--ptylab-device", args.ptylab_device,
        "--noise", args.noise,
        "--snr", str(args.snr),
        "--noise-seed", str(args.noise_seed),
        "--quad-sign", str(args.quad_sign),
        "--amp-image", args.amp_image,
        "--phs-image", args.phs_image,
        "--obj-phase-rad", str(args.obj_phase_rad),
        "--outdir", str(outdir),
    ]
    if args.save_dataset:
        command.append("--save-dataset")
    return command


def _resume_mismatches(result: Path, args: argparse.Namespace, step: int,
                       grid: int, seed: int) -> list[str]:
    """Return scientific settings that differ from an existing result."""
    try:
        with np.load(result, allow_pickle=False) as data:
            saved = json.loads(str(data["cfg"].item()))
    except Exception as exc:
        return [f"cannot read saved cfg ({exc})"]
    expected = {
        "step_px": step,
        "grid": grid,
        "obj_size": args.obj_size,
        "seed": seed,
        "epie_epochs": args.epochs,
        "eval_size": args.eval_size,
        "beta_object": args.beta_object,
        "beta_probe": args.beta_probe,
        "noise": args.noise,
        "snr_db": args.snr,
        "noise_seed": args.noise_seed,
        "quad_sign": args.quad_sign,
        "amp_image": args.amp_image,
        "phs_image": args.phs_image,
        "obj_phase_rad": args.obj_phase_rad,
    }
    mismatches = []
    for key, wanted in expected.items():
        got = saved.get(key, "<missing>")
        same = (
            math.isclose(float(got), float(wanted), rel_tol=1e-12, abs_tol=1e-12)
            if isinstance(wanted, float) and got != "<missing>"
            else got == wanted
        )
        if not same:
            mismatches.append(f"{key}: saved={got!r}, requested={wanted!r}")
    return mismatches


def run_sweep(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    overlaps = _parse_overlaps(args.overlaps)
    seeds = _parse_int_csv(args.seeds)
    jobs = []
    for target in overlaps:
        step, default_grid = OVERLAP_CASES[target]
        grid = args.fixed_grid or default_grid
        scan_span = 512 + (grid - 1) * step
        if scan_span > args.obj_size:
            raise SystemExit(
                f"ov{target}: scan span {scan_span} exceeds obj_size={args.obj_size}; "
                "increase --obj-size or lower --fixed-grid"
            )
        for seed in seeds:
            outdir = root / f"ov{target:02d}" / "ptylab_epie" / f"seed_{seed}"
            jobs.append((target, step, grid, seed, outdir,
                         _single_command(args, target, step, grid, seed, outdir)))

    print("=" * 78)
    print(f"PtyLab-ePIE overlap sweep: {len(jobs)} runs")
    print(f"root={root} overlaps={overlaps} seeds={seeds} epochs={args.epochs}")
    print("official blind ePIE only; no known probe, support, TV, or position correction")
    print("=" * 78)
    if args.dry_run:
        for i, (target, step, grid, seed, outdir, command) in enumerate(jobs, 1):
            print(f"[{i:02d}/{len(jobs):02d}] ov{target} step={step} grid={grid} seed={seed}")
            print("    " + " ".join(command))
        return 0

    _require_ptylab()  # fail before starting the matrix if Colab setup is incomplete

    failures = []
    for i, (target, step, grid, seed, outdir, command) in enumerate(jobs, 1):
        result = outdir / "epie_result.npz"
        title = f"[{i:02d}/{len(jobs):02d}] ov{target} step={step} grid={grid} seed={seed}"
        if result.exists() and not args.force:
            mismatches = _resume_mismatches(result, args, step, grid, seed)
            if not mismatches:
                print(title + " -> 已完成且设置一致，跳过")
                continue
            print(title + " -> 旧结果的设置与本次命令不同：")
            for mismatch in mismatches:
                print("    " + mismatch)
            print("请换一个 --root，或确认覆盖时加 --force。")
            return 2
        if result.exists() and args.force:
            result.unlink()
            figure = outdir / "epie_result.png"
            if figure.exists():
                figure.unlink()
        outdir.mkdir(parents=True, exist_ok=True)
        actual = 100.0 * (1.0 - step / (800e-6 / (632e-9 * 0.165 / (512 * 15.04e-6))))
        manifest = {
            "schema_version": 1,
            "status": "running",
            "started_at": _now(),
            "target_overlap_pct": target,
            "actual_overlap_pct": actual,
            "step_px": step,
            "grid": grid,
            "n_patterns": grid * grid,
            "condition_key": "ptylab_epie",
            "condition_label": "PtyLab-ePIE",
            "mode": "epie",
            "probe_knowledge": "blind",
            "probe_mode": "ptylab_blind",
            "seed": seed,
            "outdir": str(outdir),
            "command": command,
        }
        _json_dump(outdir / "run_spec.json", manifest)
        print(title + " -> 开始")
        try:
            code = _stream_subprocess(command, outdir / "log.txt", args.verbose_logs)
        except KeyboardInterrupt:
            manifest.update(status="interrupted", finished_at=_now())
            _json_dump(outdir / "run_spec.json", manifest)
            print("收到中断；再次执行相同命令即可续跑。")
            return 130
        ok = code == 0 and result.exists()
        manifest.update(status="complete" if ok else "failed", returncode=code, finished_at=_now())
        _json_dump(outdir / "run_spec.json", manifest)
        if not ok:
            failures.append((outdir, code))
            print(f"    失败；查看 {outdir / 'log.txt'}")
            if not args.keep_going:
                break
        else:
            print(f"    完成: {result}")

    if failures:
        for path, code in failures:
            print(f"[{code}] {path}")
        return 1
    return 0


def _add_scene_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=50, help="ePIE full scan passes; PtyLab default is 50")
    parser.add_argument("--obj-size", type=int, default=612)
    parser.add_argument("--eval-size", type=int, default=0,
                        help="fixed centered metric ROI; 0 keeps adaptive illumination ROI")
    parser.add_argument("--beta-object", type=float, default=0.25, help="PtyLab ePIE default")
    parser.add_argument("--beta-probe", type=float, default=0.25, help="PtyLab ePIE default")
    parser.add_argument("--seed", type=int, default=0, help="PtyLab random scan-order seed")
    parser.add_argument("--torch-device", default="auto", help="device used only to generate/check the simulation")
    parser.add_argument("--ptylab-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--noise", choices=("none", "gaussian", "poisson", "mixed"), default="none")
    parser.add_argument("--snr", type=float, default=30.0)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--quad-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--amp-image", default="USAF.jpg")
    parser.add_argument("--phs-image", default="siemens")
    parser.add_argument("--obj-phase-rad", type=float, default=0.8)
    parser.add_argument("--save-dataset", action="store_true", help="keep the large PtyLab input HDF5 in each result directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official PtyLab ePIE baseline on ProPtyNet_paper simulation data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    one.add_argument("--step-px", type=int, required=True)
    one.add_argument("--grid", type=int, required=True)
    one.add_argument("--outdir", required=True)
    _add_scene_args(one)

    sweep = sub.add_parser("sweep", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sweep.add_argument("--root", default="runs_paper_overlap")
    sweep.add_argument("--overlaps", default="80,70,60,50,40")
    sweep.add_argument("--fixed-grid", type=int, default=0,
                       help="use one grid×grid scan count at every overlap; 0 keeps original mapping")
    sweep.add_argument("--seeds", default="0", help="comma-separated ePIE scan-order seeds")
    _add_scene_args(sweep)
    sweep.add_argument("--dry-run", action="store_true")
    sweep.add_argument("--force", action="store_true")
    sweep.add_argument("--keep-going", action="store_true")
    sweep.add_argument("--verbose-logs", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        return run_one(args)
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
