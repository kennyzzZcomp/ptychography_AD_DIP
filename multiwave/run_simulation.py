"""Run with python -m multiwave.run_simulation, or directly by file path."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import hashlib
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multiwave.config import Config, METHODS, SCENES, LEGACY_METHODS, FEEDBACK_METHODS
from multiwave.physics import MultiwaveOperator
from multiwave.scene import simulate
from multiwave.reconstruct import reconstruct
from multiwave.report import plot_scene, plot_result, plot_probes, save_summary


@torch.no_grad()
def audit_scene(cfg, scene):
    op = scene.operator
    larger = MultiwaveOperator(replace(cfg, pad_factor=2*cfg.pad_factor), op.probes, op.positions)
    reference = larger(scene.objects)
    difference = float((reference-scene.clean).norm()/reference.norm())
    return {"padding_relative_intensity_difference": difference,
            "padding_checked": [cfg.pad_factor, 2*cfg.pad_factor],
            "transfer_retained_frequency_fraction": (op.transfer.abs() > 0).float().mean((-1, -2)).cpu().tolist(),
            "probe_powers": op.probes.abs().square().sum((-1, -2)).cpu().tolist(),
            "train_count": len(scene.train), "holdout_count": len(scene.holdout),
            "roi_pixels": int(scene.roi.sum()),
            "mean_detected_photons_per_scan": float(scene.clean.sum((-1, -2)).mean()*cfg.photons_per_scan),
            "known_probes": cfg.probe_mode == "known", "spectral_mode": cfg.spectral_mode,
            "explicit_spectral_weights": cfg.spectral_mode == "weighted",
            "holdout_excluded_from_network_input": True}


def run_experiment(cfg, methods=METHODS, outdir=None, progress_callback=None):
    cfg.validate()
    if cfg.probe_mode in ("pixel", "basis") and cfg.scene != "usaf_zero_phase":
        raise ValueError("blind pixel probes are currently scoped to the zero-phase USAF task")
    methods = tuple(methods)
    allowed = METHODS + LEGACY_METHODS + FEEDBACK_METHODS
    if not methods or len(set(methods)) != len(methods) or any(m not in allowed for m in methods):
        raise ValueError(f"methods must be distinct members of {allowed}")
    if cfg.scene != "usaf_zero_phase" and any(m in METHODS for m in methods):
        raise ValueError("shared_amp methods require --scene usaf_zero_phase; archived scenes need explicit legacy methods")
    if (cfg.atv_weight or cfg.tgv_weight or cfg.unet_skip != "concat") and any(m not in METHODS for m in methods):
        raise ValueError("DWT/ATV/TGV options require shared_amp methods")
    if cfg.unet_skip != "concat" and "unet_shared_amp" not in methods:
        raise ValueError("--unet-skip requires unet_shared_amp in --methods")
    if cfg.unet_detail != "none" and ("unet_shared_amp" not in methods or any(m not in METHODS for m in methods)):
        raise ValueError("--unet-detail requires shared_amp methods including unet_shared_amp")
    if any(m in FEEDBACK_METHODS for m in methods):
        if cfg.probe_mode == "basis":
            raise ValueError("basis probes are not enabled for feedback pilot")
        if cfg.probe_smooth_weight:
            raise ValueError("probe smoothness is not enabled for feedback pilot")
        if (cfg.scene != "usaf_zero_phase" or cfg.loss != "poisson" or cfg.tv_weight
                or cfg.atv_weight or cfg.tgv_weight or cfg.unet_detail != "none" or cfg.unet_skip != "concat"
                or cfg.lr_net_decay_after):
            raise ValueError("feedback requires zero-phase Poisson, concat, no detail/TV/ATV/TGV/LR decay")
    torch.set_num_threads(cfg.threads)
    device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu or auto")
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    out = Path(outdir) if outdir is not None else Path(__file__).parent/"results"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty run directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    scene = simulate(cfg, device)
    audit = audit_scene(cfg, scene)
    if audit["padding_relative_intensity_difference"] > .01:
        print("WARNING: padding intensity difference exceeds 1%; increase --pad-factor before interpreting results.", flush=True)
    metadata = {"config": asdict(cfg), "methods": methods, "actual_device": device,
                "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                "platform": platform.platform(), "audit": audit,
                "timestamp": datetime.now().astimezone().isoformat(), "command": sys.argv}
    if cfg.spectral_mode == "equal_power":
        metadata["config"].pop("weights")
        metadata["probe_power_per_mode"] = cfg.probe_power
    if cfg.scene == "usaf_zero_phase":
        asset = Path(cfg.usaf_path) if cfg.usaf_path else Path(__file__).resolve().parents[1]/"USAF.jpg"
        metadata["object_model"] = "one_shared_real_amplitude_zero_phase"
        metadata["usaf_asset"] = {"path": str(asset.resolve()), "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                                  "grayscale_convention": "amplitude", "resize": "BOX, aspect ratio preserved"}
    (out/"config.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    arr = lambda t: t.detach().cpu().numpy()
    np.savez_compressed(out/"data.npz", objects=arr(scene.objects), opd_um=arr(scene.opd_um),
                        optical_depth=arr(scene.optical_depth), probes=arr(scene.operator.probes),
                        positions_px=arr(scene.operator.positions),
                        positions_m=arr(scene.operator.positions)*cfg.pixel_um*1e-6,
                        intensity_clean=arr(scene.clean), intensity_measured=arr(scene.measured),
                        train_indices=arr(scene.train), holdout_indices=arr(scene.holdout),
                        roi=arr(scene.roi), markers=arr(scene.markers),
                        wavelengths_nm=np.array(cfg.wavelengths_nm), mixing_coefficients=np.array(cfg.mixing_coefficients))
    print(f"Device={device}, probes={cfg.probe_mode}, spectrum={cfg.spectral_mode}, {len(cfg.wavelengths_nm)} wavelengths, "
          f"train/holdout={len(scene.train)}/{len(scene.holdout)}, "
          f"padding difference={audit['padding_relative_intensity_difference']:.3g}", flush=True)
    plot_scene(scene, cfg, out)
    results = []
    for method in methods:
        result = reconstruct(cfg, scene, method, progress_callback=progress_callback)
        np.savez_compressed(out/f"{method}_fields.npz", **{k: result[k] for k in
                            ("objects", "optical_depth", "opd_um", "predicted_intensity", "probes", "initial_probes")})
        torch.save({"method": method, "config": metadata["config"], "state_dict": result["state_dict"],
                    "probe_state_dict": result["probe_state_dict"], "tgv_state_dict": result["tgv_state_dict"]},
                   out/f"{method}_model.pth")
        plot_result(result, scene, cfg, out)
        plot_probes(result, scene, cfg, out)
        results.append(result)
    summary = save_summary(results, cfg, audit, out)
    print(f"Saved: {out.resolve()}", flush=True)
    return out.resolve(), summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("smoke", "standard", "highres", "resolved"), default="smoke")
    p.add_argument("--pixel-parameterization", choices=("sigmoid", "softplus", "direct"))
    p.add_argument("--unet-activation", choices=("sigmoid", "softplus"))
    p.add_argument("--unet-skip", choices=("concat", "dwt_concat"))
    p.add_argument("--unet-detail", choices=("none", "residual"),
                   help="optional full-resolution signed residual amplitude branch")
    p.add_argument("--feedback-mode", choices=("learned", "identity", "no_gradient"))
    p.add_argument("--feedback-step", type=float)
    p.add_argument("--loss", choices=("amplitude", "poisson"),
                   help="training data loss; poisson requires a positive photon budget")
    p.add_argument("--methods", nargs="+", choices=METHODS+LEGACY_METHODS+FEEDBACK_METHODS, default=list(METHODS))
    p.add_argument("--scene", choices=SCENES)
    p.add_argument("--usaf-path", type=str)
    p.add_argument("--wavelengths-nm", nargs="+", type=float)
    p.add_argument("--weights", nargs="+", type=float)
    p.add_argument("--probe-mode", choices=("pixel", "known", "basis"))
    p.add_argument("--spectral-mode", choices=("equal_power", "weighted"))
    p.add_argument("--outdir", type=Path)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"))
    for name in ("iterations", "eval_every", "object_size", "patch_size", "grid", "step", "jitter",
                 "base_channels", "pad_factor", "chunk", "threads", "scene_seed", "noise_seed", "network_seed", "scan_quantum", "detector_size", "lr_net_decay_after", "tgv_inner_steps", "probe_grid_size", "probe_amp_order", "probe_phase_order"):
        p.add_argument("--"+name.replace("_", "-"), type=int)
    for name in ("pixel_um", "distance_mm", "photons_per_scan", "lr_pixel", "lr_net", "tv_weight", "atv_weight",
                 "opd_scale_um", "holdout_fraction", "usaf_fill", "lr_probe", "lr_net_decay_factor",
                 "tgv_weight", "tgv_alpha0", "tgv_alpha1", "tgv_eps", "tgv_lr", "probe_smooth_weight"):
        p.add_argument("--"+name.replace("_", "-"), type=float)
    args = p.parse_args()
    if args.weights is not None and args.spectral_mode != "weighted":
        p.error("--weights is legacy-only: explicitly select --spectral-mode weighted. Default blind mode has no spectral weights.")
    cfg = Config.preset(args.preset)
    for key, value in vars(args).items():
        if hasattr(cfg, key) and value is not None:
            setattr(cfg, key, tuple(value) if key in ("weights", "wavelengths_nm") else value)
    run_experiment(cfg, args.methods, args.outdir)


if __name__ == "__main__":
    main()
