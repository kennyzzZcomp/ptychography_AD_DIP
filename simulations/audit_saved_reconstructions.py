"""Read-only saved-array audit. No torch, simulation, training or source changes.

python simulations/audit_saved_reconstructions.py --directory /path/to/results
Optional --output creates a NEW JSON file and refuses to overwrite any file.
Missing truth/positions stay missing: historical scenes are never regenerated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from functions.paperrepro.evaluate import evaluate, probe_relerr


def array_sha(a):
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def center_roi(shape, size):
    if len(shape) != 2 or min(shape) < size or size < 7:
        raise ValueError("Need a 2D object and a central ROI of at least 7 pixels")
    y, x = ((n - size) // 2 for n in shape)
    return (slice(y, y + size), slice(x, x + size))


def weighted_overlap(probe, step):
    """Intensity autocorrelation for an integer axial displacement, zero outside.

    A descriptive coupling proxy, NOT linear/areal overlap or identifiability.
    """
    w = np.abs(probe).astype(np.float64) ** 2
    den = np.sum(w * w)
    if step < 0 or int(step) != step or den == 0:
        raise ValueError("Requires nonnegative integer displacement and nonzero probe")
    d = int(step)
    if d == 0:
        return [1.0, 1.0]
    return [float(np.sum(w[d:, :] * w[:-d, :]) / den) if d < w.shape[0] else 0.,
            float(np.sum(w[:, d:] * w[:, :-d]) / den) if d < w.shape[1] else 0.]


def probe_diagnostics(rec, gt):
    w = np.abs(gt).astype(np.float64) ** 2
    y, x = np.indices(gt.shape)
    r = np.hypot(y - gt.shape[0] / 2, x - gt.shape[1] / 2)
    ix = np.argsort(r.ravel())
    cumulative = np.cumsum(w.ravel()[ix]) / w.sum()
    support = w > 0
    wr = np.abs(rec).astype(np.float64) ** 2
    return {
        "full_relative_error": probe_relerr(rec, gt),
        "gt_support_relative_error_separately_aligned": probe_relerr(rec[support], gt[support]),
        "outside_gt_support_energy_fraction": float(wr[~support].sum() / wr.sum()) if wr.sum() else None,
        "centered_energy_diameters_px": {
            str(f): float(2 * r.ravel()[ix[min(np.searchsorted(cumulative, f), len(ix)-1)]])
            for f in (.90, .95, .99)},
        "gt_support_pixels": int(support.sum()),
        "warning": "GT-assisted diagnostics; masked error has its own complex alignment, not the official full error.",
    }


def coverage_stats(shape, probe, positions, roi):
    if not np.issubdtype(positions.dtype, np.integer):
        return {"unavailable": "Only stored integer top-left positions are supported"}
    cov = np.zeros(shape, np.float64)
    count = np.zeros(shape, np.int32)
    w = np.abs(probe).astype(np.float64) ** 2
    h, k = w.shape
    for y, x in positions:
        if y < 0 or x < 0 or y+h > shape[0] or x+k > shape[1]:
            raise ValueError("Stored top-left position outside canvas")
        cov[y:y+h, x:x+k] += w
        count[y:y+h, x:x+k] += w > 0
    c = cov[roi]
    return {"roi_min_support_count": int(count[roi].min()),
            "roi_max_support_count": int(count[roi].max()),
            "roi_mean_support_count": float(count[roi].mean()),
            "roi_unilluminated_fraction": float(np.mean(c == 0)),
            "roi_coverage_min_over_max": float(c.min() / c.max()) if c.max() else None}


def inspect_file(path, size=96):
    with np.load(path, allow_pickle=False) as z:
        cfg = json.loads(str(z["cfg"])) if "cfg" in z.files else {}
        hist = json.loads(str(z["hist"])) if "hist" in z.files else []
        out = {"file": str(path.resolve()), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "config": cfg, "keys": z.files,
               "last_saved_history": hist[-1] if isinstance(hist, list) and hist else None,
               "array_sha256": {}, "finite": {}, "warnings": []}
        for key in ("obj_rec", "obj_gt", "probe_rec", "probe_gt", "positions"):
            if key in z.files:
                out["array_sha256"][key] = array_sha(z[key])
                out["finite"][key] = bool(np.isfinite(z[key]).all())
        if all(k in z.files for k in ("obj_rec", "obj_gt")):
            roi = center_roi(z["obj_gt"].shape, size)
            out["central_roi"] = [roi[0].start, roi[0].stop, roi[1].start, roi[1].stop]
            out["central_metrics_paper_protocol"] = evaluate(z["obj_rec"][roi], z["obj_gt"][roi])
            if "roi" in z.files:
                a, b, c, d = (int(v) for v in z["roi"])
                out["stored_roi"] = [a, b, c, d]
                out["stored_roi_metrics"] = evaluate(z["obj_rec"][a:b, c:d], z["obj_gt"][a:b, c:d])
            if "probe_gt" in z.files and "positions" in z.files:
                out["central_coverage"] = coverage_stats(z["obj_gt"].shape, z["probe_gt"], z["positions"], roi)
        else:
            out["warnings"].append("No stored object truth: cannot re-evaluate; history is not newly verified ground truth.")
        if all(k in z.files for k in ("probe_rec", "probe_gt")):
            out["probe_diagnostics"] = probe_diagnostics(z["probe_rec"], z["probe_gt"])
            if "step_px" in cfg:
                out["probe_diagnostics"]["intensity_autocorrelation_at_step_y_x"] = weighted_overlap(z["probe_gt"], cfg["step_px"])
        if "probe_diam_um" in cfg:
            dx = cfg["wlength"] * cfg["z"] / (cfg["N"] * cfg["det_pixel"])
            diameter = cfg["probe_diam_um"] * 1e-6 / dx
            out["nominal_geometry"] = {"diameter_px": diameter, "dx_m": dx,
                                       "linear_overlap": 1 - cfg["step_px"] / diameter}
        elif "probe_dia" in cfg:
            out["nominal_geometry"] = {"configured_diameter_px": cfg["probe_dia"],
                                       "linear_overlap_from_config_only": 1 - cfg["scan_step"] / cfg["probe_dia"]}
        if "positions" not in z.files:
            out["warnings"].append("Actual scan positions absent; do not regenerate from current code to represent an old run.")
        out["warnings"].append("No saved solver revision; current source audit does not prove the historical execution path.")
        return out


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, required=True)
    ap.add_argument("--eval-size", type=int, default=96)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.output and args.output.exists():
        ap.error("Output already exists; choose a new filename")
    if not args.directory.is_dir():
        ap.error("Input directory does not exist")
    result = {"protocol": "Saved-array audit, center ROI, existing paper evaluate(); no training or historical scene regeneration.",
              "eval_size": args.eval_size, "files": [], "errors": []}
    for path in sorted(args.directory.glob("*.npz")):
        try:
            result["files"].append(inspect_file(path, args.eval_size))
        except Exception as exc:
            result["errors"].append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
    result["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__).resolve(), ROOT / "functions/paperrepro/evaluate.py",
                                         ROOT / "functions/common/metrics.py")}
    text = json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Audited {len(result['files'])} files; {len(result['errors'])} errors; created {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
