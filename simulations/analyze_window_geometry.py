"""Read-only audit of measurement grouping, not a reconstruction experiment.

Run from the repository with --outdir output/window_geometry_audit.
Uses the production grouping, probe generator and physical scan coordinates.
Writes reproducible numbers and coverage arrays; does not change any solver.

Definitions: d_k(x)=sum_{j in group k}|P(x-r_j)|^2, M_k=1[d_k>0].
Pair support overlap = |M_k & M_l| / min(|M_k|,|M_l|).
Pair weighted overlap = sum min(d_k,d_l) / min(sum d_k,sum d_l).
Graph weights are the latter, NOT measured Fisher information.
The local complex-field frame identity is verified numerically with an FFT.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulations.ProPtyNet_paper import Cfg, PRESETS
from functions.paperrepro.sample import make_truth, make_positions
from functions.paperrepro.window_training import measurement_groups


def graph_metrics(weights):
    w = np.array(weights, dtype=float, copy=True)
    np.fill_diagonal(w, 0)
    lap = np.diag(w.sum(axis=1)) - w
    eig = np.linalg.eigvalsh(lap)
    return dict(weights=w.tolist(), laplacian_eigenvalues=eig.tolist(),
                lambda2=float(eig[1]),
                phase_offset_variance_proxy=float(np.sum(1 / eig[1:]))
                if eig[1] > 1e-12 else None)


def profiled_consensus_graph(d):
    """Schur-complement graph in a surrogate local phase-offset model.

    Profiling a shared nuisance field out of sum_k d_k(x)*(a_k-b(x))**2
    gives sum_{k<l} w_kl*(a_k-a_l)**2, w_kl=sum_x d_k*d_l/sum_i d_i.
    Normalize by total mass to compare coupling per fixed total exposure.
    This is an exact quadratic identity, not the actual network Hessian.
    """
    total = d.sum(axis=0)
    safe_total = np.where(total > 0, total, 1)
    weights = np.zeros((len(d), len(d)))
    for k in range(len(d)):
        for l in range(k+1, len(d)):
            weights[k, l] = weights[l, k] = np.sum(d[k]*d[l]/safe_total) / total.sum()
    return graph_metrics(weights)


def audit_groups(groups, maps, roi):
    d = np.stack([maps[g].sum(axis=0) for g in groups])
    support = d > 0
    count = support.sum(axis=0)
    total = d.sum(axis=0)
    pairs = []
    weights = np.eye(len(groups))
    for k in range(len(groups)):
        for l in range(k + 1, len(groups)):
            inter = np.count_nonzero(support[k] & support[l])
            union = np.count_nonzero(support[k] | support[l])
            weighted = float(np.minimum(d[k], d[l]).sum() /
                             min(d[k].sum(), d[l].sum()))
            weights[k, l] = weights[l, k] = weighted
            pairs.append(dict(groups=[k, l], common_indices=len(set(groups[k]) & set(groups[l])),
                              support_overlap=float(inter / min(support[k].sum(), support[l].sum())),
                              support_iou=float(inter / union), weighted_overlap=weighted))
    roi_d = total[roi]
    roi_group_d = d[:, roi[0], roi[1]]
    union = count > 0
    # Sum physical sensitivity contributed to the ROI per measurement visit.
    visits = sum(map(len, groups))
    mass = maps[0].sum()
    result = dict(groups=groups, group_sizes=list(map(len, groups)),
                  unique_frames=len(set(sum(groups, []))), visits_per_cycle=visits,
                  group_support_pixels=support.sum(axis=(1, 2)).tolist(),
                  union_pixels=int(union.sum()),
                  all_groups_intersection_pixels=int(np.sum(count == len(groups))),
                  all_groups_intersection_over_union=float(np.sum(count == len(groups)) / union.sum()),
                  roi_covered_by_all_groups_fraction=float(np.mean(count[roi] == len(groups))),
                  roi_group_covered_fraction=support[:, roi[0], roi[1]].mean(axis=(1, 2)).tolist(),
                  roi_total_sensitivity_mean=float(roi_d.mean()),
                  roi_total_sensitivity_min=float(roi_d.min()),
                  roi_total_sensitivity_cv=float(roi_d.std() / roi_d.mean()),
                  roi_group_sensitivity_min=roi_group_d.min(axis=(1, 2)).tolist(),
                  roi_sensitivity_fraction_per_visit=float(roi_d.sum() / (visits * mass)),
                  roi_mean_sensitivity_per_visit=float(roi_d.mean() / visits),
                  roi_inverse_sensitivity_mean=float(np.mean(1 / roi_d)) if np.all(roi_d > 0) else None,
                  pairs=pairs, graph=graph_metrics(weights),
                  profiled_consensus_graph=profiled_consensus_graph(d),
                  roi_profiled_consensus_graph=profiled_consensus_graph(roi_group_d))
    return result, d


def frame_identity_check(probe, positions, maps, obj_size):
    rng = np.random.default_rng(12345)
    perturb = rng.normal(size=(obj_size, obj_size)) + 1j*rng.normal(size=(obj_size, obj_size))
    # Arbitrary unit-modulus phase tests the same unitary algebra as the Fresnel chirp.
    q = np.exp(1j*rng.normal(size=probe.shape))
    lhs = 0.
    n = probe.shape[0]
    for y, x in positions:
        field = perturb[y:y+n, x:x+n] * probe * q
        lhs += float(np.sum(np.abs(np.fft.fft2(field, norm="ortho"))**2))
    rhs = float(np.sum(maps.sum(axis=0) * np.abs(perturb)**2))
    return dict(lhs=lhs, rhs=rhs, relative_error=abs(lhs-rhs)/rhs)


def calculate():
    params = dict(PRESETS["paper"])
    params.update(grid=10, step_px=12, obj_size=624)
    cfg = Cfg(**params, preset="paper", eval_size=96, seed=0, device="cpu")
    _, probe, _, rr = make_truth(cfg)
    positions = make_positions(cfg)
    n, m = cfg.N, cfg.obj_size
    probe_intensity = np.abs(probe.astype(np.complex128))**2
    aperture = (rr <= cfg.probe_diam_px/2).astype(float)
    # Only the compact illuminated global bounding box is allocated per frame.
    half = int(np.ceil(cfg.probe_diam_px/2))
    centers = positions + n//2
    lo = centers.min(axis=0) - half
    hi = centers.max(axis=0) + half + 1
    shape = tuple(hi-lo)
    kernels = {"binary_disk": aperture, "true_probe_intensity": probe_intensity}
    specs = {"corner4": ("sparse", 4), "inner4": ("inner", 4),
             "sparse5": ("sparse", 5), "compact64": ("compact-matched", 4),
             "compact100": ("compact-full", 4)}
    groups_by_name = {name: measurement_groups(10, layout, side)
                      for name, (layout, side) in specs.items()}
    roi_begin = (m-cfg.eval_size)//2
    roi = tuple(slice(roi_begin-int(v), roi_begin+cfg.eval_size-int(v)) for v in lo)
    out = dict(geometry=dict(obj_size=m, grid=10, step_px=12, N=n,
                            probe_diameter_px=cfg.probe_diam_px,
                            ROI=[roi_begin, roi_begin+cfg.eval_size],
                            crop_origin=lo.tolist(), crop_shape=[int(v) for v in shape],
                            centers=centers.tolist(), probe_seed=0,
                            probe_texture=cfg.probe_amp_image or "lowpass-noise",
                            probe_phase=cfg.probe_phase_mode), metrics={})
    arrays = {}
    for kernel_name, kernel in kernels.items():
        maps = np.zeros((len(positions), *shape), dtype=float)
        small = kernel[n//2-half:n//2+half+1, n//2-half:n//2+half+1]
        for j, (cy, cx) in enumerate(centers-lo):
            maps[j, cy-half:cy+half+1, cx-half:cx+half+1] = small
        metrics = {}
        for name, groups in groups_by_name.items():
            metrics[name], arrays[kernel_name+"_"+name] = audit_groups(groups, maps, roi)
        out["metrics"][kernel_name] = metrics
        # Exact translation invariance: all corner and inner groups have identical
        # within-group pair displacement multisets and illumination histograms.
        for name in ("corner4", "inner4"):
            d = arrays[kernel_name+"_"+name]
            assert np.allclose(np.sort(d[0].ravel()), np.sort(d[1].ravel()))
        assert np.allclose(np.sort(arrays[kernel_name+"_corner4"][0].ravel()),
                           np.sort(arrays[kernel_name+"_inner4"][0].ravel()))
        if kernel_name == "true_probe_intensity":
            global_maps = np.zeros((len(positions), m, m), dtype=float)
            global_maps[:, lo[0]:hi[0], lo[1]:hi[1]] = maps
            out["frame_identity_check"] = frame_identity_check(probe, positions, global_maps, m)
            assert out["frame_identity_check"]["relative_error"] < 1e-10
    assert [out["metrics"]["binary_disk"][k]["unique_frames"]
            for k in ("corner4", "inner4", "sparse5")] == [64, 64, 100]
    assert all(p["common_indices"] == 0 for v in out["metrics"]["binary_disk"].values() for p in v["pairs"])
    arrays["crop_origin"] = lo
    arrays["probe_intensity"] = probe_intensity
    return out, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", required=True, help="New directory; refuse to overwrite")
    args = p.parse_args()
    target = Path(args.outdir)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite: {target}")
    result, arrays = calculate()
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    target.mkdir(parents=True, exist_ok=False)
    (target/"geometry.json").write_text(serialized, encoding="utf-8")
    np.savez_compressed(target/"coverage_maps.npz", **arrays)
    print(f"Saved {target.resolve()}")
    for mode, layouts in result["metrics"].items():
        for name, row in layouts.items():
            print(mode, name, "union", row["union_pixels"], "ROI mean", round(row["roi_total_sensitivity_mean"], 5),
                  "ROI all-group fraction", round(row["roi_covered_by_all_groups_fraction"], 5),
                  "profiled spectral gap", round(row["profiled_consensus_graph"]["lambda2"], 5))


if __name__ == "__main__":
    main()
