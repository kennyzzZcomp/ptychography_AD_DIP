"""Independent stage-1 experiment: fixed random groups, coordinate-matched channels.

Default pair: balanced_random vs balanced_random_template, 1000 updates each.
All existing source files stay unchanged. The existing solver is reused through
a private function-global namespace; no module-level monkey-patch is installed.
Only its channel-input selector is replaced. Loss groups/positions, fusion,
TGV, network initialization and the training loop are unchanged.

This first experiment deliberately uses the existing 10x10 raster simulation.
The common 25-slot template is the set of centers of its 2x2 scan cells.
Matching minimizes sum_k ||position[input[k]] - template[k]||^2 independently
in each fixed group. No diffraction intensities or object/probe truth are used.
It is a permutation, not regrouping, interpolation, or image-template matching.

Examples (from any directory, in an environment with project dependencies):
  python simulations/run_window_template_comparison.py --self-test
  python simulations/run_window_template_comparison.py --check-only
  python simulations/run_window_template_comparison.py --seed 0 \
      --outdir /content/window_template_seed0 --zip-results

--methods random-template runs only the new treatment. Optional abcd and
abcd-template modes are available for a training-level identity check.
--self-test runs geometry tests plus three TINY 4-update CPU checks, not a
full reconstruction. It uses an automatically cleaned temporary directory.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import FunctionType

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulations.run_paper_windows import WindowCfg, PRESETS
from functions.paperrepro import window_training as solver
from functions.paperrepro.coverage_balanced_batching import random_balanced_groups
from functions.paperrepro.sample import make_positions


METHODS = {
    "random": "balanced_random",
    "random-template": "balanced_random_template",
    "abcd": "abcd_original",
    "abcd-template": "abcd_template",
}
TEMPLATE_SPEC = "25 row-major 2x2 raster-cell centers; squared Euclidean one-to-one assignment"


@dataclass
class TemplateCfg(WindowCfg):
    template_method: str = "random-template"
    template_spec: str = TEMPLATE_SPEC
    template_adapter_version: int = 1


def validate_partition(groups, n_points):
    array = np.asarray(groups)
    if (array.ndim != 2 or array.shape[0] != 4 or array.shape[1] < 1
            or not np.issubdtype(array.dtype, np.integer)
            or not np.array_equal(np.sort(array.ravel()), np.arange(n_points))):
        raise ValueError("Require four equal groups, each scan index used exactly once")
    return array.astype(np.int64, copy=True)


def match_to_template(points_yx, groups, template_yx):
    """Return new-slot -> old-slot permutations and reordered scan indices."""
    points = np.asarray(points_yx, dtype=np.float64)
    template = np.asarray(template_yx, dtype=np.float64)
    group_array = validate_partition(groups, len(points))
    if (points.shape != (len(points), 2) or template.shape != (group_array.shape[1], 2)
            or not np.isfinite(points).all() or not np.isfinite(template).all()):
        raise ValueError("Finite (N,2) points and (N/4,2) template required")
    permutations, inputs = [], []
    for group in group_array:
        # Rows = template slots; columns = OLD channels within this group.
        cost = ((template[:, None, :] - points[group][None, :, :]) ** 2).sum(-1)
        rows, cols = linear_sum_assignment(cost)
        perm = np.empty(len(group), dtype=np.int64)
        perm[rows] = cols
        if not np.array_equal(np.sort(perm), np.arange(len(group))):
            raise AssertionError("Assignment was not a permutation")
        if cost[rows, cols].sum() > np.trace(cost) + 1e-8:
            raise AssertionError("Matching increased its own assignment objective")
        permutations.append(perm.tolist())
        inputs.append(group[perm].tolist())
    return permutations, inputs


def correspondence_metrics(points, input_groups, template):
    """Position proxies only; neither coverage metrics nor reconstruction quality."""
    ordered = np.asarray(points, dtype=np.float64)[np.asarray(input_groups)]
    squared = ((ordered - template[None, :, :]) ** 2).sum(-1)
    residual = ordered - ordered.mean(0, keepdims=True)
    centered = ordered - ordered.mean(1, keepdims=True)
    centered -= centered.mean(0, keepdims=True)
    return dict(
        template_rms_px=float(np.sqrt(squared.mean())),
        template_worst_distance_px=float(np.sqrt(squared.max())),
        per_group_template_rms_px=np.sqrt(squared.mean(1)).tolist(),
        slot_spread_rms_px=float(np.sqrt((residual ** 2).sum(-1).mean())),
        translation_removed_slot_spread_rms_px=float(np.sqrt((centered ** 2).sum(-1).mean())),
    )


def make_plan(cfg):
    started = time.perf_counter()
    if cfg.grid != 10:
        raise ValueError("This controlled experiment supports the original 10x10 raster only")
    points = make_positions(cfg)
    abcd = solver.measurement_groups(10, "sparse", 5)
    # No per-group recentering/fitting: one frozen template from known scan positions.
    template = points[np.asarray(abcd)].mean(axis=0)
    permutations, reordered = match_to_template(points, abcd, template)
    if reordered != abcd or permutations != [list(range(25)) for _ in range(4)]:
        raise AssertionError("ABCD template self-check failed: expected the original slot order")
    groups = ([sorted(g.tolist()) for g in random_balanced_groups(100, 4, cfg.coverage_seed)]
              if cfg.template_method.startswith("random") else abcd)
    validate_partition(groups, len(points))
    matched_perms, matched_inputs = match_to_template(points, groups, template)
    use_matching = cfg.template_method.endswith("-template")
    input_groups = matched_inputs if use_matching else [g.copy() for g in groups]
    input_perms = matched_perms if use_matching else [list(range(25)) for _ in range(4)]
    for original, reordered in zip(groups, input_groups):
        if sorted(original) != sorted(reordered):
            raise AssertionError("Matching changed group membership")
    return dict(
        method=cfg.template_method, template_spec=TEMPLATE_SPEC,
        positions_yx=points.tolist(), template_yx=template.tolist(),
        coordinate_convention="probe-patch top-left (y,x), object pixels; common translations do not affect matching",
        loss_groups=groups, input_groups=input_groups, permutations=input_perms,
        before=correspondence_metrics(points, groups, template),
        after=correspondence_metrics(points, input_groups, template),
        abcd_identity_check_passed=True, fixed_during_training=True,
        all_measurements_retained=True, group_membership_unchanged=True,
        physical_loss_pairing_unchanged=True, truth_used_for_matching=False,
        matching_objective="sum of squared distances to the common fixed template",
        metric_note="Only template distance is optimized; cross-group spread need not decrease. No coverage or overlap is changed.",
        setup_seconds=time.perf_counter() - started,
    )


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def isolated_runner(overrides):
    """Reuse the exact training bytecode with an isolated dependency dictionary."""
    original = solver.run_windows
    namespace = dict(original.__globals__)
    namespace.update(overrides)
    return FunctionType(original.__code__, namespace, original.__name__,
                        original.__defaults__, original.__closure__)


def run_case(cfg, plan, *, test_overrides=None):
    """Inject INPUT indices only. cfg/output metadata name the effective ordering."""
    overrides = dict(test_overrides or {})
    base_scene = overrides.get("build_scene", solver.build_scene)
    base_save = overrides.get("_save", solver._save)
    expected_order = "template" if cfg.template_method.endswith("-template") else "original"
    if cfg.channel_order != expected_order:
        raise ValueError("cfg.channel_order must describe the effective input order")
    used = dict(scene=False, channels=False)

    def checked_scene(*args, **kwargs):
        sc = base_scene(*args, **kwargs)
        np.testing.assert_array_equal(sc.pos, plan["positions_yx"])
        used["scene"] = True
        return sc

    def select_channels(groups, order="original", seed=0):
        if not used["scene"] or order != "original" or groups != plan["loss_groups"]:
            raise AssertionError("Runtime scene/loss groups differ from the matching plan")
        used["channels"] = True
        audit = dict(plan, runtime_scene_positions_verified=True, runtime_loss_groups_verified=True)
        write_json(Path(cfg.outdir) / "template_matching.json", audit)
        print(f"[template] {cfg.template_method}: INPUT channels only; loss groups unchanged; "
              f"template RMS {plan['before']['template_rms_px']:.3f} -> "
              f"{plan['after']['template_rms_px']:.3f} px", flush=True)
        return ([p.copy() for p in plan["permutations"]],
                [g.copy() for g in plan["input_groups"]])

    def save_effective_config(runtime_cfg, *args, **kwargs):
        # The stock solver validates only its built-in modes. Its original flag
        # is an internal adapter setting, NOT the effective order written to NPZ.
        return base_save(replace(runtime_cfg, channel_order=expected_order), *args, **kwargs)

    overrides.update(build_scene=checked_scene, channel_input_groups=select_channels,
                     _save=save_effective_config)
    hist = isolated_runner(overrides)(replace(cfg, channel_order="original"))
    if not all(used.values()):
        raise AssertionError("The solver did not use the input-channel adapter")
    # Relabel this run's output metadata only; existing source files are untouched.
    out = Path(cfg.outdir)
    channel = json.loads((out / "channel_order.json").read_text(encoding="utf-8"))
    if channel["input_groups"] != plan["input_groups"] or channel["loss_groups"] != plan["loss_groups"]:
        raise AssertionError("Saved input/loss group audit does not match actual experiment")
    channel.update(mode=expected_order, backend_validation_mode="original",
                   experiment_method=cfg.template_method, template_audit="template_matching.json")
    write_json(out / "channel_order.json", channel)
    meta = json.loads((out / "window_metadata.json").read_text(encoding="utf-8"))
    meta.update(channel_order=channel, experiment_method=cfg.template_method,
                template_audit="template_matching.json",
                template_preprocessing_seconds=plan["setup_seconds"])
    write_json(out / "window_metadata.json", meta)
    partition_file = out / "random_partition.json"
    if partition_file.exists():
        partition = json.loads(partition_file.read_text(encoding="utf-8"))
        partition.update(channel_order="effective input order is in channel_order.json; groups here are loss groups")
        write_json(partition_file, partition)
    transfer_file = out / "cross_group_transfer.json"
    if transfer_file.exists():
        transfer = json.loads(transfer_file.read_text(encoding="utf-8"))
        for report in transfer["reports"]:
            report.update(channel_order=expected_order, experiment_method=cfg.template_method)
        write_json(transfer_file, transfer)
        for report in transfer["reports"]:
            write_json(out / "cross_group_transfer" / f"step_{report['it']:06d}.json", report)
    return hist


def build_cfg(args, method, outdir):
    geometry = dict(PRESETS[args.preset])
    geometry.update(grid=10, step_px=args.step_px if args.step_px is not None else (12 if args.preset == "paper" else 2),
                    obj_size=args.obj_size if args.obj_size is not None else (624 if args.preset == "paper" else 152))
    return TemplateCfg(**geometry, preset=args.preset, template_method=method,
        channel_order="template" if method.endswith("-template") else "original",
        iters=args.iters, eval_every=args.eval_every, base_ch=args.base_ch,
        eval_size=args.eval_size if args.eval_size is not None else (96 if args.preset == "paper" else 16),
        seed=args.seed, network_seed=args.seed if args.network_seed is None else args.network_seed,
        device=args.device, lr_net=args.lr_net, lr_probe=args.lr_probe, tgv_amp=args.tgv_amp,
        probe_mode="pixel", probe_init="ones", window_side=5,
        window_layout="balanced-random" if method.startswith("random") else "sparse",
        window_update="sequential", window_consistency=0.0, window_switch_after=0,
        coverage_seed=args.partition_seed, coverage_diameter_px=59.1,
        transfer_every=args.transfer_every, outdir=str(outdir))


def self_test():
    """No persistent results, no GPU, no paper-size training."""
    import torch
    before_torch = torch.random.get_rng_state().clone()
    before_numpy = np.random.get_state()
    for seed in range(5):
        cfg = TemplateCfg(grid=10, step_px=12, obj_size=624, coverage_seed=seed)
        plan = make_plan(cfg)
        if plan["after"]["template_rms_px"] > plan["before"]["template_rms_px"] + 1e-10:
            raise AssertionError("Geometry objective increased")
        points = np.asarray(plan["positions_yx"])
        template = np.asarray(plan["template_yx"])
        _, shifted = match_to_template(points + [5, -9], plan["loss_groups"], template + [5, -9])
        if shifted != plan["input_groups"]:
            raise AssertionError("Common translation changed assignment")
    if not torch.equal(before_torch, torch.random.get_rng_state()):
        raise AssertionError("Matching consumed Torch initialization RNG")
    after_numpy = np.random.get_state()
    if before_numpy[0] != after_numpy[0] or before_numpy[2:] != after_numpy[2:]:
        raise AssertionError("Matching consumed NumPy global RNG")
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    try:
        match_to_template(points, [[0] * 25] * 4, template)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid duplicate group indices were accepted")

    original_functions = (solver.run_windows, solver.channel_input_groups, solver.build_scene, solver._save)
    records = {}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with tempfile.TemporaryDirectory(prefix="window_template_selftest_") as td:
            for method in ("stock", "random", "random-template"):
                cfg = TemplateCfg(N=32, grid=10, step_px=2, obj_size=56, eval_size=8,
                    probe_diam_um=2500, base_ch=2, device="cpu", iters=4, eval_every=4,
                    seed=0, network_seed=0, window_layout="balanced-random", window_side=5,
                    tgv_amp=.001, transfer_every=4, outdir=str(Path(td) / method),
                    template_method="random" if method == "stock" else method,
                    channel_order="template" if method.endswith("-template") else "original")
                plan = make_plan(cfg)
                inputs, physics, initial, saved = [], [], [], {}

                def build_net(*args, **kwargs):
                    net = solver.ProPtyUNet(*args, **kwargs)
                    initial.append({k: v.detach().clone() for k, v in net.state_dict().items()})
                    net.register_forward_pre_hook(lambda model, x: inputs.append(x[0].detach().clone()))
                    return net

                def capture_fwd(*args):
                    physics.append(args[3].detach().clone())
                    return solver._fwd(*args)

                def capture_save(runtime_cfg, rec, pc, *args, **kwargs):
                    saved.update(cfg=asdict(runtime_cfg), rec=rec.copy(), probe=pc.copy())

                overrides = dict(ProPtyUNet=build_net, _fwd=capture_fwd, _save=capture_save)
                with contextlib.redirect_stdout(io.StringIO()):
                    hist = (isolated_runner(overrides)(cfg) if method == "stock"
                            else run_case(cfg, plan, test_overrides=overrides))
                if not np.isfinite(hist[-1]["psnr_amp"]):
                    raise AssertionError("Nonfinite smoke-test output")
                if method != "stock":
                    audit = json.loads((Path(cfg.outdir) / "channel_order.json").read_text(encoding="utf-8"))
                    transfer = json.loads((Path(cfg.outdir) / "cross_group_transfer.json").read_text(encoding="utf-8"))
                    if audit["mode"] != cfg.channel_order or saved["cfg"]["channel_order"] != cfg.channel_order:
                        raise AssertionError("Effective mode was not saved correctly")
                    if transfer["reports"][0]["input_groups"] != plan["input_groups"]:
                        raise AssertionError("Transfer diagnostics used wrong inputs")
                    with np.load(Path(cfg.outdir) / "window_fields.npz", allow_pickle=False) as data:
                        np.testing.assert_array_equal(data["input_groups"], plan["input_groups"])
                        np.testing.assert_array_equal(data["training_groups"], plan["loss_groups"])
                records[method] = dict(inputs=inputs, physics=physics, initial=initial[0], saved=saved, plan=plan)
            stock, baseline, matched = (records[k] for k in ("stock", "random", "random-template"))
            for key in ("rec", "probe"):
                np.testing.assert_array_equal(stock["saved"][key], baseline["saved"][key])
            for rec in (baseline, matched):
                for key, value in stock["initial"].items():
                    torch.testing.assert_close(rec["initial"][key], value, rtol=0, atol=0)
                if len(stock["physics"]) != len(rec["physics"]):
                    raise AssertionError("Physics call count changed")
                for actual, expected in zip(rec["physics"], stock["physics"]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected_inputs = [baseline["inputs"][k][:, matched["plan"]["permutations"][k]] for k in range(4)]
            for k in range(4):
                torch.testing.assert_close(matched["inputs"][k], expected_inputs[k], rtol=0, atol=0)
            for actual in matched["inputs"]:
                if not any(torch.equal(actual, expected) for expected in expected_inputs):
                    raise AssertionError("Training/readout/diagnostics used an unexpected input")
    finally:
        torch.set_num_threads(old_threads)
    if original_functions != (solver.run_windows, solver.channel_input_groups, solver.build_scene, solver._save):
        raise AssertionError("Existing solver module was modified")
    print("PASS: ABCD identity, bijective matching, fixed memberships, translation invariance, RNG isolation")
    print("PASS: stock baseline exact reproduction; identical initialization/physics indices; matched inputs in training/readout/diagnostics")
    print("PASS: effective metadata, unchanged solver module; three tiny 4-update CPU runs; temporary files cleaned")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=["random", "random-template"])
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--seed", type=int, default=0, help="Scene seed; default network seed also follows it")
    parser.add_argument("--network-seed", type=int, help="Optional independent network initialization seed")
    parser.add_argument("--partition-seed", type=int, default=0, help="Same as old --coverage-seed")
    parser.add_argument("--preset", choices=("paper", "smoke"), default="paper")
    parser.add_argument("--step-px", type=int)
    parser.add_argument("--obj-size", type=int)
    parser.add_argument("--eval-size", type=int)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--lr-net", type=float, default=.001)
    parser.add_argument("--lr-probe", type=float, default=.01)
    parser.add_argument("--tgv-amp", type=float, default=.001)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--transfer-every", type=int, default=250,
                        help="Copied-state diagnostic, separately timed; 0 disables. Default matches archived ablations")
    parser.add_argument("--check-only", action="store_true", help="Validate/print geometry; no training or output files")
    parser.add_argument("--self-test", action="store_true", help="Geometry and tiny CPU integration tests only")
    parser.add_argument("--zip-results", action="store_true", help="Archive completed output folder beside itself")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if len(set(args.methods)) != len(args.methods):
        parser.error("Do not repeat methods")
    if min(args.seed, args.partition_seed) < 0 or (args.network_seed is not None and args.network_seed < 0):
        parser.error("Seeds must be nonnegative")
    if args.iters < 4 or args.iters % 4 or args.eval_every < 1 or args.transfer_every < 0:
        parser.error("iters must be a positive multiple of 4; eval-every > 0; transfer-every >= 0")
    if (not np.isfinite([args.lr_net, args.lr_probe, args.tgv_amp]).all()
            or min(args.lr_net, args.lr_probe) <= 0 or args.tgv_amp < 0):
        parser.error("Learning rates must be finite/positive; TGV finite/nonnegative")
    if not args.check_only and args.outdir is None:
        parser.error("--outdir is required for training")
    root = args.outdir.resolve() if args.outdir else Path("template_check_only")
    runs = [(build_cfg(args, method, root / METHODS[method]), method) for method in args.methods]
    plans = [make_plan(cfg) for cfg, _ in runs]
    for (cfg, method), plan in zip(runs, plans):
        print(f"[plan] {method}: 4 x 25; template RMS "
              f"{plan['before']['template_rms_px']:.3f} -> {plan['after']['template_rms_px']:.3f} px; "
              f"slot spread {plan['before']['slot_spread_rms_px']:.3f} -> {plan['after']['slot_spread_rms_px']:.3f} px")
    if args.check_only:
        print("ABCD identity check passed. No training and no output files created.")
        return
    archive_path = Path(str(root) + ".zip")
    if root.exists() or (args.zip_results and archive_path.exists()):
        parser.error("Output folder/archive already exists; choose a new --outdir (never overwrite)")
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", experiment="fixed-group coordinate-template input alignment",
        source_file=Path(__file__).name,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        solver_sha256=hashlib.sha256(Path(solver.__file__).read_bytes()).hexdigest(),
        seed=args.seed, network_seed=runs[0][0].network_seed, partition_seed=args.partition_seed,
        methods=args.methods, results=[],
        notes=["Only input order changes within each paired partition; geometry/loss unchanged.",
               "Matching uses coordinates only; no ground truth or diffraction-value matching.",
               "Stage 1 only; each iteration is one group optimizer update.",
               "Training-loop time includes normal evaluations, excludes matching/setup/save and separately timed transfer blocks.",
               "Three different scene seeds also change the truth probe; use --network-seed to vary initialization alone."],
        configs={method: asdict(cfg) for cfg, method in runs})
    manifest_path = root / "comparison_manifest.json"
    write_json(manifest_path, manifest)
    for (cfg, method), plan in zip(runs, plans):
        print(f"\n[experiment] {method} -> {cfg.outdir}", flush=True)
        try:
            hist = run_case(cfg, plan)
        except Exception as exc:
            manifest.update(status="failed", failed_method=method, error=f"{type(exc).__name__}: {exc}")
            write_json(manifest_path, manifest)
            raise
        last = hist[-1]
        metadata = json.loads((Path(cfg.outdir) / "window_metadata.json").read_text(encoding="utf-8"))
        manifest["results"].append(dict(method=method, folder=METHODS[method],
            scene_fingerprint=metadata["scene_fingerprint"], it=last["it"],
            psnr_amp=last["psnr_amp"], relerr=last["relerr"],
            train_elapsed_s=metadata["elapsed_s"], diagnostic_elapsed_s=metadata.get("diagnostic_elapsed_s", 0),
            geometry_before=plan["before"], geometry_after=plan["after"]))
        write_json(manifest_path, manifest)
    if len({r["scene_fingerprint"] for r in manifest["results"]}) != 1:
        raise AssertionError("Paired methods unexpectedly used different scenes")
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print("\nCompleted stage-1 comparison:")
    for result in manifest["results"]:
        print(f"  {result['method']}: {result['psnr_amp']:.3f} dB; "
              f"object error {result['relerr']:.5f}; loop {result['train_elapsed_s']:.2f} s")
    if args.zip_results:
        archive = shutil.make_archive(str(root), "zip", root_dir=root.parent, base_dir=root.name)
        print(f"Send this archive for analysis: {archive}")
    else:
        print(f"Results: {root}")


if __name__ == "__main__":
    # Native Windows consoles may use GBK; the existing scene banner contains mu.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    main()
