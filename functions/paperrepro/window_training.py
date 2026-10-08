"""Four measurement windows, full-canvas CNN outputs, one independent probe.

This is a separate pilot, not a change to the paper or progressive solver.
Sequential mode: one window and one optimizer step per iteration. Joint mode:
four windows at the same weights and one optimizer step per iteration.
Cached-fusion loss: one current prediction plus three detached stored predictions
compose the training object, without additional network/physics forwards.
Optional consistency uses conservative computational probe footprints, NOT
ground-truth illumination or evaluation ROI. These masks are not confidence maps.
"""
from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from functions.addip.model import ProPtyUNet, make_field
from functions.addip.losses import cabs
from functions.paperrepro.scene import build_scene
from functions.paperrepro.evaluate import evaluate, probe_relerr
from functions.paperrepro.report import _save
from functions.paperrepro.reconstruction_input import snapshot_readout, reconstruction_channels
from functions.paperrepro.solvers_addip import (
    _probe_setup, _probe_of, _initialize_object_heads, _fwd,
)
from functions.paperrepro.tgv import ObjectAmplitudeTGV


def four_windows(grid=10):
    if grid != 10:
        raise ValueError("This pilot implements the specified 10x10 scan only")
    a, b = (0, 2, 4, 6), (3, 5, 7, 9)
    return [[r * grid + c for r in rows for c in cols]
            for rows, cols in ((a, a), (a, b), (b, a), (b, b))]


def measurement_groups(grid=10, layout="sparse", side=4):
    """Equal-size groups; defaults reproduce four_windows exactly.

    compact-matched regroups the SAME sparse measurements into quadrants.
    compact-full partitions every scan position into four 5x5 quadrants.
    These deterministic raster partitions are not a K-means implementation.
    alternating adds four inner windows: 128 visits to 92 unique positions
    per eight-update cycle. Readout remains on the original four corners.
    """
    if grid != 10 or side not in (3, 4, 5):
        raise ValueError("Window experiments require grid=10 and window-side=3, 4 or 5")
    if layout in ("alternating", "inner"):
        if side != 4:
            raise ValueError(f"{layout} requires window-side=4")
        inner = [[(r + dy)*grid + c + dx for r in (0, 2, 4, 6)
                  for c in (0, 2, 4, 6)]
                 for dy, dx in ((1, 1), (1, 2), (2, 1), (2, 2))]
        return inner if layout == "inner" else four_windows(grid) + inner
    if layout == "compact-full":
        if side != 4:
            raise ValueError("compact-full uses 5x5 groups; omit --window-side")
        a, b = tuple(range(5)), tuple(range(5, 10))
    else:
        a = tuple(range(0, 2 * side, 2))
        b = tuple(range(grid - 1 - 2 * (side - 1), grid, 2))
        if layout == "compact-matched":
            axis = sorted(set(a + b))
            a, b = tuple(axis[:side]), tuple(axis[side:])
        elif layout != "sparse":
            raise ValueError("Unknown window layout")
    return [[r * grid + c for r in rows for c in cols]
            for rows, cols in ((a, a), (a, b), (b, a), (b, b))]


def update_windows(update, mode):
    if mode == "sequential":
        return [update % 4]
    if mode == "joint":
        return list(range(4))
    raise ValueError("mode must be sequential or joint")


def footprint_masks(positions, groups, obj_size, probe_size, device):
    masks = torch.zeros((4, obj_size, obj_size), dtype=torch.bool, device=device)
    for k, group in enumerate(groups):
        for index in group:
            y, x = map(int, positions[index])
            if min(y, x) < 0 or max(y + probe_size, x + probe_size) > obj_size:
                raise ValueError("Probe footprint outside object canvas")
            masks[k, y:y + probe_size, x:x + probe_size] = True
    return masks


def fusion_weights(masks):
    """Partition of unity on existing computational footprints; no GT/support."""
    return masks.float() / masks.sum(0).clamp_min(1)


def fuse_fields(fields, masks, weights=None):
    fields = torch.stack(list(fields))
    count = masks.sum(0)
    merged = ((fields * masks).sum(0) / count.clamp_min(1) if weights is None
              else (fields * weights).sum(0))
    return torch.where(count > 0, merged, torch.ones_like(merged))


class WindowFieldCache:
    """Latest PRE-update prediction per window, detached from prior graphs.

    A cache is a numerical memory, not a snapshot of the current shared network.
    Evaluation must not overwrite it: otherwise evaluation frequency changes
    the training algorithm. Only the selected training forward refreshes it.
    """

    def __init__(self, masks):
        self.masks = masks
        self.weights = fusion_weights(masks)
        self.fields = torch.ones(masks.shape, dtype=torch.complex64, device=masks.device)
        self.source_updates = [0]*len(masks)
        self.visited = [False]*len(masks)

    def compose(self, active, field):
        fields = [field if k == active else self.fields[k].detach()
                  for k in range(len(self.fields))]
        return fuse_fields(fields, self.masks, self.weights)

    @torch.no_grad()
    def commit(self, active, field, completed_updates_before_forward):
        # copy_ must happen after backward; compose's saved graph no longer lives.
        self.fields[active].copy_(field.detach())
        self.source_updates[active] = int(completed_updates_before_forward)
        self.visited[active] = True

    def audit(self, completed_updates):
        return dict(cache_source_updates=list(self.source_updates),
                    cache_ages=[int(completed_updates)-u for u in self.source_updates],
                    cache_visited=list(self.visited))


def consistency_loss(fields, masks, active=None):
    """Raw complex agreement at common GLOBAL coordinates; no GT alignment.

    Sequential callers detach current peer predictions; joint callers differentiate
    all predictions. No stale prediction cache. Divide by non-detached pair
    energy so common object shrinkage cannot trivially reduce the penalty.
    """
    terms = []
    for i in range(4):
        for j in range(i + 1, 4):
            if active is not None and active not in (i, j):
                continue
            mask = masks[i] & masks[j]
            if bool(mask.any()):
                left, right = fields[i][mask], fields[j][mask]
                energy = .5*(left.abs().square().mean() + right.abs().square().mean())
                terms.append((left-right).abs().square().mean() / energy.clamp_min(1e-20))
    return torch.stack(terms).mean() if terms else fields[0].real.sum() * 0


def scaled_amplitude_loss(amplitude, target):
    q = (amplitude * target).sum() / amplitude.square().sum().clamp_min(1e-20)
    return F.mse_loss(q * amplitude, target), q * amplitude


def residual_object(base, raw_real, raw_imag):
    """Unrestricted complex correction to a fixed, UNNORMALIZED stage-1 field.

    No softplus/phase activation is applied to residuals. The returned amplitude
    belongs to the FINAL object, not the correction, and uses the existing safe
    complex magnitude for TGV (including at exact cancellation to zero).
    """
    field = base.detach() + torch.complex(raw_real, raw_imag)
    return field, cabs(field)


def initialize_residual_heads(net):
    """Zero only the two linear output heads; keep the backbone random/trainable."""
    with torch.no_grad():
        for head in (net.head_amp, net.head_phs):
            head.weight.zero_()
            head.bias.zero_()


def run_windows(cfg):
    if (cfg.network_type != "real" or cfg.probe_mode not in ("pixel", "support", "truth")
            or cfg.measurement_schedule or cfg.lr_schedule or cfg.lr_cosine
            or cfg.tgv_amp_schedule or cfg.tgv_phase or cfg.half_res_until
            or cfg.stage2_input != "diffraction"):
        raise ValueError("Window pilot uses real CNN, independent probe, fixed LR/TGV, no stages")
    if cfg.iters < 1 or cfg.eval_every < 1:
        raise ValueError("iters and eval_every must be positive")
    if not np.isfinite(cfg.window_consistency) or cfg.window_consistency < 0:
        raise ValueError("window_consistency must be finite and nonnegative")
    loss_mode = getattr(cfg, "window_loss_mode", "independent")
    transfer_every = getattr(cfg, "transfer_every", 0)
    if transfer_every < 0:
        raise ValueError("transfer-every must be nonnegative")
    if transfer_every and (cfg.window_update != 'sequential' or loss_mode != 'independent'
                           or cfg.window_consistency or cfg.window_layout == 'alternating'):
        raise ValueError("Transfer diagnostic requires four fixed groups, sequential independent loss and consistency=0")
    if loss_mode not in ("independent", "cached-fusion"):
        raise ValueError("window_loss_mode must be independent or cached-fusion")
    if loss_mode == "cached-fusion" and (cfg.window_update != "sequential" or cfg.window_consistency):
        raise ValueError("Cached fusion requires sequential updates and consistency-weight=0")
    if loss_mode == "cached-fusion" and cfg.obj_init_alpha != 0:
        raise ValueError("Cached fusion initializes four fields to ones; requires obj-init-alpha=0")
    switch = getattr(cfg, "window_switch_after", 0)
    stage2_readout = getattr(cfg, "window_stage2_readout", "direct")
    if stage2_readout not in ("direct", "complex-residual"):
        raise ValueError("stage2-readout must be direct or complex-residual")
    if stage2_readout == "complex-residual" and not switch:
        raise ValueError("Complex-residual readout requires an enabled second stage")
    if switch and not 0 < switch < cfg.iters:
        raise ValueError("switch-after must be positive and less than total iters (or 0 to disable)")
    if switch and cfg.window_update == "sequential" and switch % 4:
        raise ValueError("Sequential stage 1 must end after a complete four-window sweep")
    if switch:
        if any(not np.isfinite(v) or v <= 0 for v in (cfg.window_stage2_lr_net, cfg.window_stage2_lr_probe)):
            raise ValueError("Stage-2 learning rates must be finite and positive")
        if not np.isfinite(cfg.window_stage2_tgv) or cfg.window_stage2_tgv < 0:
            raise ValueError("Stage-2 TGV must be finite and nonnegative")
    layout = getattr(cfg, "window_layout", "sparse")
    side = getattr(cfg, "window_side", 4)
    coverage_balanced = layout == "coverage-balanced"
    random_balanced = layout == "balanced-random"
    if (coverage_balanced or random_balanced) and (cfg.n_pat < 4 or cfg.n_pat % 4):
        raise ValueError("Partition training requires measurement count divisible by 4; unequal-channel groups are not supported")
    groups = None if (coverage_balanced or random_balanced) else measurement_groups(cfg.grid, layout, side)
    alternating = layout == "alternating"
    if alternating:
        if cfg.window_update != "sequential" or loss_mode != "independent" or cfg.window_consistency:
            raise ValueError("alternating requires sequential, independent loss and consistency-weight=0")
        if switch and switch % 8:
            raise ValueError("alternating stage 1 must end after a complete 8-update cycle")
    update_windows(0, cfg.window_update)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    device = cfg.dev()
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    sc = build_scene(cfg, device)
    coverage_audit = None
    if random_balanced:
        from functions.paperrepro.coverage_balanced_batching import random_balanced_groups
        groups = [sorted(g.tolist()) for g in random_balanced_groups(cfg.n_pat, 4, cfg.coverage_seed)]
        (outdir / "random_partition.json").write_text(json.dumps(dict(
            groups=groups, seed=cfg.coverage_seed, positions_yx=sc.pos.tolist(),
            all_measurements_retained=True, fixed_during_training=True,
            channel_order='ascending original measurement index', truth_used=False), indent=2), encoding='utf-8')
        print(f"[windows] balanced-random: fixed partition seed={cfg.coverage_seed}; window-side ignored.", flush=True)
    if coverage_balanced:
        from functions.paperrepro.coverage_balanced_batching import disk_window_partition
        diameter = getattr(cfg, "coverage_diameter_px", None)
        groups, coverage_audit = disk_window_partition(
            sc.pos, cfg.probe_diam_px if diameter is None else diameter,
            anchor_step=getattr(cfg, "coverage_anchor_step", 2.0),
            seed=getattr(cfg, "coverage_seed", 0))
        (outdir / "coverage_partition.json").write_text(
            json.dumps(dict(groups=groups, positions_yx=sc.pos.tolist(), **coverage_audit), indent=2),
            encoding="utf-8")
        print(f"[windows] coverage-balanced: fixed nominal disk D={coverage_audit['diameter_px']:.6g}px; "
              f"partition seed={coverage_audit['seed']}; eval E={coverage_audit['E_percent']:.2f}% "
              f"(initial random {coverage_audit['initial_random_E_percent']:.2f}%); "
              f"setup incl. diagnostics={coverage_audit['setup_with_diagnostics_seconds']:.3f}s. "
              "No truth probe or added reconstruction support; window-side ignored.", flush=True)
    patterns_per_group = len(groups[0])
    unique_patterns = len(set(sum(groups, [])))
    if cfg.network_seed is not None:
        torch.manual_seed(cfg.network_seed)
    net = ProPtyUNet(patterns_per_group, cfg.base_ch, n_fields=1, ph_ch=2,
                    skip_mode=cfg.skip_mode, wavelet_threshold=cfg.wavelet_threshold).to(device)
    _initialize_object_heads(net, cfg.obj_init_alpha)
    mode, fixed, pr, pi, support = _probe_setup(cfg, sc, sc.probe, device, "windows")
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr_net, weight_decay=cfg.weight_decay)
    opt_p = None if mode == "truth" else torch.optim.Adam([pr, pi], lr=cfg.lr_probe)
    selections = [torch.tensor(g, device=device) for g in groups]
    # Match the existing full-canvas diffraction padding, including odd dimensions.
    m, ns = cfg.obj_size, cfg.net_size
    d = m - cfg.N
    stack = F.pad(sc.Imt[None], (d//2, d-d//2, d//2, d-d//2))
    p = ns - m
    stack = F.pad(stack, (p//2, p-p//2, p//2, p-p//2))
    stack = stack / stack.amax((2, 3), keepdim=True).clamp_min(1e-12)
    inputs = [stack[:, s] for s in selections]
    crop = slice(p//2, p//2 + m)
    # Use the same four groups for input, loss and readout; alternating alone
    # retains the original four-corner readout to isolate schedule changes.
    masks = footprint_masks(sc.pos, groups[:4], m, cfg.N, device)
    cache = WindowFieldCache(masks) if loss_mode == "cached-fusion" else None
    # Independent auxiliary TGV state per window; nominal domains, as in net.
    tgvs = [ObjectAmplitudeTGV(cfg, sc.pos[g], device) for g in groups] if cfg.tgv_amp else [None]*len(groups)
    energy = sc.sqrtIm.square().mean().detach()
    residual_base = None

    def decode(k):
        a, ph = net(inputs[k])
        if residual_base is not None:
            return residual_object(residual_base, a[0][crop, crop], ph[0][crop, crop])
        return make_field(a[0], ph[:2])[crop, crop], F.softplus(a[0])[crop, crop]

    def probe():
        return _probe_of(mode, fixed, pr, pi, support)

    counts = dict(optimizer_updates=0, window_visits=0, training_patterns=0,
                  training_network_forwards=0, consistency_network_forwards=0,
                  evaluation_network_forwards=0, evaluation_patterns=0,
                  transition_network_forwards=0, stage2_updates=0, cache_updates=0)
    print(f"[windows] {cfg.window_update}; iteration = ONE optimizer update; "
          f"Stage 1: layout={layout}; {len(groups)} groups x {patterns_per_group} patterns; {unique_patterns} unique training frames, "
          f"remaining {cfg.n_pat-unique_patterns} evaluation-only in stage 1. "
          f"consistency={cfg.window_consistency:g}; full-canvas outputs, no added support.", flush=True)
    if alternating:
        print("[windows] alternating: 4 corner updates + 4 inner updates = 1 cycle; "
              "evaluation/stage-2 conditioning use fresh predictions of the original 4 corner windows.", flush=True)
    if cache is not None:
        print("[windows] loss-mode=cached-fusion: ONE current window forward + 3 detached cached fields; "
              f"data/TGV use the fused object, {patterns_per_group} measurements/update. Cache starts at ones, "
              "stores pre-update predictions; evaluation NEVER refreshes training cache.", flush=True)
    if cfg.window_update == "sequential" and cfg.window_consistency:
        print("[windows] Consistency adds 3 current-weight no-grad peer forwards per update; "
              "BN buffers restored. No stale cache; extra work counted.", flush=True)
    metadata = dict(groups=groups, missing_indices=sorted(set(range(cfg.n_pat))-set(sum(groups, []))),
                    update_mode=cfg.window_update, consistency_weight=cfg.window_consistency,
                    loss_mode=loss_mode,
                    input_channels=patterns_per_group, window_layout=layout, window_side=side,
                    unique_training_patterns=unique_patterns,
                    output="full global canvas for each window",
                    fusion="complex mean on full computational footprints; ones outside union",
                    consistency_domain="full computational footprint intersections, NOT confidence/support",
                    calibration="independent analytic amplitude scale per training window; global scale for fused evaluation",
                    evaluation="fresh post-update fields, train-mode BN with restored buffers; no GT in training",
                    iteration="one network/probe optimizer update", timing="loop includes evaluation; excludes setup/save",
                    device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    network_parameters=sum(p.numel() for p in net.parameters()), scene_fingerprint=sc.fp)
    metadata.update(switch_after=switch, total_updates=cfg.iters,
                    evaluation_groups=groups[:4], schedule_period_updates=len(groups),
                    window_schedule="four corners then four inner windows" if alternating else "fixed groups",
                    stage2_readout=stage2_readout if switch else None,
                    stage2="fresh object-only input, full measurements" if switch else None)
    if coverage_audit is not None:
        metadata.update(coverage_partition=coverage_audit, window_side=None)
    if random_balanced:
        metadata.update(partition_seed=cfg.coverage_seed, window_side=None)
    if transfer_every:
        metadata.update(transfer_every=transfer_every,
                        timing='elapsed_s excludes separately recorded diagnostic blocks; includes normal evaluation; excludes setup/save',
                        transfer_note='copied-state fixed-probe Adam+TGV interventions, stage 1 only; not normal joint object/probe updates')
    if cache is not None:
        metadata.update(training_object="partition-weighted complex fusion; active prediction + detached peers",
                        training_tgv="amplitude of fused object on active window TGV domain",
                        cache_initialization="four constant ones fields",
                        cache_commit="pre-update active prediction, copied after optimizer step",
                        cache_refresh="training visits only; no extra peer forwards",
                        evaluation="fresh post-update fusion; cache gap reported, cache never changed",
                        fusion_weight_geometry="existing full computational probe footprints; flat normalized overlap weights")
    (outdir / "window_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    hist = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    diagnostic_seconds = 0.0
    diagnostic_records = []
    diagnostic_reports = []
    transition = None
    snapshot = None
    for it in range(cfg.iters):
        stage2 = bool(switch and it >= switch)
        if switch and it == switch:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            transition_start = time.perf_counter()
            with snapshot_readout(net):
                stage1_fields = [decode(k)[0].detach().clone() for k in range(4)]
                stage1_object = fuse_fields(stage1_fields, masks, cache.weights if cache else None).detach().clone()
                stage1_probe = probe().detach().clone()
                cache_gap_at_switch = (float((stage1_object-fuse_fields(cache.fields, masks, cache.weights)).norm()
                                            / stage1_object.norm().clamp_min(1e-12)) if cache else None)
            counts["transition_network_forwards"] += 4
            conditioning, encoding = reconstruction_channels(stage1_object, None, ns, include_probe=False)
            snapshot = dict(obj=stage1_object.cpu().numpy(), probe=stage1_probe.cpu().numpy(),
                            fields=torch.stack(stage1_fields).cpu().numpy(),
                            network_input=conditioning.cpu().numpy())
            # Same fresh backbone and fixed conditioning; optional two-head residual.
            seed = cfg.seed if cfg.network_seed is None else cfg.network_seed
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(seed))
                new_net = ProPtyUNet(3, cfg.base_ch, n_fields=1,
                                    ph_ch=1 if stage2_readout == "complex-residual" else 2,
                                    skip_mode=cfg.skip_mode, wavelet_threshold=cfg.wavelet_threshold)
                if stage2_readout == "complex-residual":
                    initialize_residual_heads(new_net)
                else:
                    _initialize_object_heads(new_net, cfg.obj_init_alpha)
            net = new_net.to(device)
            if stage2_readout == "complex-residual":
                # Retain the raw complex scale; normalized conditioning is INPUT only.
                residual_base = stage1_object
            opt = torch.optim.Adam(net.parameters(), lr=cfg.window_stage2_lr_net, weight_decay=cfg.weight_decay)
            probe_steps = ([float(opt_p.state[v]["step"]) for v in (pr, pi)] if opt_p else [])
            if opt_p:
                for group in opt_p.param_groups:
                    group["lr"] = cfg.window_stage2_lr_probe
            inputs = [conditioning]
            selections = [torch.arange(cfg.n_pat, device=device)]
            # Window auxiliaries describe different domains; use a NEW full-data
            # TGV auxiliary, never silently copy one window's vector field.
            tgvs = ([ObjectAmplitudeTGV(cfg, sc.pos, device)] if cfg.window_stage2_tgv else [None])
            with snapshot_readout(net):
                initial_obj = decode(0)[0]
                obj_jump = float((initial_obj-stage1_object).norm()/stage1_object.norm().clamp_min(1e-12))
                probe_jump = float((probe()-stage1_probe).norm()/stage1_probe.norm().clamp_min(1e-12))
                snapshot["stage2_initial_obj"] = initial_obj.cpu().numpy()
            counts["transition_network_forwards"] += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            transition = dict(**encoding, after_update=switch, seed=seed,
                              new_object_network=True, new_object_optimizer=True,
                              probe_optimizer_retained=True, probe_adam_steps_at_switch=probe_steps,
                              probe_relative_jump=probe_jump, object_relative_jump=obj_jump,
                              stage2_readout=stage2_readout,
                              output_channels=2 if residual_base is not None else 3,
                              object_initialization=("O1 + zero linear complex residual" if residual_base is not None
                                                     else "neutral heads, NOT residual warm start"),
                              tgv_amplitude_readout=("safe magnitude of final O1 + residual" if residual_base is not None
                                                     else "softplus object amplitude head"),
                              new_full_domain_tgv_auxiliary=bool(cfg.window_stage2_tgv),
                              cache_fresh_relative_difference=cache_gap_at_switch,
                              stage1_cache=cache.audit(switch) if cache else None,
                              lr_net=cfg.window_stage2_lr_net, lr_probe=cfg.window_stage2_lr_probe,
                              tgv_amp=cfg.window_stage2_tgv,
                              stage1_elapsed_s=transition_start-start-diagnostic_seconds,
                              transition_elapsed_s=time.perf_counter()-transition_start,
                              network_parameters=sum(v.numel() for v in net.parameters()))
            print(f"[windows] update {it+1}: fresh U-Net; fixed object-only 3-channel input; "
                  f"readout={stage2_readout}, object jump={obj_jump:g}; "
                  f"100 measurements; retained probe/Adam, probe jump={probe_jump:g}; "
                  f"lr_net={cfg.window_stage2_lr_net:g}, lr_probe={cfg.window_stage2_lr_probe:g}, "
                  f"TGV={cfg.window_stage2_tgv:g}", flush=True)
            del stage1_fields, stage1_object, stage1_probe, initial_obj, new_net
        active = [0] if stage2 else ([it % 8] if alternating else update_windows(it, cfg.window_update))
        consistency_weight = 0. if stage2 else cfg.window_consistency
        tgv_weight = cfg.window_stage2_tgv if stage2 else cfg.tgv_amp
        # Read peers BEFORE any grad-enabled forward: restoring BN buffers after
        # such a forward would otherwise invalidate autograd saved tensors.
        fields = [None]*len(groups)
        if consistency_weight and len(active) == 1:
            with snapshot_readout(net):
                for k in range(4):
                    if k not in active:
                        fields[k] = decode(k)[0].detach()
                        counts["consistency_network_forwards"] += 1
        opt.zero_grad(set_to_none=True)
        if opt_p:
            opt_p.zero_grad(set_to_none=True)
        losses, regs = [], []
        current_probe = probe()
        for k in active:
            fields[k], amp = decode(k)
            counts["training_network_forwards"] += 1
            training_object = cache.compose(k, fields[k]) if cache is not None and not stage2 else fields[k]
            if cache is not None and not stage2:
                amp = cabs(training_object)
            data, _ = scaled_amplitude_loss(cabs(_fwd(cfg, training_object, current_probe,
                                                     sc.post[selections[k]], sc.Q)), sc.sqrtIm[selections[k]])
            reg = tgvs[k](amp)[0] if tgvs[k] is not None else data.new_zeros(())
            losses.append(data)
            regs.append(reg)
        data = torch.stack(losses).mean()
        reg = torch.stack(regs).mean()
        con = (consistency_loss(fields, masks, active[0] if len(active) == 1 else None)
               if consistency_weight else data.new_zeros(()))
        loss = data + energy * (tgv_weight*reg + consistency_weight*con)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite loss at update {it+1}")
        loss.backward()
        opt.step()
        if opt_p:
            opt_p.step()
        if cache is not None and not stage2:
            cache.commit(active[0], fields[active[0]], it)
            counts["cache_updates"] += 1
        counts["optimizer_updates"] += 1
        counts["window_visits"] += 0 if stage2 else len(active)
        counts["stage2_updates"] += int(stage2)
        counts["training_patterns"] += cfg.n_pat if stage2 else patterns_per_group*len(active)
        # Avoid retaining four autograd graphs across iterations/evaluation.
        loss_value, data_value, con_value, reg_value = [t.item() for t in (loss, data, con, reg)]
        del fields, losses, regs, loss, data, con, reg, amp, current_probe, training_object
        if (it+1) % cfg.eval_every == 0 or it+1 == cfg.iters or it+1 == switch:
            with snapshot_readout(net):
                final_fields = [decode(k)[0] for k in range(1 if stage2 else 4)]
                fused = final_fields[0] if stage2 else fuse_fields(final_fields, masks, cache.weights if cache else None)
                cache_gap = (float((fused-fuse_fields(cache.fields, masks, cache.weights)).norm()
                                   / fused.norm().clamp_min(1e-12)) if cache and not stage2 else None)
                pp = probe()
                full_loss, predicted = scaled_amplitude_loss(cabs(_fwd(cfg, fused, pp, sc.post, sc.Q)), sc.sqrtIm)
                real = torch.linalg.vector_norm(predicted.square()-sc.Iclt).item()
                disagreement = None if stage2 else consistency_loss(final_fields, masks).item()
                rec, pc = fused.cpu().numpy(), pp.cpu().numpy()
                final_array = torch.stack(final_fields).cpu().numpy()
            counts["evaluation_network_forwards"] += 1 if stage2 else 4
            counts["evaluation_patterns"] += cfg.n_pat
            rs, cs = sc.roi
            metrics = evaluate(rec[rs, cs], sc.obj[rs, cs])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row = dict(it=it+1, loss=loss_value, data_loss=data_value, tgv_amp=reg_value,
                       consistency=con_value, post_update_disagreement=disagreement,
                       full_data_loss=full_loss.item(), real=real, relerr_p=probe_relerr(pc, sc.probe),
                       active_windows=[] if stage2 else active, stage=2 if stage2 else 1,
                       input_channels=3 if stage2 else patterns_per_group, active_patterns=cfg.n_pat if stage2 else patterns_per_group*len(active),
                       object_readout=stage2_readout if stage2 else "window-direct",
                       tgv_amp_weight=tgv_weight,
                       loss_mode=loss_mode if not stage2 else "full-data",
                       cache_fresh_relative_difference=cache_gap,
                       completed_sweeps=counts["window_visits"]//4,
                       completed_schedule_cycles=counts["window_visits"]//len(groups),
                       sweep_progress=counts["window_visits"]/4,
                       elapsed_s=time.perf_counter()-start-diagnostic_seconds, **counts, **metrics)
            if transfer_every:
                row.update(diagnostic_elapsed_s=diagnostic_seconds,
                           wall_elapsed_s=time.perf_counter()-start)
            hist.append(row)
            if cache and not stage2:
                row.update(cache.audit(it+1))
            cache_note = f" | cache/fresh gap {cache_gap:.3e}" if cache_gap is not None else ""
            print(f"[windows] stage {row['stage']} update {it+1} | stage1 sweeps {row['sweep_progress']:g} | "
                  f"amp PSNR {metrics['psnr_amp']:.2f} | object err {metrics['relerr']:.4f} | "
                  f"probe err {row['relerr_p']:.4f} | full loss {row['full_data_loss']:.4e} | "
                  f"{row['elapsed_s']:.2f}s{cache_note}", flush=True)
        if transfer_every and not stage2 and ((it+1) % transfer_every == 0 or it+1 == (switch or cfg.iters)):
            from functions.paperrepro.window_transfer import cross_group_transfer
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            diag_start = time.perf_counter()
            frozen_probe = probe().detach().clone()

            def diagnostic_decode(model, k):
                a, ph = model(inputs[k])
                return make_field(a[0], ph[:2])[crop, crop], F.softplus(a[0])[crop, crop]

            def diagnostic_loss(model, k, reg_copy):
                field, amplitude = diagnostic_decode(model, k)
                data, _ = scaled_amplitude_loss(cabs(_fwd(cfg, field, frozen_probe,
                    sc.post[selections[k]], sc.Q)), sc.sqrtIm[selections[k]])
                penalty = reg_copy(amplitude)[0] if reg_copy is not None else data.new_zeros(())
                return data + energy * cfg.tgv_amp * penalty

            def diagnostic_evaluate(model):
                rows, fields = [], []
                rs, cs = sc.roi
                with snapshot_readout(model):
                    for k in range(4):
                        field, _ = diagnostic_decode(model, k)
                        fields.append(field)
                        data, _ = scaled_amplitude_loss(cabs(_fwd(cfg, field, frozen_probe,
                            sc.post[selections[k]], sc.Q)), sc.sqrtIm[selections[k]])
                        rows.append(dict(data_loss=float(data), **evaluate(
                            field[rs, cs].cpu().numpy(), sc.obj[rs, cs])))
                    fused = fuse_fields(fields, masks)
                    fm = evaluate(fused[rs, cs].cpu().numpy(), sc.obj[rs, cs])
                return dict(groups=rows, fused=fm)

            report = cross_group_transfer(net, opt, tgvs, diagnostic_loss, diagnostic_evaluate)
            report.update(it=it+1, groups=groups,
                          roi=[sc.roi[0].start, sc.roi[0].stop, sc.roi[1].start, sc.roi[1].stop],
                          truth_metrics='simulation truth used only for evaluation; per-field global complex alignment',
                          readout='train-mode BN as production snapshot; copied buffers only')
            diagdir = outdir / 'cross_group_transfer'
            diagdir.mkdir(exist_ok=True)
            target = diagdir / f'step_{it+1:06d}.json'
            target.write_text(json.dumps(report, indent=2), encoding='utf-8')
            diagnostic_reports.append(report)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            duration = time.perf_counter()-diag_start
            diagnostic_seconds += duration
            diagnostic_records.append(dict(it=it+1, seconds=duration, file=str(target.name)))
            print(f"[transfer] update {it+1}: copied-state 4x4 diagnostic saved; extra {duration:.2f}s excluded from training time", flush=True)
            del frozen_probe
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_elapsed = time.perf_counter()-start
    elapsed = wall_elapsed-diagnostic_seconds
    print(f"[windows] training-loop time {elapsed:.2f}s / {cfg.iters} updates; "
          f"training net forwards={counts['training_network_forwards']}, "
          f"training patterns={counts['training_patterns']}, cache commits={counts['cache_updates']}", flush=True)
    _save(cfg, rec, pc, sc.obj, sc.probe, hist, sc.roi, sc.pos, tag="windows", train_elapsed_s=elapsed)
    np.savez_compressed(outdir / "window_fields.npz", fields=snapshot['fields'] if snapshot else final_array,
                        masks=masks.cpu().numpy(), groups=np.asarray(groups[:4]),
                        training_groups=np.asarray(groups),
                        weights=cache.weights.cpu().numpy() if cache else fusion_weights(masks).cpu().numpy())
    if cache is not None:
        np.savez_compressed(outdir / "window_cache.npz", fields=cache.fields.cpu().numpy(),
                            source_updates=np.asarray(cache.source_updates), visited=np.asarray(cache.visited),
                            weights=cache.weights.cpu().numpy(), masks=masks.cpu().numpy())
    if snapshot is not None:
        np.savez_compressed(outdir / "stage1_conditioning.npz", **snapshot)
        (outdir / "reconstruction_input.json").write_text(json.dumps(transition, indent=2), encoding="utf-8")
        if residual_base is not None:
            np.savez_compressed(outdir / "stage2_residual.npz", base=snapshot['obj'],
                                residual=rec-snapshot['obj'], final_object=rec)
    metadata.update(transition=transition, window_fields_stage=1,
                    final_object="single stage-2 output" if switch else "four-window fusion")
    metadata.update(counts=counts, elapsed_s=elapsed)
    if transfer_every:
        (outdir / 'cross_group_transfer.json').write_text(
            json.dumps(dict(reports=diagnostic_reports, timing=diagnostic_records), indent=2), encoding='utf-8')
        metadata.update(diagnostic_elapsed_s=diagnostic_seconds, wall_elapsed_s=wall_elapsed,
                        transfer_records=diagnostic_records,
                        diagnostic_timing_caveat='subtracts diagnostic blocks; GPU allocator/thermal/cache effects may remain; use diagnostic-off runs for throughput claims')
    if cache is not None:
        metadata.update(final_stage1_cache=cache.audit(switch or cfg.iters))
    (outdir / "window_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return hist
