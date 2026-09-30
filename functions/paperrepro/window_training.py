"""Four measurement windows, full-canvas CNN outputs, one independent probe.

This is a separate pilot, not a change to the paper or progressive solver.
Sequential mode: one window and one optimizer step per iteration. Joint mode:
four windows at the same weights and one optimizer step per iteration.
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


def fuse_fields(fields, masks):
    fields = torch.stack(list(fields))
    count = masks.sum(0)
    merged = (fields * masks).sum(0) / count.clamp_min(1)
    return torch.where(count > 0, merged, torch.ones_like(merged))


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
    switch = getattr(cfg, "window_switch_after", 0)
    if switch and not 0 < switch < cfg.iters:
        raise ValueError("switch-after must be positive and less than total iters (or 0 to disable)")
    if switch and cfg.window_update == "sequential" and switch % 4:
        raise ValueError("Sequential stage 1 must end after a complete four-window sweep")
    if switch:
        if any(not np.isfinite(v) or v <= 0 for v in (cfg.window_stage2_lr_net, cfg.window_stage2_lr_probe)):
            raise ValueError("Stage-2 learning rates must be finite and positive")
        if not np.isfinite(cfg.window_stage2_tgv) or cfg.window_stage2_tgv < 0:
            raise ValueError("Stage-2 TGV must be finite and nonnegative")
    groups = four_windows(cfg.grid)
    update_windows(0, cfg.window_update)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    device = cfg.dev()
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    sc = build_scene(cfg, device)
    if cfg.network_seed is not None:
        torch.manual_seed(cfg.network_seed)
    net = ProPtyUNet(16, cfg.base_ch, n_fields=1, ph_ch=2,
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
    masks = footprint_masks(sc.pos, groups, m, cfg.N, device)
    # Independent auxiliary TGV state per window; nominal domains, as in net.
    tgvs = [ObjectAmplitudeTGV(cfg, sc.pos[g], device) for g in groups] if cfg.tgv_amp else [None]*4
    energy = sc.sqrtIm.square().mean().detach()

    def decode(k):
        a, ph = net(inputs[k])
        return make_field(a[0], ph[:2])[crop, crop], F.softplus(a[0])[crop, crop]

    def probe():
        return _probe_of(mode, fixed, pr, pi, support)

    counts = dict(optimizer_updates=0, window_visits=0, training_patterns=0,
                  training_network_forwards=0, consistency_network_forwards=0,
                  evaluation_network_forwards=0, evaluation_patterns=0,
                  transition_network_forwards=0, stage2_updates=0)
    print(f"[windows] {cfg.window_update}; iteration = ONE optimizer update; "
          "Stage 1: 4x16 = 64 unique training frames, remaining 36 evaluation-only in stage 1. "
          f"consistency={cfg.window_consistency:g}; full-canvas outputs, no added support.", flush=True)
    if cfg.window_update == "sequential" and cfg.window_consistency:
        print("[windows] Consistency adds 3 current-weight no-grad peer forwards per update; "
              "BN buffers restored. No stale cache; extra work counted.", flush=True)
    metadata = dict(groups=groups, missing_indices=sorted(set(range(cfg.n_pat))-set(sum(groups, []))),
                    update_mode=cfg.window_update, consistency_weight=cfg.window_consistency,
                    input_channels=16, output="full global canvas for each window",
                    fusion="complex mean on full computational footprints; ones outside union",
                    consistency_domain="full computational footprint intersections, NOT confidence/support",
                    calibration="independent analytic amplitude scale per training window; global scale for fused evaluation",
                    evaluation="fresh post-update fields, train-mode BN with restored buffers; no GT in training",
                    iteration="one network/probe optimizer update", timing="loop includes evaluation; excludes setup/save",
                    device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    network_parameters=sum(p.numel() for p in net.parameters()), scene_fingerprint=sc.fp)
    metadata.update(switch_after=switch, total_updates=cfg.iters,
                    stage2="fresh object-only input, full measurements" if switch else None)
    (outdir / "window_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    hist = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
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
                stage1_object = fuse_fields(stage1_fields, masks).detach().clone()
                stage1_probe = probe().detach().clone()
            counts["transition_network_forwards"] += 4
            conditioning, encoding = reconstruction_channels(stage1_object, None, ns, include_probe=False)
            snapshot = dict(obj=stage1_object.cpu().numpy(), probe=stage1_probe.cpu().numpy(),
                            fields=torch.stack(stage1_fields).cpu().numpy(),
                            network_input=conditioning.cpu().numpy())
            # Same isolated initialization and neutral readout as progressive fresh.
            seed = cfg.seed if cfg.network_seed is None else cfg.network_seed
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(seed))
                new_net = ProPtyUNet(3, cfg.base_ch, n_fields=1, ph_ch=2,
                                    skip_mode=cfg.skip_mode, wavelet_threshold=cfg.wavelet_threshold)
                _initialize_object_heads(new_net, cfg.obj_init_alpha)
            net = new_net.to(device)
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
            counts["transition_network_forwards"] += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            transition = dict(**encoding, after_update=switch, seed=seed,
                              new_object_network=True, new_object_optimizer=True,
                              probe_optimizer_retained=True, probe_adam_steps_at_switch=probe_steps,
                              probe_relative_jump=probe_jump, object_relative_jump=obj_jump,
                              object_initialization="neutral heads, NOT residual warm start",
                              new_full_domain_tgv_auxiliary=bool(cfg.window_stage2_tgv),
                              lr_net=cfg.window_stage2_lr_net, lr_probe=cfg.window_stage2_lr_probe,
                              tgv_amp=cfg.window_stage2_tgv,
                              stage1_elapsed_s=transition_start-start,
                              transition_elapsed_s=time.perf_counter()-transition_start,
                              network_parameters=sum(v.numel() for v in net.parameters()))
            print(f"[windows] update {it+1}: fresh U-Net; fixed object-only 3-channel input; "
                  f"100 measurements; retained probe/Adam, probe jump={probe_jump:g}; "
                  f"lr_net={cfg.window_stage2_lr_net:g}, lr_probe={cfg.window_stage2_lr_probe:g}, "
                  f"TGV={cfg.window_stage2_tgv:g}", flush=True)
            del stage1_fields, stage1_object, stage1_probe, initial_obj, new_net
        active = [0] if stage2 else update_windows(it, cfg.window_update)
        consistency_weight = 0. if stage2 else cfg.window_consistency
        tgv_weight = cfg.window_stage2_tgv if stage2 else cfg.tgv_amp
        # Read peers BEFORE any grad-enabled forward: restoring BN buffers after
        # such a forward would otherwise invalidate autograd saved tensors.
        fields = [None]*4
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
            data, _ = scaled_amplitude_loss(cabs(_fwd(cfg, fields[k], current_probe,
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
        counts["optimizer_updates"] += 1
        counts["window_visits"] += 0 if stage2 else len(active)
        counts["stage2_updates"] += int(stage2)
        counts["training_patterns"] += cfg.n_pat if stage2 else 16*len(active)
        # Avoid retaining four autograd graphs across iterations/evaluation.
        loss_value, data_value, con_value, reg_value = [t.item() for t in (loss, data, con, reg)]
        del fields, losses, regs, loss, data, con, reg, amp, current_probe
        if (it+1) % cfg.eval_every == 0 or it+1 == cfg.iters or it+1 == switch:
            with snapshot_readout(net):
                final_fields = [decode(k)[0] for k in range(1 if stage2 else 4)]
                fused = final_fields[0] if stage2 else fuse_fields(final_fields, masks)
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
                       input_channels=3 if stage2 else 16, active_patterns=cfg.n_pat if stage2 else 16*len(active),
                       tgv_amp_weight=tgv_weight,
                       completed_sweeps=counts["window_visits"]//4,
                       sweep_progress=counts["window_visits"]/4,
                       elapsed_s=time.perf_counter()-start, **counts, **metrics)
            hist.append(row)
            print(f"[windows] stage {row['stage']} update {it+1} | stage1 sweeps {row['sweep_progress']:g} | "
                  f"amp PSNR {metrics['psnr_amp']:.2f} | object err {metrics['relerr']:.4f} | "
                  f"probe err {row['relerr_p']:.4f} | full loss {row['full_data_loss']:.4e} | "
                  f"{row['elapsed_s']:.2f}s", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-start
    _save(cfg, rec, pc, sc.obj, sc.probe, hist, sc.roi, sc.pos, tag="windows", train_elapsed_s=elapsed)
    np.savez_compressed(outdir / "window_fields.npz", fields=snapshot['fields'] if snapshot else final_array,
                        masks=masks.cpu().numpy(), groups=np.asarray(groups))
    if snapshot is not None:
        np.savez_compressed(outdir / "stage1_conditioning.npz", **snapshot)
        (outdir / "reconstruction_input.json").write_text(json.dumps(transition, indent=2), encoding="utf-8")
    metadata.update(transition=transition, window_fields_stage=1,
                    final_object="single stage-2 output" if switch else "four-window fusion")
    metadata.update(counts=counts, elapsed_s=elapsed)
    (outdir / "window_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return hist
