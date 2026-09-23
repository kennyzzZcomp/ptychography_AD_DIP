"""Opt-in, full-batch constant-LR checkpoint intervention experiment.

Legacy solvers are unchanged. Saved arrays AND evaluation refer to post-update
states. Additional decoding preserves BN buffers/RNG; it cannot perturb training.
Checkpoints contain only tensors and primitive containers (weights_only load).
"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from functions.addip.model import ProPtyUNet, make_field
from functions.addip.losses import cabs
from functions.paperrepro.scene import build_scene
from functions.paperrepro.evaluate import evaluate, probe_relerr
from functions.paperrepro.report import _save
from functions.paperrepro.tgv import ObjectAmplitudeTGV, tgv2_terms


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "paper-branch-v1":
        raise ValueError("Not a full paper branch checkpoint; result NPZ cannot resume a network")
    return state


def resume_config(args, cfg_type):
    state = load_checkpoint(args.resume)
    allowed = {"mode", "resume", "branch", "iters", "outdir", "device",
               "eval_every", "checkpoint_out", "fwd_chunk", "lr_obj"}
    for key, value in vars(args).items():
        if value is not None and key not in allowed:
            raise ValueError(f"Resume inherits its scene and solver settings; do not pass --{key.replace('_', '-')}")
    if args.iters is None or args.outdir is None:
        raise ValueError("Resume requires explicit --iters (additional steps) and a NEW --outdir")
    branch = args.branch or "continue"
    if getattr(args, "lr_obj", None) is not None and branch not in ("C", "D"):
        raise ValueError("--lr-obj override is only allowed for pixel branches C/D")
    if branch != "continue" and state["representation"] != "net":
        raise ValueError("A/B/C/D must all start from the same NETWORK checkpoint")
    kw = dict(state["cfg"])
    quad_sign = kw.pop("quad_sign", -1.)
    for key in allowed - {"mode", "branch"}:
        value = getattr(args, key, None)
        if value is not None:
            kw[key] = value
    kw["checkpoint_out"] = args.checkpoint_out or str(Path(args.outdir) / "checkpoint.pt")
    kw["branch"] = branch
    if branch in ("B", "D"):
        kw["tgv_amp"] = 0.
    if branch in ("A", "C") and kw["tgv_amp"] <= 0:
        raise ValueError("A/C retain TGV and require a TGV-enabled parent")
    cfg = cfg_type(**kw)
    cfg.quad_sign = quad_sign
    return cfg


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), root/'functions/addip/model.py',
             root/'functions/common/unet.py', root/'functions/paperrepro/tgv.py',
             root/'functions/paperrepro/optics.py', root/'functions/paperrepro/solvers_addip.py']
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def data_loss(cfg, obj, probe, scene):
    from functions.paperrepro.solvers_addip import _fwd
    predicted = cabs(_fwd(cfg, obj, probe, scene["post"], scene["Q"]))
    target = scene["sqrtIm"]
    predicted = predicted * ((predicted*target).sum() / predicted.square().sum().clamp_min(1e-20))
    return F.mse_loss(predicted, target)


def observational_decode(net, decode):
    """Training-mode output without extra BN updates or RNG consumption."""
    buffers = {} if net is None else {k: b.clone() for k, b in net.named_buffers()}
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        with torch.no_grad():
            return tuple(t.detach().clone() for t in decode())
    finally:
        if net is not None:
            for k, b in net.named_buffers():
                b.copy_(buffers[k])
        torch.set_rng_state(rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)


def run_branch(cfg, scene_factory=build_scene):
    # Deliberately bounded protocol, rather than silently mishandling schedules.
    if cfg.measurement_schedule or cfg.input_policy != "full" or cfg.lr_cosine or cfg.tgv_phase:
        raise ValueError("Branch protocol supports full input, constant LR, amplitude TGV only")
    if cfg.iters < 1 or cfg.eval_every < 1:
        raise ValueError("iters and eval_every must be positive")
    out = Path(cfg.outdir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose a NEW empty output directory: {out}")
    target = Path(cfg.checkpoint_out)
    if target.exists():
        raise FileExistsError(f"Will not overwrite checkpoint: {target}")
    parent = load_checkpoint(cfg.resume) if cfg.resume else None
    if parent and parent["sources"] != source_hashes():
        raise ValueError("Solver sources differ from checkpoint; use the same code revision for this experiment")
    device = cfg.dev()
    torch.manual_seed(cfg.seed)
    # Stable test/continuation semantics; same environment still required for bitwise replay.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if parent:
        scene = {k: v.to(device) for k, v in parent["scene"].items()}
        roi_values = parent["roi"]
        x = parent["input"].to(device)
        start = parent["step"]
        representation = parent["representation"] if cfg.branch == "continue" else (
            "pixel" if cfg.branch in ("C", "D") else "net")
    else:
        sc = scene_factory(cfg, device)
        scene = {"obj_gt": torch.as_tensor(sc.obj, device=device),
                 "probe_gt": torch.as_tensor(sc.probe, device=device),
                 "post": sc.post, "Q": sc.Q, "sqrtIm": sc.sqrtIm,
                 "Im": sc.Imt, "Icl": sc.Iclt, "P0": torch.as_tensor(sc.P0, device=device)}
        rs, cs = sc.roi
        roi_values = [rs.start, rs.stop, cs.start, cs.stop]
        pad = (cfg.obj_size-cfg.N)//2
        pad_n = (cfg.net_size-cfg.obj_size)//2
        x = F.pad(scene["Im"][None], (pad,)*4)
        x = F.pad(x, (pad_n, cfg.net_size-cfg.obj_size-pad_n)*2)
        x = x / x.amax(dim=(2, 3), keepdim=True).clamp_min(1e-12)
        start, representation = 0, "net"
    rs = slice(*roi_values[:2]); cs = slice(*roi_values[2:])
    pos = scene["post"].cpu().numpy()
    obj_gt = scene["obj_gt"].cpu().numpy()
    probe_gt = scene["probe_gt"].cpu().numpy()
    net = None
    crop = slice((cfg.net_size-cfg.obj_size)//2, (cfg.net_size-cfg.obj_size)//2+cfg.obj_size)
    if representation == "net":
        net = ProPtyUNet(cfg.n_pat, cfg.base_ch, n_fields=1, ph_ch=2).to(device)
        if parent:
            net.load_state_dict(parent["network"])
        else:
            with torch.no_grad():
                net.head_amp.weight[0].mul_(cfg.obj_init_alpha)
                net.head_phs.weight[:2].mul_(cfg.obj_init_alpha)
                net.head_amp.bias[0] = math.log(math.e-1.)
                net.head_phs.bias[:2] = 0.
                net.head_phs.bias[0] = 1.
        object_params = list(net.parameters())
    else:
        saved = parent["object"].to(device)
        Or, Oi = nn.Parameter(saved.real.clone()), nn.Parameter(saved.imag.clone())
        object_params = [Or, Oi]
    from functions.paperrepro.solvers_addip import _probe_setup, _probe_of
    mode, fixed, Pr, Pi, mask = _probe_setup(
        cfg, SimpleNamespace(P0=scene["P0"].cpu().numpy()), probe_gt, device, "branch")
    if parent and mode != "truth":
        with torch.no_grad():
            Pr.copy_(parent["probe_real"].to(device)); Pi.copy_(parent["probe_imag"].to(device))
    opt_obj = torch.optim.Adam(object_params, lr=cfg.lr_net if net is not None else cfg.lr_obj,
                               weight_decay=cfg.weight_decay if net is not None else 0.)
    # Deliberately SAME probe learning rate in all branches (not legacy AD lr_prb).
    opt_probe = None if mode == "truth" else torch.optim.Adam([Pr, Pi], lr=cfg.lr_probe)
    tgv = ObjectAmplitudeTGV(cfg, pos, device) if cfg.tgv_amp > 0 else None
    if parent and tgv is not None:
        with torch.no_grad():
            tgv.v.copy_(parent["tgv_vector"].to(device))
    if parent and cfg.branch == "continue":
        opt_obj.load_state_dict(parent["optimizer_object"])
        if opt_probe is not None:
            opt_probe.load_state_dict(parent["optimizer_probe"])
        if tgv is not None:
            tgv.opt.load_state_dict(parent["optimizer_tgv"])
    if parent:
        torch.set_rng_state(parent["rng_cpu"])
        if device.type == "cuda" and parent["rng_cuda"]:
            torch.cuda.set_rng_state_all(parent["rng_cuda"])

    def decode():
        if net is not None:
            a, p = net(x)
            O = make_field(a[0], p[:2])[crop, crop]
            amplitude = F.softplus(a[0])[crop, crop]
        else:
            O = torch.complex(Or, Oi)
            amplitude = O.abs()
        return O, _probe_of(mode, fixed, Pr, Pi, mask), amplitude

    initial_O, initial_P, _ = observational_decode(net, decode)
    checks = {}
    if parent:
        for key, actual, expected in [("object", initial_O, parent["object"]),
                                      ("probe", initial_P, parent["probe"])]:
            expected = expected.to(device)
            checks[key+"_max_abs_difference"] = float((actual-expected).abs().max())
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        with torch.no_grad():
            initial_loss = float(data_loss(cfg, initial_O, initial_P, scene))
        checks["data_loss"] = initial_loss
        if not math.isclose(initial_loss, parent["final_data_loss"], rel_tol=1e-4, abs_tol=1e-10):
            raise ValueError("Branch starting data loss differs from parent")
    out.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"parent": str(Path(cfg.resume).resolve()) if parent else None,
                "parent_sha256": hashlib.sha256(Path(cfg.resume).read_bytes()).hexdigest() if parent else None,
                "branch": cfg.branch if parent else "prefix", "representation": representation,
                "start_step": start, "additional_steps": cfg.iters,
                "optimizer_reset": bool(parent and cfg.branch != "continue"),
                "start_checks": checks, "sources": source_hashes(),
                "protocol": "post-update metrics; BN-preserving observation; full-batch constant LR",
                "probe_lr": cfg.lr_probe, "object_lr": opt_obj.param_groups[0]["lr"]}
    print(f"[branch] {metadata['branch']} | {representation} | start={start} + {cfg.iters} | TGV={cfg.tgv_amp}")
    print(f"[branch] inherited probe_mode={mode}; start checks={checks}", flush=True)
    energy = scene["sqrtIm"].square().mean().detach()
    history = []
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    sync(); began = time.perf_counter()
    training_seconds = 0.
    for i in range(cfg.iters):
        sync(); tick = time.perf_counter()
        O, P, amplitude = decode()
        before_O, before_P = O.detach(), P.detach()
        data = data_loss(cfg, O, P, scene)
        loss = data
        if tgv is not None:
            reg, _, _ = tgv(amplitude)
            loss = loss + cfg.tgv_amp*energy*reg
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite loss; do not interpret as a mechanism result")
        opt_obj.zero_grad(set_to_none=True)
        if opt_probe is not None:
            opt_probe.zero_grad(set_to_none=True)
        loss.backward(); opt_obj.step()
        if opt_probe is not None:
            opt_probe.step()
        sync(); training_seconds += time.perf_counter()-tick
        if i == 0 or (i+1) % cfg.eval_every == 0 or i == cfg.iters-1:
            Oa, Pa, Aa = observational_decode(net, decode)
            with torch.no_grad():
                dl = float(data_loss(cfg, Oa, Pa, scene))
                reg_now = float(tgv2_terms(tgv._prepare(Aa), tgv.v, tgv.mask,
                                          cfg.tgv_alpha0, cfg.tgv_alpha1, cfg.tgv_eps)[0]) if tgv else 0.
            rec, pc = Oa.cpu().numpy(), Pa.cpu().numpy()
            record = {"it": start+i+1, "branch_step": i+1,
                      "data_loss": dl, "relative_data_loss": dl/float(energy.clamp_min(1e-20)),
                      "tgv_amp": reg_now, "tgv_weighted": cfg.tgv_amp*float(energy)*reg_now,
                      "loss": dl+cfg.tgv_amp*float(energy)*reg_now,
                      "relerr_p": probe_relerr(pc, probe_gt),
                      "object_relative_update": float(torch.linalg.vector_norm(Oa-before_O)/torch.linalg.vector_norm(before_O).clamp_min(1e-20)),
                      "probe_relative_update": float(torch.linalg.vector_norm(Pa-before_P)/torch.linalg.vector_norm(before_P).clamp_min(1e-20)),
                      **evaluate(rec[rs, cs], obj_gt[rs, cs])}
            sync()
            record.update(elapsed_s=time.perf_counter()-began, training_s=training_seconds)
            history.append(record)
            print(f"[branch] it {record['it']} | data {dl:.4e} | PSNR {record['psnr_amp']:.2f} | obj {record['relerr']:.4f} | probe {record['relerr_p']:.4f}", flush=True)
    sync()
    elapsed = time.perf_counter()-began
    state_cfg = asdict(cfg); state_cfg["quad_sign"] = cfg.quad_sign
    state = {"format": "paper-branch-v1", "cfg": state_cfg, "step": start+cfg.iters,
             "representation": representation, "scene": scene, "roi": roi_values, "input": x,
             "network": net.state_dict() if net is not None else None,
             "object": Oa, "probe": Pa, "probe_real": Pr, "probe_imag": Pi,
             "optimizer_object": opt_obj.state_dict(),
             "optimizer_probe": opt_probe.state_dict() if opt_probe else None,
             "tgv_vector": tgv.v if tgv else None, "optimizer_tgv": tgv.opt.state_dict() if tgv else None,
             "rng_cpu": torch.get_rng_state(),
             "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
             "history": history, "final_data_loss": dl, "sources": source_hashes(),
             "cumulative_elapsed_s": (parent["cumulative_elapsed_s"] if parent else 0.)+elapsed}
    # Exclusive creation: never overwrite a parent or existing checkpoint.
    with target.open('xb') as f:
        torch.save(cpu_tree(state), f)
    metadata.update(elapsed_s=elapsed, training_s=training_seconds,
                    cumulative_elapsed_s=state["cumulative_elapsed_s"], checkpoint=str(target))
    (out/'branch_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    _save(cfg, rec, pc, obj_gt, probe_gt, history, (rs, cs), pos,
          tag='net' if representation == 'net' else 'ad')
    print(f"[branch] saved {target}; elapsed {elapsed:.1f}s (excludes setup/checkpoint/plots)")
    return history
