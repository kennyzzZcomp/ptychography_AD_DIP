"""DeePIE on the unmodified ProPtyNet scene, optics and evaluation contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from functions.paperrepro.deepie_model import DeePIEObject, ModelConfig, coordinate_grid
from functions.paperrepro.evaluate import evaluate, probe_relerr
from functions.paperrepro.optics import forward_field
from functions.paperrepro.report import _save, _save_convergence
from functions.paperrepro.scene import build_scene


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    probe_lr: float = 1e-4
    decay_every: int = 100
    decay_factor: float = 0.5
    scan_chunk: int = 4
    coordinate_chunk: int = 4096
    scan_weight: str = "intensity"
    balance: str = "equal-norm"
    probe_scale: str = "measurement-rms"
    network_seed: int = 0
    save_scene: bool = False

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in (self.lr, self.probe_lr, self.decay_factor)):
            raise ValueError("learning rates and decay_factor must be finite and positive")
        if self.decay_factor > 1 or min(self.decay_every, self.scan_chunk, self.coordinate_chunk) < 1:
            raise ValueError("chunks/decay_every must be positive and decay_factor <= 1")
        if self.scan_weight not in ("uniform", "intensity") or self.balance not in ("none", "equal-norm"):
            raise ValueError("invalid scan_weight/balance")
        if self.probe_scale not in ("none", "measurement-rms"):
            raise ValueError("invalid probe_scale")


def scan_weights(measured_amplitude, mode):
    if mode == "uniform":
        return torch.ones(measured_amplitude.shape[0], device=measured_amplitude.device,
                          dtype=measured_amplitude.dtype)
    energy = measured_amplitude.square().mean((-2, -1))
    if energy.mean() <= 0:
        raise ValueError("all measurements are zero")
    # Explicit implementation choice, not an author-specified weighting formula.
    return (energy / energy.mean()).detach()


def measurement_loss(obj, probe, positions, q, measured, weights, chunk, backward=False):
    """Full-data mean weighted amplitude error; chunks do NOT change optimizer steps."""
    total = measured.new_zeros(())
    for start in range(0, len(positions), chunk):
        stop = start + chunk
        predicted = forward_field(obj, probe, positions[start:stop], q, probe.shape[-1]).abs()
        per_scan = (predicted - measured[start:stop]).square().mean((-2, -1))
        loss = (per_scan * weights[start:stop]).sum() / len(positions)
        if backward:
            loss.backward()
        total += loss.detach()
    return total


@torch.no_grad()
def initial_probe_scale(obj, probe, positions, q, measured, chunk):
    predicted_energy = measured.new_zeros(())
    for start in range(0, len(positions), chunk):
        value = forward_field(obj, probe, positions[start:start + chunk], q, probe.shape[-1])
        predicted_energy += value.abs().square().sum()
    if predicted_energy <= 0 or measured.square().sum() <= 0:
        raise ValueError("cannot calibrate zero-energy data or initialization")
    return (measured.square().sum() / predicted_energy).sqrt().item()


@torch.no_grad()
def balance_network_gradients(model, mode):
    groups = [list(model.amplitude.parameters()), list(model.phase.parameters())]
    norms = [torch.sqrt(sum(p.grad.square().sum() for p in group)).item() for group in groups]
    factors = [1.0, 1.0]
    if mode == "equal-norm" and min(norms) > 1e-20:
        target = math.sqrt(norms[0] * norms[1])
        factors = [min(10., max(.1, target / n)) for n in norms]
        for group, scale in zip(groups, factors):
            for p in group:
                p.grad.mul_(scale)
    return dict(amp_gradient_norm=norms[0], phase_gradient_norm=norms[1],
                amp_gradient_factor=factors[0], phase_gradient_factor=factors[1])


def exact_chunked_gradients(model, probe, coords, size, positions, q, measured, weights, tc):
    """Exact chain rule in two stages: physics -> complex field -> Fourier coefficients.

    O is held fixed throughout a full-data update. Coordinate chunks accumulate
    dL/dW; one W->Lambda pullback follows. No stale fields or stochastic batches.
    """
    effective = model.materialize()
    obj = model.field(coords, size, tc.coordinate_chunk, effective).requires_grad_()
    loss = measurement_loss(obj, probe, positions, q, measured, weights, tc.scan_chunk, True)
    model.field_vjp(coords, obj.grad.detach(), tc.coordinate_chunk, effective)
    return loss


def _source_hashes():
    root = Path(__file__).resolve().parents[2]
    names = ["simulations/DeePIE.py", "functions/paperrepro/deepie_model.py",
             "functions/paperrepro/deepie_solver.py", "simulations/ProPtyNet_paper.py",
             "functions/paperrepro/scene.py", "functions/paperrepro/sample.py",
             "functions/paperrepro/optics.py", "functions/paperrepro/evaluate.py"]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def run_deepie(cfg, mc: ModelConfig, tc: TrainingConfig, *, scene=None, scene_metadata=None):
    if cfg.iters < 1 or cfg.eval_every < 1:
        raise ValueError("iters and eval_every must be positive")
    if cfg.probe_mode != "pixel":
        raise ValueError("DeePIE baseline uses a learned pixel probe; no truth/support mode")
    out = Path(cfg.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "deepie_manifest.json").exists():
        raise FileExistsError(f"Existing DeePIE run: {out}. Choose a new --outdir.")
    device = cfg.dev()
    if device.type == "cuda":
        # FP32 reference behavior, not GPU-dependent TF32 approximation.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if scene is None:
        scene = build_scene(cfg, device)
    torch.manual_seed(tc.network_seed)
    model = DeePIEObject(mc).to(device)
    coords = coordinate_grid(cfg.obj_size, device)
    probe = torch.nn.Parameter(torch.from_numpy(scene.P0.copy()).to(device))
    scale = 1.0
    if tc.probe_scale == "measurement-rms":
        obj = model.field(coords, cfg.obj_size, tc.coordinate_chunk, model.materialize())
        scale = initial_probe_scale(obj, probe, scene.post, scene.Q, scene.sqrtIm, tc.scan_chunk)
        with torch.no_grad():
            probe.mul_(scale)
    weights = scan_weights(scene.sqrtIm, tc.scan_weight)
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": tc.lr},
                                  {"params": [probe], "lr": tc.probe_lr}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, tc.decay_every, gamma=tc.decay_factor)
    manifest = {
        "method": "DeePIE independent reproduction, adapted to ProPtyNet Fresnel simulation",
        "author_code": False, "status": "running",
        "paper_doi": "10.1364/OL.578784", "supplement_doi": "10.6084/m9.figshare.30454637",
        "scene_fingerprint": scene.fp, "scene_config": {**asdict(cfg), "quad_sign": cfg.quad_sign},
        "model_config": asdict(mc), "training_config": asdict(tc),
        "initial_probe_scale": scale, "initialization_uses_ground_truth": False,
        "object_parameters": sum(p.numel() for p in model.parameters()),
        "probe_real_parameters": 2 * probe.numel(),
        "frozen_basis_elements": sum(b.numel() for n, b in model.named_buffers() if n.endswith("basis")),
        "device": str(device), "torch_version": torch.__version__, "source_sha256": _source_hashes(),
        "unresolved_author_details": ["activation formula/role of alpha", "PE convention",
                                      "coefficient initialization", "scan-weight formula",
                                      "gradient-balancing rule", "LR decay milestones",
                                      "phase scaling inconsistency between main Eqs.4 and 5"],
        "implementation_choices": {
            "activation": "sin(omega*x); alpha unused" if mc.activation == "sine" else
                          "leaky_relu(sin(omega*x), alpha); unverified hypothesis",
            "weights": "raw cosine basis; all affine layers including output reparameterized",
            "initialization": "variance-matched random coefficients, small output weights; amplitude bias 1",
            "balance": "geometric-mean gradient norm for amplitude/phase, factors clipped [0.1,10]; NOT full GradNorm",
            "scan_weight": "mean measured intensity / grand mean, or uniform as configured",
            "probe_scale": "one-time measured RMS calibration, or none; never per-update fitting",
            "iteration": "one Adam update after all scan positions and coordinate chunks",
            "metrics": "shared evaluate() on shared scene ROI; post-update fields; truth only for reporting",
        },
    }
    manifest_path = out / "deepie_manifest.json"
    if scene_metadata is not None:
        manifest['imported_scene'] = scene_metadata
    def write_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest()
    if tc.save_scene:
        np.savez_compressed(out / "deepie_scene.npz", Im=scene.Im, Icl=scene.Icl, P0=scene.P0,
                            positions=scene.pos, Q=scene.Q.cpu().numpy(), fingerprint=scene.fp)
    print("[deepie] Independent reproduction; unspecified details are listed in deepie_manifest.json", flush=True)
    print(f"[deepie] object parameters={manifest['object_parameters']:,}; "
          f"probe real parameters={manifest['probe_real_parameters']:,}; P0 scale={scale:.6g}", flush=True)
    print(f"[deepie] {cfg.n_pat} scans/update; scan chunk={tc.scan_chunk}, coordinate chunk={tc.coordinate_chunk}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    history = []
    rs, cs = scene.roi
    try:
        for it in range(cfg.iters):
            optimizer.zero_grad(set_to_none=True)
            pre_loss = exact_chunked_gradients(model, probe, coords, cfg.obj_size, scene.post,
                                              scene.Q, scene.sqrtIm, weights, tc)
            stats = balance_network_gradients(model, tc.balance)
            if not torch.isfinite(pre_loss) or any(not torch.isfinite(p.grad).all()
                                                   for p in [*model.parameters(), probe]):
                raise FloatingPointError(f"Non-finite loss/gradient at update {it + 1}")
            used_lrs = [g["lr"] for g in optimizer.param_groups]
            optimizer.step()
            scheduler.step()
            if (it + 1) % cfg.eval_every == 0 or it == cfg.iters - 1:
                with torch.no_grad():
                    obj = model.field(coords, cfg.obj_size, tc.coordinate_chunk, model.materialize())
                    post_loss = measurement_loss(obj, probe, scene.post, scene.Q, scene.sqrtIm,
                                                 weights, tc.scan_chunk)
                    raw_loss = measurement_loss(obj, probe, scene.post, scene.Q, scene.sqrtIm,
                                                torch.ones_like(weights), tc.scan_chunk)
                    if not torch.isfinite(post_loss):
                        raise FloatingPointError("Non-finite post-update loss")
                    rec, pc = obj.cpu().numpy(), probe.detach().cpu().numpy().copy()
                metrics = evaluate(rec[rs, cs], scene.obj[rs, cs])
                row = {"it": it + 1, "loss": post_loss.item(), "pre_update_loss": pre_loss.item(),
                       "data_loss": raw_loss.item(), "relerr_p": probe_relerr(pc, scene.probe),
                       "lr_object": used_lrs[0], "lr_probe": used_lrs[1], **stats, **metrics}
                history.append(row)
                print(f"[deepie] update {it+1}/{cfg.iters} | amplitude MSE {row['data_loss']:.5e} | "
                      f"amp PSNR {row['psnr_amp']:.3f} | phase PSNR {row['psnr_phs']:.3f}", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        manifest.update(status="complete", completed_updates=cfg.iters, train_elapsed_s=elapsed,
                        timing_scope="training loop including evaluation, excluding simulation and final output",
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                        final_metrics=history[-1])
        # Full trainable representation supports arbitrary-coordinate evaluation.
        torch.save({"model": model.state_dict(), "probe": probe.detach().cpu(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "model_config": asdict(mc), "training_config": asdict(tc),
                    "scene_config": manifest["scene_config"], "completed_updates": cfg.iters,
                    "scene_fingerprint": scene.fp}, out / "deepie_checkpoint.pt")
        _save(cfg, rec, pc, scene.obj, scene.probe, history, scene.roi, scene.pos,
              tag="deepie", train_elapsed_s=elapsed)
        # Preserve the common NPZ layout, while making this result self-describing.
        result_path = out / "deepie_result.npz"
        with np.load(result_path, allow_pickle=False) as saved:
            result = {key: saved[key] for key in saved.files}
        result.update(cfg=json.dumps(manifest["scene_config"]),
                      model_config=json.dumps(asdict(mc)), training_config=json.dumps(asdict(tc)),
                      scene_fingerprint=scene.fp, initial_probe_scale=scale)
        np.savez_compressed(result_path, **result)
        import matplotlib.pyplot as plt
        _save_convergence(cfg, history, "deepie", plt)
        (out / "deepie_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        write_manifest()
        print(f"[deepie] completed in {elapsed:.2f}s; outputs: {out}", flush=True)
        return history
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_manifest()
        raise
