#!/usr/bin/env python
"""Isolate DeePIE failure: audit, supervised object fit, or known-probe inversion.

These use ground truth for diagnosis and MUST NOT be reported as blind baselines.
Examples and interpretation: simulations/DeePIE_README.md, diagnostic section.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulations.DeePIE import parser as base_parser, configurations
from functions.paperrepro.deepie_model import DeePIEObject, coordinate_grid
from functions.paperrepro.deepie_solver import (exact_chunked_gradients, measurement_loss,
                                               balance_network_gradients, scan_weights)
from functions.paperrepro.evaluate import evaluate
from functions.paperrepro.optics import forward_field
from functions.paperrepro.scene import build_scene


def _json_write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def field_statistics(field):
    """Scale-invariant departure from the best complex constant; no phase unwrap."""
    field = np.asarray(field, dtype=np.complex128)
    energy = np.linalg.norm(field)
    amp = np.abs(field)
    return {
        "relative_distance_to_constant": float(np.linalg.norm(field - field.mean()) / max(energy, 1e-30)),
        "amplitude_cv": float(amp.std() / max(amp.mean(), 1e-30)),
        "amplitude_mean": float(amp.mean()),
        "amplitude_std": float(amp.std()),
        "complex_rms": float(np.sqrt(np.mean(np.abs(field)**2))),
    }


def audit_result(result_path, outdir):
    """Use saved arrays only: no regenerated scene, checkpoint loading or training."""
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / "diagnostic.json"
    if target.exists():
        raise FileExistsError(f"Choose a new output directory: {outdir}")
    with np.load(result_path, allow_pickle=False) as data:
        rec, gt = data["obj_rec"], data["obj_gt"]
        a, b, c, d = map(int, data["roi"])
        pc, pg = data["probe_rec"], data["probe_gt"]
        history = json.loads(str(data["hist"].item())) if "hist" in data else []
    if rec.shape != gt.shape or not (0 <= a < b <= gt.shape[0] and 0 <= c < d <= gt.shape[1]):
        raise ValueError("Invalid object shapes or ROI in result")
    rec_roi, gt_roi = rec[a:b,c:d], gt[a:b,c:d]
    metrics = evaluate(rec_roi, gt_roi)
    constant_metrics = evaluate(np.ones_like(gt_roi), gt_roi)
    report = {
        "diagnostic": "audit", "source": str(result_path.resolve()),
        "roi": [a,b,c,d], "metrics": metrics, "constant_object_metrics": constant_metrics,
        "gain_over_constant_db": {k: metrics[k]-constant_metrics[k] for k in ("psnr_amp","psnr_phs")},
        "object_statistics": field_statistics(rec_roi), "truth_statistics": field_statistics(gt_roi),
        "probe_statistics": field_statistics(pc), "probe_truth_statistics": field_statistics(pg),
        "first_recorded_update": history[0] if history else None,
        "last_recorded_update": history[-1] if history else None,
        "interpretation": "Low gain over constant AND low spatial variation support a nearly constant object. "
                          "PSNR alone does not prove collapse. Probe-only explanation requires a separate control.",
    }
    manifest = result_path.parent / "deepie_manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        report["training_config"] = saved.get("training_config")
        report["model_config"] = saved.get("model_config")
        report["scene_fingerprint"] = saved.get("scene_fingerprint")
    _json_write(target, report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("psnr_amp", "psnr_phs")):
        if history:
            ax.plot([r["it"] for r in history], [r[metric] for r in history], label="saved reconstruction")
        ax.axhline(constant_metrics[metric], color="red", linestyle="--", label="constant object")
        ax.set(xlabel="Completed updates", ylabel=metric + " (dB)")
        ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(outdir / "constant_comparison.png", dpi=130); plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


@torch.no_grad()
def scaled_truth_probe(scene, cfg, chunk):
    """Undo simulator's global max-intensity normalization using truth, DIAGNOSTIC ONLY."""
    obj = torch.from_numpy(scene.obj).to(scene.Q.device)
    probe = torch.from_numpy(scene.probe).to(scene.Q.device)
    maximum = obj.real.new_zeros(())
    for start in range(0, len(scene.post), chunk):
        u = forward_field(obj, probe, scene.post[start:start+chunk], scene.Q, cfg.N)
        maximum = torch.maximum(maximum, (u.real.square()+u.imag.square()).max())
    if maximum <= 0:
        raise ValueError("Zero truth intensity")
    return probe * maximum.rsqrt()


def supervised_gradients(model, coords, amp_target, phase_target, chunk):
    """Direct amplitude and phase MSE, no diffraction, no learned probe, no field detach."""
    effective = model.materialize()
    total = coords.new_zeros(())
    for start in range(0, len(coords), chunk):
        xy = coords[start:start+chunk]
        amp = model.amplitude(xy, effective[0])
        phase = model.phase(xy, effective[1]) * model.cfg.phase_scale
        loss = ((amp-amp_target[start:start+chunk]).square().sum()
                + (phase-phase_target[start:start+chunk]).square().sum()) / len(coords)
        loss.backward()
        total += loss.detach()
    model.amplitude.pullback(effective[0]); model.phase.pullback(effective[1])
    return total


@torch.no_grad()
def effective_snapshot(model):
    return {f"{name}.{i}": (layer.effective_weight() if hasattr(layer,"effective_weight")
                            else layer.weight).detach().clone()
            for name, net in (("amplitude",model.amplitude),("phase",model.phase))
            for i,layer in enumerate(net.layers)}


@torch.no_grad()
def effective_update_stats(model, before):
    after = effective_snapshot(model)
    return {key: {"before_rms": old.square().mean().sqrt().item(),
                  "delta_rms": (after[key]-old).square().mean().sqrt().item(),
                  "relative_update": ((after[key]-old).norm()/old.norm().clamp_min(1e-30)).item()}
            for key,old in before.items()}


def run_diagnostic(mode, cfg, mc, tc, fit_domain="roi", source_manifest=None):
    if mode not in ("fit-object", "known-probe", "pixel-known-probe"):
        raise ValueError(f"Unknown diagnostic: {mode}")
    pixel_mode = mode == "pixel-known-probe"
    out = Path(cfg.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "diagnostic.json").exists():
        raise FileExistsError(f"Choose a new diagnostic output directory: {out}")
    device = cfg.dev()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    scene = build_scene(cfg, device)
    if source_manifest:
        original = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
        expected = original.get("scene_fingerprint")
        print(f"[diagnostic] source fingerprint={expected}; regenerated={scene.fp}")
        if expected and expected != scene.fp:
            print("[diagnostic] WARNING: scene differs from source run; check overrides/assets/backend before comparison.")
    torch.manual_seed(tc.network_seed)
    model = DeePIEObject(mc).to(device)
    coords = coordinate_grid(cfg.obj_size, device)
    if pixel_mode:
        # Match the network control's initial complex field exactly, then free each pixel.
        with torch.no_grad():
            pixels = torch.nn.Parameter(model.field(coords, cfg.obj_size, tc.coordinate_chunk,
                                                     model.materialize()).clone())
        del model
        model = None
    parameters = [pixels] if pixel_mode else list(model.parameters())
    rs, cs = scene.roi
    selection = (rs,cs) if fit_domain == "roi" else (slice(None),slice(None))
    fit_coords = coords.reshape(cfg.obj_size,cfg.obj_size,2)[selection].reshape(-1,2)
    gt_field = torch.from_numpy(scene.obj).to(device)
    amp_target, phase_target = gt_field[selection].abs().reshape(-1), gt_field[selection].angle().reshape(-1)
    probe = scaled_truth_probe(scene,cfg,tc.scan_chunk)
    weights = scan_weights(scene.sqrtIm,tc.scan_weight)
    with torch.no_grad():
        truth_loss = measurement_loss(gt_field,probe,scene.post,scene.Q,scene.Iclt.sqrt(),
                                      torch.ones_like(weights),tc.scan_chunk).item()
    measured_energy = scene.sqrtIm.square().mean().clamp_min(1e-30).item()
    print(f"[diagnostic] truth clean amplitude MSE={truth_loss:.6g}")
    opt = torch.optim.Adam(parameters,lr=tc.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt,tc.decay_every,gamma=tc.decay_factor)
    metadata = {
        "diagnostic": mode, "not_a_blind_baseline": True, "status": "running",
        "fit_domain": fit_domain if mode == "fit-object" else None,
        "source_manifest": str(source_manifest) if source_manifest else None,
        "source_model_weights_loaded": False,
        "object_parameterization": "complex pixels" if pixel_mode else "coordinate network",
        "network_used_for_initialization_only": pixel_mode,
        "scene_fingerprint": scene.fp,
        "scene_config": {**asdict(cfg),"quad_sign":cfg.quad_sign},
        "model_config":asdict(mc),"training_config":asdict(tc),
        "truth_clean_amplitude_mse":truth_loss,
        "constant_object_metrics":evaluate(np.ones_like(scene.obj[rs,cs]),scene.obj[rs,cs]),
        "source_sha256": {str(Path(p).name):hashlib.sha256(Path(p).read_bytes()).hexdigest()
                          for p in (__file__, Path(__file__).parents[1]/"functions/paperrepro/deepie_model.py")},
    }
    history=[]
    def record(it, pre_loss=None, gradient_stats=None, weight_stats=None):
        with torch.no_grad():
            if pixel_mode:
                obj=pixels.detach()
            else:
                eff=model.materialize()
                obj=model.field(coords,cfg.obj_size,tc.coordinate_chunk,eff)
            relative_data_loss=None
            if mode == "fit-object":
                loss=coords.new_zeros(())
                for start in range(0,len(fit_coords),tc.coordinate_chunk):
                    xy=fit_coords[start:start+tc.coordinate_chunk]
                    a=model.amplitude(xy,eff[0]);p=model.phase(xy,eff[1])*mc.phase_scale
                    loss+=((a-amp_target[start:start+len(xy)]).square().sum()
                           +(p-phase_target[start:start+len(xy)]).square().sum())/len(fit_coords)
            else:
                loss=measurement_loss(obj,probe,scene.post,scene.Q,scene.sqrtIm,weights,tc.scan_chunk)
                data_loss = loss if tc.scan_weight == "uniform" else measurement_loss(
                    obj,probe,scene.post,scene.Q,scene.sqrtIm,torch.ones_like(weights),tc.scan_chunk)
                relative_data_loss=data_loss.item()/measured_energy
            rec=obj.cpu().numpy()
        row={"it":it,"loss":loss.item(),"pre_update_loss":pre_loss,
             "relative_amplitude_mse":relative_data_loss,
             "metrics":evaluate(rec[rs,cs],scene.obj[rs,cs]),
             "object_statistics":field_statistics(rec[rs,cs]),
             "gradient_stats":gradient_stats,"effective_weight_update":weight_stats}
        history.append(row)
        _json_write(out/"history.json",history)
        print(f"[diagnostic:{mode}] {it}/{cfg.iters}: loss={row['loss']:.6g}, "
              f"amp PSNR={row['metrics']['psnr_amp']:.3f}, "
              f"phase PSNR={row['metrics']['psnr_phs']:.3f}, "
              f"distance to constant={row['object_statistics']['relative_distance_to_constant']:.5g}",flush=True)
        return rec
    _json_write(out/"diagnostic.json",metadata)
    start=time.perf_counter()
    try:
        rec=record(0)
        for it in range(cfg.iters):
            log_step=it==0 or (it+1)%cfg.eval_every==0 or it==cfg.iters-1
            before=effective_snapshot(model) if log_step and not pixel_mode else None
            opt.zero_grad(set_to_none=True)
            if mode == "fit-object":
                loss=supervised_gradients(model,fit_coords,amp_target,phase_target,tc.coordinate_chunk)
            elif pixel_mode:
                loss=measurement_loss(pixels,probe,scene.post,scene.Q,scene.sqrtIm,weights,
                                      tc.scan_chunk,backward=True)
            else:
                loss=exact_chunked_gradients(model,probe,coords,cfg.obj_size,scene.post,scene.Q,
                                             scene.sqrtIm,weights,tc)
            stats=({"pixel_gradient_norm":pixels.grad.norm().item()} if pixel_mode
                   else balance_network_gradients(model,tc.balance))
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError("Non-finite loss/gradient")
            used_lr=opt.param_groups[0]["lr"]
            opt.step();scheduler.step()
            if log_step:
                rec=record(it+1,loss.item(),stats,
                           None if pixel_mode else effective_update_stats(model,before))
                history[-1]["lr_used"]=used_lr
                if not np.isfinite(history[-1]["loss"]):
                    raise FloatingPointError("Non-finite post-update loss")
        if device.type=="cuda":torch.cuda.synchronize(device)
        metadata.update(status="complete",elapsed_s=time.perf_counter()-start,
                        completed_updates=cfg.iters)
        np.savez_compressed(out/"diagnostic_fields.npz",obj_rec=rec,obj_gt=scene.obj,
                            probe_fixed=probe.cpu().numpy(),roi=[rs.start,rs.stop,cs.start,cs.stop])
        _json_write(out/"history.json",history)
        _json_write(out/"diagnostic.json",metadata)
        print(f"[diagnostic] outputs: {out}; these are NOT blind-reconstruction results.")
        return metadata,history
    except BaseException as exc:
        metadata.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        _json_write(out/"diagnostic.json",metadata)
        raise


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    modes=("audit","fit-object","known-probe","pixel-known-probe")
    if not argv or argv[0] not in modes:
        raise SystemExit("Usage: DeePIE_diagnose.py {audit|fit-object|known-probe|pixel-known-probe} [options]; append --help")
    mode=argv.pop(0)
    if mode=="audit":
        p=argparse.ArgumentParser(description="Inspect a saved DeePIE result without training")
        p.add_argument("--result",type=Path,required=True)
        p.add_argument("--outdir",type=Path,required=True)
        a=p.parse_args(argv)
        audit_result(a.result,a.outdir)
        return
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument("--from-run",type=Path)
    inherited,_=pre.parse_known_args(argv)
    p=base_parser()
    p.description="DIAGNOSTIC ONLY. Inherits source model/training settings; defaults below disable decay/weighting/balance."
    p.add_argument("--from-run",type=Path,help="run directory containing deepie_manifest.json")
    p.add_argument("--fit-domain",choices=["roi","full"],default="roi")
    source=None
    if inherited.from_run:
        source=inherited.from_run/"deepie_manifest.json"
        saved=json.loads(source.read_text(encoding="utf-8"))
        p.set_defaults(**saved["model_config"],**saved["training_config"],scene_config=source)
    # Remove poorly specified training heuristics in this deliberately easier test.
    # lr/model/seed are inherited, unless explicitly overridden on the command line.
    p.set_defaults(iters=100,eval_every=10,decay_factor=1.,scan_weight="uniform",balance="none",
                   outdir=f"results_paper/deepie_diagnostic_{mode}")
    if mode == "pixel-known-probe":
        # Pixel values and Fourier coefficients have different scales; explicit --lr overrides.
        p.set_defaults(lr=1e-2)
    args=p.parse_args(["run",*argv])
    cfg,mc,tc=configurations(args)
    run_diagnostic(mode,cfg,mc,tc,args.fit_domain,source)


if __name__=="__main__":
    main()
