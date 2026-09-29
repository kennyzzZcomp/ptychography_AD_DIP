"""Full-batch reconstruction, post-update metrics, and no truth-based selection."""
import math
import time
import numpy as np
import torch

from .models import ObjectModel, SharedAmplitudeModel, masked_tv, amplitude_tv
from .physics import amplitude_loss


def field_metrics(objects, tau, opd, scene, cfg):
    roi = scene.roi
    channels = []
    for l, wave in enumerate(cfg.wavelengths_nm):
        rec, truth = objects[l][roi], scene.objects[l][roi]
        piston = torch.angle((rec.conj()*truth).sum())
        aligned = rec*torch.exp(1j*piston)  # phase only; never rescale amplitudes
        phase_error = torch.angle(aligned*truth.conj())
        delta_opd = (opd[l]-scene.opd_um[l])[roi]
        delta_opd = delta_opd-delta_opd.mean()
        channels.append({
            "wavelength_nm": wave,
            "amplitude_rmse": float((rec.abs()-truth.abs()).square().mean().sqrt()),
            "complex_relative_error": float((aligned-truth).abs().norm()/truth.abs().norm()),
            "wrapped_phase_rmse_rad": float(phase_error.square().mean().sqrt()),
            "centered_opd_rmse_nm": float(delta_opd.square().mean().sqrt()*1000),
            "phase_piston_removed_rad": float(piston),
        })
    result = {"channels": channels, "roi_pixels": int(roi.sum()),
              "mean_complex_relative_error": float(np.mean([x["complex_relative_error"] for x in channels]))}
    if cfg.scene == "usaf_zero_phase":
        rec, truth = objects[0].abs()[roi], scene.objects[0].abs()[roi]
        mse = (rec-truth).square().mean()
        result["shared_amplitude_rmse"] = float(mse.sqrt())
        result["shared_amplitude_relative_error"] = float((rec-truth).norm()/truth.norm())
        result["shared_amplitude_psnr_db"] = float(-10*torch.log10(mse.clamp_min(1e-20)))
        dark, bright = truth <= .1, truth >= .9
        if dark.any() and bright.any():
            lo, hi = rec[dark].mean(), rec[bright].mean()
            result["amplitude_region_contrast"] = float((hi-lo)/(hi+lo).clamp_min(1e-12))
        result["max_object_imaginary_abs"] = float(objects.imag.abs().max())
        result["max_interwavelength_object_difference"] = float((objects-objects[:1]).abs().max())
    if cfg.scene in ("spectral_absorption", "dispersive"):
        # Joint regression removes constant/common texture before testing channel markers.
        # This diagnostic uses phantom truth ONLY during evaluation, never optimization.
        basis = torch.cat((torch.ones_like(scene.common_texture)[None],
                           scene.common_texture[None], scene.markers), 0)
        design = basis[:, roi].T.cpu().numpy().astype(np.float64)
        coeff = np.linalg.lstsq(design, tau[:, roi].T.cpu().numpy(), rcond=None)[0]
        matrix = coeff[2:].T/.35  # ground truth is identity
        off = matrix[~np.eye(len(matrix), dtype=bool)]
        result["marker_transfer_matrix"] = matrix.tolist()
        result["marker_offdiagonal_abs_mean"] = float(np.abs(off).mean())
        result["marker_diagonal_mean"] = float(np.diag(matrix).mean())
        result["marker_identity_rmse"] = float(np.sqrt(np.mean((matrix-np.eye(len(matrix)))**2)))
    return result


@torch.no_grad()
def evaluate(model, scene, cfg):
    # Keep BatchNorm in its fitting mode; changing to running stats changes DIP output.
    objects, tau, opd = model()
    pred = scene.operator(objects)
    metrics = field_metrics(objects, tau, opd, scene, cfg)
    for label, ids in (("train", scene.train), ("holdout", scene.holdout)):
        metrics[f"{label}_observed_amplitude_nrmse"] = float(amplitude_loss(pred[ids], scene.measured[ids]).sqrt())
        metrics[f"{label}_clean_amplitude_nrmse"] = float(amplitude_loss(pred[ids], scene.clean[ids]).sqrt())
    return metrics, objects, tau, opd, pred


def reconstruct(cfg, scene, method):
    torch.manual_seed(cfg.network_seed)
    shared_amp = method.endswith("_shared_amp")
    model_class = SharedAmplitudeModel if shared_amp else ObjectModel
    model = model_class(cfg, method, scene.input_stack).to(scene.objects.device)
    lr = cfg.lr_net if method.startswith("unet") else cfg.lr_pixel
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    def record(iteration):
        metrics, *fields = evaluate(model, scene, cfg)
        history.append({"iteration": iteration, **metrics})
        print(f"{method:19s} {iteration:4d}/{cfg.iterations} "
              f"train={metrics['train_observed_amplitude_nrmse']:.4f} "
              f"heldout={metrics['holdout_clean_amplitude_nrmse']:.4f} "
              f"object={metrics['mean_complex_relative_error']:.4f}", flush=True)
        return metrics, fields

    initial, _ = record(0)
    if scene.objects.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for iteration in range(1, cfg.iterations+1):
        optimizer.zero_grad(set_to_none=True)
        objects, tau, opd = model()
        prediction = scene.operator(objects, scene.train)
        loss = amplitude_loss(prediction, scene.measured[scene.train])
        if cfg.tv_weight:
            prior = amplitude_tv(objects.abs(), scene.roi) if shared_amp else masked_tv(tau, opd, scene.roi, cfg.opd_scale_um)
            loss = loss + cfg.tv_weight*prior
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at {method} iteration {iteration}")
        loss.backward()
        optimizer.step()
        if iteration % cfg.eval_every == 0 or iteration == cfg.iterations:
            final, fields = record(iteration)
    if scene.objects.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    if not math.isfinite(final["mean_complex_relative_error"]):
        raise FloatingPointError("non-finite final reconstruction")
    arrays = [v.detach().cpu().numpy() for v in fields]
    return {"method": method, "initial": initial, "final": final, "history": history,
            "elapsed_s_including_evaluation": elapsed,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "learning_rate": lr, "objects": arrays[0], "optical_depth": arrays[1],
            "opd_um": arrays[2], "predicted_intensity": arrays[3],
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()
                           if k != "input_stack"}}
