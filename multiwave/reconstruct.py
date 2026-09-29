"""Full-batch reconstruction, post-update metrics, and no truth-based selection."""
import math
import time
import numpy as np
import torch

from .models import ObjectModel, SharedAmplitudeModel, masked_tv, amplitude_tv
from .physics import amplitude_loss
from .losses import poisson_loss, poisson_nll_sum, poisson_normalizer
from .probes import PixelProbes, probe_metrics


def backward_shared_data(objects, probes, scene, cfg, regularizer=None):
    """Exact full-batch field VJP; FFT graphs are freed per scan chunk.

    The U-Net is evaluated once (including BatchNorm). Detached field gradients
    are accumulated, then passed through the original network/probe once.
    """
    a = objects.detach().requires_grad_(True)
    p = probes.detach().requires_grad_(probes.requires_grad)
    denominator = scene.measured[scene.train].mean().clamp_min(1e-12)
    elements = scene.measured[scene.train].numel()
    count_normalizer = (poisson_normalizer(scene.measured[scene.train], cfg.photons_per_scan)
                        if cfg.loss == "poisson" else None)
    for ids in scene.train.split(cfg.chunk):
        pred = scene.operator(a, ids, probes=p)
        if cfg.loss == "poisson":
            loss = poisson_nll_sum(pred, scene.measured[ids], cfg.photons_per_scan)/count_normalizer
        else:
            loss = ((pred+1e-12).sqrt()-(scene.measured[ids]+1e-12).sqrt()).square().sum()/elements/denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite chunk loss")
        loss.backward()
    if cfg.tv_weight:
        (cfg.tv_weight*amplitude_tv(a.abs(), scene.roi)).backward()
    if regularizer is not None:
        (cfg.tgv_weight*regularizer.penalty(a.real[0])).backward()
    outputs, gradients = [objects], [a.grad]
    if probes.requires_grad:
        outputs.append(probes)
        gradients.append(p.grad)
    torch.autograd.backward(outputs, gradients)


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
        if cfg.object_size == 384:
            from .diagnose_resolution import metrics as detail_metrics
            detail = detail_metrics(objects[0].real, scene.objects[0].real, roi)
            result["center_amplitude_rmse"] = detail["center_rmse"]
            result["high_frequency_relative_error"] = detail["high_frequency_error"]
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
def evaluate(model, scene, cfg, probe_model=None):
    # Keep BatchNorm in its fitting mode; changing to running stats changes DIP output.
    objects, tau, opd = model()
    probes = scene.operator.probes if probe_model is None else probe_model()
    pred = scene.operator(objects, probes=probes)
    metrics = field_metrics(objects, tau, opd, scene, cfg)
    metrics["probes"] = probe_metrics(probes, scene.operator.probes, cfg.wavelengths_nm)
    for label, ids in (("train", scene.train), ("holdout", scene.holdout)):
        metrics[f"{label}_observed_amplitude_nrmse"] = float(amplitude_loss(pred[ids], scene.measured[ids]).sqrt())
        metrics[f"{label}_clean_amplitude_nrmse"] = float(amplitude_loss(pred[ids], scene.clean[ids]).sqrt())
    metrics["train_data_loss"] = float(poisson_loss(pred[scene.train], scene.measured[scene.train], cfg.photons_per_scan)
                                       if cfg.loss == "poisson" else
                                       amplitude_loss(pred[scene.train], scene.measured[scene.train]))
    return metrics, objects, tau, opd, pred, probes


def reconstruct(cfg, scene, method, progress_callback=None):
    torch.manual_seed(cfg.network_seed)
    shared_amp = method.endswith("_shared_amp")
    feedback = method == "feedback_shared_amp"
    model_class = SharedAmplitudeModel if shared_amp else ObjectModel
    if feedback:
        from .feedback import FeedbackAmplitudeModel
        if cfg.loss != "poisson" or cfg.tv_weight or cfg.tgv_weight:
            raise ValueError("feedback supports unregularized Poisson loss only")
        model_class = FeedbackAmplitudeModel
    model = model_class(cfg, method, scene.input_stack).to(scene.objects.device)
    lr = cfg.lr_net if method.startswith("unet") or feedback else cfg.lr_pixel
    probe_model = PixelProbes(cfg, scene.objects.device) if cfg.probe_mode == "pixel" else None
    regularizer = None
    if cfg.tgv_weight:
        if not shared_amp:
            raise ValueError("TGV is supported only for shared_amp methods")
        from .regularization import AmplitudeTGV
        regularizer = AmplitudeTGV(cfg, scene)
    groups = [{"params": model.parameters(), "lr": lr}]
    if probe_model is not None:
        groups.append({"params": probe_model.parameters(), "lr": cfg.lr_probe})
    optimizer = torch.optim.Adam(groups)
    initial_probes = (scene.operator.probes if probe_model is None else probe_model()).detach().cpu().numpy().copy()
    history = []

    def record(iteration):
        metrics, *fields = evaluate(model, scene, cfg, probe_model)
        metrics["object_learning_rate"] = optimizer.param_groups[0]["lr"]
        metrics["probe_learning_rate"] = optimizer.param_groups[1]["lr"] if probe_model is not None else None
        metrics["train_total_objective"] = metrics["train_data_loss"]
        if cfg.tv_weight:
            with torch.no_grad():
                penalty = (amplitude_tv(fields[0].abs(), scene.roi) if shared_amp else
                           masked_tv(fields[1], fields[2], scene.roi, cfg.opd_scale_um))
            metrics["train_total_objective"] += float(cfg.tv_weight*penalty)
        if regularizer is not None:
            with torch.no_grad():
                total, first, second = regularizer.terms(fields[0][0].real)
            metrics.update(tgv_penalty=float(total), tgv_first_order=float(first),
                           tgv_second_order=float(second), tgv_weighted_penalty=float(cfg.tgv_weight*total),
                           tgv_aux_learning_rate=cfg.tgv_lr)
            metrics["train_total_objective"] += metrics["tgv_weighted_penalty"]
        if feedback and iteration:
            metrics["feedback_gradient_rms"] = model.gradient_rms
            metrics["feedback_weight_mean"] = model.weight_mean
        history.append({"iteration": iteration, **metrics})
        if progress_callback is not None:
            progress_callback(method, iteration, metrics)
        print(f"{method:19s} {iteration:4d}/{cfg.iterations} "
              f"loss[{cfg.loss}]={metrics['train_data_loss']:.5g} "
              f"train={metrics['train_observed_amplitude_nrmse']:.4f} "
              f"heldout={metrics['holdout_clean_amplitude_nrmse']:.4f} "
              f"object={metrics['mean_complex_relative_error']:.4f} "
              f"probe={np.mean([p['complex_relative_error'] for p in metrics['probes']]):.4f}", flush=True)
        return metrics, fields

    initial, _ = record(0)
    if scene.objects.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for iteration in range(1, cfg.iterations+1):
        if (method.startswith("unet") and cfg.lr_net_decay_after
                and iteration == cfg.lr_net_decay_after + 1):
            optimizer.param_groups[0]["lr"] = cfg.lr_net * cfg.lr_net_decay_factor
            print(f"{method}: update {iteration}, network lr -> {optimizer.param_groups[0]['lr']:.6g}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        if feedback:
            current_probes = scene.operator.probes if probe_model is None else probe_model()
            model.prepare(scene, current_probes)
        objects, tau, opd = model()
        probes = scene.operator.probes if probe_model is None else probe_model()
        if shared_amp:
            backward_shared_data(objects, probes, scene, cfg, regularizer)
        else:
            prediction = scene.operator(objects, scene.train, probes=probes)
            loss = (poisson_loss(prediction, scene.measured[scene.train], cfg.photons_per_scan)
                    if cfg.loss == "poisson" else amplitude_loss(prediction, scene.measured[scene.train]))
            if cfg.tv_weight:
                loss = loss + cfg.tv_weight*masked_tv(tau, opd, scene.roi, cfg.opd_scale_um)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at {method} iteration {iteration}")
            loss.backward()
        optimizer.step()
        if feedback:
            model.commit()
        if shared_amp and model.net is None and cfg.pixel_parameterization == "direct":
            with torch.no_grad():
                model.raw_amp.clamp_(0, 1)
        if iteration % cfg.eval_every == 0 or iteration == cfg.iterations:
            final, fields = record(iteration)
    if scene.objects.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    if not math.isfinite(final["mean_complex_relative_error"]):
        raise FloatingPointError("non-finite final reconstruction")
    arrays = [v.detach().cpu().numpy() for v in fields]
    return {"method": method, "initial": initial, "final": final, "history": history,
            "loss": cfg.loss,
            "feedback_mode": cfg.feedback_mode if feedback else None,
            "feedback_step": cfg.feedback_step if feedback else None,
            "training_data_gradient_passes": cfg.iterations * (2 if feedback else 1),
            "feedback_gradient": "globally RMS-normalized; detached one-step" if feedback else None,
            "unet_skip": cfg.unet_skip if method.startswith("unet") else None,
            "unet_detail": cfg.unet_detail if method.startswith("unet") else None,
            "detail_parameter_count": (sum(p.numel() for p in model.net.detail_head.parameters())
                                       if model.net is not None and hasattr(model.net, "detail_head") else 0),
            "tgv_weight": cfg.tgv_weight,
            "tgv_state_dict": regularizer.state_dict() if regularizer is not None else None,
            "tgv_auxiliary_parameter_count": regularizer.v.numel() if regularizer is not None else 0,
            "tgv_domain_pixels": int(regularizer.mask.sum()) if regularizer is not None else 0,
            "elapsed_s_including_evaluation": elapsed,
            "parameter_count": sum(p.numel() for p in model.parameters()) + (sum(p.numel() for p in probe_model.parameters()) if probe_model is not None else 0),
            "probe_mode": cfg.probe_mode,
            "probe_learning_rate": cfg.lr_probe if probe_model is not None else None,
            "probe_state_dict": None if probe_model is None else {k: v.detach().cpu() for k, v in probe_model.state_dict().items()},
            "probes": arrays[4], "initial_probes": initial_probes,
            "learning_rate": lr, "objects": arrays[0], "optical_depth": arrays[1],
            "opd_um": arrays[2], "predicted_intensity": arrays[3],
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()
                           if k != "input_stack"}}
