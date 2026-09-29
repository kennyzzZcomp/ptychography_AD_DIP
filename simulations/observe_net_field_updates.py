"""Observe a support-probe net run without replacing a gradient/update.

Wraps audit_net_reproducibility.py. Extra FFTs and host transfers make traced
runtime unsuitable for performance comparisons. Truth metrics are diagnostics,
not inputs to training or to a controller. Full-data, no-TGV real net only.
Short mathematical/passivity tests do not validate a full GPU trajectory.
Compare a long replay against the unobserved reference before interpreting it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def probe_gradient_contributions(grad_u, obj, positions, quad, rows, cols):
    """Adjoint of the shifted orthonormal FFT, in shared probe coordinates.

    grad_u is the ACTUAL full-loss U gradient, including global VarPro. No
    per-position amplitude fit or extra autograd pass is introduced.
    """
    exit_grad = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(grad_u, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1))
    patches = obj[positions[:, 0:1] + rows[None],
                  positions[:, 1:2] + cols[None]]
    return exit_grad[:, rows, cols] * (patches * quad[rows, cols]).conj()


def update_stats(current, previous):
    current, previous = current.flatten(), previous.flatten()
    denominator = torch.linalg.vector_norm(previous).clamp_min(1e-30)
    scalar = (current.conj() * previous).sum() / current.abs().square().sum().clamp_min(1e-30)
    return {
        "relative_raw_change": (torch.linalg.vector_norm(current-previous) / denominator).item(),
        "relative_gauge_removed_change": (
            torch.linalg.vector_norm(scalar*current-previous) / denominator).item(),
        "alignment_scalar_real": scalar.real.item(),
        "alignment_scalar_imag": scalar.imag.item(),
    }


def coherence_stats(contributions):
    norms = torch.linalg.vector_norm(contributions, dim=1)
    total = contributions.sum(0)
    norm_sum = norms.sum().clamp_min(1e-30)
    pixel_sum = contributions.abs().sum(0)
    active = norms > 1e-30
    unit = contributions[active] / norms[active, None]
    count = int(active.sum())
    pair_cosine = ((unit.sum(0).abs().square().sum()-count) / (count*(count-1))).item() if count > 1 else None
    return {
        "norm_coherence": (torch.linalg.vector_norm(total) / norm_sum).item(),
        "pixel_weighted_coherence": (total.abs().sum() / pixel_sum.sum().clamp_min(1e-30)).item(),
        "mean_pair_real_cosine": pair_cosine,
        "per_position_gradient_norm": norms.cpu().tolist(),
        "active_positions": count,
        "total_probe_gradient_norm": torch.linalg.vector_norm(total).item(),
    }


def network_gradient_norms(net):
    groups = {"amplitude_head": [], "phase_head": [], "backbone": []}
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            if parameter.grad is None:
                continue
            group = ("amplitude_head" if name.startswith("head_amp.") else
                     "phase_head" if name.startswith("head_phs.") else "backbone")
            groups[group].append(parameter.grad.detach().square().sum())
        return {key: torch.stack(values).sum().sqrt().item() if values else 0.
                for key, values in groups.items()}


class FieldObserver:
    def __init__(self, samples):
        self.samples = set(samples)
        self.count = 0
        self.rows = []
        self.scene = None
        self.previous_object = self.previous_probe = None
        self.mask_rows = self.mask_cols = None

    def set_scene(self, cfg, scene):
        from functions.paperrepro.crosstalk import pinhole_mask
        if (cfg.probe_mode != "support" or cfg.network_type != "real"
                or cfg.measurement_schedule or cfg.tgv_amp or cfg.tgv_phase):
            raise ValueError("Observer requires full-data, no-TGV, real net with support probe")
        self.scene = scene
        mask = torch.as_tensor(pinhole_mask(cfg, cfg.probe_support_margin), device=scene.Q.device)
        self.mask_rows, self.mask_cols = torch.where(mask > 0)

    def observe(self, cfg, obj, probe, positions, quad, field):
        self.count += 1
        rs, cs = self.scene.roi
        rr, cc = self.mask_rows, self.mask_cols
        with torch.no_grad():
            current_obj = obj[rs, cs].detach()
            current_probe = probe[rr, cc].detach()
            row = None
            if self.count in self.samples:
                from functions.paperrepro.evaluate import evaluate, probe_relerr
                row = {"forward_iteration": self.count,
                       "field_state_after_updates": self.count-1,
                       "object_roi": [rs.start, rs.stop, cs.start, cs.stop]}
                if self.previous_object is not None:
                    row["object_update"] = update_stats(current_obj, self.previous_object)
                    row["probe_update"] = update_stats(current_probe, self.previous_probe)
                # Replicate the data term only for read-only reporting.
                from functions.addip.losses import cabs
                amplitude = cabs(field.detach())
                target = self.scene.sqrtIm
                scale = (amplitude*target).sum() / amplitude.square().sum().clamp_min(1e-20)
                row["data_loss"] = ((scale*amplitude-target).square().mean()).item()
                metrics = evaluate(current_obj.cpu().numpy(), self.scene.obj[rs, cs])
                row["truth_diagnostic_only"] = {
                    "psnr_amp": metrics["psnr_amp"], "object_error": metrics["relerr"],
                    "probe_error": probe_relerr(probe.detach().cpu().numpy(), self.scene.probe)}
                self.rows.append(row)
            # Store values, not graphs; no RNG or network forward call.
            self.previous_object = current_obj.clone()
            self.previous_probe = current_probe.clone()
        if row is None:
            return

        gradient_state = {}

        def on_field_gradient(gradient):
            with torch.no_grad():
                per_position = probe_gradient_contributions(
                    gradient.detach(), obj.detach(), positions, quad, rr, cc)
                row["probe_gradient"] = coherence_stats(per_position)
                gradient_state["expected_probe"] = per_position.sum(0)
            # None: pass the original gradient unchanged.

        def on_probe_gradient(gradient):
            with torch.no_grad():
                actual = gradient.detach()[rr, cc]
                expected = gradient_state.pop("expected_probe")
                error = torch.linalg.vector_norm(actual-expected)
                relative = error / torch.linalg.vector_norm(actual).clamp_min(1e-30)
                row["probe_gradient"]["adjoint_sum_relative_error"] = relative.item()
                row["probe_gradient"]["adjoint_sum_max_absolute_error"] = (actual-expected).abs().max().item()
                if relative > 1e-4 and error > 1e-10:
                    raise RuntimeError("Per-position adjoint does not match actual total probe gradient")

        def on_object_gradient(gradient):
            with torch.no_grad():
                row["object_roi_gradient_norm"] = torch.linalg.vector_norm(gradient.detach()[rs, cs]).item()

        field.register_hook(on_field_gradient)
        probe.register_hook(on_probe_gradient)
        obj.register_hook(on_object_gradient)

    def save(self, outdir):
        path = Path(outdir) / "field_update_trace.json"
        if path.exists():
            raise FileExistsError(path)
        value = {
            "observational_only": True,
            "warning": "No explicit gradient/update replacement; full-run trajectory passivity must be validated against an unobserved replay. Traced runtime includes diagnostic overhead. Truth is evaluation only.",
            "sample_convention": "Decode before update; iteration i fields reflect i-1 parameter updates.",
            "probe_coordinates": "Existing hard support; same physical probe pixel across scan positions.",
            "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "samples_requested": sorted(self.samples), "forwards_observed": self.count,
            "rows": self.rows,
        }
        path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    default_samples = sorted(set(range(1, 21)) | set(range(50, 2001, 50))
                             | set(range(51, 2000, 50)))
    parser.add_argument("--field-samples", default=",".join(map(str, default_samples)))
    args, remaining = parser.parse_known_args()
    samples = [int(value) for value in args.field_samples.split(",")]
    if not samples or min(samples) < 1:
        parser.error("Samples must be positive forward iteration numbers")
    from simulations import audit_net_reproducibility as audit
    from functions.paperrepro import solvers_addip as solver
    observer = FieldObserver(samples)
    original_scene, original_forward, original_save = solver.build_scene, solver._fwd, solver._save
    original_optimizers = solver._net_optimizers

    def build_scene(cfg, device):
        scene = original_scene(cfg, device)
        observer.set_scene(cfg, scene)
        return scene

    def forward(cfg, obj, probe, positions, quad):
        field = original_forward(cfg, obj, probe, positions, quad)
        observer.observe(cfg, obj, probe, positions, quad, field)
        return field

    def save(*args, **kwargs):
        original_save(*args, **kwargs)
        observer.save(args[0].outdir)

    def optimizers(cfg, net, mode, probe_real, probe_imag):
        net_optimizer, probe_optimizer = original_optimizers(cfg, net, mode, probe_real, probe_imag)
        original_step = net_optimizer.step

        def read_then_step(*args, **kwargs):
            if observer.count in observer.samples:
                row = observer.rows[-1]
                row["network_parameter_gradient_norms"] = network_gradient_norms(net)
                row["actual_lr_net"] = net_optimizer.param_groups[0]["lr"]
            return original_step(*args, **kwargs)

        net_optimizer.step = read_then_step
        return net_optimizer, probe_optimizer

    with patch.object(solver, "build_scene", build_scene), \
         patch.object(solver, "_fwd", forward), \
         patch.object(solver, "_save", save), \
         patch.object(solver, "_net_optimizers", optimizers), \
         patch.object(sys, "argv", [str(audit.__file__), *remaining]):
        audit.main()


if __name__ == "__main__":
    main()
