"""Smoothed discrete second-order TGV for the net object's amplitude only.

R(u, v) = alpha1 mean |grad(u)-v|_2 + alpha0 mean |E(v)|_F.
E(v) is the symmetric gradient, including both off-diagonal entries. Forward
differences use only valid interior stencils (no periodic wrap or zero padding).
The vector field is approximately minimized by a few warm-started Adam steps
per network iteration; this is not an exact inner minimizer.
"""
from __future__ import annotations

import numpy as np
import torch


def tgv2_terms(u, v, mask, alpha0=2.0, alpha1=1.0, eps=1e-3):
    """Return total, first-order, second-order terms on a boolean domain mask.

    u: (H,W); v: (2,H,W), components in row/column order.
    Smooth vector norms are sqrt(sum(x**2) + eps**2) - eps.
    """
    valid = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:]
    dy = u[1:, :-1] - u[:-1, :-1]
    dx = u[:-1, 1:] - u[:-1, :-1]
    vy, vx = v[0], v[1]
    ry, rx = dy - vy[:-1, :-1], dx - vx[:-1, :-1]
    eyy = vy[1:, :-1] - vy[:-1, :-1]
    exx = vx[:-1, 1:] - vx[:-1, :-1]
    eyx = 0.5 * (
        vy[:-1, 1:] - vy[:-1, :-1] + vx[1:, :-1] - vx[:-1, :-1]
    )
    first = (torch.sqrt(ry.square() + rx.square() + eps**2) - eps)[valid].mean()
    second = (torch.sqrt(eyy.square() + exx.square() + 2 * eyx.square() + eps**2)
              - eps)[valid].mean()
    return alpha1 * first + alpha0 * second, first, second


def scan_domain(cfg, positions):
    """Nominal illuminated disk union; uses geometry, not GT/recovered probe.

    All valid stencils have equal weight. No regularization edge is introduced
    at the outside of the scan. The evaluation ROI plays no role here.
    """
    mask = np.zeros((cfg.obj_size, cfg.obj_size), dtype=bool)
    yy, xx = np.mgrid[:cfg.N, :cfg.N] - cfg.N // 2
    disk = yy**2 + xx**2 <= (cfg.probe_diam_px / 2)**2
    for y, x in positions:
        mask[y:y + cfg.N, x:x + cfg.N] |= disk
    rows, cols = np.where(mask)
    if not len(rows):
        raise ValueError("TGV scan domain is empty")
    roi = (slice(int(rows.min()), int(rows.max()) + 1),
           slice(int(cols.min()), int(cols.max()) + 1))
    return roi, mask[roi]


class ObjectAmplitudeTGV:
    """Warm-started auxiliary solve; gradients returned only through amplitude."""

    def __init__(self, cfg, positions, device):
        self.cfg = cfg
        self.roi, domain = scan_domain(cfg, positions)
        self.mask = torch.as_tensor(domain, dtype=torch.bool, device=device)
        valid = self.mask[:-1, :-1] & self.mask[1:, :-1] & self.mask[:-1, 1:]
        if not bool(valid.any()):
            raise ValueError("TGV domain has no valid finite-difference stencil")
        self.v = torch.nn.Parameter(torch.zeros((2, *domain.shape), device=device))
        self.opt = torch.optim.Adam([self.v], lr=cfg.tgv_lr)

    def __call__(self, amplitude):
        u = amplitude[self.roi]
        # Do NOT detach the mean: the prior must be invariant to object scale,
        # just like the VarPro data term. Otherwise shrinking O can lower R.
        u = u / u[self.mask].mean().clamp_min(1e-12)
        for _ in range(self.cfg.tgv_inner_steps):
            self.opt.zero_grad(set_to_none=True)
            inner, _, _ = tgv2_terms(
                u.detach(), self.v, self.mask,
                self.cfg.tgv_alpha0, self.cfg.tgv_alpha1, self.cfg.tgv_eps,
            )
            inner.backward()
            self.opt.step()
        return tgv2_terms(
            u, self.v.detach(), self.mask,
            self.cfg.tgv_alpha0, self.cfg.tgv_alpha1, self.cfg.tgv_eps,
        )

    def save(self, path):
        rs, cs = self.roi
        np.savez_compressed(
            path, vector=self.v.detach().cpu().numpy(),
            mask=self.mask.cpu().numpy(),
            roi=np.array([rs.start, rs.stop, cs.start, cs.stop]),
        )
