"""Truth-independent, smooth zero-phase initialization; independent complex pixels."""
import torch
from torch import nn


def probe_smoothness(probes):
    """Mean per-mode nearest-neighbor complex roughness / mode power.

    Pixel-unit differences, no wrap-around or artificial zero boundary.
    Invariant to each mode's global phase and nonzero amplitude scale.
    """
    dx = probes[..., :, 1:] - probes[..., :, :-1]
    dy = probes[..., 1:, :] - probes[..., :-1, :]
    energy = dx.abs().square().sum((-1, -2)) + dy.abs().square().sum((-1, -2))
    power = probes.abs().square().sum((-1, -2)).clamp_min(1e-20)
    return (energy / power).mean()


def initial_probe_field(cfg, device="cpu"):
    n = cfg.patch_size
    x = torch.arange(n, device=device, dtype=torch.float32)-(n-1)/2
    y, x = torch.meshgrid(x, x, indexing="ij")
    radius = torch.sqrt(x*x+y*y)
    # Approximate smooth disk, not the true amplitude or its true phase screen.
    amp = torch.sigmoid((.33*n-radius)/(.06*n))
    amp = amp/amp.square().sum().sqrt()*cfg.probe_power**.5
    return torch.complex(amp, torch.zeros_like(amp))[None].repeat(len(cfg.wavelengths_nm), 1, 1)


class PixelProbes(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        initial = initial_probe_field(cfg, device)
        # Dimensionless O(1) raw parameters, rather than pixels O(1/N).
        self.real = nn.Parameter(initial.real*cfg.patch_size)
        self.imag = nn.Parameter(initial.imag*cfg.patch_size)
        self.register_buffer("power", torch.tensor(cfg.probe_power, device=device))

    def forward(self):
        raw = torch.complex(self.real, self.imag)
        norm = raw.abs().square().sum((-1, -2), keepdim=True).clamp_min(1e-20).sqrt()
        return raw/norm*self.power.sqrt()


@torch.no_grad()
def probe_metrics(probes, truth, wavelengths):
    values = []
    for l, wavelength in enumerate(wavelengths):
        rec, gt = probes[l], truth[l]
        piston = torch.angle((rec.conj()*gt).sum())
        aligned = rec*torch.exp(1j*piston)
        weight = gt.abs().square()
        phase_error = torch.angle(aligned*gt.conj())
        values.append({"wavelength_nm": wavelength,
                       "complex_relative_error": float((aligned-gt).abs().norm()/gt.abs().norm()),
                       "amplitude_relative_error": float((rec.abs()-gt.abs()).norm()/gt.abs().norm()),
                       "weighted_phase_rmse_rad": float((weight*phase_error.square()).sum().div(weight.sum()).sqrt()),
                       "power": float(rec.abs().square().sum())})
    return values
