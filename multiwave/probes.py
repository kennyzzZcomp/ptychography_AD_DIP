"""Truth-independent, smooth zero-phase initialization; independent complex pixels."""
import torch
from torch import nn
from torch.nn import functional as F


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
        self.patch_size = cfg.patch_size
        self.grid_size = cfg.probe_grid_size or cfg.patch_size
        if self.grid_size != cfg.patch_size:
            # Downsample the nominal initialization, never the true probe.
            initial = torch.complex(
                F.interpolate(initial.real[:, None], size=self.grid_size, mode="area")[:, 0],
                F.interpolate(initial.imag[:, None], size=self.grid_size, mode="area")[:, 0])
        # Dimensionless O(1) raw parameters, rather than pixels O(1/N).
        self.real = nn.Parameter(initial.real*cfg.patch_size)
        self.imag = nn.Parameter(initial.imag*cfg.patch_size)
        self.register_buffer("power", torch.tensor(cfg.probe_power, device=device))

    def forward(self):
        real, imag = self.real, self.imag
        if self.grid_size != self.patch_size:
            real = F.interpolate(real[:, None], size=self.patch_size, mode="bilinear", align_corners=False)[:, 0]
            imag = F.interpolate(imag[:, None], size=self.patch_size, mode="bilinear", align_corners=False)[:, 0]
        raw = torch.complex(real, imag)
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


class BasisProbes(nn.Module):
    """Independent smooth amplitude/phase cosine expansions, without truth input.

    Orders count frequencies per axis including zero. Phase excludes piston.
    This is a low-frequency prior, not an exact physical aberration model.
    """
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        n = cfg.patch_size
        self.grid_size = None
        t = (torch.arange(n, device=device, dtype=torch.float32)+.5)/n
        def basis(order):
            k = torch.arange(order, device=device, dtype=torch.float32)
            b = torch.cos(torch.pi*k[:, None]*t[None])
            b[1:] *= 2**.5
            return torch.einsum("iy,jx->ijyx", b, b).reshape(order*order,n,n)
        self.register_buffer("amp_basis", basis(cfg.probe_amp_order))
        self.register_buffer("phase_basis", basis(cfg.probe_phase_order)[1:])
        nominal = initial_probe_field(cfg,device).abs()*n
        self.register_buffer("nominal_raw", torch.log(torch.expm1(nominal)))
        self.register_buffer("power", torch.tensor(cfg.probe_power,device=device))
        count = len(cfg.wavelengths_nm)
        self.amp_coeff = nn.Parameter(torch.zeros(count,cfg.probe_amp_order**2,device=device))
        self.phase_coeff = nn.Parameter(torch.zeros(count,cfg.probe_phase_order**2-1,device=device))

    def forward(self):
        raw = self.nominal_raw + torch.einsum("lk,kyx->lyx",self.amp_coeff,self.amp_basis)
        amp = F.softplus(raw)
        amp = amp/amp.square().sum((-1,-2),keepdim=True).clamp_min(1e-20).sqrt()*self.power.sqrt()
        phase = torch.einsum("lk,kyx->lyx",self.phase_coeff,self.phase_basis)
        return torch.polar(amp,phase)
