"""Common-grid, zero-padded angular spectrum; wavelengths add in intensity."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def asm_transfer(n, pixel_m, wavelength_m, distance_m, *, device="cpu",
                 dtype=torch.complex64):
    """Piston-free propagating spectrum with a 2D phase-sampling cutoff.

    The cutoff follows |d phase/df_axis| * df <= pi. It is conservative
    for a finite window, not a guarantee that the output is window-independent.
    Evanescent terms are discarded: the intended regime is scalar paraxial optics.
    """
    if n < 2 or pixel_m <= 0 or wavelength_m <= 0 or distance_m < 0:
        raise ValueError("invalid propagation geometry")
    f = torch.fft.fftfreq(n, d=pixel_m, device=device, dtype=torch.float64)
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    q = wavelength_m**2 * (fx.square() + fy.square())
    root = torch.sqrt((1-q).clamp_min(1e-30))
    # sqrt(1-q)-1 = -q/(sqrt(1-q)+1), avoiding subtractive cancellation.
    phase = -2*math.pi * distance_m/wavelength_m * q/(root+1)
    half_window = n*pixel_m/2
    mask = ((q < 1) & (distance_m*wavelength_m*fx.abs()/root <= half_window)
            & (distance_m*wavelength_m*fy.abs()/root <= half_window))
    return (torch.polar(torch.ones_like(phase), phase)*mask).to(dtype)


def propagate_padded(field, transfer, detector_size=None):
    n = field.shape[-1]
    if field.shape[-2] != n or transfer.shape[-1] < n:
        raise ValueError("propagation requires square fields and a larger FFT grid")
    delta = transfer.shape[-1]-n
    lo, hi = delta//2, delta-delta//2
    padded = F.pad(field, (lo, hi, lo, hi))
    propagated = torch.fft.ifft2(torch.fft.fft2(padded)*transfer)
    detector_size = n if detector_size is None else detector_size
    if detector_size > transfer.shape[-1] or detector_size < 1:
        raise ValueError("detector must fit propagation grid")
    crop = (transfer.shape[-1]-detector_size)//2
    return propagated[..., crop:crop+detector_size, crop:crop+detector_size]


class MultiwaveOperator(nn.Module):
    """Object [L,M,M], detector [J,N,N]; optional estimated probes override truth."""
    def __init__(self, cfg, probes, positions):
        super().__init__()
        self.n = cfg.patch_size
        self.detector_size = cfg.detector_pixels
        self.object_size = cfg.object_size
        self.chunk = cfg.chunk
        if probes.shape != (len(cfg.wavelengths_nm), self.n, self.n):
            raise ValueError("probe shape does not match configuration")
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape [J,2]")
        if not torch.equal(positions, positions.round()) or positions.min() < 0:
            raise ValueError("this prototype requires nonnegative integer positions")
        if (positions+self.n > cfg.object_size).any():
            raise ValueError("scan patch extends beyond object")
        device = probes.device
        self.register_buffer("probes", probes)
        self.register_buffer("positions", positions.to(device=device, dtype=torch.long))
        self.register_buffer("weights", torch.tensor(cfg.mixing_coefficients, device=device,
                                                     dtype=probes.real.dtype))
        self.register_buffer("transfer", torch.stack([
            asm_transfer(self.n*cfg.pad_factor, cfg.pixel_um*1e-6, w*1e-9,
                         cfg.distance_mm*1e-3, device=device, dtype=probes.dtype)
            for w in cfg.wavelengths_nm]))
        a = torch.arange(self.n, device=device)
        self.register_buffer("rows", self.positions[:, :1]+a)
        self.register_buffer("cols", self.positions[:, 1:]+a)

    def components(self, objects, indices=None, probes=None):
        probes = self.probes if probes is None else probes
        if probes.shape != self.probes.shape:
            raise ValueError("replacement probe shape mismatch")
        if objects.shape != (len(self.weights), self.object_size, self.object_size):
            raise ValueError("object shape does not match configuration")
        rows = self.rows if indices is None else self.rows[indices]
        cols = self.cols if indices is None else self.cols[indices]
        outputs = []
        for start in range(0, len(rows), self.chunk):
            r, c = rows[start:start+self.chunk], cols[start:start+self.chunk]
            patches = objects[:, r[:, :, None], c[:, None, :]]
            wave = propagate_padded(patches*probes[:, None], self.transfer[:, None], self.detector_size)
            outputs.append(wave.abs().square())
        return torch.cat(outputs, dim=1)

    def forward(self, objects, indices=None, probes=None):
        return (self.components(objects, indices, probes)*self.weights[:, None, None, None]).sum(0)

    def illumination(self, indices, probes=None):
        """Training-only exposure map; contains no object truth."""
        result = torch.zeros(self.object_size, self.object_size, device=self.probes.device)
        probes = self.probes if probes is None else probes
        power = (probes.abs().square()*self.weights[:, None, None]).sum(0)
        for row, col in self.positions[indices].tolist():
            result[row:row+self.n, col:col+self.n] += power
        return result


def amplitude_loss(prediction, target):
    """One global scale, with the same smoothing in both square roots."""
    return ((prediction+1e-12).sqrt()-(target+1e-12).sqrt()).square().mean() / target.mean().clamp_min(1e-12)
