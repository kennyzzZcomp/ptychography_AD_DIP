"""Analytic phantoms: truth is used only for simulation and evaluation."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from pathlib import Path

from .physics import MultiwaveOperator
from .probes import initial_probe_field


@dataclass
class Scene:
    operator: MultiwaveOperator
    objects: torch.Tensor
    opd_um: torch.Tensor  # effective OPD per wavelength, also for mismatch scenes
    optical_depth: torch.Tensor  # AMPLITUDE optical depth: A=exp(-tau)
    markers: torch.Tensor
    common_texture: torch.Tensor
    clean: torch.Tensor
    measured: torch.Tensor
    train: torch.Tensor
    holdout: torch.Tensor
    roi: torch.Tensor
    input_stack: torch.Tensor


def phantom(cfg, device):
    m, count = cfg.object_size, len(cfg.wavelengths_nm)
    if cfg.scene == "usaf_zero_phase":
        from PIL import Image
        path = Path(cfg.usaf_path) if cfg.usaf_path else Path(__file__).resolve().parents[1]/"USAF.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"USAF image missing: {path}. Supply --usaf-path.")
        with Image.open(path) as im:
            im = im.convert("L")
            scale = max(1, round(m*cfg.usaf_fill))/max(im.size)
            size = tuple(max(1, round(v*scale)) for v in im.size)
            # Treat grayscale directly as amplitude, not measured intensity.
            small = np.asarray(im.resize(size, Image.Resampling.BOX), dtype=np.float32)/255
        amplitude = torch.ones(m, m, device=device)
        h, w = small.shape
        top, left = (m-h)//2, (m-w)//2
        amplitude[top:top+h, left:left+w] = torch.tensor(small, device=device)
        objects = torch.complex(amplitude, torch.zeros_like(amplitude))[None].repeat(count, 1, 1)
        opd = torch.zeros_like(objects.real)
        tau = -amplitude.clamp_min(1e-8).log()[None].repeat(count, 1, 1)
        return objects, opd, tau, torch.empty(0, m, m, device=device), amplitude
    a = torch.linspace(-1, 1, m, device=device)
    y, x = torch.meshgrid(a, a, indexing="ij")
    texture = (0.65*torch.exp(-((x+.20)**2+(y-.06)**2)/.10)
               + .25*torch.exp(-((x-.22)**2+(y+.19)**2)/.045)
               + .10*(1+torch.sin(15*x)*torch.cos(13*y))*torch.exp(-(x*x+y*y)/.32))
    d0 = (.16*torch.exp(-((x-.08)**2+(y+.12)**2)/.13)
          - .07*torch.exp(-((x+.28)**2+(y-.22)**2)/.04)
          + .025*torch.sin(14*x)*torch.cos(12*y)*torch.exp(-(x*x+y*y)/.22))
    markers = []
    for l in range(count):
        angle = 2*math.pi*l/count + .35
        cx, cy = .32*math.cos(angle), .32*math.sin(angle)
        radius = torch.sqrt((x-cx)**2+(y-cy)**2)
        markers.append(torch.sigmoid((.105-radius)/.012))
    markers = torch.stack(markers)
    tau = (.08+.38*texture)[None].repeat(count, 1, 1)
    if cfg.scene in ("spectral_absorption", "dispersive"):
        tau = tau + .35*markers
    opd = d0[None].repeat(count, 1, 1)
    wave_um = torch.tensor(cfg.wavelengths_nm, device=device)*1e-3
    if cfg.scene == "shared_complex":
        # Artificial control: same complex object, not a nondispersive OPD sample.
        opd = opd * (wave_um/wave_um[0])[:, None, None]
    elif cfg.scene == "dispersive":
        basis = (wave_um[0]/wave_um).square()-1
        d1 = .12*torch.exp(-((x+.12)**2+(y+.16)**2)/.035)
        opd = opd+basis[:, None, None]*d1
    phase = 2*math.pi*opd/wave_um[:, None, None]
    return torch.polar(torch.exp(-tau), phase), opd, tau, markers, texture


def known_probes(cfg, device):
    """Finite, smooth aperture plus an achromatic OPD phase screen at sample.

    Equal-power mode: each probe has power 1/L, intensities add directly.
    Legacy weighted mode: each power is 1 and fixed weights set fractions.
    These are simulation truth; the blind solver initializes independently.
    """
    n = cfg.patch_size
    a = torch.arange(n, device=device)-(n-1)/2
    y, x = torch.meshgrid(a, a, indexing="ij")
    r = torch.sqrt(x*x+y*y)
    amp = (1-(r/(.46*n))**8).clamp_min(0).square()*torch.exp(-(r/(.34*n))**2)
    amp = amp / amp.square().sum().sqrt()*cfg.probe_power**.5
    plate_opd = .10*torch.sin(2*math.pi*x/(.43*n))*torch.cos(2*math.pi*y/(.51*n))
    plate_opd += .04*(x*x-y*y)/(.46*n)**2
    return torch.stack([torch.polar(amp, 2*math.pi*plate_opd/(w*1e-3))
                        for w in cfg.wavelengths_nm])


def simulate(cfg, device="cpu"):
    cfg.validate()
    m, n = cfg.object_size, cfg.patch_size
    rng = np.random.default_rng(cfg.scene_seed)
    start = (m-n-(cfg.grid-1)*cfg.step)//2
    positions = np.array([(start+r*cfg.step, start+c*cfg.step)
                          for r in range(cfg.grid) for c in range(cfg.grid)])
    positions += rng.integers(-cfg.jitter, cfg.jitter+1, positions.shape)
    count = len(positions)
    shuffled = rng.permutation(count)
    k = max(1, round(count*cfg.holdout_fraction))
    holdout = torch.tensor(np.sort(shuffled[:k]), device=device)
    train = torch.tensor(np.sort(shuffled[k:]), device=device)
    op = MultiwaveOperator(cfg, known_probes(cfg, device), torch.tensor(positions, device=device))
    objects, opd, tau, markers, texture = phantom(cfg, device)
    with torch.no_grad():
        clean = op(objects)
    if cfg.photons_per_scan > 0:
        # A single fixed photon conversion for every scan. No per-frame normalization.
        noise = np.random.default_rng(cfg.noise_seed)
        counts = noise.poisson(clean.cpu().numpy().astype(np.float64)*cfg.photons_per_scan)
        measured = torch.tensor(counts/cfg.photons_per_scan, device=device, dtype=clean.dtype)
    else:
        measured = clean.clone()
    # Use a nominal initialization footprint, never true probe shape, for the
    # default blind task's evaluation/regularization region and its known control.
    nominal = initial_probe_field(cfg, device) if cfg.spectral_mode == "equal_power" else None
    exposure = op.illumination(train, nominal)
    roi = exposure > .20*exposure.max()
    # The holdout measurements never enter the U-Net input or training loss.
    inp = measured[train]/measured[train].max().clamp_min(1e-12)
    pad = (m-n)//2
    inp = torch.nn.functional.pad(inp, (pad, m-n-pad, pad, m-n-pad))[None]
    return Scene(op, objects, opd, tau, markers, texture, clean, measured,
                 train, holdout, roi, inp)
