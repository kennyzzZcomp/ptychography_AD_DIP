"""Independent DeePIE implementation of Supplement 1, Eqs. S3-S5.

The supplement does not define the activation formula, coefficient initialization,
or precise PE convention. These choices are explicit in ModelConfig; see
simulations/DeePIE_README.md. This is not the authors' source code.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    width: int = 256
    hidden_layers: int = 3
    high_frequencies: int = 128
    low_frequencies: int = 128
    phases: int = 32
    encoding_side: int = 256
    positional_encoding: bool = True
    fourier_weights: bool = True
    omega: float = 30.0
    activation: str = "sine"
    alpha: float = 0.05
    phase_scale: float = 2 * math.pi
    head_scale: float = 1e-3

    def __post_init__(self):
        for key in ("width", "hidden_layers", "high_frequencies", "low_frequencies", "phases"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.encoding_side < 4:
            raise ValueError("encoding_side must be >= 4")
        if self.activation not in ("sine", "leaky-sine"):
            raise ValueError("activation must be sine or leaky-sine")
        if any(not math.isfinite(v) or v <= 0 for v in (self.omega, self.phase_scale, self.head_scale)):
            raise ValueError("omega, phase_scale and head_scale must be finite and positive")
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0,1]")


def fourier_basis(in_features, high, low, phases):
    """S4, literal angular frequencies (no implicit 2*pi frequency multiplier).

    z includes endpoints [0,1]; phase samples exclude the duplicate 2*pi endpoint.
    Frequency 1 appears in BOTH low and high lists, as specified in the supplement.
    No normalization of B: that would change the coefficient optimization geometry.
    """
    z = torch.linspace(0, 1, in_features, dtype=torch.float64)
    omega = torch.cat((torch.arange(1, low + 1, dtype=torch.float64) / low,
                       torch.arange(1, high + 1, dtype=torch.float64)))
    phi = torch.arange(phases, dtype=torch.float64) * (2 * math.pi / phases)
    return torch.cos(omega[:, None, None] * z + phi[None, :, None]).reshape(-1, in_features).float()


class FourierLinear(nn.Module):
    def __init__(self, in_features, out_features, cfg, first=False, output=False, bias=0.0):
        super().__init__()
        self.register_buffer("basis", fourier_basis(in_features, cfg.high_frequencies,
                                                   cfg.low_frequencies, cfg.phases))
        self.coefficients = nn.Parameter(torch.empty(out_features, self.basis.shape[0]))
        self.bias = nn.Parameter(torch.full((out_features,), float(bias)))
        # Disclosed choice: match average effective-weight variance to SIREN init.
        # IID coefficients cannot make all W entries independent; B fixes correlations.
        bound = 1 / in_features if first else math.sqrt(6 / in_features) / cfg.omega
        if output:
            bound *= cfg.head_scale
        std = bound / math.sqrt(3 * self.basis.square().sum(0).mean().item())
        nn.init.normal_(self.coefficients, std=std)

    def effective_weight(self):
        return self.coefficients @ self.basis

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)


class CoordinateNetwork(nn.Module):
    def __init__(self, cfg: ModelConfig, amplitude=False):
        super().__init__()
        self.cfg = cfg
        # Coordinates span [-1,1]. pi*2**k has 2**k cycles over that interval.
        # Both endpoints are sampled: Nyquist is (encoding_side - 1) / 2 cycles.
        bands = int(math.floor(math.log2((cfg.encoding_side - 1) / 2))) + 1
        self.register_buffer("pe_frequencies", math.pi * 2.0 ** torch.arange(bands))
        dim = 2 + 4 * bands if cfg.positional_encoding else 2
        dims = [dim] + [cfg.width] * cfg.hidden_layers + [1]
        layers = []
        for i, (di, do) in enumerate(zip(dims, dims[1:])):
            output = i == len(dims) - 2
            bias = 1.0 if amplitude and output else 0.0
            if cfg.fourier_weights:
                layer = FourierLinear(di, do, cfg, first=i == 0, output=output, bias=bias)
            else:
                layer = nn.Linear(di, do)
                bound = 1 / di if i == 0 else math.sqrt(6 / di) / cfg.omega
                with torch.no_grad():
                    layer.weight.uniform_(-bound, bound)
                    if output:
                        layer.weight.mul_(cfg.head_scale)
                    layer.bias.fill_(bias)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def encode(self, xy):
        if not self.cfg.positional_encoding:
            return xy
        angles = xy[..., :, None] * self.pe_frequencies
        return torch.cat((xy, angles.sin().flatten(-2), angles.cos().flatten(-2)), -1)

    def materialize(self):
        """Small dense W leaves used for exact memory-bounded chain-rule evaluation."""
        with torch.no_grad():
            return [(layer.effective_weight().detach().requires_grad_() if isinstance(layer, FourierLinear)
                     else layer.weight.detach().clone().requires_grad_(),
                     layer.bias.detach().clone().requires_grad_()) for layer in self.layers]

    def forward(self, xy, effective=None):
        x = self.encode(xy)
        for i, layer in enumerate(self.layers):
            x = layer(x) if effective is None else F.linear(x, *effective[i])
            if i < len(self.layers) - 1:
                x = torch.sin(self.cfg.omega * x)
                if self.cfg.activation == "leaky-sine":
                    # Optional hypothesis, NOT a formula disclosed by DeePIE.
                    x = F.leaky_relu(x, self.cfg.alpha)
        return x.squeeze(-1)

    @torch.no_grad()
    def pullback(self, effective):
        """dL/dLambda = (dL/dW) B^T; do not update the materialized W leaves."""
        for layer, (w, b) in zip(self.layers, effective):
            if isinstance(layer, FourierLinear):
                layer.coefficients.grad = w.grad @ layer.basis.T
            else:
                layer.weight.grad = w.grad.clone()
            layer.bias.grad = b.grad.clone()


def coordinate_grid(size, device, dtype=torch.float32):
    axis = torch.linspace(-1, 1, size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), -1).reshape(-1, 2)


class DeePIEObject(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.amplitude = CoordinateNetwork(cfg, amplitude=True)
        self.phase = CoordinateNetwork(cfg)

    def materialize(self):
        return self.amplitude.materialize(), self.phase.materialize()

    def forward(self, xy, effective=None):
        a, p = (None, None) if effective is None else effective
        amp = self.amplitude(xy, a)
        phase = self.phase(xy, p) * self.cfg.phase_scale
        # Linear amplitude head follows Eq.4; no added positivity activation.
        return torch.complex(amp * phase.cos(), amp * phase.sin())

    @torch.no_grad()
    def field(self, coords, size, chunk, effective=None):
        out = torch.empty(coords.shape[0], dtype=(torch.complex128 if coords.dtype == torch.float64
                                                else torch.complex64), device=coords.device)
        for start in range(0, len(coords), chunk):
            out[start:start + chunk] = self(coords[start:start + chunk], effective)
        return out.reshape(size, size)

    def field_vjp(self, coords, gradient, chunk, effective):
        upstream = gradient.reshape(-1)
        for start in range(0, len(coords), chunk):
            value = self(coords[start:start + chunk], effective)
            torch.autograd.backward(value, upstream[start:start + chunk])
        self.amplitude.pullback(effective[0])
        self.phase.pullback(effective[1])
