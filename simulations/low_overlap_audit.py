"""Tiny CPU diagnostics; NOT a reconstruction or an experiment-quality benchmark.

Run in Colab: python simulations/low_overlap_audit.py
Prints JSON to stdout; does not create files or import the reconstruction solver.
"""
from __future__ import annotations

import json
import numpy as np


def unet_conv_macs(channels: int, size: int = 624, base: int = 32) -> int:
    """Forward convolution MAC estimate for functions.addip.model.ProPtyUNet.

    Batch=1, n_fields=1, ph_ch=2. Excludes boundary effects, BN, activations,
    backward, optimizer, physical propagation, evaluation and TGV. NOT timing.
    Transposed convolution counted using its input spatial size.
    """
    if min(channels, base, size) <= 0 or size % 8:
        raise ValueError("Positive channels/base and size divisible by 8 required")
    total = 0

    def conv(cin, cout, side):
        return side * side * cin * cout * 9

    def double(cin, cout, side):
        return conv(cin, cout, side) + conv(cout, cout, side)

    cin = channels
    for level in range(4):
        cout = base * 2 ** level
        total += double(cin, cout, size // 2 ** level)
        cin = cout
    for level in (2, 1, 0):
        cout = base * 2 ** level
        total += conv(cout * 2, cout, size // 2 ** (level + 1))
        total += double(cout * 2, cout, size // 2 ** level)
    return total + conv(base, 3, size)


def periodic_gauge(shape, period):
    """Nonzero complex gauge, periodic independently along both coordinates."""
    y, x = np.indices(shape)
    amplitude = 0.15 * (np.cos(2 * np.pi * y / period)
                        + np.sin(2 * np.pi * x / period))
    phase = 0.3 * (np.sin(2 * np.pi * y / period)
                   + np.cos(2 * np.pi * x / period))
    return np.exp(amplitude + 1j * phase)


def gauge_errors(positions, period, seed=7):
    """O'=O*g, P'=P/g for reference scan origin (0,0).

    Every patch stays inside the canvas. Tests exit waves and their FFT
    intensities. Equal exit waves imply equal intensities for ANY common
    linear propagator, including the paper's fixed Fresnel propagator.
    """
    positions = np.asarray(positions, dtype=int)
    if positions.ndim != 2 or positions.shape[1] != 2 or np.any(positions < 0):
        raise ValueError("Expected nonnegative (n, 2) integer positions")
    if period <= 0:
        raise ValueError("period must be positive")
    n = 24
    m = int(positions.max()) + n
    rng = np.random.default_rng(seed)
    obj = (0.2 + rng.random((m, m))) * np.exp(1j * rng.normal(size=(m, m)))
    yy, xx = np.indices((n, n)) - n / 2
    probe = np.exp(-(xx ** 2 + yy ** 2) / 60) * np.exp(0.02j * (xx ** 2 + yy ** 2))
    probe[xx ** 2 + yy ** 2 > 100] = 0
    g = periodic_gauge((m, m), period)
    altered_obj = obj * g
    altered_probe = probe / g[:n, :n]
    numer_exit = denom_exit = numer_int = denom_int = 0.0
    for y, x in positions:
        exit0 = obj[y:y+n, x:x+n] * probe
        exit1 = altered_obj[y:y+n, x:x+n] * altered_probe
        i0 = np.abs(np.fft.fft2(exit0)) ** 2
        i1 = np.abs(np.fft.fft2(exit1)) ** 2
        numer_exit += float(np.sum(np.abs(exit1 - exit0) ** 2))
        denom_exit += float(np.sum(np.abs(exit0) ** 2))
        numer_int += float(np.sum((i1 - i0) ** 2))
        denom_int += float(np.sum(i0 ** 2))
    return {"exit_relative_error": float(np.sqrt(numer_exit / denom_exit)),
            "intensity_relative_error": float(np.sqrt(numer_int / denom_int))}


def audit():
    d = 4
    master = [(y*d, x*d) for y in range(7) for x in range(7)]
    stride = [(y*d, x*d) for y in range(0, 7, 2) for x in range(0, 7, 2)]
    # Two axis-offset measurements from the existing master dataset, NOT new acquisitions.
    mixed = stride + [(d, 0), (0, d)]
    # These are NEW positions not available by selecting the native raster.
    off_grid = stride + [(1, 0), (0, 1)]
    cases = {}
    for name, pos in (("master", master), ("stride2", stride),
                      ("stride2_plus_native_offsets", mixed),
                      ("stride2_plus_new_off_grid_positions", off_grid)):
        cases[name] = {"n_patterns": len(pos),
                       "native_period_d": gauge_errors(pos, d),
                       "enlarged_period_2d": gauge_errors(pos, 2*d)}
    macs = []
    for j in (16, 64, 100):
        original = unet_conv_macs(j)
        reduced = unet_conv_macs(4)
        macs.append({"input_channels_before": j, "input_channels_after": 4,
                     "forward_conv_GMAC_before": original / 1e9,
                     "forward_conv_GMAC_after": reduced / 1e9,
                     "conv_MAC_reduction_fraction": 1 - reduced / original,
                     "conv_only_ideal_ratio": original / reduced})
    return {"warning": "Analytical/toy diagnostics only; no recovery or GPU speed claim.",
            "gauge_cases": cases, "input_channel_cost": macs}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
