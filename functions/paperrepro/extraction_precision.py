"""Opt-in accumulation precision diagnostic, not a new reconstruction prior.

Promote only the linear object-window extraction and cast its output back BEFORE
probe multiplication and FFT. Autograd then sums overlapping window gradients
in float64, while fields, FFT, network and optimizer retain their original dtype.
Both casts remain differentiable. Costs and memory must be measured explicitly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class _RasterWithPromotedAdjoint(torch.autograd.Function):
    """Exact linear raster VJP, promoted only in backward (experimental).

    Uses ATen's own unfold adjoint for both indexed/unfold forward controls.
    No learned/approximate gradients, saved truth or extra optical backward.
    PyTorch-version-specific low-level operator: tested against autograd and
    both first/second derivative checks, not advertised as compile/vmap ready.
    """
    @staticmethod
    def forward(ctx, obj, geometry, extraction):
        ctx.geometry = geometry
        if extraction == "unfold":
            return geometry.patches(obj)
        if extraction == "reference":
            idx = torch.arange(geometry.patch_size, device=obj.device)
            # Derive the indexed forward from the SAME geometry as its VJP;
            # accepting independently supplied coordinates could silently
            # differentiate the wrong raster. Only fixed integer rasters apply.
            rr = geometry.origin_row+torch.arange(geometry.rows, device=obj.device)*geometry.row_step
            cc = geometry.origin_col+torch.arange(geometry.cols, device=obj.device)*geometry.col_step
            rows = rr.repeat_interleave(geometry.cols)[:, None]+idx[None]
            cols = cc.repeat(geometry.rows)[:, None]+idx[None]
            return obj[rows[:, :, None], cols[:, None, :]]
        raise ValueError("Backward-only precision supports indexed/unfold, not slices")

    @staticmethod
    def backward(ctx, grad):
        geometry = ctx.geometry
        n, (h, w) = geometry.patch_size, geometry.extent
        windows = extraction_source(grad, "float64").reshape(geometry.rows, geometry.cols, n, n)
        # Reverse the two forward unfold views in reverse order. Each window
        # contribution is included exactly once; overlap is summed, not averaged.
        first = torch.ops.aten.unfold_backward.default(windows, [geometry.rows, w, n], 1, n, geometry.col_step)
        crop = torch.ops.aten.unfold_backward.default(first, [h, w], 0, n, geometry.row_step)
        r, c = geometry.origin_row, geometry.origin_col
        result = F.pad(crop, (c, geometry.object_shape[1]-c-w, r, geometry.object_shape[0]-r-h))
        return result.to(grad.dtype), None, None


def backward_promoted_patches(obj, geometry, extraction="unfold"):
    """Native-valued forward windows; float64 exact overlap accumulation VJP."""
    if tuple(obj.shape) != geometry.object_shape:
        raise ValueError("Object differs from precomputed raster")
    # Validate dtype without allocating a promoted forward tensor.
    if obj.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        raise ValueError("Unsupported object dtype for promoted linear adjoint")
    return _RasterWithPromotedAdjoint.apply(obj, geometry, extraction)


def forward_indexed_backward_precise_field(obj, probe, Q, geometry, chunk=0):
    """Indexed-forward control with the SAME explicit raster adjoint as unfold.

    This is NOT the original optics operator. Propagation chunking occurs after
    full extraction, as in the unfold experiment. Fixed integer raster only.
    """
    patches = backward_promoted_patches(obj, geometry, "reference")

    def propagate(part):
        psi = part * probe[None] * Q[None]
        return torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(psi, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))

    if chunk <= 0 or chunk >= geometry.count:
        return propagate(patches)
    return torch.cat([propagate(patches[start:start+chunk])
                      for start in range(0, geometry.count, chunk)], 0)


def extraction_source(obj, precision="native"):
    if precision == "native":
        return obj
    if precision != "float64":
        raise ValueError("adjoint precision must be native or float64")
    if obj.dtype in (torch.complex64, torch.complex128):
        return obj.to(torch.complex128)
    if obj.dtype in (torch.float32, torch.float64):
        return obj.to(torch.float64)
    raise ValueError("float64 accumulation supports float32/64 or complex64/128 objects")


def forward_indexed_precise_field(obj, probe, positions, Q, n, chunk=0):
    """Original indexed extraction with promoted linear adjoint; FP32 physics.

    This is the matched precision control, not the original production baseline.
    The original optics.forward_field is untouched. Chunking matches its graph.
    """
    source = extraction_source(obj, "float64")
    idx = torch.arange(n, device=obj.device)

    def propagate(pos):
        rows = pos[:, 0:1]+idx[None, :]
        cols = pos[:, 1:2]+idx[None, :]
        patches = source[rows[:, :, None], cols[:, None, :]].to(obj.dtype)
        psi = patches * probe[None] * Q[None]
        return torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(psi, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1))

    if chunk <= 0 or chunk >= positions.shape[0]:
        return propagate(positions)
    return torch.cat([propagate(positions[start:start+chunk])
                      for start in range(0, positions.shape[0], chunk)], 0)
