"""Experimental exact raster extraction; original optics/defaults stay unchanged.

This changes the patch extraction graph, not the measurements or optical model.
Two Tensor.unfold views expose the rectangular scan geometry to autograd. The
adjoint is the sum over all overlapping windows (no measurement subsampling).
Geometry is validated/precomputed once, not copied from CUDA every iteration.
No probe support or probe truth is assumed. Integer, row-major uniform rasters
only; irregular/optimized/subpixel positions must use the original operator.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RasterGeometry:
    origin_row: int
    origin_col: int
    rows: int
    cols: int
    row_step: int
    col_step: int
    patch_size: int
    object_shape: tuple[int, int]

    @classmethod
    def from_positions(cls, positions, object_shape, patch_size):
        if isinstance(positions, torch.Tensor) and positions.requires_grad:
            raise ValueError("Raster geometry requires fixed, non-optimized positions")
        values = torch.as_tensor(positions).detach().cpu()
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
            raise ValueError("positions must be a nonempty (J, 2) array")
        if values.is_complex() or values.dtype == torch.bool:
            raise ValueError("positions must be finite integer coordinates")
        if not torch.isfinite(values).all() or not torch.equal(values, values.round()):
            raise ValueError("positions must be finite integer coordinates")
        points = [tuple(map(int, row)) for row in values.tolist()]
        rr, cc = sorted({p[0] for p in points}), sorted({p[1] for p in points})
        if points != [(r, c) for r in rr for c in cc]:
            raise ValueError("requires a complete row-major rectangular raster")

        def spacing(axis):
            step = axis[1] - axis[0] if len(axis) > 1 else 1
            if any(b-a != step for a, b in zip(axis, axis[1:])):
                raise ValueError("requires uniformly spaced raster axes")
            return step

        if len(object_shape) != 2 or patch_size < 1:
            raise ValueError("requires a 2D object and positive patch size")
        shape = tuple(map(int, object_shape))
        if rr[0] < 0 or cc[0] < 0 or rr[-1]+patch_size > shape[0] or cc[-1]+patch_size > shape[1]:
            raise ValueError("scan patches are outside the object canvas")
        return cls(rr[0], cc[0], len(rr), len(cc), spacing(rr), spacing(cc),
                   int(patch_size), shape)

    @property
    def count(self):
        return self.rows * self.cols

    @property
    def extent(self):
        return (self.patch_size + (self.rows-1)*self.row_step,
                self.patch_size + (self.cols-1)*self.col_step)

    def patches(self, obj):
        if tuple(obj.shape) != self.object_shape:
            raise ValueError("object shape differs from precomputed raster")
        h, w = self.extent
        crop = obj[self.origin_row:self.origin_row+h, self.origin_col:self.origin_col+w]
        windows = crop.unfold(0, self.patch_size, self.row_step).unfold(1, self.patch_size, self.col_step)
        return windows.reshape(self.count, self.patch_size, self.patch_size)

    def sliced_patches(self, obj):
        """Simple stack-of-slices control, with the same precomputed geometry."""
        if tuple(obj.shape) != self.object_shape:
            raise ValueError("object shape differs from precomputed raster")
        n = self.patch_size
        return torch.stack([
            obj[self.origin_row+i*self.row_step:self.origin_row+i*self.row_step+n,
                self.origin_col+j*self.col_step:self.origin_col+j*self.col_step+n]
            for i in range(self.rows) for j in range(self.cols)])


def forward_raster_field(obj, probe, geometry, Q, chunk=0, extraction="unfold",
                         adjoint_precision="native"):
    """Same FFT, phase factor and frame order as optics.forward_field.

    Autograd handles complex O/P/Q gradients and upstream U-Net gradients.
    No gradients are detached. Chunking here divides propagation, not extraction;
    peak-memory tradeoffs must be measured rather than assumed equivalent.
    """
    n = geometry.patch_size
    if probe.shape != (n, n) or Q.shape != (n, n):
        raise ValueError("probe and Q must match the raster patch size")
    if extraction not in ("unfold", "slices"):
        raise ValueError("extraction must be unfold or slices")
    from .extraction_precision import extraction_source, backward_promoted_patches
    if adjoint_precision == "native-order":
        if extraction != "unfold":
            raise ValueError("Native-order adjoint requires unfold extraction")
        from .raster_native_order import native_order_patches
        patches = native_order_patches(obj, geometry)
    elif adjoint_precision == "float64-backward":
        if extraction != "unfold":
            raise ValueError("Backward-only precision requires unfold extraction")
        patches = backward_promoted_patches(obj, geometry)
    else:
        source = extraction_source(obj, adjoint_precision)
        patches = geometry.patches(source) if extraction == "unfold" else geometry.sliced_patches(source)
        # Cast BEFORE all nonlinear/optical operations. Only the extraction
        # adjoint is promoted; never accidentally run a double FFT benchmark.
        patches = patches.to(obj.dtype)

    def propagate(part):
        psi = part * probe[None] * Q[None]
        return torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(psi, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1))

    if chunk <= 0 or chunk >= geometry.count:
        return propagate(patches)
    return torch.cat([propagate(patches[start:start+chunk])
                      for start in range(0, geometry.count, chunk)], 0)
