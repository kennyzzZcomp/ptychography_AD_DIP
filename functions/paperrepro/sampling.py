"""Explicit, reproducible measurement curricula (not acquisition reduction)."""
from __future__ import annotations

import numpy as np


class MeasurementSchedule:
    """Schedule syntax: '0:4,500:2,1000:1' = start iteration : axis stride.

    Iterations are zero based. Each stride must divide grid, giving equal
    cardinality for fixed/random/rotating controls. Final stage must use all.
    Local RNG never changes scene generation or network initialization.
    """
    def __init__(self, spec, grid, iters, policy="fixed", seed=0):
        spec = spec or "0:1"
        if policy not in ("fixed", "random", "rotate"):
            raise ValueError("measurement policy must be fixed, random or rotate")
        self.grid, self.policy = grid, policy
        if grid < 1 or iters < 1:
            raise ValueError("grid and iters must be positive")
        self.rng = np.random.default_rng(seed)
        try:
            self.stages = [(int(a), int(b)) for a, b in
                           (part.strip().split(":") for part in spec.split(","))]
        except (ValueError, TypeError):
            raise ValueError("measurement schedule must look like 0:2,1000:1") from None
        starts = [a for a, _ in self.stages]
        if not starts or starts[0] != 0 or any(b <= a for a, b in zip(starts, starts[1:])):
            raise ValueError("schedule starts must begin at 0 and strictly increase")
        if starts[-1] >= iters:
            raise ValueError("final full-data stage must start before iters")
        strides = [s for _, s in self.stages]
        if any(s < 1 or grid % s for s in strides):
            raise ValueError("each stride must be positive and divide grid")
        if any(a % b for a, b in zip(strides, strides[1:])) or strides[-1] != 1:
            raise ValueError("strides must form nested subsets ending in stride 1")

    def select(self, iteration):
        start, stride = max((v for v in self.stages if v[0] <= iteration), key=lambda v: v[0])
        if stride == 1:
            return np.arange(self.grid**2, dtype=np.int64), stride
        if self.policy == "random":
            return np.sort(self.rng.choice(self.grid**2, (self.grid//stride)**2,
                                           replace=False)), stride
        ry = rx = 0
        if self.policy == "rotate":
            ry, rx = divmod((iteration - start) % (stride**2), stride)
        y = np.arange(ry, self.grid, stride)
        x = np.arange(rx, self.grid, stride)
        return (y[:, None]*self.grid + x[None, :]).ravel().astype(np.int64), stride
