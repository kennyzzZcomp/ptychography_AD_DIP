"""All lengths at the CLI/config boundary are explicitly in um, nm or mm."""
from dataclasses import dataclass
import math


METHODS = ("pixel_independent", "pixel_coupled", "unet_independent", "unet_coupled")
SCENES = ("shared_complex", "shared_opd", "spectral_absorption", "dispersive")


@dataclass
class Config:
    object_size: int = 96
    patch_size: int = 48
    grid: int = 5
    step: int = 8
    jitter: int = 1
    pixel_um: float = 4.0
    distance_mm: float = 1.5
    wavelengths_nm: tuple = (515.0, 633.0)
    weights: tuple = (0.65, 0.35)
    pad_factor: int = 2
    photons_per_scan: float = 200000.0  # 0 = noiseless, otherwise incident photons
    scene: str = "spectral_absorption"
    scene_seed: int = 17
    noise_seed: int = 23
    network_seed: int = 31
    holdout_fraction: float = 0.2
    iterations: int = 500
    eval_every: int = 25
    base_channels: int = 8
    lr_pixel: float = 0.03
    lr_net: float = 0.002
    opd_scale_um: float = 0.15
    tv_weight: float = 0.0
    chunk: int = 8
    device: str = "auto"
    threads: int = 2

    @classmethod
    def preset(cls, name):
        if name == "smoke":
            return cls(object_size=64, patch_size=32, step=6, iterations=80,
                       eval_every=20, base_channels=4, photons_per_scan=0.0)
        if name == "standard":
            return cls()
        raise ValueError(f"Unknown preset: {name}")

    def validate(self):
        for name in ("object_size", "patch_size", "grid", "step", "pad_factor",
                     "iterations", "eval_every", "base_channels", "chunk", "threads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.object_size % 8 or self.object_size < 16:
            raise ValueError("object_size must be >=16 and divisible by 8 (U-Net)")
        if self.patch_size % 2 or self.patch_size < 8:
            raise ValueError("patch_size must be even and >=8")
        if not isinstance(self.jitter, int) or self.jitter < 0 or 2*self.jitter >= self.step:
            raise ValueError("jitter must be a nonnegative integer with 2*jitter < step")
        extent = self.patch_size + (self.grid-1)*self.step + 2*self.jitter
        if extent > self.object_size or self.grid < 3:
            raise ValueError("scan must fit object_size, and grid must be >=3")
        if self.pad_factor < 2:
            raise ValueError("pad_factor >=2 is required for this finite-window pilot")
        for name in ("pixel_um", "distance_mm", "lr_pixel", "lr_net", "opd_scale_um"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("photons_per_scan", "tv_weight"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.holdout_fraction < 0.5:
            raise ValueError("holdout_fraction must be in (0, 0.5)")
        if not 2 <= len(self.wavelengths_nm) <= 6:
            raise ValueError("this pilot supports 2 to 6 wavelengths")
        if any(not math.isfinite(v) or v <= 0 for v in self.wavelengths_nm):
            raise ValueError("wavelengths must be finite and positive")
        if len(set(self.wavelengths_nm)) != len(self.wavelengths_nm):
            raise ValueError("wavelengths must be distinct")
        if len(self.weights) != len(self.wavelengths_nm):
            raise ValueError("one weight is required per wavelength")
        if any(not math.isfinite(v) or v <= 0 for v in self.weights):
            raise ValueError("weights must be finite and positive")
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-6):
            raise ValueError("weights must sum to 1; they are fixed physical fractions")
        if self.scene not in SCENES:
            raise ValueError(f"scene must be one of {SCENES}")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        return self
