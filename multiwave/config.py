"""All lengths at the CLI/config boundary are explicitly in um, nm or mm."""
from dataclasses import dataclass
import math


METHODS = ("pixel_shared_amp", "unet_shared_amp")
FEEDBACK_METHODS = ("feedback_shared_amp",)
LEGACY_METHODS = ("pixel_independent", "pixel_coupled", "unet_independent", "unet_coupled", "pixel_common", "unet_common")
SCENES = ("usaf_zero_phase", "shared_complex", "shared_opd", "spectral_absorption", "dispersive")


@dataclass
class Config:
    object_size: int = 96
    patch_size: int = 48
    detector_size: int = 0  # 0 preserves the historical patch-sized detector
    pixel_parameterization: str = "sigmoid"
    unet_activation: str = "sigmoid"
    grid: int = 5
    step: int = 8
    jitter: int = 1
    scan_quantum: int = 1  # preserve the same physical jitter on refined grids
    pixel_um: float = 4.0
    distance_mm: float = 1.5
    wavelengths_nm: tuple = (515.0, 633.0)
    weights: tuple = (0.65, 0.35)
    spectral_mode: str = "equal_power"  # no explicit weights; power is in probe amplitude
    probe_mode: str = "pixel"  # known remains an oracle control
    lr_probe: float = 0.01
    pad_factor: int = 2
    photons_per_scan: float = 200000.0  # 0 = noiseless, otherwise incident photons
    scene: str = "usaf_zero_phase"
    usaf_path: str = ""  # defaults to project-root USAF.jpg
    usaf_fill: float = 0.55  # longest image side / object side; aspect ratio preserved
    scene_seed: int = 17
    noise_seed: int = 23
    network_seed: int = 31
    holdout_fraction: float = 0.2
    iterations: int = 500
    eval_every: int = 25
    base_channels: int = 8
    lr_pixel: float = 0.03
    lr_net: float = 0.002
    lr_net_decay_after: int = 0  # 0 disables; decay starts on update after this count
    lr_net_decay_factor: float = 0.2
    opd_scale_um: float = 0.15
    tv_weight: float = 0.0
    loss: str = "amplitude"
    unet_skip: str = "concat"
    unet_detail: str = "none"
    feedback_mode: str = "learned"
    feedback_step: float = 0.01
    tgv_weight: float = 0.0
    tgv_alpha0: float = 2.0
    tgv_alpha1: float = 1.0
    tgv_eps: float = 1e-3
    tgv_lr: float = 0.01
    tgv_inner_steps: int = 5
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
        if name == "highres":
            return cls(object_size=384, patch_size=192, step=32, jitter=4,
                       scan_quantum=4, pixel_um=1.0, chunk=2,
                       photons_per_scan=0.0, iterations=1000, eval_every=50)
        if name == "resolved":
            return replace_config_highres()
        raise ValueError(f"Unknown preset: {name}")

    def validate(self):
        if self.feedback_mode not in ("learned", "identity", "no_gradient"):
            raise ValueError("unsupported feedback_mode")
        if not math.isfinite(self.feedback_step) or self.feedback_step <= 0:
            raise ValueError("feedback_step must be finite and positive")
        if self.unet_detail not in ("none", "residual"):
            raise ValueError("unet_detail must be none or residual")
        if self.unet_detail != "none" and self.scene != "usaf_zero_phase":
            raise ValueError("detail branch requires shared zero-phase amplitude scene")
        if self.loss not in ("amplitude", "poisson"):
            raise ValueError("loss must be amplitude or poisson")
        if self.loss == "poisson" and self.photons_per_scan <= 0:
            raise ValueError("--loss poisson requires --photons-per-scan > 0 (count data)")
        if self.unet_skip not in ("concat", "dwt_concat"):
            raise ValueError("unet_skip must be concat or dwt_concat")
        if self.scene != "usaf_zero_phase" and (self.tgv_weight or self.unet_skip != "concat"):
            raise ValueError("DWT/TGV options are scoped to the shared zero-phase amplitude scene")
        if self.tv_weight and self.tgv_weight:
            raise ValueError("test TV and TGV separately; do not enable both")
        if (not isinstance(self.lr_net_decay_after, int) or isinstance(self.lr_net_decay_after, bool)
                or self.lr_net_decay_after < 0
                or self.lr_net_decay_after >= self.iterations):
            raise ValueError("lr_net_decay_after must be 0 (disabled) or less than iterations")
        if not math.isfinite(self.lr_net_decay_factor) or not 0 < self.lr_net_decay_factor <= 1:
            raise ValueError("lr_net_decay_factor must be in (0, 1]")
        for name in ("object_size", "patch_size", "grid", "step", "pad_factor",
                     "iterations", "eval_every", "base_channels", "chunk", "threads", "scan_quantum", "tgv_inner_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.object_size % 8 or self.object_size < 16:
            raise ValueError("object_size must be >=16 and divisible by 8 (U-Net)")
        if self.patch_size % 2 or self.patch_size < 8:
            raise ValueError("patch_size must be even and >=8")
        if not isinstance(self.detector_size, int) or self.detector_size < 0 or self.detector_size % 2:
            raise ValueError("detector_size must be zero or a positive even integer")
        if self.detector_pixels > self.patch_size*self.pad_factor:
            raise ValueError("detector_size must fit the padded propagation grid")
        if self.pixel_parameterization not in ("sigmoid", "softplus", "direct") or self.unet_activation not in ("sigmoid", "softplus"):
            raise ValueError("unsupported amplitude parameterization")
        if not isinstance(self.jitter, int) or self.jitter < 0 or 2*self.jitter >= self.step:
            raise ValueError("jitter must be a nonnegative integer with 2*jitter < step")
        if self.jitter % self.scan_quantum or self.step % self.scan_quantum:
            raise ValueError("jitter and step must be divisible by scan_quantum")
        extent = self.patch_size + (self.grid-1)*self.step + 2*self.jitter
        if extent > self.object_size or self.grid < 3:
            raise ValueError("scan must fit object_size, and grid must be >=3")
        if self.pad_factor < 2:
            raise ValueError("pad_factor >=2 is required for this finite-window pilot")
        for name in ("pixel_um", "distance_mm", "lr_pixel", "lr_net", "lr_probe", "opd_scale_um",
                     "tgv_alpha0", "tgv_alpha1", "tgv_eps", "tgv_lr"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("photons_per_scan", "tv_weight", "tgv_weight"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.holdout_fraction < 0.5:
            raise ValueError("holdout_fraction must be in (0, 0.5)")
        if not 1 <= len(self.wavelengths_nm) <= 6:
            raise ValueError("this pilot supports 1 to 6 wavelengths")
        if any(not math.isfinite(v) or v <= 0 for v in self.wavelengths_nm):
            raise ValueError("wavelengths must be finite and positive")
        if len(set(self.wavelengths_nm)) != len(self.wavelengths_nm):
            raise ValueError("wavelengths must be distinct")
        if self.spectral_mode not in ("equal_power", "weighted"):
            raise ValueError("spectral_mode must be equal_power or weighted")
        if self.probe_mode not in ("known", "pixel"):
            raise ValueError("probe_mode must be known or pixel")
        if self.spectral_mode == "weighted" and len(self.weights) != len(self.wavelengths_nm):
            raise ValueError("one weight is required per wavelength")
        if any(not math.isfinite(v) or v <= 0 for v in self.weights):
            raise ValueError("weights must be finite and positive")
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-6):
            raise ValueError("weights must sum to 1; they are fixed physical fractions")
        if self.scene not in SCENES:
            raise ValueError(f"scene must be one of {SCENES}")
        if not math.isfinite(self.usaf_fill) or not 0.1 <= self.usaf_fill <= 1:
            raise ValueError("usaf_fill must be in [0.1, 1]")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        return self

    @property
    def detector_pixels(self):
        return self.detector_size or self.patch_size

    @property
    def probe_power(self):
        return 1/len(self.wavelengths_nm) if self.spectral_mode == "equal_power" else 1.0

    @property
    def mixing_coefficients(self):
        return (1.,)*len(self.wavelengths_nm) if self.spectral_mode == "equal_power" else self.weights


def replace_config_highres():
    from dataclasses import replace
    return replace(Config.preset("highres"), detector_size=768, pad_factor=8,
                   base_channels=16, pixel_parameterization="direct", unet_activation="softplus")
