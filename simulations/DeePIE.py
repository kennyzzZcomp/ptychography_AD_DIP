#!/usr/bin/env python
"""DeePIE independent reproduction on ProPtyNet_paper's exact scene builder.

See DeePIE_README.md for source mapping, disclosed assumptions and run commands.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulations.ProPtyNet_paper import Cfg, PRESETS as BASE_PRESETS
from functions.paperrepro.deepie_model import ModelConfig
from functions.paperrepro.deepie_solver import TrainingConfig, run_deepie
from functions.paperrepro.scene import build_scene

# Local to DeePIE: do not alter the baseline's defaults or CLI presets.
# Match PtyINR's default ARRAY/SCAN layout, not its X-ray specimen or probe.
# Keeping wavelength/z/detector pitch means N=64 increases object pixel pitch 8x.
PRESETS = {**BASE_PRESETS, "ptyinr-layout": {
    **BASE_PRESETS["paper"], "N":64, "obj_size":241, "grid":60, "step_px":3,
    "eval_size":0,
}}


# Only measurement/geometry/initial-probe/evaluation fields are inherited.
# The other solver's network and training settings must not leak into DeePIE.
SCENE_FIELDS = {
    "wlength": float, "N": int, "det_pixel": float, "z": float,
    "grid": int, "step_px": int, "probe_diam_um": float, "obj_size": int,
    "probe_amp_image": str, "probe_phase_mode": str, "probe_phase_rad": float,
    "obj_amp_floor": float, "amp_image": str, "phs_image": str, "obj_phase_rad": float,
    "noise": str, "snr_db": float, "noise_seed": int, "seed": int, "assets": str,
    "probe_init": str, "probe_init_sigma": float, "s1_margin": float, "eval_size": int,
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["check", "run"], help="check: scene only; run: optimize DeePIE")
    p.add_argument("--preset", choices=PRESETS, default=None)
    p.add_argument("--scene-config", type=Path, help="baseline result .npz, config .json, or DeePIE manifest")
    p.add_argument("--sample-pixel-um", type=float, default=None,
                   help="Set object-plane pixel pitch by deriving detector pitch from lambda*z/(N*dx); changes measurement sampling")
    for name, kind in SCENE_FIELDS.items():
        p.add_argument("--" + name.replace("_", "-"), dest=name, type=kind, default=None)
    p.add_argument("--obj-amp-binary-invert", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--quad-sign", type=float, choices=[-1., 1.], default=None)
    p.add_argument("--snr", dest="snr_db", type=float, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--outdir", default="results_paper/deepie")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--hidden-layers", type=int, default=3)
    p.add_argument("--high-frequencies", type=int, default=128)
    p.add_argument("--low-frequencies", type=int, default=128)
    p.add_argument("--phases", type=int, default=32)
    p.add_argument("--encoding-side", type=int, default=256)
    p.add_argument("--positional-encoding", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fourier-weights", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--omega", type=float, default=30.)
    p.add_argument("--activation", choices=["sine", "leaky-sine"], default="sine",
                   help="disclosed hypotheses; supplement does not define the activation")
    p.add_argument("--alpha", type=float, default=.05,
                   help="only used with leaky-sine; not silently interpreted as a paper formula")
    p.add_argument("--phase-scale", type=float, default=2 * np.pi)
    p.add_argument("--head-scale", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--probe-lr", type=float, default=1e-4)
    p.add_argument("--decay-every", type=int, default=100)
    p.add_argument("--decay-factor", type=float, default=.5)
    p.add_argument("--scan-chunk", type=int, default=4)
    p.add_argument("--coordinate-chunk", type=int, default=4096)
    p.add_argument("--scan-weight", choices=["intensity", "uniform"], default="intensity")
    p.add_argument("--balance", choices=["equal-norm", "none"], default="equal-norm")
    p.add_argument("--probe-scale", choices=["measurement-rms", "none"], default="measurement-rms")
    p.add_argument("--network-seed", type=int, default=0)
    p.add_argument("--save-scene", action="store_true")
    return p


def configurations(args):
    loaded = {}
    if args.scene_config:
        if args.scene_config.suffix.lower() == ".npz":
            with np.load(args.scene_config, allow_pickle=False) as data:
                loaded = json.loads(str(data["cfg"].item()))
        else:
            loaded = json.loads(args.scene_config.read_text(encoding="utf-8-sig"))
            loaded = loaded.get("scene_config", loaded)
    preset = args.preset or loaded.get("preset", "paper")
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset}")
    kw = dict(PRESETS[preset], preset=preset)
    accepted = {*SCENE_FIELDS, "obj_amp_binary_invert"}
    kw.update({k: v for k, v in loaded.items() if k in accepted})
    # Explicit preset overrides geometry inherited from an existing result.
    if args.preset is not None:
        kw.update(PRESETS[args.preset])
    kw.update({k: getattr(args, k) for k in accepted if getattr(args, k) is not None})
    if args.sample_pixel_um is not None:
        if not np.isfinite(args.sample_pixel_um) or args.sample_pixel_um <= 0:
            raise ValueError("sample-pixel-um must be finite and positive")
        if args.det_pixel is not None:
            raise ValueError("Use either --sample-pixel-um or --det-pixel, not both")
        base=Cfg()
        kw["det_pixel"] = (kw.get("wlength",base.wlength)*kw.get("z",base.z)
                           / (kw.get("N",base.N)*args.sample_pixel_um*1e-6))
    kw.update(device=args.device, iters=args.iters, eval_every=args.eval_every,
              outdir=args.outdir, probe_mode="pixel")
    if kw.get("noise", "none") not in ("none", "gaussian", "poisson", "mixed"):
        raise ValueError("unknown noise type")
    if kw.get("probe_init", "ones") not in ("ones", "disk"):
        raise ValueError("probe_init must be ones or disk")
    cfg = Cfg(**kw)
    cfg.quad_sign = args.quad_sign if args.quad_sign is not None else loaded.get("quad_sign", -1.)
    if cfg.quad_sign not in (-1., 1.):
        raise ValueError("quad_sign must be -1 or +1")
    if args.scene_config and "quad_sign" not in loaded:
        print("[deepie] Input config has no quad_sign; using -1 unless explicitly overridden.")
    if cfg.iters < 1 or cfg.eval_every < 1:
        raise ValueError("iters/eval_every must be positive")
    mc = ModelConfig(**{k: getattr(args, k) for k in ModelConfig.__dataclass_fields__})
    tc = TrainingConfig(**{k: getattr(args, k) for k in TrainingConfig.__dataclass_fields__})
    return cfg, mc, tc


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        cfg, mc, tc = configurations(args)
    except (ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    if args.mode == "check":
        scene = build_scene(cfg, cfg.dev())
        print(json.dumps({"scene_fingerprint": scene.fp,
                          "scene_config": {**asdict(cfg),"quad_sign":cfg.quad_sign},
                          "sampling": {"object_pixel_um":cfg.dx1*1e6,
                                       "probe_diameter_px":cfg.probe_diam_px,
                                       "window_overlap":1-cfg.step_px/cfg.N,
                                       "scan_displacement_px":(cfg.grid-1)*cfg.step_px},
                          "model_config": asdict(mc),
                          "training_config": asdict(tc)}, indent=2))
    else:
        run_deepie(cfg, mc, tc)


if __name__ == "__main__":
    main()
