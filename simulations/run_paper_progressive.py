"""Editable Colab launcher for sparse-subset or four-window progressive training.

Run this file after editing the settings below. No training occurs on import.
Initially input channels and measurement loss follow the same 2D subset.
Optional reconstructed-field conditioning replaces only the stage-2 input.
"""
from pathlib import Path
import subprocess
import sys


# -------- Edit these settings, then run this file in Colab --------
STAGE1_MODE = "windows"      # "windows": sequential four windows; "subset": original sparse subset
GRID = 10
STEP_PX = 12                 # full ~79.7%; windows step=24px; subset stride=3 gives 36px
OBJ_SIZE = 624               # must fit N + (GRID-1)*STEP_PX = 620
EVAL_SIZE = 96
SWITCH_AFTER = 1000          # stage 2 begins at update 1001
HALF_RES_STAGE1 = False      # optional: half spatial object network, unchanged physical forward
STAGE1_STRIDE = 3            # skip TWO positions: row/col indices 0,3,6,9
SEED = 0
BASE_CH = 32
RESET_TGV_AT_SWITCH = False  # subset/object mode only; windows already creates a new full-domain TGV
# No support mask; no cosine on top of these piecewise-constant learning rates.

# Edit the block for your selected mode. All iteration totals include BOTH stages.
WINDOW_UPDATE = "sequential" # one window loss and optimizer update per iteration
WINDOW_CONSISTENCY = 0.0
STAGE2_READOUT = "complex-residual" # windows only: "direct" reproduces the original stage-2 readout
if STAGE1_MODE == "windows":
    TOTAL_ITERS = 2000       # 1000 window updates + 1000 full-data updates
    STAGE2_INPUT = "object"  # fixed fused amplitude/cos-phase/sin-phase input
    STAGE2_NETWORK = "fresh" # new backbone; STAGE2_READOUT chooses direct vs residual output
    STAGE1_TGV = 0.001       # successful 30.50 dB run settings, held fixed
    STAGE2_TGV = 0.001
    STAGE1_LR_NET = 0.001
    STAGE2_LR_NET = 0.002
    STAGE1_LR_PROBE = 0.01
    STAGE2_LR_PROBE = 0.01
    PROBE_INIT = "ones"
    EVAL_EVERY = 100
    OUTDIR = f"ov80_windows_progressive_{STAGE2_READOUT}_tgv001_seed0"
else:                       # original sparse-subset defaults, preserved
    TOTAL_ITERS = 4000
    STAGE2_INPUT = "diffraction" # also supports "object" or "reconstruction"
    STAGE2_NETWORK = "reuse"     # "fresh" requires object/reconstruction input
    STAGE1_TGV = 0.1
    STAGE2_TGV = 0.0
    STAGE1_LR_NET = 5e-3
    STAGE2_LR_NET = 8e-4
    STAGE1_LR_PROBE = 2e-2
    STAGE2_LR_PROBE = 2e-2
    PROBE_INIT = "disk"
    EVAL_EVERY = 25
    OUTDIR = "ov80_grid10_progressive"


def build_command():
    if STAGE1_MODE not in ("subset", "windows"):
        raise ValueError("STAGE1_MODE must be 'subset' or 'windows'")
    if STAGE1_MODE == "windows":
        if RESET_TGV_AT_SWITCH:
            raise ValueError("Windows already resets TGV; this comparison switch is for subset mode")
        if HALF_RES_STAGE1 or STAGE2_INPUT != "object" or STAGE2_NETWORK != "fresh":
            raise ValueError("Windows mode requires full-resolution stage 1 and fresh object-input stage 2")
        if GRID != 10 or not 0 < SWITCH_AFTER < TOTAL_ITERS:
            raise ValueError("Windows progressive requires GRID=10 and 0 < SWITCH_AFTER < TOTAL_ITERS")
        if WINDOW_UPDATE == "sequential" and SWITCH_AFTER % 4:
            raise ValueError("Sequential stage 1 must complete a four-window sweep")
        if STAGE2_READOUT not in ("direct", "complex-residual"):
            raise ValueError("STAGE2_READOUT must be direct or complex-residual")
        script = Path(__file__).resolve().with_name("run_paper_windows.py")
        options = {
            "preset": "paper", "grid": GRID, "step-px": STEP_PX, "obj-size": OBJ_SIZE,
            "eval-size": EVAL_SIZE, "eval-every": EVAL_EVERY, "iters": TOTAL_ITERS,
            "switch-after": SWITCH_AFTER, "base-ch": BASE_CH, "seed": SEED, "device": "cuda",
            "update-mode": WINDOW_UPDATE, "consistency-weight": WINDOW_CONSISTENCY,
            "probe-mode": "pixel", "probe-init": PROBE_INIT,
            "lr-net": STAGE1_LR_NET, "lr-probe": STAGE1_LR_PROBE, "tgv-amp": STAGE1_TGV,
            "stage2-lr-net": STAGE2_LR_NET, "stage2-lr-probe": STAGE2_LR_PROBE,
            "stage2-tgv-amp": STAGE2_TGV, "outdir": OUTDIR,
            "stage2-readout": STAGE2_READOUT,
        }
        return [sys.executable, "-u", str(script),
                *[s for key, value in options.items() for s in (f"--{key}", str(value))]]
    script = Path(__file__).resolve().with_name("ProPtyNet_paper.py")
    options = {
        "preset": "paper", "obj-size": OBJ_SIZE, "grid": GRID, "step-px": STEP_PX,
        "eval-size": EVAL_SIZE, "eval-every": EVAL_EVERY, "iters": TOTAL_ITERS,
        "base-ch": BASE_CH, "seed": SEED, "device": "cuda",
        "probe-mode": "pixel", "probe-init": PROBE_INIT,
        "measurement-schedule": f"0:{STAGE1_STRIDE},{SWITCH_AFTER}:1",
        "measurement-policy": "fixed", "input-policy": "follow_measurements",
        "tgv-amp": STAGE1_TGV, "tgv-phase": 0,
        "tgv-amp-schedule": f"{SWITCH_AFTER}:{STAGE2_TGV}",
        "lr-net": STAGE1_LR_NET, "lr-probe": STAGE1_LR_PROBE,
        "lr-schedule": f"{SWITCH_AFTER}:{STAGE2_LR_NET}:{STAGE2_LR_PROBE}",
        "outdir": OUTDIR,
    }
    if HALF_RES_STAGE1:
        options["half-res-until"] = SWITCH_AFTER
    if STAGE2_INPUT != "diffraction":
        options["stage2-input"] = STAGE2_INPUT
    if STAGE2_NETWORK != "reuse":
        options["stage2-network"] = STAGE2_NETWORK
    return [sys.executable, "-u", str(script), "net",
            *(["--reset-tgv-at-switch"] if RESET_TGV_AT_SWITCH else []),
            *[s for key, value in options.items() for s in (f"--{key}", str(value))]]


def main():
    if Path(OUTDIR).exists():
        raise FileExistsError(f"Choose a new OUTDIR to preserve existing results: {OUTDIR}")
    command = build_command()
    print("Running:", " ".join(command), flush=True)
    # Forward child output; launching via !python in Colab displays traceback/logs.
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
