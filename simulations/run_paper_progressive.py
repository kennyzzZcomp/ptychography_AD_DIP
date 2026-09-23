"""Editable Colab launcher: 10x10 acquisition, 4x4 -> 10x10 net training.

Run this file after editing the settings below. No training occurs on import.
Both input channels and measurement loss follow the same 2D subset.
"""
from pathlib import Path
import subprocess
import sys


# -------- Edit these settings, then run this file in Colab --------
GRID = 10
STEP_PX = 12                 # full ~79.7%; stage 1 step=36px, ~39.1%
OBJ_SIZE = 624               # must fit N + (GRID-1)*STEP_PX = 620
EVAL_SIZE = 96
TOTAL_ITERS = 4000           # total, not 4000 additional stage-2 updates
SWITCH_AFTER = 1000          # stage 2 begins at update 1001
STAGE1_STRIDE = 3            # skip TWO positions: row/col indices 0,3,6,9
STAGE1_TGV = 0.1
STAGE2_TGV = 0.0
STAGE1_LR_NET = 5e-3
STAGE2_LR_NET = 8e-4
STAGE1_LR_PROBE = 2e-2
STAGE2_LR_PROBE = 2e-2        # intentionally unchanged
PROBE_INIT = "disk"
SEED = 0
BASE_CH = 32
EVAL_EVERY = 25
OUTDIR = "ov80_grid10_progressive"  # relative to current Colab working directory
# No support mask; no cosine on top of these piecewise-constant learning rates.


def build_command():
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
    return [sys.executable, "-u", str(script), "net",
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
