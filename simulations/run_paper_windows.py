"""Independent four-window pilot; does not modify run/net/progressive defaults.

Colab: !python /content/ptychography_AD_DIP/simulations/run_paper_windows.py \
    --update-mode sequential --iters 1000 --outdir /content/windows_seq_seed0

iters counts optimizer updates in BOTH modes: sequential 1000 = 250 sweeps;
joint 250 = 250 sweeps. Compare elapsed time and frame visits as well as updates.
Default consistency=0 first isolates sequential grouped conditioning; explicitly
enable --consistency-weight 0.01 to test agreement (extra peer forward cost).
No spatial cropping, known support, TGV, cosine schedule, or disk is added by default.
Optional --switch-after 1000 --iters 2000: replace the window network with a
fresh object-conditioned U-Net after 1000 updates; full data in stage 2.
--stage2-readout complex-residual: zero-initialized linear real/imag corrections
to the fixed raw stage-1 object. Default direct preserves the previous solver.
--loss-mode cached-fusion: preserve sequential 16-pattern updates, but train
the fused object using the current prediction and three detached cached fields.
Caches start at ones; no extra training forwards or extra optimizer updates.
Fusion uses existing computational footprints, not known illumination/support.

--window-layout coverage-balanced: fixed four-way partition of ALL measurements
using the coverage demo's capacity-preserving pair swaps. For 10x10 this is
4x25 channels/patterns, matching sparse --window-side 5, NOT inner 4x4's 64 frames.
--window-side is ignored in this mode. Use --coverage-diameter-px 59.1 and
--coverage-seed 0 to reproduce the geometry demo (grid 10, step 12).
The disk is a nominal grouping proxy only, not an object/probe support or a GT
probe. Grouping is computed once and frozen; channels stay in ascending original
scan-index order. Input, loss and readout all use the same new groups.
coverage_partition.json stores exact groups, E and preprocessing times (excluded
from the existing training-loop time). Other layouts/progressive defaults stay
unchanged. The training adapter currently requires equal group lengths (N%4=0).

--window-layout balanced-random: fixed random equal-capacity partition of ALL
measurements; --coverage-seed controls its isolated RNG. Same initial partition
as coverage-balanced with that seed; channels sorted by original scan index.
--transfer-every 250: copied-state, fixed-probe cross-group one-step diagnostics
at stage-1 updates 250,500,... and its final update. Saves cross_group_transfer.json;
extra diagnostic blocks are timed separately and excluded from elapsed_s.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulations.ProPtyNet_paper import Cfg, PRESETS
from functions.paperrepro.window_training import run_windows


@dataclass
class WindowCfg(Cfg):
    window_layout: str = "sparse"
    window_side: int = 4
    coverage_diameter_px: float = None
    coverage_anchor_step: float = 2.0
    coverage_seed: int = 0
    transfer_every: int = 0
    window_update: str = "sequential"
    window_loss_mode: str = "independent"
    window_consistency: float = 0.0
    window_switch_after: int = 0
    window_stage2_lr_net: float = .002
    window_stage2_lr_probe: float = .01
    window_stage2_tgv: float = .0001
    window_stage2_readout: str = "direct"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--update-mode", choices=("sequential", "joint"), default="sequential")
    p.add_argument("--window-layout", choices=("sparse", "compact-matched", "compact-full", "alternating", "inner", "coverage-balanced", "balanced-random"), default="sparse",
                   help="coverage-balanced: disk-coverage partition of ALL measurements into four equal groups; inner: fixed inner windows; alternating: corners then inner")
    p.add_argument("--window-side", type=int, choices=(3, 4, 5), default=4,
                   help="3, 4 or 5 scan positions per axis (stride 2); ignored by coverage-balanced and balanced-random; omit for compact-full")
    p.add_argument("--coverage-diameter-px", type=float, default=None,
                   help="coverage-balanced ONLY: nominal uniform-disk diameter for grouping, not a reconstruction support. Default: configured physical diameter in pixels; use 59.1 to reproduce the demo")
    p.add_argument("--coverage-anchor-step", type=float, default=2.0,
                   help="coverage-balanced ONLY: grouping feature-grid spacing in object pixels")
    p.add_argument("--coverage-seed", type=int, default=0,
                   help="coverage-balanced / balanced-random: isolated fixed partition RNG seed")
    p.add_argument("--transfer-every", type=int, default=0,
                   help="0 disables; stage-1 copied-state cross-group Adam diagnostic every N updates and at stage-1 end; sequential independent loss only")
    p.add_argument("--loss-mode", choices=("independent", "cached-fusion"), default="independent",
                   help="stage-1 training object: individual window or current + cached peer fusion")
    p.add_argument("--consistency-weight", type=float, default=0.0)
    p.add_argument("--preset", choices=("paper", "smoke"), default="paper")
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--step-px", type=int)
    p.add_argument("--obj-size", type=int)
    p.add_argument("--eval-size", type=int)
    p.add_argument("--iters", type=int, default=1000, help="optimizer updates, NOT sweeps")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--network-seed", type=int)
    p.add_argument("--device", default="cuda", choices=("auto", "cuda", "cpu"))
    p.add_argument("--probe-init", default="ones", choices=("ones", "disk"))
    p.add_argument("--probe-mode", default="pixel", choices=("pixel", "support", "truth"))
    p.add_argument("--lr-net", type=float, default=.001)
    p.add_argument("--lr-probe", type=float, default=.01)
    p.add_argument("--tgv-amp", type=float, default=0)
    p.add_argument("--switch-after", type=int, default=0, help="completed stage-1 updates; 0 disables stage 2")
    p.add_argument("--stage2-lr-net", type=float, default=.002)
    p.add_argument("--stage2-lr-probe", type=float, default=.01)
    p.add_argument("--stage2-tgv-amp", type=float, default=.0001)
    p.add_argument("--stage2-readout", choices=("direct", "complex-residual"), default="direct",
                   help="fresh stage-2 output: original amplitude/phase or O1 + zero-initialized complex correction")
    p.add_argument("--outdir", required=True, help="must not exist; never overwrite")
    a = p.parse_args()
    geometry = dict(PRESETS[a.preset])
    geometry.update(grid=a.grid,
                    step_px=a.step_px if a.step_px is not None else (12 if a.preset == "paper" else 2),
                    obj_size=a.obj_size if a.obj_size is not None else (624 if a.preset == "paper" else 152))
    cfg = WindowCfg(**geometry, preset=a.preset, iters=a.iters, eval_every=a.eval_every,
                    eval_size=a.eval_size if a.eval_size is not None else (96 if a.preset == "paper" else 16), base_ch=a.base_ch,
                    seed=a.seed, network_seed=a.network_seed, device=a.device,
                    probe_init=a.probe_init, probe_mode=a.probe_mode,
                    lr_net=a.lr_net, lr_probe=a.lr_probe, tgv_amp=a.tgv_amp,
                    outdir=a.outdir, window_update=a.update_mode,
                    window_layout=a.window_layout, window_side=a.window_side,
                    coverage_diameter_px=a.coverage_diameter_px,
                    coverage_anchor_step=a.coverage_anchor_step, coverage_seed=a.coverage_seed,
                    transfer_every=a.transfer_every,
                    window_loss_mode=a.loss_mode,
                    window_consistency=a.consistency_weight,
                    window_switch_after=a.switch_after, window_stage2_lr_net=a.stage2_lr_net,
                    window_stage2_lr_probe=a.stage2_lr_probe, window_stage2_tgv=a.stage2_tgv_amp,
                    window_stage2_readout=a.stage2_readout)
    run_windows(cfg)


if __name__ == "__main__":
    main()
