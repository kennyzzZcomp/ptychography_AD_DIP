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
    p.add_argument("--window-layout", choices=("sparse", "compact-matched", "compact-full", "alternating"), default="sparse",
                   help="alternating: four corner then four inner 4x4 windows; fixed corner fusion. Other layouts: sparse, compact-matched, compact-full")
    p.add_argument("--window-side", type=int, choices=(3, 4, 5), default=4,
                   help="3, 4 or 5 scan positions per axis (stride 2); omit for compact-full")
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
                    window_loss_mode=a.loss_mode,
                    window_consistency=a.consistency_weight,
                    window_switch_after=a.switch_after, window_stage2_lr_net=a.stage2_lr_net,
                    window_stage2_lr_probe=a.stage2_lr_probe, window_stage2_tgv=a.stage2_tgv_amp,
                    window_stage2_readout=a.stage2_readout)
    run_windows(cfg)


if __name__ == "__main__":
    main()
