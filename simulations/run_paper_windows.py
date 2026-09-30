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
    window_update: str = "sequential"
    window_consistency: float = 0.0
    window_switch_after: int = 0
    window_stage2_lr_net: float = .002
    window_stage2_lr_probe: float = .01
    window_stage2_tgv: float = .0001


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--update-mode", choices=("sequential", "joint"), default="sequential")
    p.add_argument("--consistency-weight", type=float, default=0.0)
    p.add_argument("--preset", choices=("paper", "smoke"), default="paper")
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
    p.add_argument("--outdir", required=True, help="must not exist; never overwrite")
    a = p.parse_args()
    geometry = dict(PRESETS[a.preset])
    geometry.update(grid=10, step_px=12 if a.preset == "paper" else 2,
                    obj_size=624 if a.preset == "paper" else 152)
    cfg = WindowCfg(**geometry, preset=a.preset, iters=a.iters, eval_every=a.eval_every,
                    eval_size=96 if a.preset == "paper" else 16, base_ch=a.base_ch,
                    seed=a.seed, network_seed=a.network_seed, device=a.device,
                    probe_init=a.probe_init, probe_mode=a.probe_mode,
                    lr_net=a.lr_net, lr_probe=a.lr_probe, tgv_amp=a.tgv_amp,
                    outdir=a.outdir, window_update=a.update_mode,
                    window_consistency=a.consistency_weight,
                    window_switch_after=a.switch_after, window_stage2_lr_net=a.stage2_lr_net,
                    window_stage2_lr_probe=a.stage2_lr_probe, window_stage2_tgv=a.stage2_tgv_amp)
    run_windows(cfg)


if __name__ == "__main__":
    main()
