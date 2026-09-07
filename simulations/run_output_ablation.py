# -*- coding: utf-8 -*-
"""顺序运行 ProPtyNet_torch 的四组输出参数化消融。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CASES = (
    ("softplus", "cossin"),
    ("leaky", "cossin"),
    ("softplus", "tanh"),
    ("leaky", "tanh"),
)


def main():
    ap = argparse.ArgumentParser(description="运行四组 GT-probe 输出参数化消融")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-root", default="output_param_ablation")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    target = here / "ProPtyNet_torch.py"
    out_root = Path(args.out_root).resolve()

    for amp, phase in CASES:
        outdir = out_root / f"{amp}_{phase}"
        cmd = [
            sys.executable, str(target), "net",
            "--probe-mode", "truth",
            "--amp-output", amp,
            "--phase-output", phase,
            "--iters", str(args.iters),
            "--seed", str(args.seed),
            "--device", args.device,
            "--outdir", str(outdir),
        ]
        print("\n$ " + subprocess.list2cmdline(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=here.parent, check=True)


if __name__ == "__main__":
    main()
