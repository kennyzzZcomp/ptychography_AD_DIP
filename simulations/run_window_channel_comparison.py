"""ABCD input-order ablation: original / shared shuffle / independent shuffles.

Every run retains the SAME four ABCD groups, 25 inputs and measurements per step.
Only network input channel order changes; loss positions/targets are untouched.
Defaults match the previous group comparison: 1000 stage-1 updates, no stage 2,
LR net=.001, probe=.01, TGV=.001, diagnostics every 250 updates. Permutations are
fixed, not redrawn each iteration. Network weights are NOT permuted to compensate.
Repeat with different --seed and --channel-seed for robustness; these are separate
RNGs. The output directory and final archive must not exist.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def commands(args):
    common = [sys.executable, '-u', str(Path(__file__).with_name('run_paper_windows.py')),
              '--preset', 'paper', '--grid', '10', '--step-px', '12', '--obj-size', '624',
              '--eval-size', '96', '--window-layout', 'sparse', '--window-side', '5',
              '--update-mode', 'sequential', '--loss-mode', 'independent',
              '--iters', str(args.iters), '--switch-after', '0', '--eval-every', '100',
              '--base-ch', '32', '--seed', str(args.seed), '--network-seed', str(args.seed),
              '--channel-seed', str(args.channel_seed), '--device', args.device,
              '--probe-mode', 'pixel', '--probe-init', 'ones', '--consistency-weight', '0',
              '--lr-net', str(args.lr_net), '--lr-probe', str(args.lr_probe),
              '--tgv-amp', str(args.tgv_amp), '--transfer-every', str(args.transfer_every)]
    return [(name, common + ['--channel-order', mode, '--outdir', str(Path(args.outdir)/name)])
            for name, mode in [('a_original', 'original'), ('b_shared_shuffle', 'shared'),
                               ('c_independent_shuffle', 'independent')]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--outdir', required=True)
    p.add_argument('--device', choices=('cuda', 'cpu', 'auto'), default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--channel-seed', type=int, default=0)
    p.add_argument('--iters', type=int, default=1000)
    p.add_argument('--transfer-every', type=int, default=250)
    p.add_argument('--lr-net', type=float, default=.001)
    p.add_argument('--lr-probe', type=float, default=.01)
    p.add_argument('--tgv-amp', type=float, default=.001)
    args = p.parse_args()
    if args.iters < 1 or args.transfer_every < 0 or args.channel_seed < 0:
        p.error('Require positive iters and nonnegative transfer-every/channel-seed')
    root = Path(args.outdir).resolve()
    archive = root.parent/(root.name + '.zip')
    if root.exists() or archive.exists():
        p.error('Output directory or zip exists; choose a new --outdir (no overwrite).')
    args.outdir = str(root)
    runs = commands(args)
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(settings=vars(args), runs=[dict(name=n, command=c) for n, c in runs],
                    note='ABCD geometry and physics loss fixed; only input channels permuted; first-layer weights not reparameterized')
    (root/'comparison_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for name, command in runs:
        print(f'\n=== {name}: stage 1, {args.iters} updates ===', flush=True)
        subprocess.run(command, check=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(root.rglob('*')):
            if file.is_file():
                z.write(file, arcname=str(file.relative_to(root.parent)))
    print(f'\nAll three runs complete. Send this archive for analysis: {archive}', flush=True)


if __name__ == '__main__':
    main()
