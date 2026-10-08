"""Run three stage-1-only grouping experiments with identical training settings.

Colab: !python /content/ptychography_AD_DIP/simulations/run_window_group_comparison.py \
    --outdir /content/window_group_comparison_seed0

Defaults: 1000 updates, sparse ABCD 5x5 / balanced random / coverage balanced,
four fixed groups of 25, diagnostics every 250 updates. No stage 2. The output
root and final .zip must not already exist. Upload the resulting zip for analysis.
Use --transfer-every 0 for timing-only runs without diagnostic overhead.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def commands(args):
    entry = Path(__file__).with_name('run_paper_windows.py')
    common = [sys.executable, '-u', str(entry), '--preset', 'paper',
              '--grid', '10', '--step-px', '12', '--obj-size', '624', '--eval-size', '96',
              '--window-side', '5', '--update-mode', 'sequential', '--loss-mode', 'independent',
              '--iters', str(args.iters), '--switch-after', '0', '--eval-every', '100',
              '--base-ch', '32', '--seed', str(args.seed), '--network-seed', str(args.seed),
              '--device', args.device, '--probe-mode', 'pixel', '--probe-init', 'ones',
              '--lr-net', str(args.lr_net), '--lr-probe', str(args.lr_probe), '--tgv-amp', str(args.tgv_amp),
              '--consistency-weight', '0', '--coverage-seed', str(args.partition_seed),
              '--coverage-diameter-px', '59.1', '--coverage-anchor-step', '2',
              '--transfer-every', str(args.transfer_every)]
    return [(name, common + ['--window-layout', layout, '--outdir', str(Path(args.outdir)/name)])
            for name, layout in [('abcd_5x5', 'sparse'), ('balanced_random', 'balanced-random'),
                                 ('coverage_balanced', 'coverage-balanced')]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu', 'auto'), default='cuda')
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--partition-seed', type=int, default=0)
    parser.add_argument('--transfer-every', type=int, default=250)
    parser.add_argument('--lr-net', type=float, default=.001)
    parser.add_argument('--lr-probe', type=float, default=.01)
    parser.add_argument('--tgv-amp', type=float, default=.001)
    args = parser.parse_args()
    if args.iters < 1 or args.transfer_every < 0:
        parser.error('iters must be positive; transfer-every must be nonnegative')
    root = Path(args.outdir).resolve()
    archive = root.parent / (root.name + '.zip')
    if root.exists() or archive.exists():
        parser.error('Output directory or zip already exists; choose a new --outdir (no overwrite).')
    args.outdir = str(root)
    runs = commands(args)
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(settings=vars(args), runs=[dict(name=n, command=c) for n, c in runs],
                    note='all 100 frames, 4 fixed groups of 25; stage1 only; diagnostic blocks timed separately')
    (root/'comparison_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for name, command in runs:
        print(f'\n=== {name}: stage 1, {args.iters} updates ===', flush=True)
        subprocess.run(command, check=True)
    # Exclusive creation avoids silently replacing any archive created during training.
    import zipfile
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(root.rglob('*')):
            if file.is_file():
                z.write(file, arcname=str(file.relative_to(root.parent)))
    print(f'\nAll three runs complete. Results: {archive}', flush=True)


if __name__ == '__main__':
    main()
