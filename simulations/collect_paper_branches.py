"""Collect completed checkpoint branches; never runs reconstruction."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def collect(root):
    rows, histories = [], {}
    parents = set()
    for label in ('continue', 'A', 'B', 'C', 'D'):
        folder = root / label
        meta = json.loads((folder/'branch_metadata.json').read_text(encoding='utf-8'))
        parents.add(meta['parent_sha256'])
        filename = 'ad_result.npz' if meta['representation'] == 'pixel' else 'net_result.npz'
        with np.load(folder/filename, allow_pickle=False) as z:
            history = json.loads(str(z['hist']))
            cfg = json.loads(str(z['cfg']))
        if meta['branch'] != label or history[-1]['branch_step'] != meta['additional_steps']:
            raise ValueError(f'Incomplete or mislabeled branch {label}')
        end = history[-1]
        tail = [r for r in history if r['branch_step'] > end['branch_step']-200]
        rows.append({'branch': label, 'representation': meta['representation'],
                     'probe_mode': cfg['probe_mode'], 'tgv_weight': cfg['tgv_amp'],
                     'object_lr': meta['object_lr'], 'probe_lr': meta['probe_lr'],
                     'psnr_amp': end['psnr_amp'], 'ssim_amp': end['ssim_amp'],
                     'ssim_phs': end['ssim_phs'], 'object_relerr': end['relerr'],
                     'probe_relerr': end['relerr_p'], 'data_loss': end['data_loss'],
                     'tail200_logged_mean_objerr': float(np.mean([r['relerr'] for r in tail])),
                     'tail_logged_samples': len(tail), 'elapsed_s': meta['elapsed_s'],
                     'training_s': meta['training_s'],
                     'with_shared_prefix_s': meta['cumulative_elapsed_s'],
                     'first_object_update': history[0]['object_relative_update'],
                     'first_probe_update': history[0]['probe_relative_update']})
        histories[label] = history
    if len(parents) != 1:
        raise ValueError('Branches did not originate from the same checkpoint')
    output = root/'summary.csv'
    with output.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for label, history in histories.items():
        for ax, key in zip(axes, ('psnr_amp', 'relerr', 'relerr_p')):
            ax.plot([r['it'] for r in history], [r[key] for r in history], label=label)
            ax.set_xlabel('Total updates (includes shared prefix)')
            ax.set_title(key); ax.grid(alpha=.3)
    axes[0].legend()
    image = root/'branch_curves.png'
    if image.exists():
        raise FileExistsError(image)
    fig.savefig(image, dpi=150); plt.close(fig)
    print(output)
    for row in rows:
        print(f"{row['branch']:>8}: PSNR={row['psnr_amp']:.2f}, object={row['object_relerr']:.4f}, probe={row['probe_relerr']:.4f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    collect(ap.parse_args().root)
