"""Finite saved-array diagnostics; no torch, reconstruction, or training.

Fits descriptive periodic residual templates on alternating spatial tiles and
scores ONLY held-out tiles. These are not corrected reconstruction metrics.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from functions.common.metrics import align_global_factor
from functions.paperrepro.evaluate import evaluate


def coverage(shape, probe, positions):
    out = np.zeros(shape, dtype=float)
    w = np.abs(probe.astype(complex)) ** 2
    if not np.issubdtype(positions.dtype, np.integer):
        raise ValueError('Only integer saved top-left positions are supported')
    for y, x in positions:
        h, k = w.shape
        if min(y, x) < 0 or y+h > shape[0] or x+k > shape[1]:
            raise ValueError('Probe patch outside saved canvas')
        out[y:y+h, x:x+k] += w
    return out


def periodic_cv(values, weights, period, valid=None):
    """Two-fold tile holdout, weighted complex residue-class means.

    Every residue class is fitted without its test tile. Training means and
    constant baselines are fit on train pixels only. Negative skill is allowed.
    This is spatial cross-validation, NOT independent experimental replication.
    """
    y, x = np.indices(values.shape)
    labels = (y % period) * period + x % period
    fold = ((y // period) + (x // period)) % 2
    valid = np.ones(values.shape, bool) if valid is None else valid
    total, baseline, count = 0., 0., 0
    minimum = values.size
    for f in (0, 1):
        train = valid & (fold != f)
        test = valid & (fold == f)
        den = np.bincount(labels[train], weights=weights[train], minlength=period**2)
        n = np.bincount(labels[train], minlength=period**2)
        real = np.bincount(labels[train], weights=(weights*values.real)[train], minlength=period**2)
        imag = np.bincount(labels[train], weights=(weights*values.imag)[train], minlength=period**2)
        pred = (real + 1j*imag) / np.maximum(den, 1e-30)
        test &= den[labels] > 0
        minimum = min(minimum, int(n[labels[test]].min()))
        mean = np.sum(weights[train]*values[train]) / np.sum(weights[train])
        total += float(np.sum(weights[test]*np.abs(values[test]-pred[labels[test]])**2))
        baseline += float(np.sum(weights[test]*np.abs(values[test]-mean)**2))
        count += int(test.sum())
    return {'heldout_skill_vs_train_constant': 1-total/baseline if baseline > 1e-25 else None,
            'scored_pixels': count, 'min_train_pixels_per_scored_residue': minimum,
            'heldout_sse': total, 'constant_sse': baseline}


def analyze(path):
    with np.load(path, allow_pickle=False) as z:
        cfg = json.loads(str(z['cfg']))
        gt = z['obj_gt'].astype(complex)
        rec = z['obj_rec'].astype(complex)
        start = (gt.shape[0]-96)//2
        roi = (slice(start, start+96), slice(start, start+96))
        illum = coverage(gt.shape, z['probe_gt'], z['positions'])[roi]
        gt, rec = gt[roi], rec[roi]
        aligned, _ = align_global_factor(rec, gt)
        residual = aligned-gt
        power = np.abs(residual)**2
        amp_error = np.abs(aligned)-np.abs(gt)
        phase_error = np.angle(aligned*np.conj(gt))
        valid = np.abs(gt) >= .05*np.abs(gt).max()
        ratio = np.zeros_like(gt)
        ratio[valid] = residual[valid]/gt[valid]
        bins = np.array_split(np.argsort(illum.ravel(), kind='stable'), 4)
        quartiles = []
        for ids in bins:
            quartiles.append({'n': len(ids), 'mean_illumination': float(illum.ravel()[ids].mean()),
                             'complex_rmse': float(np.sqrt(power.ravel()[ids].mean())),
                             'amp_rmse': float(np.sqrt((amp_error.ravel()[ids]**2).mean())),
                             'wrapped_phase_rmse': float(np.sqrt((phase_error.ravel()[ids]**2).mean())),
                             'error_energy_fraction': float(power.ravel()[ids].sum()/power.sum())})
        d = int(cfg['step_px'])
        period_scores = {}
        for period in (d-2, d, d+2):
            period_scores[str(period)] = {
                'relative_complex_residual': periodic_cv(ratio, np.abs(gt)**2, period, valid),
                'additive_complex_residual': periodic_cv(residual, np.ones(gt.shape), period),
                'gt_amplitude_control': periodic_cv(np.abs(gt).astype(complex), np.ones(gt.shape), period)}
        # Structural stratification reduces (but cannot remove) object confounding.
        gy, gx = np.gradient(np.abs(gt))
        edge = np.hypot(gy, gx)
        strata = {}
        for label, mask in [('dark', np.abs(gt) <= .5), ('bright', np.abs(gt) > .5),
                            ('low_amp_gradient', edge <= .05), ('high_amp_gradient', edge > .05)]:
            strata[label] = {'n': int(mask.sum()), 'illumination_error_spearman':
                            float(spearmanr(illum[mask], power[mask]).statistic) if mask.sum()>2 else None}
        result = {'file': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'step_px': d, 'tgv_amp': cfg.get('tgv_amp'),
                  'lr_net': cfg.get('lr_net'), 'lr_probe': cfg.get('lr_probe'),
                  'official_metrics': evaluate(rec, gt), 'roi': [start, start+96, start, start+96],
                  'illumination_error_spearman': float(spearmanr(illum.ravel(), power.ravel()).statistic),
                  'zero_illumination_pixels': int((illum==0).sum()),
                  'ratio_valid_pixels': int(valid.sum()), 'gt_min_amp': float(np.abs(gt).min()),
                  'illumination_quartiles_low_to_high': quartiles, 'structure_strata': strata,
                  'periodic_cross_validation': period_scores}
        maps = (np.abs(gt), np.abs(aligned), illum/illum.mean(), np.abs(residual), phase_error)
        return result, maps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(exist_ok=False, parents=True)
    names = [4, 5, 10, 8]
    results, maps = [], []
    for n in names:
        result, fields = analyze(args.directory / f'net_result ({n}).npz')
        results.append(result)
        maps.append(fields)
    payload = {'protocol': 'Center96, official global complex alignment; GT-assisted descriptive diagnostics only. No corrected scores.',
               'periodic_warning': 'Tile holdout is not independent sampling. Neighbor periods and GT control are descriptive, not a significance test or proof of ambiguity.',
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'results': results}
    with (args.output_dir/'diagnostics.json').open('x', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 5, figsize=(13, 10), constrained_layout=True)
    titles = ['GT amplitude', 'Aligned amplitude', 'Illumination / mean', 'Complex error magnitude', 'Wrapped phase error']
    limits = [(0, 1), (0, 1), (0, max(m[2].max() for m in maps)),
              (0, max(m[3].max() for m in maps)), (-np.pi, np.pi)]
    for i, fields in enumerate(maps):
        for j, field in enumerate(fields):
            im = axes[i,j].imshow(field, vmin=limits[j][0], vmax=limits[j][1], cmap='twilight' if j==4 else 'viridis')
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            if i==0:
                axes[i,j].set_title(titles[j])
            if j==0:
                axes[i,j].set_ylabel(f'File {names[i]} | step {results[i]["step_px"]}\nTGV {results[i]["tgv_amp"]}')
            if i==3:
                fig.colorbar(im, ax=axes[:,j], shrink=.6)
    fig.savefig(args.output_dir/'error_maps.png', dpi=160)
    plt.close(fig)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
