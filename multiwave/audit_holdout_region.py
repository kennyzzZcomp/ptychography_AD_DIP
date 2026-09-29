"""Truth substitution only for post-hoc attribution, NEVER used for fitting."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .diagnose_resolution import setup, DiagnosticOperator, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    root = Path(args.directory)
    cfg, scene = setup()
    op = DiagnosticOperator(scene, cfg)
    gt = scene.objects[0].real
    report = {}
    with torch.no_grad():
        for n in (192,768):
            rec = torch.tensor(np.load(root/f'direct_flat_{n}'/'fields.npz')['reconstruction'])
            fields = {'reconstruction':rec,
                      'truth_outside_roi':torch.where(scene.roi,rec,gt),
                      'truth_inside_roi':torch.where(scene.roi,gt,rec)}
            rows = {}
            for name,a in fields.items():
                numerator, denominator = 0.,0.
                for ids in scene.holdout.split(2):
                    target, pred = op(gt,ids,n),op(a,ids,n)
                    numerator += float(((pred+1e-12).sqrt()-(target+1e-12).sqrt()).square().sum())
                    denominator += float(target.sum())
                rows[name] = (numerator/denominator)**.5
            report[str(n)] = rows
    save_json(root/'holdout_attribution.json',report)
    print(report)


if __name__ == '__main__': main()
