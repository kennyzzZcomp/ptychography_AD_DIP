"""Render saved diagnostic arrays without changing them."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory')
    args = p.parse_args()
    root = Path(args.directory)
    rows = []
    table = ['# Saved reconstruction diagnostics', '',
             '| Run | wavelengths nm | updates | ROI relative error | center RMSE | high-frequency relative error | train amplitude NRMSE | holdout amplitude NRMSE |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for path in sorted(root.glob('*/fit.json')):
        record = json.loads(path.read_text())
        rows.append(dict(name=path.parent.name, **record['history'][-1]))
        r = record['history'][-1]
        table.append(f"| {path.parent.name} | {record['config']['wavelengths_nm']} | {r['step']} | {r['object_error']:.8g} | {r['center_rmse']:.8g} | {r['high_frequency_error']:.8g} | {r['train']:.8g} | {r['holdout']:.8g} |")
    table += ['', 'All metrics use saved final iterates, without truth-based checkpoint selection. Different detector sizes correspond to different measured data; compare common-object metrics, not measurement residuals alone. See RESOLUTION_DIAGNOSIS.md for definitions and limits.']
    (root/'experiment_table.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    (root/'table.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    names = ['direct_flat_192','direct_flat_768','sigmoid_flat_768']
    available = [n for n in names if (root/n/'fields.npz').exists()]
    first = np.load(root/available[0]/'fields.npz')
    images = [('Truth',first['truth'])]
    for name in available:
        arr = np.load(root/name/'fields.npz')
        info = next(r for r in rows if r['name']==name)
        images.append((f"{name}\n{info['step']} updates; ROI error {info['object_error']:.5f}",arr['reconstruction']))
    fig, axes = plt.subplots(2,len(images),figsize=(4*len(images),7),constrained_layout=True)
    for col,(label,a) in enumerate(images):
        axes[0,col].imshow(a[134:250,134:250],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        axes[0,col].set_title(label,fontsize=9)
        axes[1,col].imshow(a[170:214,170:234],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        axes[1,col].set_title('Identical center crop; no display smoothing',fontsize=8)
        for ax in axes[:,col]: ax.set_axis_off()
    fig.savefig(root/'detector_comparison.png',dpi=160);plt.close(fig)
    for prefix, filename in [('', 'detector_window_only.png'), ('dual_', 'dual_detector_window_only.png')]:
        group = [prefix+'direct_flat_192', prefix+'direct_flat_768']
        if not all((root/n/'fields.npz').exists() for n in group):
            continue
        fields = [('Truth', first['truth'])]
        for name in group:
            record = next(r for r in rows if r['name']==name)
            fields.append((f"Detector {name.rsplit('_',1)[-1]} um; {record['step']} updates\nROI relative error {record['object_error']:.6f}",
                           np.load(root/name/'fields.npz')['reconstruction']))
        fig, axes = plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
        for col,(label,a) in enumerate(fields):
            axes[0,col].imshow(a[134:250,134:250],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            axes[0,col].set_title(label,fontsize=10)
            axes[1,col].imshow(a[170:214,170:234],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            for ax in axes[:,col]: ax.set_axis_off()
        fig.suptitle(('515 + 633 nm' if prefix else '515 nm')+' | known probes, identical direct pixels / optimizer / initialization')
        fig.savefig(root/filename,dpi=160);plt.close(fig)
    print(json.dumps(rows,indent=2))


if __name__ == '__main__': main()
