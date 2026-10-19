"""Collect already-completed diagnostics and make exploratory previews (CPU only)."""
import argparse
import json
from pathlib import Path
import textwrap

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'results/report_20261019')
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    names = ['official_ad100_seed0','official_defaults_ad100_seed0','abcd_mse500_updates_seed0',
             'fixed_reference_probe_ad100_seed0','fixed_reference_probe_positive_ad100_seed0']
    labels = ['PtyRAD blind: P LR 0.005','PtyRAD blind: P LR 0.0001','PtyRAD ABCD+MSE diagnostic',
              'Fixed reference P (domain issue)','Fixed reference P + positive domain']
    rows=[];histories=[]
    for name,label in zip(names,labels):
        folder = HERE/'results'/name
        summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
        history = json.loads((folder/'history.json').read_text(encoding='utf-8'))
        checkpoint = torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        amp = checkpoint['model']['opt_obja']
        rows.append(dict(name=name,label=label,folder=str(folder.resolve()),
                         summary=summary,actual_args=config['args'],
                         signed_object_amplitude_min=float(amp.min()),
                         negative_amplitude_fraction=float((amp<0).float().mean())))
        histories.append(history)
    cross = json.loads((HERE/'results/all_blind_controls_20261019/metrics.json').read_text(encoding='utf-8'))
    (args.out/'summary.json').write_text(json.dumps(dict(local_AD_cases=rows,uploaded_comparators=cross),indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    cases = [('Uploaded NN (structure crosstalk)',Path('C:/Users/kennyzz/Downloads/reconstruction (1).npz')),
             ('PtyRAD blind, same circular initial P',HERE/'results/official_ad100_seed0/reconstruction.npz'),
             ('PtyRAD fixed reference P + domain guard\nDIAGNOSTIC, not blind reconstruction',
              HERE/'results/fixed_reference_probe_positive_ad100_seed0/reconstruction.npz'),
             ('mPIE reference reconstruction',Path('C:/Users/kennyzz/Downloads/reconstruction_mpie_100ITER.npz'))]
    fig,axs = plt.subplots(4,2,figsize=(9,12),layout='constrained')
    for row,(label,path) in zip(axs,cases):
        with np.load(path,allow_pickle=False) as z:
            obj,probe = z['obj_rec'],z['probe_rec']
        if obj.shape[0]==2454:
            obj = obj[510:1950,510:1950]
        for ax,field,title in [(row[0],obj[420:1020,420:1020],'Object: same central 600 px'),
                               (row[1],probe,'Probe: full 1024 px')]:
            amp = np.abs(field)
            im = ax.imshow(amp,cmap='gray',vmin=0,vmax=np.percentile(amp,99.8))
            ax.set_title(label+'\n'+title,fontsize=9)
            ax.set(xlabel='x (px)',ylabel='y (px)')
            fig.colorbar(im,ax=ax,shrink=.72)
    fig.suptitle('Diagnostic overview | independent display scales | no experimental ground truth',fontsize=11)
    fig.savefig(args.out/'diagnosis_overview.png',dpi=160)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(8,5),layout='constrained')
    for label,history in zip(labels,histories):
        ax.plot([h['optimizer_updates'] for h in history],
                [h['relative_amplitude_rmse'] for h in history],marker='.',label=label)
    ax.set(xlabel='Actual optimizer updates (different grouping schedules)',
           ylabel='Common full-data amplitude rRMSE',
           title='Exploratory traces; no network trace inferred from final NPZ')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(args.out/'ad_diagnostic_curves.png',dpi=160)
    plt.close(fig)
    print(json.dumps([dict(label=r['label'],rrmse=r['summary']['final']['relative_amplitude_rmse'],
                           updates=r['summary']['optimizer_updates'],seconds=r['summary']['training_seconds'],
                           negative_fraction=r['negative_amplitude_fraction']) for r in rows],indent=2))
    print('OUTPUT',args.out.resolve())


if __name__=='__main__':
    main()
