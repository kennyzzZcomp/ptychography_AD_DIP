"""CPU-only exploratory views of saved controls; no experimental ground truth."""
import argparse
import json
from pathlib import Path
import textwrap

import numpy as np

HERE = Path(__file__).resolve().parent


def load_object(path):
    with np.load(path,allow_pickle=False) as z:
        obj = z['obj_rec'].copy()
    if obj.shape==(2454,2454):
        obj = obj[510:1950,510:1950]
    if obj.shape!=(1440,1440):
        raise ValueError(f'Unexpected field/canvas {obj.shape}')
    return obj


def field_stats(obj,reference):
    a,b = np.abs(obj[420:1020,420:1020]),np.abs(reference[420:1020,420:1020])
    correlation = float(np.corrcoef(a.ravel(),b.ravel())[0,1]) if a.std()>1e-12 else None
    amp = np.abs(obj)
    return dict(raw_object_amp_max=float(amp.max()),
                center128_energy_fraction=float(np.sum(amp[656:784,656:784]**2)/np.sum(amp**2)),
                reference_amplitude_pearson_unregistered_central600=correlation,
                caveat='Reference agreement only, NOT ground-truth accuracy; no shift/deformation alignment fitted')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--network',type=Path,default=HERE/'results/nn_fixed_reference_probe_500_200_seed0')
    p.add_argument('--lr-screen',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    reference = load_object(Path('C:/Users/kennyzz/Downloads/reconstruction_mpie_100ITER.npz'))
    cases = [('Uploaded NN, joint O/P\nold bilinear operator',Path('C:/Users/kennyzz/Downloads/reconstruction (1).npz'))]
    if (args.network/'stage1.npz').is_file():
        cases.append(('Original U16, FIXED reference P\nstage 1: 500 updates',args.network/'stage1.npz'))
    else:
        latest = sorted((args.network/'snapshots').glob('stage1_update*.npz'))[-1]
        cases.append(('U16 fixed-P progress, not final\n'+latest.stem,latest))
    if (args.network/'reconstruction.npz').is_file():
        cases.append(('Original U16 -> U32, FIXED reference P\n500 + 200 updates',args.network/'reconstruction.npz'))
    else:
        stage2 = sorted((args.network/'snapshots').glob('stage2_update*.npz'))
        if stage2:
            cases.append(('U32 fixed-P progress, not final\n'+stage2[-1].stem,stage2[-1]))
    cases.append(('Successful mPIE reference\nNOT experimental ground truth',Path('C:/Users/kennyzz/Downloads/reconstruction_mpie_100ITER.npz')))
    rows=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(1,len(cases),figsize=(4*len(cases),4.4),layout='constrained',squeeze=False)
    for ax,(label,path) in zip(axs[0],cases):
        obj = load_object(path)
        rows.append(dict(label=label,file=str(path.resolve()),**field_stats(obj,reference)))
        amp = np.abs(obj[420:1020,420:1020])
        im = ax.imshow(amp,cmap='gray',vmin=0,vmax=max(float(np.percentile(amp,99.8)),1e-12))
        ax.set(title=label,xlabel='x (px)',ylabel='y (px)')
        ax.title.set_fontsize(9)
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('Exploratory diagnostic | same central 600 px | independent color scales | no ground truth',fontsize=11)
    fig.savefig(args.out/'network_objects.png',dpi=140)
    plt.close(fig)
    history = json.loads((args.network/'history.json').read_text(encoding='utf-8'))
    fig,ax = plt.subplots(figsize=(7,4),layout='constrained')
    for stage in (1,2):
        hs = [h for h in history if h['stage']==stage]
        if hs:
            ax.plot([h['total_update'] for h in hs],[h['relative_amplitude_rmse'] for h in hs],
                    marker='o',markersize=3,label=f'Stage {stage}, fixed reference P')
    ax.set(xlabel='Network Adam updates (iteration)',ylabel='Common full-data amplitude rRMSE',
           title='Original network, fixed reference P; not a blind baseline')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(args.out/'network_fit_curve.png',dpi=150);plt.close(fig)
    result = dict(network_fields=rows,network_history=history)
    if args.lr_screen is not None:
        screen = json.loads((args.lr_screen/'summary.json').read_text(encoding='utf-8'))
        result['lr_screen'] = screen
        fig,axs = plt.subplots(4,2,figsize=(9,12),layout='constrained')
        full_objects = []
        for axes,row in zip(axs[:3],screen):
            folder = args.lr_screen/row['name']
            with np.load(folder/'reconstruction.npz',allow_pickle=False) as z:
                obj,probe = z['obj_rec'].copy(),z['probe_rec'].copy()
            cfg = json.loads((folder/'config.json').read_text(encoding='utf-8'))['args']
            label=f"LR O/P {cfg['lr_object']:g}/{cfg['lr_probe']:g}; 50 sweeps"
            full_objects.append((label,obj))
            result.setdefault('lr_field_stats',[]).append(dict(name=row['name'],**field_stats(obj,reference)))
            for ax,field,title in ((axes[0],obj[420:1020,420:1020],'Object, central 600 px'),
                                   (axes[1],probe,'Probe, full 1024 px')):
                a = np.abs(field)
                im = ax.imshow(a,cmap='gray',vmin=0,vmax=max(float(np.percentile(a,99.8)),1e-12))
                ax.set_title(label+'\n'+title,fontsize=9)
                ax.set(xlabel='x (px)',ylabel='y (px)')
                fig.colorbar(im,ax=ax,shrink=.7)
        with np.load('C:/Users/kennyzz/Downloads/reconstruction_mpie_100ITER.npz',allow_pickle=False) as z:
            refp = z['probe_rec'].copy()
        for ax,field,title in ((axs[3,0],reference[420:1020,420:1020],'Object'),(axs[3,1],refp,'Probe')):
            a=np.abs(field)
            im=ax.imshow(a,cmap='gray',vmin=0,vmax=np.percentile(a,99.8))
            ax.set_title('mPIE reference (not ground truth)\n'+title,fontsize=9)
            ax.set(xlabel='x (px)',ylabel='y (px)');fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle('Exploratory blind AD LR sensitivity | same domain guard and budget | independent color scales',fontsize=10)
        fig.savefig(args.out/'lr_objects_probes.png',dpi=145);plt.close(fig)
        # Free O/P can share a translation gauge. Show whole fields as well,
        # so a fixed central crop cannot conceal a displaced reconstruction.
        full_objects.append(('mPIE reference, mapped to same canvas',reference))
        fig,axs = plt.subplots(1,4,figsize=(16,4.5),layout='constrained')
        for ax,(label,field) in zip(axs,full_objects):
            a=np.abs(field)
            im=ax.imshow(a,cmap='gray',vmin=0,vmax=max(float(np.percentile(a,99.8)),1e-12))
            ax.set_title(textwrap.fill(label,width=35)+'\nFull 1440 px canvas',fontsize=9)
            ax.set(xlabel='x (px)',ylabel='y (px)');fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle('Whole-object diagnostic | no translation registration | independent color scales',fontsize=11)
        fig.savefig(args.out/'lr_full_objects.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for row in screen:
            hist=json.loads((args.lr_screen/row['name']/'history.json').read_text(encoding='utf-8'))
            ax.plot([h['optimizer_updates'] for h in hist],[h['relative_amplitude_rmse'] for h in hist],marker='.',label=row['name'])
        ax.set(xlabel='Actual Adam updates (iteration)',ylabel='Common full-data amplitude rRMSE',
               title='PtyRAD blind sensitivity: both LRs co-scaled; positive-domain guard')
        ax.grid(alpha=.2);ax.legend()
        fig.savefig(args.out/'lr_fit_curves.png',dpi=150);plt.close(fig)
    (args.out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))
    print('OUTPUT',args.out.resolve())


if __name__=='__main__':
    main()
