"""Read-only inspection of user-provided aligned result and exported measurements."""
import json
from pathlib import Path
import sys
import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from functions.common.metrics import align_global_factor
from functions.paperrepro.optics import forward_field

root=Path(__file__).resolve().parents[1]
out=root/'results_paper/ptyinr_result_audit'
out.mkdir(parents=True,exist_ok=True)
with np.load('C:/Users/kennyzz/Downloads/result.npz',allow_pickle=False) as d:
    obj,gt,probe,pgt=[d[k].copy() for k in ('obj_rec','obj_gt','probe_rec','probe_gt')]
    a,b,c,e=map(int,d['roi']);hist=json.loads(str(d['hist'].item()))
roi=(slice(a,b),slice(c,e));rec=align_global_factor(obj[roi],gt[roi])[0]
palign=align_global_factor(probe,pgt)[0]
support=np.abs(pgt)>np.abs(pgt).max()*1e-6
def stats(f):
    amp=np.abs(f)
    return dict(amp_mean=float(amp.mean()),amp_std=float(amp.std()),
                amp_cv=float(amp.std()/max(amp.mean(),1e-30)),
                distance_to_constant=float(np.linalg.norm(f-f.mean())/np.linalg.norm(f)))
report=dict(last=hist[-1],object=stats(obj[roi]),truth=stats(gt[roi]),
    probe_energy_outside_true_support=float(np.sum(np.abs(probe[~support])**2)/np.sum(np.abs(probe)**2)),
    probe_max=float(np.abs(probe).max()),true_normalized_probe_max=float(np.abs(pgt).max()),
    object_amplitude_correlation=float(np.corrcoef(np.abs(rec).ravel(),np.abs(gt[roi]).ravel())[0,1]))
with h5py.File(root/'results_paper/ptyinr_alignment/scene.h5','r') as f:
    report['truth_arrays_equal_export']=bool(np.array_equal(gt,f['diagnostic_truth/object'][()]) and np.array_equal(pgt,f['diagnostic_truth/probe_normalized'][()]))
    if report['truth_arrays_equal_export']:
        dev='cuda'
        pos=torch.tensor(f['positions_px'][()],device=dev)
        q=torch.tensor(f['quadratic_phase'][()],device=dev)
        target=torch.tensor(f['diffamp'][()],device=dev)
        with torch.no_grad():
            for label,o,p in [('reconstruction',obj,probe),('truth',gt,pgt)]:
                ot=torch.tensor(o,device=dev);pt=torch.tensor(p,device=dev)
                err=0.
                for s in range(0,len(pos),4):
                    pred=forward_field(ot,pt,pos[s:s+4],q,512).abs()
                    err+=(pred-target[s:s+4]).square().sum().item()
                report[label+'_relative_amplitude_mse']=err/target.square().sum().item()
fig,axs=plt.subplots(2,4,figsize=(13,6),constrained_layout=True)
fields=[gt[roi],rec,pgt,palign]
for j,(title,field) in enumerate(zip(['Object truth (ROI)','Object reconstruction','Probe truth','Probe reconstruction'],fields)):
    limit=np.abs(gt[roi]).max() if j<2 else np.abs(pgt).max()
    axs[0,j].imshow(np.abs(field),cmap='gray',vmin=0,vmax=limit)
    phase=np.angle(field)
    if j<2:phase=phase-phase.mean()+np.angle(gt[roi]).mean()
    axs[1,j].imshow(phase,cmap='twilight',vmin=-np.pi,vmax=np.pi)
    axs[0,j].set_title(title)
    for ax in axs[:,j]:ax.axis('off')
fig.savefig(out/'reconstruction_audit.png',dpi=140);plt.close(fig)
(out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
