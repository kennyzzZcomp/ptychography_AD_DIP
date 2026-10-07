"""Inspect supplied result only; no reconstruction training."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.signal import correlate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from functions.paperrepro.evaluate import evaluate
from functions.common.metrics import align_global_factor

def match(image,template):
    im=np.asarray(image,dtype=np.float64); t=np.asarray(template,dtype=np.float64)
    h,w=t.shape
    def windows(a):
        s=np.pad(a,((1,0),(1,0))).cumsum(0).cumsum(1)
        return s[h:,w:]-s[:-h,w:]-s[h:,:-w]+s[:-h,:-w]
    t=t-t.mean()
    score=correlate(im,t,mode='valid',method='fft')/np.sqrt(np.maximum(windows(im**2)-windows(im)**2/(h*w),1e-25)*np.sum(t*t))
    y,x=np.unravel_index(score.argmax(),score.shape)
    return int(y),int(x),float(score[y,x])

d=np.load('C:/Users/kennyzz/Downloads/result (2).npz')
o,g,p,q=[d[k] for k in ['obj_rec','obj_gt','probe_rec','probe_gt']]
a,b,c,e=map(int,d['roi']); target=g[a:b,c:e]; h,w=target.shape
y,x,corr=match(abs(o),abs(target)); aligned=o[y:y+h,x:x+w]
support=abs(q)>abs(q).max()*1e-6
rows,cols=np.where(support); ya,yb=rows.min(),rows.max()+1; xa,xb=cols.min(),cols.max()+1
py,px,pcorr=match(abs(p),abs(q[ya:yb,xa:xb]))
shift=(py-int(ya),px-int(xa))
moved=np.roll(p,(-shift[0],-shift[1]),axis=(0,1))
def energy_outside(v): return float(np.sum(abs(v[~support])**2)/np.sum(abs(v)**2))
report=dict(source='result (2).npz',roi=[a,b,c,e],last=json.loads(str(d['hist']))[-1],
    raw_metrics=evaluate(o[a:b,c:e],target),
    best_amplitude_ncc_offset_yx=[y-a,x-c],best_amplitude_ncc=corr,
    best_translation_metrics=evaluate(aligned,target),
    probe_shape_match_offset_yx=shift,probe_shape_ncc=pcorr,
    probe_energy_outside_true_support=energy_outside(p),
    probe_energy_outside_support_after_periodic_shift=energy_outside(moved),
    note='Translation selected using truth amplitude NCC over all valid ROI positions; diagnostic best match, no phase-ramp correction. Probe energy shift uses periodic roll.')
out=Path('results_paper/ptyinr_large_result2_audit');out.mkdir(parents=True,exist_ok=True)
(out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
fig,ax=plt.subplots(2,5,figsize=(15,6),constrained_layout=True)
fields=[target,align_global_factor(o[a:b,c:e],target)[0],align_global_factor(aligned,target)[0],q,p]
for k,(title,v) in enumerate(zip(['Object truth ROI','Original ROI','Best translation ROI','True probe','Recovered probe'],fields)):
    amp=abs(v); amp=amp/(amp.max() if k>=3 else 1)
    ax[0,k].imshow(amp,cmap='gray',vmin=0,vmax=1)
    ax[1,k].imshow(np.angle(v),cmap='twilight',vmin=-np.pi,vmax=np.pi)
    ax[0,k].set_title(title)
fig.savefig(out/'audit.png',dpi=140);plt.close(fig)
print(json.dumps(report,indent=2))
