"""Exploratory common-metric and finite-window dual-field diagnostics.

Reference mPIE is a comparison reconstruction, never ground truth. A reflected
conjugate factorization is tested numerically, not assumed to be an exact
ambiguity for finite arrays or bilinear sampling. No optimizer runs here.
"""
import argparse
import json
from pathlib import Path
import sys
import textwrap

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'real_ptycho'))
from core import Propagator


def dual_fields(obj,probe,center):
    """O'(x)=conj(P(C-x)); P'(u)=conj(O(C-u)), zero beyond stored fields."""
    side,n = obj.shape[-1],probe.shape[-1]
    cy,cx = map(int,center)
    new_o = np.zeros_like(obj)
    src_y,src_x = cy-np.arange(side),cx-np.arange(side)
    good_y,good_x = np.flatnonzero((src_y>=0)&(src_y<n)),np.flatnonzero((src_x>=0)&(src_x<n))
    new_o[np.ix_(good_y,good_x)] = probe[np.ix_(src_y[good_y],src_x[good_x])].conj()
    new_p = np.zeros_like(probe)
    src_y,src_x = cy-np.arange(n),cx-np.arange(n)
    good_y,good_x = np.flatnonzero((src_y>=0)&(src_y<side)),np.flatnonzero((src_x>=0)&(src_x<side))
    new_p[np.ix_(good_y,good_x)] = obj[np.ix_(src_y[good_y],src_x[good_x])].conj()
    return new_o,new_p


@torch.no_grad()
def compare_pair(obj,probe,positions,ids,data,lookup,sampling,dual=False,device='cuda'):
    prop = Propagator(positions,probe.shape[-1],sampling=sampling)
    o,p = torch.from_numpy(obj).to(device),torch.from_numpy(probe).to(device)
    if dual:
        center = np.rint(np.mean(positions,axis=0)).astype(int)+probe.shape[-1]-1
        od,pd = dual_fields(obj,probe,center)
        odt,pdt = torch.from_numpy(od).to(device),torch.from_numpy(pd).to(device)
    se=energy=dual_se=pred_energy=relative_dual=0.
    for start in range(0,len(ids),4):
        part = np.arange(start,min(start+4,len(ids)))
        target = torch.from_numpy(np.array(data[[lookup[int(ids[i])] for i in part]],copy=True)).to(device).sqrt()
        a = prop.amplitude(o,p,part)
        se += float((a-target).square().sum(dtype=torch.float64))
        energy += float(target.square().sum(dtype=torch.float64))
        if dual:
            b = prop.amplitude(odt,pdt,part)
            dual_se += float((b-target).square().sum(dtype=torch.float64))
            relative_dual += float((b-a).square().sum(dtype=torch.float64))
            pred_energy += float(a.square().sum(dtype=torch.float64))
    result = dict(relative_amplitude_rmse=float(np.sqrt(se/energy)),sampling=sampling)
    if dual:
        result.update(dual_relative_amplitude_rmse=float(np.sqrt(dual_se/energy)),
                      dual_vs_original_predicted_amplitude_relative_error=float(np.sqrt(relative_dual/pred_energy)),
                      dual_center_yx=center.tolist(),
                      caveat='Finite-window numerical test; bilinear sampling can break the algebraic identity')
        return result,(od,pd)
    return result,None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--case',nargs=2,action='append',metavar=('LABEL','NPZ'),required=True)
    p.add_argument('--dual-case',action='append',help='Labels to test with the reflected-conjugate transformation; default NN_final')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    data = np.load(args.prepared/'intensity.npy',mmap_mode='r')
    meta = json.loads((args.prepared/'prepared.json').read_text(encoding='utf-8'))
    lookup = {int(k):i for i,k in enumerate(meta['original_frame_ids'])}
    rows=[];images=[]
    for label,path in args.case:
        with np.load(path,allow_pickle=False) as z:
            obj,probe,positions,ids = [z[k].copy() for k in ('obj_rec','probe_rec','positions_px','original_frame_ids')]
        sampling = 'mpie-integer' if np.all(positions==np.round(positions)) else 'bilinear'
        do_dual = label in (args.dual_case or ['NN_final'])
        values,alternative = compare_pair(obj,probe,positions,ids,data,lookup,sampling,dual=do_dual,device=args.device)
        rows.append(dict(label=label,file=str(Path(path).resolve()),**values))
        if do_dual and sampling!='mpie-integer':
            integer_values,_ = compare_pair(obj,probe,np.rint(positions).astype(int),ids,data,lookup,
                                            'mpie-integer',dual=True,device=args.device)
            rows[-1]['rounded_positions_diagnostic'] = integer_values
        if obj.shape[0] == 2454:
            obj = obj[510:510+1440,510:510+1440]
        images.append((label,obj,probe))
        if alternative is not None:
            od,pd = alternative
            safe_label = ''.join(c if c.isalnum() else '_' for c in label)
            np.savez(args.out/f'diagnostic_dual_{safe_label}.npz',obj=od,probe=pd,
                     positions_px=positions,original_frame_ids=ids,
                     note='Diagnostic algebraic field transformation; NOT an optimized or validated reconstruction')
            if od.shape[0] == 2454:
                od = od[510:510+1440,510:510+1440]
            images.append((label+' dual-field diagnostic (not optimized)',od,pd))
    (args.out/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(len(images),2,figsize=(10,3.4*len(images)),squeeze=False,layout='constrained')
    for axes,(label,obj,probe) in zip(axs,images):
        # A shared physical crop, not an automatically chosen attractive ROI.
        oc = obj[420:1020,420:1020]
        for ax,field,title in [(axes[0],oc,'Object, central 600 px'),
                               (axes[1],probe,'Probe, full 1024 px')]:
            a = np.abs(field)
            im = ax.imshow(a,cmap='gray',vmin=0,vmax=np.percentile(a,99.8))
            ax.set(title=textwrap.fill(label,width=43)+'\n'+title,xlabel='x (px)',ylabel='y (px)')
            ax.title.set_fontsize(9)
            fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle('Exploratory reconstruction comparison; independent display scales; no ground truth')
    fig.savefig(args.out/'comparison.png',dpi=140)
    plt.close(fig)
    print(json.dumps(rows,indent=2),flush=True)
    print('OUTPUT',args.out.resolve(),flush=True)


if __name__=='__main__':
    main()
