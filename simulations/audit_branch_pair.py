"""Read-only A/B saved-array inspection, no torch or training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import uniform_filter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from functions.common.metrics import align_global_factor, psnr, ssim
from functions.paperrepro.evaluate import evaluate, probe_relerr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results, maps, probes = [], [], []
    for number in (11, 12):
        path = args.directory/f'net_result ({number}).npz'
        with np.load(path, allow_pickle=False) as z:
            cfg = json.loads(str(z['cfg']))
            roi = tuple(slice(int(z['roi'][i]), int(z['roi'][i+1])) for i in (0, 2))
            gt = z['obj_gt'][roi].astype(complex)
            rec = z['obj_rec'][roi].astype(complex)
            rec, factor = align_global_factor(rec, gt)
            ga, ra = np.abs(gt), np.abs(rec)
            err = ra-ga
            # Reproduce existing local SSIM map exactly, then summarize interior.
            f = lambda x: uniform_filter(x, size=7)
            ux, uy = f(ra), f(ga)
            vx, vy = (f(ra*ra)-ux*ux)*49/48, (f(ga*ga)-uy*uy)*49/48
            cov = (f(ra*ga)-ux*uy)*49/48
            c1, c2 = (.01*np.ptp(ga))**2, (.03*np.ptp(ga))**2
            sm = ((2*ux*uy+c1)*(2*cov+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2))
            interior = np.zeros_like(ga, bool); interior[3:-3, 3:-3] = True
            flat = (vy < 1e-6) & interior
            structured = (~flat) & interior
            subsets = {}
            for name, mask in [('flat_gt_windows', flat), ('structured_gt_windows', structured)]:
                subsets[name] = dict(n=int(mask.sum()), mean_ssim=float(sm[mask].mean()),
                                    amp_rmse=float(np.sqrt(np.mean(err[mask]**2))),
                                    local_rec_std=float(np.sqrt(np.maximum(vx[mask],0)).mean()))
            pr, pg = z['probe_rec'].astype(complex), z['probe_gt'].astype(complex)
            pa, _ = align_global_factor(pr, pg)
            support = np.abs(pg)>0
            energy = np.abs(pr)**2
            # Separate amplitude scale-only diagnostic, not replacement official metric.
            rawamp = np.abs(z['obj_rec'][roi]).astype(float)
            scale = np.sum(rawamp*ga)/np.sum(rawamp**2)
            results.append(dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                branch=cfg['branch'], metrics=evaluate(z['obj_rec'][roi],z['obj_gt'][roi]),
                amplitude_only_alignment_diagnostic=dict(psnr=psnr(rawamp*scale,ga),ssim=ssim(rawamp*scale,ga)),
                wrapped_phase_rmse=float(np.sqrt(np.mean(np.angle(rec*np.conj(gt))**2))),
                subsets=subsets, probe_relerr=probe_relerr(pr,pg),
                probe_outside_gt_support_energy_fraction=float(energy[~support].sum()/energy.sum()),
                probe_inside_separately_aligned_relerr=probe_relerr(pr[support],pg[support])))
            maps.append([ga,ra,np.abs(err),np.angle(gt),np.angle(rec),sm])
            probes.append([np.abs(pg)[196:316,196:316],np.abs(pa)[196:316,196:316],
                           np.where(support,np.angle(pa*np.conj(pg)),np.nan)[196:316,196:316]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    titles=['GT amplitude','Recovered amplitude','Absolute amp error','GT phase','Recovered phase','Local amplitude SSIM']
    limits=[(0,1),(0,1),(0,max(m[2].max() for m in maps)),(-np.pi,np.pi),(-np.pi,np.pi),(-1,1)]
    fig, axes=plt.subplots(2,6,figsize=(17,6),constrained_layout=True)
    for i,fields in enumerate(maps):
        for j,field in enumerate(fields):
            im=axes[i,j].imshow(field,vmin=limits[j][0],vmax=limits[j][1],cmap='twilight' if j in (3,4) else 'viridis')
            axes[i,j].set_title(titles[j]); axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
            if j==0: axes[i,j].set_ylabel(results[i]['branch'])
            if i==1: fig.colorbar(im,ax=axes[:,j],shrink=.5)
    fig.savefig(args.output/'object_comparison.png',dpi=150); plt.close(fig)
    fig, axes=plt.subplots(2,3,figsize=(10,6),constrained_layout=True)
    for i,fields in enumerate(probes):
        for j,field in enumerate(fields):
            im=axes[i,j].imshow(field,vmin=-np.pi if j==2 else 0,vmax=np.pi if j==2 else max(p[0].max() for p in probes),cmap='twilight' if j==2 else 'viridis')
            axes[i,j].set_title(['GT probe amplitude','Aligned probe amplitude','Phase error inside GT support'][j]);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if j==0: axes[i,j].set_ylabel(results[i]['branch'])
            if i==1: fig.colorbar(im,ax=axes[:,j],shrink=.5)
    fig.savefig(args.output/'probe_comparison.png',dpi=150);plt.close(fig)
    with (args.output/'audit.json').open('x',encoding='utf-8') as f: json.dump(results,f,indent=2)
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
