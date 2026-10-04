"""Offline endpoint audit of saved window snapshots; never trains a model.

Regenerates noiseless data from saved truth using the production forward model.
Geometry masks derived from true probe are diagnostic only, not training inputs.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from functions.addip.losses import cabs
from functions.paperrepro.evaluate import evaluate, probe_relerr
from functions.paperrepro.optics import forward_field, make_quad_phase
from functions.paperrepro.window_training import (
    measurement_groups, footprint_masks, fuse_fields, consistency_loss,
)


def scale_audit(pred, target, groups):
    """Double-precision sufficient statistics, with one common O/P per call."""
    a, b = pred.astype(np.float64), target.astype(np.float64)
    axes = (1, 2)
    e = np.sum(a*a, axis=axes)
    h = np.sum(a*b, axis=axes)
    energy = np.sum(b*b, axis=axes)
    selected = np.unique(np.concatenate(groups))
    remaining = np.setdiff1d(np.arange(len(a)), selected)
    ek = np.array([e[g].sum() for g in groups])
    hk = np.array([h[g].sum() for g in groups])
    qk = hk / ek
    qsel, qall = h[selected].sum()/e[selected].sum(), h.sum()/e.sum()
    gap = float(np.sum(ek*(qk-qsel)**2))
    # Compute residuals directly to avoid cancellation in E*q^2-2H*q+B.
    grouped_sse = sum(float(np.sum((q*a[g]-b[g])**2)) for q, g in zip(qk, groups))
    common_sse = float(np.sum((qsel*a[selected]-b[selected])**2))
    def loss(ids):
        if len(ids) == 0:
            return None
        residual = qall*a[ids]-b[ids]
        return dict(mse=float(np.mean(residual**2)),
                    relative_amplitude_error=float(np.sqrt(np.sum(residual**2)/energy[ids].sum())))
    return dict(q_groups=qk.tolist(), q_selected=float(qsel), q_all=float(qall),
                relative_q_groups=(qk/qsel).tolist(),
                q_cv=float(qk.std()/qk.mean()),
                q_energy_weighted_cv=float(np.sqrt(gap/ek.sum())/qsel),
                profiled_loss_gap_mse=gap/(len(selected)*a.shape[1]*a.shape[2]),
                gap_fraction_of_common_selected_sse=gap/common_sse if common_sse else 0.,
                identity_absolute_error=abs(common_sse-grouped_sse-gap),
                all=loss(np.arange(len(a))), selected=loss(selected), remaining=loss(remaining))


def regional_errors(rec, gt, regions, alignment_mask):
    """ONE complex scalar per endpoint, shared by every reported region."""
    r, g = rec.astype(np.complex128), gt.astype(np.complex128)
    factor = np.vdot(r[alignment_mask], g[alignment_mask])/np.vdot(r[alignment_mask], r[alignment_mask])
    aligned = factor*r
    dynamic_range = np.ptp(np.abs(g[alignment_mask]))
    result = dict(alignment_factor=[float(factor.real), float(factor.imag)], regions={})
    for name, mask in regions.items():
        rr, gg = aligned[mask], g[mask]
        mse = float(np.mean((np.abs(rr)-np.abs(gg))**2))
        result['regions'][name] = dict(pixels=int(mask.sum()), amplitude_mse=mse,
            amplitude_psnr=float(10*np.log10(dynamic_range**2/mse)) if mse else float('inf'),
            complex_relerr=float(np.linalg.norm(rr-gg)/np.linalg.norm(gg)))
    return result


def pair_disagreement(fields, region, group_coverage):
    rows = []
    for i in range(4):
        for j in range(i+1, 4):
            mask = region & (group_coverage[i]>0) & (group_coverage[j]>0)
            l, r = fields[i][mask].astype(np.complex128), fields[j][mask].astype(np.complex128)
            rows.append(dict(pair=[i, j], pixels=int(mask.sum()), normalized_squared_disagreement=
                float(np.sum(np.abs(l-r)**2)/(.5*np.sum(np.abs(l)**2+np.abs(r)**2)))))
    return dict(pairs=rows, mean=float(np.mean([r['normalized_squared_disagreement'] for r in rows])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--conditioning', type=Path, required=True)
    ap.add_argument('--result', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    with np.load(args.conditioning, allow_pickle=False) as f:
        snap = {k: f[k] for k in f.files}
    with np.load(args.result, allow_pickle=False) as f:
        final = {k: f[k] for k in f.files}
    cfg = json.loads(str(final['cfg'].item()))
    hist = json.loads(str(final['hist'].item()))
    if cfg['noise'] != 'none':
        raise ValueError('This audit reconstructs noiseless targets only.')
    if cfg['window_layout'] == 'alternating':
        raise ValueError('Alternating uses different training/readout groups; not supported here.')
    groups = measurement_groups(cfg['grid'], cfg['window_layout'], cfg['window_side'])
    pos = final['positions']
    n, size = cfg['N'], cfg['obj_size']
    masks = footprint_masks(pos, groups, size, n, 'cpu')
    fields = torch.from_numpy(snap['fields'])
    fused = fuse_fields(fields, masks).numpy()
    y0,y1,x0,x1 = map(int, final['roi'])
    roi = np.zeros((size,size), bool)
    roi[y0:y1,x0:x1] = True
    gt, pgt = final['obj_gt'], final['probe_gt']
    coverage = np.zeros((len(pos), size, size), np.float32)
    for i,(y,x) in enumerate(pos):
        coverage[i,y:y+n,x:x+n] = np.abs(pgt)**2
    total = coverage.sum(0)
    group_cov = np.stack([coverage[g].sum(0) for g in groups])
    support = total>0
    regions = dict(center=roi & support, periphery=support & ~roi, all_illuminated=support,
                   periphery_above_10pct=(total>=.1*total.max()) & ~roi)
    del coverage
    params = SimpleNamespace(**cfg)
    params.dx1 = cfg['wlength']*cfg['z']/(n*cfg['det_pixel'])
    params.quad_sign = cfg.get('quad_sign', -1.)
    quad = make_quad_phase(params, 'cpu')
    positions = torch.from_numpy(pos.astype(np.int64))
    def amplitude(o,p,safe=True):
        with torch.no_grad():
            u = forward_field(torch.from_numpy(o),torch.from_numpy(p),positions,quad,n,chunk=8)
            return (cabs(u) if safe else torch.sqrt(u.real**2+u.imag**2)).numpy()
    clean = amplitude(gt,pgt,safe=False)**2
    target = np.sqrt(np.clip(clean/clean.max(),0,1)).astype(np.float32)
    del clean
    output = dict(inputs=dict(conditioning=str(args.conditioning.resolve()), result=str(args.result.resolve())),
        config=cfg, assumptions=dict(noise='none', quad_sign=params.quad_sign,
        alignment='one scalar fit on all illuminated pixels per endpoint; fixed across regions',
        peripheral_support='true-probe nonzero union minus fixed saved ROI',
        pairing='snapshot has no run ID; verify by metric matches below'),
        checks=dict(fusion_max_abs_difference=float(np.max(np.abs(fused-snap['obj']))),
                    zero_jump_exact=bool(np.array_equal(snap['obj'],snap['stage2_initial_obj'])),
                    stage1_computational_disagreement=float(consistency_loss(fields,masks))),
        groups=groups, endpoints={},
        disagreement={name:pair_disagreement(snap['fields'],mask,group_cov) for name,mask in regions.items()})
    stage1_rows = [r for r in hist if r['stage']==1]
    for name,obj,probe,record in [('stage1',snap['obj'],snap['probe'],stage1_rows[-1]),
                                ('stage2',final['obj_rec'],final['probe_rec'],hist[-1])]:
        print('Auditing',name,flush=True)
        pred = amplitude(obj,probe)
        output['endpoints'][name] = dict(
            recorded=record,
            reproduced_roi=evaluate(obj[y0:y1,x0:x1],gt[y0:y1,x0:x1]),
            reproduced_probe_relerr=probe_relerr(probe,pgt),
            regional=regional_errors(obj,gt,regions,support),
            regional_center_aligned_sensitivity=regional_errors(obj,gt,regions,roi & support),
            scales=scale_audit(pred,target,groups))
        del pred
    output['stage1_recorded_disagreement'] = [dict(iter=r['it'],disagreement=r['post_update_disagreement'])
        for r in stage1_rows]
    args.outdir.mkdir(parents=True,exist_ok=True)
    (args.outdir/'audit.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    with (args.outdir/'history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in hist for k in r)))
        writer.writeheader()
        writer.writerows(hist)
    print(json.dumps(output,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
