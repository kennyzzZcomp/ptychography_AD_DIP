"""Independent H5 adapter for run_paper_progressive's four-inner-window mode.

Reads exported measurements verbatim; never regenerates a scene.
No training on import. --check-only validates data and grouping without training.
"""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulations import run_paper_progressive as settings
from simulations.run_paper_windows import WindowCfg
from functions.paperrepro import window_training as solver
from functions.paperrepro.optics import forward_field


def inner_groups(grid):
    if grid < 4 or grid % 2:
        raise ValueError('Four inner groups require an even raster grid >=4')
    a,b=range(1,grid-1,2),range(2,grid-1,2)
    return [[r*grid+c for r in rows for c in cols]
            for rows,cols in ((a,a),(a,b),(b,a),(b,b))]


def load_scene(path, device):
    with h5py.File(path,'r') as f:
        if f.attrs.get('schema')!='proptynet-ptyinr-v1':
            raise ValueError('Expected exported proptynet-ptyinr-v1 scene')
        config=json.loads(f.attrs['scene_config_json'])
        amp=f['diffamp'][()].astype(np.float32)
        intensity=f['intensity'][()].astype(np.float32)
        pos=f['positions_px'][()].astype(np.int64)
        obj=f['diagnostic_truth/object'][()]
        probe=f['diagnostic_truth/probe_normalized'][()].astype(np.complex64)
        clean=f['diagnostic_truth/clean_intensity'][()].astype(np.float32)
        q=f['quadratic_phase'][()]
        p0=f['probe_init'][()].astype(np.complex64)
        roi=list(map(int,f['roi'][()])); fp=str(f.attrs['scene_fingerprint'])
    if not all(np.isfinite(x).all() for x in (amp,intensity,obj,probe,q,p0,clean)):
        raise ValueError('Nonfinite scene values')
    if np.any(amp<0) or not np.allclose(amp**2,intensity,rtol=2e-5,atol=1e-7):
        raise ValueError('Inconsistent measured amplitudes/intensities')
    g=int(config['grid']); n=int(config['N']); m=int(config['obj_size'])
    if amp.shape!=(g*g,n,n) or obj.shape!=(m,m) or probe.shape!=(n,n) or q.shape!=(n,n) or p0.shape!=(n,n):
        raise ValueError('Scene dimensions disagree with configuration')
    if not np.allclose(p0,1):
        raise ValueError('This adapter preserves progressive ones initialization; scene probe_init must be ones')
    ys=np.unique(pos[:,0]); xs=np.unique(pos[:,1])
    expected=np.array([(y,x) for y in ys for x in xs])
    if len(ys)!=g or len(xs)!=g or not np.array_equal(pos,expected):
        raise ValueError('Expected row-major square raster positions')
    if not (np.all(np.diff(ys)==config['step_px']) and np.all(np.diff(xs)==config['step_px'])):
        raise ValueError('Scan step mismatch')
    if pos.min()<0 or pos.max()+n>m:
        raise ValueError('Object windows exceed canvas')
    a,b,c,d=roi
    if not (0<=a<b<=m and 0<=c<d<=m): raise ValueError('Invalid ROI')
    tensor=lambda x:torch.from_numpy(x).to(device)
    sc=SimpleNamespace(obj=obj,probe=probe,pos=pos,P0=p0,Q=tensor(q),post=tensor(pos),
        Im=intensity,Icl=clean,Imt=tensor(intensity),Iclt=tensor(clean),sqrtIm=tensor(amp),
        roi=(slice(a,b),slice(c,d)),fp=fp)
    # Check every frame in small batches, without retaining a training graph.
    error=den=0.
    with torch.no_grad():
        ot,pt=tensor(obj),tensor(probe)
        for start in range(0,len(pos),16):
            pred=forward_field(ot,pt,sc.post[start:start+16],sc.Q,n).abs()
            truth=sc.Iclt[start:start+16].clamp_min(0).sqrt()
            error+=(pred-truth).square().sum().item(); den+=truth.square().sum().item()
    relative=(error/max(den,1e-30))**.5
    if relative>1e-5: raise ValueError(f'Truth forward mismatch: {relative}')
    return sc,config,relative


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    p.add_argument('--iters',type=int,default=settings.TOTAL_ITERS)
    p.add_argument('--switch-after',type=int,default=settings.SWITCH_AFTER)
    p.add_argument('--eval-every',type=int,default=settings.EVAL_EVERY)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.outdir.exists(): p.error('Choose a new output directory')
    if settings.STAGE1_MODE!='windows' or settings.WINDOW_LAYOUT!='inner' or settings.WINDOW_SIDE!=4:
        p.error('This adapter targets the supplied windows/inner4 progressive configuration')
    if settings.HALF_RES_STAGE1 or settings.RESET_TGV_AT_SWITCH or settings.STAGE2_INPUT!='object' or settings.STAGE2_NETWORK!='fresh':
        p.error('Unsupported changes to progressive settings')
    if not 0<a.switch_after<a.iters or a.switch_after%4: p.error('Require 0 < switch < iters and complete four-window sweeps')
    sc,source,relative=load_scene(a.scene,torch.device(a.device))
    groups=inner_groups(source['grid'])
    # Start from source geometry, but use launcher training choices explicitly.
    valid={f.name for f in fields(WindowCfg) if f.init}
    opts={k:v for k,v in source.items() if k in valid}
    opts.update(iters=a.iters,eval_every=a.eval_every,eval_size=0,device=a.device,
        outdir=str(a.outdir),seed=settings.SEED,network_seed=settings.SEED,base_ch=settings.BASE_CH,
        probe_mode='pixel',probe_init='ones',network_type='real',
        lr_net=settings.STAGE1_LR_NET,lr_probe=settings.STAGE1_LR_PROBE,tgv_amp=settings.STAGE1_TGV,
        window_layout='inner',window_side=4,window_update=settings.WINDOW_UPDATE,
        window_consistency=settings.WINDOW_CONSISTENCY,window_switch_after=a.switch_after,
        window_stage2_lr_net=settings.STAGE2_LR_NET,window_stage2_lr_probe=settings.STAGE2_LR_PROBE,
        window_stage2_tgv=settings.STAGE2_TGV,window_stage2_readout=settings.STAGE2_READOUT)
    cfg=WindowCfg(**opts)
    report=dict(scene=str(a.scene.resolve()),scene_sha256=hashlib.sha256(a.scene.read_bytes()).hexdigest(),
        scene_fingerprint=sc.fp,relative_clean_amplitude_l2=relative,
        patterns=list(sc.Im.shape),object_shape=list(sc.obj.shape),
        groups=groups,stage1_channels=len(groups[0]),stage1_unique_patterns=len(set(sum(groups,[]))),
        stage2_patterns=len(sc.pos),roi=[sc.roi[0].start,sc.roi[0].stop,sc.roi[1].start,sc.roi[1].stop],
        grouping='Inner row/column indices 1..grid-2 split by parity; exact original inner4 when grid=10',
        evaluation='Original shared ROI and evaluator; no translation registration added',
        truth_usage='Forward validation, evaluation and output only; pixel probe starts at ones',
        training='Original window_training solver; original per-frame CNN input normalization and analytic loss scale retained',
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [Path(__file__),Path(settings.__file__),Path(solver.__file__)]})
    print(json.dumps({k:v for k,v in report.items() if k!='groups'},indent=2),flush=True)
    if a.check_only:
        a.outdir.mkdir(parents=True)
        (a.outdir/'scene_adapter.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        return
    original_scene,original_groups=solver.build_scene,solver.measurement_groups
    def imported_scene(config,device):
        (a.outdir/'scene_adapter.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        return sc
    # Scoped adapters reuse the solver without changing either original file.
    try:
        solver.build_scene=imported_scene
        solver.measurement_groups=lambda grid,layout,side:groups
        solver.run_windows(cfg)
    finally:
        solver.build_scene,solver.measurement_groups=original_scene,original_groups


if __name__=='__main__': main()
