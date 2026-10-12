"""Exploratory compact residual priors + output EMA; no AD regularizer sweep.

Oracle and actual-stage1 starts are separate experiments. Only the latter is
evidence about useful refinement. No GT-based stopping or candidate pruning.
"""
import argparse
import copy
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys
import time
import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulations.run_two_stage_ablation import (
    WindowCfg,load_scene,ProPtyUNet,Stage2SkipAblation,cabs,_fwd,
    initialize_residual_heads,reconstruction_channels,snapshot_readout,
    ObjectAmplitudeTGV,evaluate,probe_relerr,poisson_profile_loss,save_json,sync)

CANDIDATES=['pixel','unet24','unet8_no_concat','dd16','dd32','cond_decoder16','cnn8']
EXTRA_CANDIDATES=['cnn8_polar','cnn16_polar']

class CompactResidual(nn.Module):
    """Small residual decoder, inspired by (not an exact replica of) Deep Decoder."""
    def __init__(self,kind,size):
        super().__init__()
        self.kind=kind;self.size=size
        if kind.startswith('cnn'):
            width=16 if kind=='cnn16_polar' else 8
            self.body=nn.Sequential(nn.Conv2d(3,width,3,padding=1),nn.LeakyReLU(.1),
                                    nn.Conv2d(width,width,3,padding=1),nn.LeakyReLU(.1),
                                    nn.Conv2d(width,width,3,padding=1),nn.LeakyReLU(.1))
        else:
            width=32 if kind=='dd32' else 16
            incoming=11 if kind=='cond_decoder16' else width
            kernel=3 if kind=='cond_decoder16' else 1
            self.blocks=nn.ModuleList()
            for _ in range(3):
                self.blocks.append(nn.Sequential(nn.Conv2d(incoming,width,kernel,padding=kernel//2),
                                  nn.ReLU(),nn.BatchNorm2d(width)))
                incoming=width
        self.head=nn.Conv2d(width,2,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,x):
        if self.kind.startswith('cnn'):x=self.body(x)
        else:
            for block,divisor in zip(self.blocks,(4,2,1)):
                x=F.interpolate(x,size=(self.size//divisor,self.size//divisor),mode='bilinear',align_corners=False)
                x=block(x)
        return self.head(x)[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--initialization',choices=('oracle','stage1'),required=True)
    p.add_argument('--candidates',nargs='+',choices=CANDIDATES+EXTRA_CANDIDATES,default=CANDIDATES)
    p.add_argument('--updates',type=int,default=500)
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    if a.outdir.exists():p.error('Existing output is never overwritten')
    if a.updates<=0 or a.eval_every<=0:p.error('Positive budgets required')
    with np.load(a.reference,allow_pickle=False) as z:
        source=json.loads(z['cfg'].item());gt=z['obj_gt'].copy();pos=z['positions'].copy()
    cfg=WindowCfg(**{k:v for k,v in source.items() if k in {f.name for f in fields(WindowCfg) if f.init}})
    cfg.device='cuda'
    device=torch.device('cuda')
    sc,scene_cfg,fwd_error=load_scene(a.scene,device)
    assert np.array_equal(gt,sc.obj) and np.array_equal(pos,sc.pos)
    for k in ('N','grid','step_px','obj_size','wlength','det_pixel','z'):assert source[k]==scene_cfg[k],k
    cfg.quad_sign=scene_cfg.get('quad_sign',-1.)
    for k in ('noise','snr_db','noise_seed'):setattr(cfg,k,scene_cfg[k])
    assert cfg.grid==10 and len(sc.pos)==100 and cfg.window_stage2_tgv==.005
    with h5py.File(a.scene,'r') as f:
        counts=torch.as_tensor(f['measurement_noise/counts'][()],dtype=torch.float32,device=device)
        alpha=float(f['measurement_noise'].attrs['global_alpha'])
    assert torch.allclose(counts/alpha,sc.Imt,rtol=2e-6,atol=1e-7)
    digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    # Verify the stage-1 checkpoint belongs to this exact noisy scene.
    previous=json.loads((a.checkpoint.parent/'summary.json').read_text(encoding='utf-8'))
    assert previous['scene_sha256']==digest(a.scene)
    assert previous['reference_sha256']==digest(a.reference)
    state=torch.load(a.checkpoint,map_location=device,weights_only=False)
    assert state['history'][-1]['it']==500
    if a.initialization=='oracle':
        base=torch.from_numpy(sc.obj.copy()).to(device)
        pbase=torch.from_numpy(sc.probe.copy()).to(device)
    else:base=state['obj'].detach().clone();pbase=state['probe'].detach().clone()
    pad=(cfg.net_size-cfg.obj_size)//2
    crop=slice(pad,pad+cfg.obj_size)
    conditioning,_=reconstruction_channels(base,None,cfg.net_size,include_probe=False)
    energy=sc.sqrtIm.square().mean().detach()
    betas=(.95,.99)
    a.outdir.mkdir(parents=True)
    manifest=dict(scene=str(a.scene.resolve()),reference=str(a.reference.resolve()),
        checkpoint=str(a.checkpoint.resolve()),scene_sha256=digest(a.scene),reference_sha256=digest(a.reference),
        checkpoint_sha256=digest(a.checkpoint),script_sha256=digest(Path(__file__)),
        cfg=asdict(cfg),scene_cfg=scene_cfg,initialization=a.initialization,
        updates=a.updates,eval_every=a.eval_every,seed=a.seed,candidates=a.candidates,
        ema_betas=betas,ema='Complex output/probe EMA of post-update fields, initialized at FIRST post-update field, never GT/initial output; no feedback to loss',
        timing='Synchronized updates INCLUDING extra output readout + both EMA updates; excludes scoring/save/setup',
        optimizer='Fresh object/network Adam; stage1 probe Adam cloned in stage1 mode, fresh probe Adam in oracle mode',
        probe='Joint optimization in every arm, same lr .01; no support added',
        lr=cfg.window_stage2_lr_net,tgv=cfg.window_stage2_tgv,
        loss='Existing profiled-gain Poisson KL/(2 alpha), energy-scaled amplitude TGV; no phase prior',
        parameterization='Candidates ending _polar use O1 * exp(delta_log_amplitude + i delta_phase); others use additive complex residual or direct pixel AD',
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,forward_relative_error=fwd_error,
        caveats=['All candidates predeclared; exploratory single data/seed, NOT a validated winner.',
                 'Oracle-mode networks can use truth in their input/base; drift preservation is not denoising proof.',
                 'Stage1 mode uses predictions only; no truth in input/loss/stopping.',
                 'Fixed 500 updates; parameter counts, timings, raw and EMA metrics all reported.',
                 'No AD TGV sweep; AD receives the same loss, fixed regularizer and EMA readouts.'])
    save_json(a.outdir/'manifest.json',manifest)
    torch.backends.cudnn.benchmark=True

    def score(obj,probe):
        rec=obj.detach().cpu().numpy();prb=probe.detach().cpu().numpy()
        metrics=evaluate(rec[sc.roi],sc.obj[sc.roi])
        metrics['probe_relerr']=probe_relerr(prb,sc.probe)
        with torch.no_grad():
            field=_fwd(cfg,obj,probe,sc.post,sc.Q)
            metrics['poisson_data_loss']=float(poisson_profile_loss(field.abs().square(),counts,alpha))
            metrics['change_from_initial_roi']=float(torch.linalg.vector_norm((obj-base)[sc.roi])/torch.linalg.vector_norm(base[sc.roi]))
        return metrics

    def run(kind):
        pr=nn.Parameter(pbase.real.clone());pi=nn.Parameter(pbase.imag.clone())
        optp=torch.optim.Adam([pr,pi],lr=cfg.window_stage2_lr_probe)
        if a.initialization=='stage1':
            optp.load_state_dict(copy.deepcopy(state['probe_optimizer']))
            for g in optp.param_groups:g['lr']=cfg.window_stage2_lr_probe
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(a.seed)
            if kind=='pixel':
                net=None;rr=nn.Parameter(base.real.clone());ri=nn.Parameter(base.imag.clone());params=[rr,ri]
            elif kind.startswith('unet'):
                net=ProPtyUNet(3,24,n_fields=1,ph_ch=1) if kind=='unet24' else Stage2SkipAblation(8,'all')
                initialize_residual_heads(net)
            else:net=CompactResidual(kind,cfg.net_size)
        if net is not None:net=net.to(device);params=list(net.parameters())
        inp=conditioning
        if kind in ('dd16','dd32','cond_decoder16'):
            width=32 if kind=='dd32' else (8 if kind=='cond_decoder16' else 16)
            rng=torch.Generator().manual_seed(a.seed+100000)
            inp=torch.rand((1,width,cfg.net_size//8,cfg.net_size//8),generator=rng).to(device)*.1
            if kind=='cond_decoder16':
                inp=torch.cat((F.interpolate(conditioning,size=inp.shape[-2:],mode='area'),inp),dim=1)
        opt=torch.optim.Adam(params,lr=cfg.window_stage2_lr_net,weight_decay=cfg.weight_decay)
        tgv=ObjectAmplitudeTGV(cfg,sc.pos,device)

        def decode():
            if net is None:return torch.complex(rr,ri)
            if kind.startswith('unet'):
                real,imag=net(inp);real,imag=real[0],imag[0]
            else:
                fields=net(inp);real,imag=fields[0],fields[1]
            if kind.endswith('_polar'):
                return base*torch.exp(torch.complex(real[crop,crop],imag[crop,crop]))
            return base+torch.complex(real[crop,crop],imag[crop,crop])

        def read():
            if net is None:
                with torch.no_grad():return decode().detach().clone()
            with snapshot_readout(net):return decode().detach().clone()

        assert torch.equal(read(),base)
        history=[dict(update=0,seconds=0.,raw=score(base,pbase))]
        emas={};seconds=0.
        for i in range(1,a.updates+1):
            sync();start=time.perf_counter()
            opt.zero_grad(set_to_none=True);optp.zero_grad(set_to_none=True)
            obj=decode();probe=torch.complex(pr,pi)
            data=poisson_profile_loss(_fwd(cfg,obj,probe,sc.post,sc.Q).abs().square(),counts,alpha)
            regularizer=tgv(cabs(obj))[0]
            loss=data+energy*cfg.window_stage2_tgv*regularizer
            if not torch.isfinite(loss):raise FloatingPointError(kind)
            loss.backward();opt.step();optp.step()
            del obj,probe,data,regularizer,loss
            with torch.no_grad():
                obj=read();probe=torch.complex(pr,pi).detach()
                for beta in betas:
                    if beta not in emas:emas[beta]=(obj.clone(),probe.clone())
                    else:
                        emas[beta][0].mul_(beta).add_(obj,alpha=1-beta)
                        emas[beta][1].mul_(beta).add_(probe,alpha=1-beta)
            sync();seconds+=time.perf_counter()-start
            if i%a.eval_every==0 or i==a.updates:
                row=dict(update=i,seconds=seconds,raw=score(obj,probe))
                for beta,(eo,ep) in emas.items():row[f'ema{beta}']=score(eo,ep)
                history.append(row)
                print(f'{a.initialization} {kind} {i}: raw={row["raw"]["psnr_amp"]:.3f}/{row["raw"]["psnr_phs"]:.3f} '
                      f'EMA99={row["ema0.99"]["psnr_amp"]:.3f}/{row["ema0.99"]["psnr_phs"]:.3f} {seconds:.2f}s',flush=True)
        dest=a.outdir/kind;dest.mkdir()
        rs,cs=sc.roi
        arrays=dict(obj_rec=obj.cpu().numpy(),probe_rec=probe.cpu().numpy(),obj_gt=sc.obj,probe_gt=sc.probe,
                    obj_initial=base.cpu().numpy(),roi=[rs.start,rs.stop,cs.start,cs.stop],hist=json.dumps(history))
        for beta,(eo,ep) in emas.items():
            arrays[f'obj_ema{beta}']=eo.cpu().numpy();arrays[f'probe_ema{beta}']=ep.cpu().numpy()
        np.savez_compressed(dest/'result.npz',**arrays)
        save_json(dest/'history.json',history)
        return dict(name=kind,parameters=sum(p.numel() for p in params),network_parameters=0 if net is None else sum(p.numel() for p in params),
                    seconds=seconds,history=history,final=history[-1],result_file=str((dest/'result.npz').resolve()))

    results=[]
    for kind in a.candidates:
        results.append(run(kind));torch.cuda.empty_cache()
        save_json(a.outdir/'summary.json',dict(manifest,results=results))
    assert digest(a.scene)==manifest['scene_sha256'] and digest(a.checkpoint)==manifest['checkpoint_sha256']
    print('COMPLETED compact prior search',flush=True)

if __name__=='__main__':main()
