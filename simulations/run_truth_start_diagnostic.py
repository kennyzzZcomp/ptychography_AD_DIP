"""Oracle-initialization diagnostic, NOT an information-bound experiment.

Explicitly launched only. All arms begin at ground-truth object and probe.
Network conditioning and external residual base ALSO contain truth. This is
privileged information and must never be presented as ordinary reconstruction.
"""
import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulations.run_two_stage_ablation import (
    WindowCfg, load_scene, ProPtyUNet, cabs, _fwd, scaled_amplitude_loss,
    initialize_residual_heads, residual_object, reconstruction_channels,
    snapshot_readout, ObjectAmplitudeTGV, evaluate, probe_relerr,
    poisson_profile_loss, save_json, sync)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--updates',type=int,default=500)
    p.add_argument('--eval-every',type=int,default=25)
    a=p.parse_args()
    if a.outdir.exists():p.error('Output directory exists; choose a fresh path')
    if a.updates<1 or a.eval_every<1:p.error('Positive updates/eval-every required')
    with np.load(a.reference,allow_pickle=False) as z:
        source=json.loads(z['cfg'].item())
        ref_truth=z['obj_gt'].copy();ref_pos=z['positions'].copy()
    cfg=WindowCfg(**{k:v for k,v in source.items() if k in {f.name for f in fields(WindowCfg) if f.init}})
    cfg.device='cuda'
    device=torch.device('cuda')
    sc,scene_cfg,forward_error=load_scene(a.scene,device)
    assert np.array_equal(ref_truth,sc.obj) and np.array_equal(ref_pos,sc.pos)
    for k in ('N','grid','step_px','obj_size','wlength','det_pixel','z'):
        assert source[k]==scene_cfg[k],k
    assert cfg.grid==10 and len(sc.pos)==100
    cfg.quad_sign=scene_cfg.get('quad_sign',-1.)
    for k in ('noise','snr_db','noise_seed'):setattr(cfg,k,scene_cfg[k])
    seed=cfg.seed if cfg.network_seed is None else cfg.network_seed
    with h5py.File(a.scene,'r') as f:
        counts=torch.as_tensor(f['measurement_noise/counts'][()],dtype=torch.float32,device=device)
        alpha=float(f['measurement_noise'].attrs['global_alpha'])
    assert torch.allclose(counts/alpha,sc.Imt,rtol=2e-6,atol=1e-7)
    truth=torch.from_numpy(sc.obj.copy()).to(device)
    truth_probe=torch.from_numpy(sc.probe.copy()).to(device)
    clean_amp=sc.Iclt.clamp_min(0).sqrt()
    clean_counts=alpha*sc.Iclt
    energy=sc.sqrtIm.square().mean().detach()
    crop_start=(cfg.net_size-cfg.obj_size)//2
    crop=slice(crop_start,crop_start+cfg.obj_size)
    variants=[
        ('A_clean_AD_amp',False,True,'amplitude',0.,False),
        ('B_noisy_AD_amp',False,False,'amplitude',0.,False),
        ('C_clean_AD_poisson',False,True,'poisson',0.,False),
        ('D_noisy_AD_poisson',False,False,'poisson',0.,False),
        ('E_noisy_AD_poisson_TGV_fixed',False,False,'poisson',cfg.window_stage2_tgv,False),
        ('F_noisy_NN24_poisson_TGV_fixed',True,False,'poisson',cfg.window_stage2_tgv,False),
        ('G_noisy_AD_poisson_TGV_joint',False,False,'poisson',cfg.window_stage2_tgv,True),
        ('H_noisy_NN24_poisson_TGV_joint',True,False,'poisson',cfg.window_stage2_tgv,True),
    ]
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    a.outdir.mkdir(parents=True)
    manifest=dict(reference=str(a.reference.resolve()),scene=str(a.scene.resolve()),
        reference_sha256=sha(a.reference),scene_sha256=sha(a.scene),script_sha256=sha(Path(__file__)),
        helper_sha256=sha(Path(__file__).with_name('run_two_stage_ablation.py')),
        cfg=asdict(cfg),scene_config=scene_cfg,updates=a.updates,eval_every=a.eval_every,
        initial_object='exact ground truth',initial_probe='exact ground truth',
        optimizer_state='fresh Adam for both object/network and probe; no inherited stage-1 moments',
        network='24 base channels, concat, fixed GT-derived amplitude/cos/sin input; output=GT+residual',
        poisson='Same profiled global gain KL/(2*alpha) as previous stage-2 tests',
        tgv='Same amplitude-only TGV and measured-energy multiplier; no phase prior',
        timing='CUDA-synchronized updates only; excludes evaluation/setup/save',
        clean_source='Stored diagnostic_truth/clean_intensity; Poisson arm uses deterministic expected counts, no new noise',
        forward_relative_error=forward_error,gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__,seed=seed,variants=variants,
        caveats=['Oracle initialization is extra information, not an achievable noise ceiling.',
                 'Networks also have GT in fixed input and external residual base.',
                 'One fixed photon-noise realization; finite 500-step horizon is not convergence proof.',
                 'No truth-based early stopping; final step is reported for every arm.'])
    save_json(a.outdir/'manifest.json',manifest)
    torch.backends.cudnn.benchmark=True

    def run(name,network,clean,loss_kind,tgv_weight,joint):
        torch.manual_seed(seed)
        pr=torch.nn.Parameter(truth_probe.real.clone(),requires_grad=joint)
        pi=torch.nn.Parameter(truth_probe.imag.clone(),requires_grad=joint)
        optp=torch.optim.Adam([pr,pi],lr=cfg.window_stage2_lr_probe) if joint else None
        if network:
            inp,_=reconstruction_channels(truth,None,cfg.net_size,include_probe=False)
            net=ProPtyUNet(3,24,n_fields=1,ph_ch=1,skip_mode='concat').to(device)
            initialize_residual_heads(net)
            parameters=list(net.parameters())
            def decode():
                real,imag=net(inp)
                return residual_object(truth,real[0][crop,crop],imag[0][crop,crop])[0]
        else:
            net=None
            real=torch.nn.Parameter(truth.real.clone());imag=torch.nn.Parameter(truth.imag.clone())
            parameters=[real,imag]
            def decode():return torch.complex(real,imag)
        opt=torch.optim.Adam(parameters,lr=cfg.window_stage2_lr_net,weight_decay=cfg.weight_decay)
        tgv=ObjectAmplitudeTGV(cfg,sc.pos,device) if tgv_weight else None
        target_amp=clean_amp if clean else sc.sqrtIm
        target_counts=clean_counts if clean else counts

        def data_loss(field):
            if clean and loss_kind=='poisson':
                # Deterministic expected counts can be fractional. The shared
                # noisy-data helper assumes integer counts and clamps log(n)
                # at n=1; that constant would misreport the clean diagnostic.
                raw=alpha*field.abs().square()
                expected=(raw*(target_counts.sum()/raw.sum().clamp_min(1e-20))).clamp_min(1e-8)
                return (expected-target_counts+target_counts*(target_counts.clamp_min(1e-30).log()-expected.log())).mean()/(2*alpha)
            return (poisson_profile_loss(field.abs().square(),target_counts,alpha) if loss_kind=='poisson' else
                    scaled_amplitude_loss(cabs(field),target_amp)[0])

        def read():
            if net is None:
                with torch.no_grad():return decode().detach().clone()
            with snapshot_readout(net):return decode().detach().clone()

        def score(i,seconds):
            obj=read();probe=torch.complex(pr,pi).detach()
            rec=obj.cpu().numpy()
            row=dict(update=i,seconds=seconds,**evaluate(rec[sc.roi],sc.obj[sc.roi]))
            row['probe_relerr']=probe_relerr(probe.cpu().numpy(),sc.probe)
            row['raw_complex_relerr_roi']=float(torch.linalg.vector_norm((obj-truth)[sc.roi])/torch.linalg.vector_norm(truth[sc.roi]))
            with torch.no_grad():
                field=_fwd(cfg,obj,probe,sc.post,sc.Q)
                row['training_data_loss']=float(data_loss(field))
                row['noisy_poisson_loss']=float(poisson_profile_loss(field.abs().square(),counts,alpha))
                row['noisy_amplitude_loss']=float(scaled_amplitude_loss(cabs(field),sc.sqrtIm)[0])
                row['clean_amplitude_loss']=float(scaled_amplitude_loss(cabs(field),clean_amp)[0])
            return row

        assert torch.equal(read(),truth)
        assert torch.equal(torch.complex(pr,pi),truth_probe)
        history=[score(0,0.)];seconds=0.
        for i in range(1,a.updates+1):
            sync();start=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            if optp:optp.zero_grad(set_to_none=True)
            obj=decode();probe=torch.complex(pr,pi)
            data=data_loss(_fwd(cfg,obj,probe,sc.post,sc.Q))
            reg=tgv(cabs(obj))[0] if tgv else data.new_zeros(())
            total=data+energy*tgv_weight*reg
            if not torch.isfinite(total):raise FloatingPointError(name)
            total.backward()
            if i==1:
                initial_grad=float(sum(p.grad.detach().square().sum() for p in parameters if p.grad is not None).sqrt())
            opt.step()
            if optp:optp.step()
            sync();seconds+=time.perf_counter()-start
            del obj,probe,data,reg,total
            if i%a.eval_every==0 or i==a.updates:
                row=score(i,seconds);history.append(row)
                print(f'{name} {i}: amp={row["psnr_amp"]:.3f} phase={row["psnr_phs"]:.3f} '
                      f'data={row["training_data_loss"]:.6g} time={seconds:.2f}s',flush=True)
        obj=read();probe=torch.complex(pr,pi).detach()
        if not joint:assert torch.equal(probe,truth_probe)
        dest=a.outdir/name;dest.mkdir()
        metadata=dict(network=network,base_ch=24 if network else None,clean_measurements=clean,
                      loss_kind=loss_kind,tgv_weight=tgv_weight,probe_joint=joint,
                      truth_in_input=network,truth_external_residual_base=network,
                      initial_grad_l2=initial_grad)
        rs,cs=sc.roi
        np.savez_compressed(dest/'result.npz',obj_rec=obj.cpu().numpy(),probe_rec=probe.cpu().numpy(),
            obj_gt=sc.obj,probe_gt=sc.probe,positions=sc.pos,roi=[rs.start,rs.stop,cs.start,cs.stop],
            hist=json.dumps(history),cfg=json.dumps(asdict(cfg)),run_metadata=json.dumps(metadata))
        save_json(dest/'history.json',history)
        return dict(name=name,metadata=metadata,history=history,final=history[-1],seconds=seconds,
                    result_file=str((dest/'result.npz').resolve()))

    results=[]
    for variant in variants:
        results.append(run(*variant))
        torch.cuda.empty_cache()
        save_json(a.outdir/'summary.json',dict(manifest,results=results))
    assert sha(a.scene)==manifest['scene_sha256']
    assert sha(a.reference)==manifest['reference_sha256']
    print('COMPLETED oracle diagnostic, NOT an information bound',flush=True)


if __name__=='__main__':main()
