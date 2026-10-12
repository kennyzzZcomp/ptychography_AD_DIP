"""Explicitly launched, isolated two-stage ablations; never edits production solvers.

Reuses original model, forward, loss, TGV, fusion, conditioning and evaluator.
Stage-2 branches share O1, P1 and probe Adam state. Timing excludes evaluation,
file I/O and setup; every update is CUDA-synchronized. Pixel AD runs at least
the same update count and continues to the residual branch's compute budget.
"""
import argparse
import copy
from dataclasses import fields, asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import h5py
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulations.run_paper_windows import WindowCfg
from simulations.run_progressive_h5 import load_scene
from functions.addip.model import ProPtyUNet, make_field
from functions.addip.losses import cabs
from functions.paperrepro.solvers_addip import _initialize_object_heads, _fwd
from functions.paperrepro.window_training import (
    measurement_groups, footprint_masks, fuse_fields, scaled_amplitude_loss,
    initialize_residual_heads, residual_object)
from functions.paperrepro.reconstruction_input import snapshot_readout, reconstruction_channels
from functions.paperrepro.tgv import ObjectAmplitudeTGV, scan_domain
from functions.paperrepro.evaluate import evaluate, probe_relerr


def sync():
    torch.cuda.synchronize()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def poisson_profile_loss(intensity, counts, alpha):
    """Profile one global gain; half Poisson deviance / (2 alpha).

    Equivalent to Poisson NLL up to constants and a positive multiplier.
    Locally matches amplitude-MSE curvature: KL(n,mu) ~ 2*(sqrt(mu)-sqrt(n))^2.
    No per-frame rescaling, no clean data, and no regenerated photon counts.
    """
    raw = float(alpha) * intensity
    gain = counts.sum() / raw.sum().clamp_min(1e-20)
    expected = (gain * raw).clamp_min(1e-8)
    # 0*log(0/expected) has value zero but can yield NaN backward through xlogy.
    # Counts are nonnegative integers; clamp only the constant numerator.
    kl = expected - counts + counts * (counts.clamp_min(1).log() - expected.log())
    return kl.mean() / (2 * float(alpha))


class FixedPhasePrior:
    """Circular nearest-neighbor prior, fixed O1 reliability; no GT/ROI weights."""
    def __init__(self, cfg, positions, base):
        self.roi, domain = scan_domain(cfg, positions)
        mask = torch.as_tensor(domain,device=base.device)
        amp = base[self.roi].abs().detach()
        reliability = (amp / amp[mask].mean().clamp_min(1e-12)).clamp(0,1).square()
        self.wy = torch.minimum(reliability[1:],reliability[:-1]) * (mask[1:] & mask[:-1])
        self.wx = torch.minimum(reliability[:,1:],reliability[:,:-1]) * (mask[:,1:] & mask[:,:-1])

    def __call__(self, obj):
        field = obj[self.roi]
        valid = field.abs() > 1e-8
        phase = torch.atan2(torch.where(valid,field.imag,torch.zeros_like(field.imag)),
                            torch.where(valid,field.real,torch.ones_like(field.real)))
        y = self.wy * (1-torch.cos(phase[1:]-phase[:-1]))
        x = self.wx * (1-torch.cos(phase[:,1:]-phase[:,:-1]))
        return (y.sum()+x.sum()) / (self.wy.sum()+self.wx.sum()).clamp_min(1e-12)


class Stage2SkipAblation(ProPtyUNet):
    """Actually remove encoder-decoder concatenations; external residual untouched.

    Only the input convolution of each affected decoder block is replaced.
    All unaffected base-32 weights retain the original paired initialization.
    """
    def __init__(self, base=32, remove='shallow'):
        super().__init__(3,base,n_fields=1,ph_ch=1,skip_mode='concat')
        if remove not in ('shallow','all'): raise ValueError(remove)
        self.remove=remove
        blocks=[(self.d1,base)] if remove=='shallow' else [(self.d1,base),(self.d2,2*base),(self.d3,4*base)]
        for block,width in blocks:
            block.f[0]=torch.nn.Conv2d(width,width,3,padding=1)

    def forward_features(self,x):
        x1=self.e1(x);x2=self.e2(self.pool(x1));x3=self.e3(self.pool(x2))
        y=self.u3(self.bot(self.pool(x3)))
        y=self.d3(y if self.remove=='all' else torch.cat([y,x3],1))
        y=self.u2(y)
        y=self.d2(y if self.remove=='all' else torch.cat([y,x2],1))
        return self.d1(self.u1(y))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--stage1-updates', type=int)
    p.add_argument('--stage2-updates', type=int)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--pixel-lr', type=float, default=None)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--scene-noise-from-h5', action='store_true', help='Use stored noisy measurements with reference optimizer settings')
    p.add_argument('--equal-updates-only', action='store_true')
    p.add_argument('--stage2-comparison-only', action='store_true')
    p.add_argument('--denoise-summary', type=Path, help='Reuse this run\'s ABCD checkpoint; test four stage2 variants')
    p.add_argument('--architecture-summary', type=Path, help='Reuse ABCD checkpoint; test widths and actual skip removal')
    p.add_argument('--architecture-suite', choices=('baseline','light-no-concat','light-poisson'), default='baseline',
                   help='light-no-concat removes all internal concats at widths 24/20; light-poisson keeps concats and uses Poisson loss at widths 24/20')
    p.add_argument('--phase-weight', type=float, default=0.001)
    p.add_argument('--scratch-control-summary', type=Path,
                   help='Run pixel AD from ones for the total time of the saved ABCD + equal-update pixel arm')
    p.add_argument('--scratch-updates', type=int,
                   help='Run ONLY full-data pixel AD from ones for exactly this many updates')
    p.add_argument('--scratch-data-loss', choices=('amplitude','poisson','auto'), default='amplitude',
                   help='Fixed-update scratch AD only. Default reproduces the old amplitude-loss baseline; auto uses Poisson for noise')
    p.add_argument('--resume-full-checkpoint', type=Path,
                   help='Own saved full-stage1 checkpoint; enforce the same outside-footprint readout as ABCD, then run only stage2')
    a = p.parse_args()
    if a.scratch_updates is not None:
        if a.scratch_updates < 1 or any((a.scratch_control_summary, a.denoise_summary,
                a.architecture_summary, a.resume_full_checkpoint, a.stage2_comparison_only)):
            p.error('--scratch-updates requires a positive count and cannot combine with other experiment modes')
    elif a.scratch_data_loss != 'amplitude':
        p.error('--scratch-data-loss requires --scratch-updates')
    if a.architecture_suite != 'baseline' and not a.architecture_summary:
        p.error('--architecture-suite requires --architecture-summary')
    if a.outdir.exists():
        p.error('Choose a new output directory; existing runs are never overwritten')
    with np.load(a.reference, allow_pickle=False) as z:
        source = json.loads(z['cfg'].item())
        reference_truth, reference_pos = z['obj_gt'].copy(), z['positions'].copy()
    cfg = WindowCfg(**{k:v for k,v in source.items() if k in {f.name for f in fields(WindowCfg) if f.init}})
    assert cfg.grid == 10 and cfg.window_side == 5
    assert cfg.window_stage2_readout == 'complex-residual' and cfg.window_consistency == 0
    assert cfg.window_loss_mode == 'independent' and cfg.probe_mode == 'pixel'
    cfg.device = 'cuda'
    n1 = a.stage1_updates or cfg.window_switch_after
    n2 = a.stage2_updates or (cfg.iters-cfg.window_switch_after)
    assert n1 > 0 and n1 % 4 == 0 and n2 > 0 and a.eval_every > 0
    pixel_lr = cfg.window_stage2_lr_net if a.pixel_lr is None else a.pixel_lr
    assert pixel_lr > 0
    sc, scene_cfg, forward_error = load_scene(a.scene, torch.device('cuda'))
    assert np.array_equal(reference_truth, sc.obj) and np.array_equal(reference_pos, sc.pos)
    for key in ('N','grid','step_px','obj_size','wlength','det_pixel','z'):
        assert source[key] == scene_cfg[key], key
    if a.scene_noise_from_h5:
        for key in ('noise', 'snr_db', 'noise_seed'):
            if key in scene_cfg and hasattr(cfg, key):
                setattr(cfg, key, scene_cfg[key])
    else:
        assert source['noise'] == scene_cfg['noise']
    cfg.quad_sign = scene_cfg.get('quad_sign', -1.)
    a.outdir.mkdir(parents=True)
    manifest = dict(reference=str(a.reference.resolve()), scene=str(a.scene.resolve()),
        reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        scene_sha256=hashlib.sha256(a.scene.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cfg=asdict(cfg), stage1_updates=n1, stage2_network_updates=n2, pixel_lr=pixel_lr,
        seed=cfg.seed, gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        scene_fingerprint=sc.fp, forward_relative_error=forward_error,
        timing='Synchronized update wall time incl. forward/backward/Adam/TGV; excludes evaluation, setup and saving',
        stored_scene_config=scene_cfg,
        stage2_comparison_only=a.stage2_comparison_only,
        pixel_budget=('Exactly the same update count' if a.equal_updates_only else 'At least network update count; continue until >= network stage2 update-time budget; may overshoot by one update'),
        caveats=['One seed per scene, inherited hyperparameters; not a tuned pixel-AD optimum',
                 'Full-batch changes input channels 25->100 as required by measurement-conditioned design',
                 'Stage2 probe parameters AND Adam moments are cloned from the shared stage1 checkpoint',
                 'Truth used only for input consistency check, evaluation and saved plots; no GT-based early stopping'])
    save_json(a.outdir/'manifest.json', manifest)
    if a.check_only:
        print('CHECK PASSED', flush=True)
        return
    torch.backends.cudnn.benchmark = True
    seed = cfg.seed if cfg.network_seed is None else cfg.network_seed
    device = torch.device('cuda')
    m, ns, d = cfg.obj_size, cfg.net_size, cfg.obj_size-cfg.N
    pad = ns-m
    crop = slice(pad//2, pad//2+m)
    stack = F.pad(sc.Imt[None], (d//2,d-d//2,d//2,d-d//2))
    stack = F.pad(stack, (pad//2,pad-pad//2,pad//2,pad-pad//2))
    stack = stack/stack.amax((2,3), keepdim=True).clamp_min(1e-12)
    energy = sc.sqrtIm.square().mean().detach()
    all_sel = torch.arange(cfg.n_pat, device=device)

    def score(obj, prb):
        rec, probe = obj.detach().cpu().numpy(), prb.detach().cpu().numpy()
        rs, cs = sc.roi
        metrics = evaluate(rec[rs,cs], sc.obj[rs,cs])
        metrics['relerr_p'] = probe_relerr(probe,sc.probe)
        with torch.no_grad():
            metrics['full_data_loss'] = float(scaled_amplitude_loss(cabs(_fwd(cfg,obj.detach(),prb.detach(),sc.post,sc.Q)),sc.sqrtIm)[0])
        return metrics

    def store_run(name, obj, prb, history, seconds, run_metadata=None):
        dest = a.outdir/name
        dest.mkdir(exist_ok=True)
        rs,cs = sc.roi
        np.savez_compressed(dest/'result.npz',obj_rec=obj.detach().cpu().numpy(),probe_rec=prb.detach().cpu().numpy(),
            obj_gt=sc.obj,probe_gt=sc.probe,positions=sc.pos,roi=[rs.start,rs.stop,cs.start,cs.stop],
            hist=json.dumps(history),cfg=json.dumps(asdict(cfg)),train_elapsed_s=seconds,
            run_metadata=json.dumps(run_metadata or {}))
        save_json(dest/'history.json',history)

    def stage1(kind):
        torch.manual_seed(seed)
        groups = measurement_groups(10,'sparse',5) if kind=='abcd' else [list(range(100))]
        selections = [torch.tensor(g,device=device) for g in groups]
        inputs = [stack[:,g] for g in selections]
        net = ProPtyUNet(len(groups[0]),cfg.base_ch,n_fields=1,ph_ch=2,
            skip_mode=cfg.skip_mode,wavelet_threshold=cfg.wavelet_threshold).to(device)
        _initialize_object_heads(net,cfg.obj_init_alpha)
        pr=torch.nn.Parameter(torch.from_numpy(sc.P0.real.copy()).to(device))
        pi=torch.nn.Parameter(torch.from_numpy(sc.P0.imag.copy()).to(device))
        opt=torch.optim.Adam(net.parameters(),lr=cfg.lr_net,weight_decay=cfg.weight_decay)
        optp=torch.optim.Adam([pr,pi],lr=cfg.lr_probe)
        masks=footprint_masks(sc.pos,groups,m,cfg.N,device)
        tgvs=[ObjectAmplitudeTGV(cfg,sc.pos[g],device) for g in groups] if cfg.tgv_amp else [None]*len(groups)
        def decode(k):
            amp,ph=net(inputs[k])
            return make_field(amp[0],ph[:2])[crop,crop],F.softplus(amp[0])[crop,crop]
        def read():
            with snapshot_readout(net):
                values=[decode(k)[0] for k in range(len(groups))]
                return (fuse_fields(values,masks) if kind=='abcd' else
                        torch.where(masks[0],values[0],torch.ones_like(values[0]))).detach().clone()
        history=[]; seconds=0.
        for i in range(1,n1+1):
            k=(i-1)%len(groups)
            sync(); start=time.perf_counter()
            opt.zero_grad(set_to_none=True);optp.zero_grad(set_to_none=True)
            obj,amp=decode(k);prb=torch.complex(pr,pi)
            data=scaled_amplitude_loss(cabs(_fwd(cfg,obj,prb,sc.post[selections[k]],sc.Q)),sc.sqrtIm[selections[k]])[0]
            reg=tgvs[k](amp)[0] if tgvs[k] else data.new_zeros(())
            loss=data+energy*cfg.tgv_amp*reg
            if not torch.isfinite(loss): raise FloatingPointError(kind)
            loss.backward();opt.step();optp.step()
            sync();seconds+=time.perf_counter()-start
            del obj,amp,prb,data,reg,loss
            if i%a.eval_every==0 or i==n1:
                obj=read();prb=torch.complex(pr,pi).detach()
                row=dict(it=i,stage=1,update_time_s=seconds,**score(obj,prb))
                history.append(row)
                print(f'{kind} S1 {i}: amp={row["psnr_amp"]:.3f} phase={row["psnr_phs"]:.3f} time={seconds:.2f}s',flush=True)
        obj=read();prb=torch.complex(pr,pi).detach().clone()
        state=dict(obj=obj,probe=prb,probe_optimizer=copy.deepcopy(optp.state_dict()),
                   history=history,stage1_seconds=seconds)
        torch.save(state,a.outdir/f'{kind}_stage1_checkpoint.pt')
        store_run(kind+'_stage1',obj,prb,history,seconds)
        return state

    def stage2(state, name, pixel=False, budget=None, freeze_probe=False, phase_weight=0., poisson=False,
               base_ch=None, remove_skip=None):
        base=state['obj'].detach().clone()
        pr=torch.nn.Parameter(state['probe'].real.clone())
        pi=torch.nn.Parameter(state['probe'].imag.clone())
        optp=torch.optim.Adam([pr,pi],lr=cfg.window_stage2_lr_probe)
        optp.load_state_dict(copy.deepcopy(state['probe_optimizer']))
        for group in optp.param_groups: group['lr']=cfg.window_stage2_lr_probe
        if freeze_probe:
            pr.requires_grad_(False); pi.requires_grad_(False)
        if pixel:
            real=torch.nn.Parameter(base.real.clone()); imag=torch.nn.Parameter(base.imag.clone())
            opt=torch.optim.Adam([real,imag],lr=pixel_lr,weight_decay=cfg.weight_decay)
            net=None
            def decode():
                obj=torch.complex(real,imag)
                return obj,cabs(obj)
        else:
            conditioning,_=reconstruction_channels(base,None,ns,include_probe=False)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                width=cfg.base_ch if base_ch is None else base_ch
                net=(Stage2SkipAblation(width,remove_skip) if remove_skip else
                     ProPtyUNet(3,width,n_fields=1,ph_ch=1,skip_mode=cfg.skip_mode,wavelet_threshold=cfg.wavelet_threshold))
                initialize_residual_heads(net)
            net=net.to(device)
            opt=torch.optim.Adam(net.parameters(),lr=cfg.window_stage2_lr_net,weight_decay=cfg.weight_decay)
            def decode():
                amp,ph=net(conditioning)
                return residual_object(base,amp[0][crop,crop],ph[0][crop,crop])
        def read():
            if net is None:
                with torch.no_grad(): return decode()[0].detach().clone()
            with snapshot_readout(net): return decode()[0].detach().clone()
        assert torch.equal(read(),base)
        assert torch.equal(torch.complex(pr,pi),state['probe'])
        run_metadata=dict(stage2_base_ch=None if pixel else width,removed_internal_skips=remove_skip,
            network_parameters=0 if pixel else sum(p.numel() for p in net.parameters()),
            external_readout='pixel' if pixel else 'O1 + complex residual',freeze_probe=freeze_probe,
            phase_weight=phase_weight,poisson=poisson)
        tgv=ObjectAmplitudeTGV(cfg,sc.pos,device) if cfg.window_stage2_tgv else None
        phase_prior=FixedPhasePrior(cfg,sc.pos,base) if phase_weight else None
        if poisson:
            with h5py.File(a.scene,'r') as f:
                counts=torch.as_tensor(f['measurement_noise/counts'][()],device=device,dtype=torch.float32)
                alpha=float(f['measurement_noise'].attrs['global_alpha'])
            assert torch.allclose(counts/alpha,sc.Imt,rtol=2e-6,atol=1e-7)
        history=[dict(it=n1,stage=2,stage2_updates=0,stage2_time_s=0.,update_time_s=state['stage1_seconds'],**score(base,state['probe']))]
        seconds=0.;i=0
        while i<n2 or (budget is not None and seconds<budget):
            sync();start=time.perf_counter()
            opt.zero_grad(set_to_none=True);optp.zero_grad(set_to_none=True)
            obj,amp=decode();prb=torch.complex(pr,pi)
            field=_fwd(cfg,obj,prb,sc.post[all_sel],sc.Q)
            data=(poisson_profile_loss(field.abs().square(),counts,alpha) if poisson else
                  scaled_amplitude_loss(cabs(field),sc.sqrtIm)[0])
            reg=tgv(amp)[0] if tgv else data.new_zeros(())
            phase_reg=phase_prior(obj) if phase_prior else data.new_zeros(())
            loss=data+energy*cfg.window_stage2_tgv*reg+energy*phase_weight*phase_reg
            if not torch.isfinite(loss): raise FloatingPointError(name)
            loss.backward();opt.step()
            if not freeze_probe: optp.step()
            sync();seconds+=time.perf_counter()-start;i+=1
            del obj,amp,prb,data,reg,loss,field,phase_reg
            done=i>=n2 and (budget is None or seconds>=budget)
            if i%a.eval_every==0 or i==n2 or done:
                obj=read();prb=torch.complex(pr,pi).detach()
                row=dict(it=n1+i,stage=2,stage2_updates=i,stage2_time_s=seconds,
                    update_time_s=state['stage1_seconds']+seconds,**score(obj,prb))
                history.append(row)
                print(f'{name} S2 {i}: amp={row["psnr_amp"]:.3f} phase={row["psnr_phs"]:.3f} time={seconds:.2f}s',flush=True)
                if i==n2: store_run(name+'_equal_updates',obj,prb,state['history']+history,seconds+state['stage1_seconds'],run_metadata)
        obj=read();prb=torch.complex(pr,pi).detach()
        if freeze_probe: assert torch.equal(prb,state['probe'])
        store_run(name,obj,prb,state['history']+history,seconds+state['stage1_seconds'],run_metadata)
        return dict(name=name,stage1_seconds=state['stage1_seconds'],stage2_seconds=seconds,
                    stage2_updates=i,final=history[-1],history=state['history']+history,
                    result_file=str((a.outdir/name/'result.npz').resolve()),
                    **run_metadata)

    if a.denoise_summary or a.architecture_summary:
        previous_path=a.denoise_summary or a.architecture_summary
        previous=json.loads(previous_path.read_text(encoding='utf-8'))
        for key in ('scene_sha256','reference_sha256'):
            assert previous[key]==manifest[key],key
        checkpoint=previous_path.parent/'abcd_stage1_checkpoint.pt'
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        assert state['history'][-1]['it']==n1
        assert a.phase_weight>=0
        manifest.update(checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            phase_weight=a.phase_weight,phase_prior='Circular nearest-neighbor, fixed O1 amplitude reliability, nominal scan support',
            poisson_loss='Profile global expected-count gain; mean KL(counts,expected)/(2*alpha); local amplitude-MSE curvature match',
            pixel_budget='Not applicable: all residual-network variants use the same update count')
        if a.architecture_summary:
            variants=[('A_width32',dict(base_ch=32)),('B_width24',dict(base_ch=24)),
                ('C_width20',dict(base_ch=20)),('D_no_shallow_concat',dict(base_ch=32,remove_skip='shallow')),
                ('E_no_concat',dict(base_ch=32,remove_skip='all'))]
            if a.architecture_suite == 'light-no-concat':
                variants=[('F_width24_no_concat',dict(base_ch=24,remove_skip='all')),
                          ('G_width20_no_concat',dict(base_ch=20,remove_skip='all'))]
            elif a.architecture_suite == 'light-poisson':
                variants=[('H_width24_poisson',dict(base_ch=24,poisson=True)),
                          ('I_width20_poisson',dict(base_ch=20,poisson=True))]
            manifest.update(experiment='Stage2 architecture; actual concat removal and decoder input-width reduction',
                variants=variants,architecture_suite=a.architecture_suite,eval_every=a.eval_every,
                caveats=manifest['caveats']+['Removing skips also reduces parameters; not a parameter-matched information-path-only ablation',
                    'Different widths have different weights; same seed and identical output O1, not identical parameter arrays',
                    'cfg is inherited baseline; per-run architecture is stored in run_metadata and summary'])
        else:
            variants=[('A_original',{}),('B_fixed_probe',dict(freeze_probe=True)),
                ('C_phase_prior',dict(phase_weight=a.phase_weight)),('D_poisson',dict(poisson=True))]
        save_json(a.outdir/'manifest.json',manifest)
        results=[]
        for name,options in variants:
            results.append(stage2(state,name,**options))
            torch.cuda.empty_cache()
            save_json(a.outdir/'summary.json',dict(manifest,results=results))
        for key in ('psnr_amp','psnr_phs','relerr_p','full_data_loss'):
            initial=[next(h[key] for h in r['history'] if h.get('stage2_updates')==0) for r in results]
            assert len(set(initial))==1,key
        assert manifest['checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert manifest['scene_sha256']==hashlib.sha256(a.scene.read_bytes()).hexdigest()
        print('COMPLETED denoise suite',flush=True)
        return

    if a.scratch_updates is not None:
        initial_probe=torch.from_numpy(sc.P0.copy()).to(device)
        assert torch.equal(initial_probe,torch.ones_like(initial_probe))
        fresh_probe_optimizer=torch.optim.Adam([
            torch.nn.Parameter(initial_probe.real.clone()),
            torch.nn.Parameter(initial_probe.imag.clone())],lr=cfg.window_stage2_lr_probe)
        state=dict(obj=torch.ones((m,m),dtype=torch.complex64,device=device),
                   probe=initial_probe,probe_optimizer=fresh_probe_optimizer.state_dict(),
                   history=[],stage1_seconds=0.)
        n1=0; n2=a.scratch_updates
        loss_kind=('amplitude' if cfg.noise=='none' else 'poisson') if a.scratch_data_loss=='auto' else a.scratch_data_loss
        manifest.update(stage1_updates=0,stage2_network_updates=0,scratch_updates=n2,
            data_loss=loss_kind,pixel_budget=f'Exactly {n2} full-data pixel updates',
            initialization='Complex object ones, probe ones, fresh object and probe Adam; no network',
            caveats=['Fixed-update baseline, not a tuned pixel-AD optimum.',
                     'All 100 measurements per update; no network or inherited optimizer state.',
                     'Same numerical learning rates do not imply equal effective steps across parameterizations.',
                     'Truth used for scoring only; no GT-selected stopping.'])
        manifest['results']=[stage2(state,'scratch_pixel',pixel=True,poisson=loss_kind=='poisson')]
        assert manifest['results'][0]['stage2_updates']==n2
        assert manifest['scene_sha256']==hashlib.sha256(a.scene.read_bytes()).hexdigest()
        save_json(a.outdir/'manifest.json',{k:v for k,v in manifest.items() if k!='results'})
        save_json(a.outdir/'summary.json',manifest)
        print('COMPLETED fixed-update scratch AD',flush=True)
        return

    if a.scratch_control_summary:
        comparison=json.loads(a.scratch_control_summary.read_text(encoding='utf-8'))
        assert comparison['scene_sha256']==manifest['scene_sha256']
        assert comparison['reference_sha256']==manifest['reference_sha256']
        assert comparison['pixel_lr']==pixel_lr
        for key in ('window_stage2_lr_probe','window_stage2_tgv','weight_decay'):
            assert comparison['cfg'][key]==asdict(cfg)[key],key
        arm=next(r for r in comparison['results'] if r['name']=='abcd_pixel')
        target_updates=comparison['stage2_network_updates']
        target=next(h for h in arm['history'] if h.get('stage2_updates')==target_updates)
        budget=target['update_time_s']
        initial_probe=torch.from_numpy(sc.P0.copy()).to(device)
        assert torch.equal(initial_probe,torch.ones_like(initial_probe))
        fresh_probe_optimizer=torch.optim.Adam([
            torch.nn.Parameter(initial_probe.real.clone()),
            torch.nn.Parameter(initial_probe.imag.clone())],lr=cfg.window_stage2_lr_probe)
        state=dict(obj=torch.ones((m,m),dtype=torch.complex64,device=device),
                   probe=initial_probe,probe_optimizer=fresh_probe_optimizer.state_dict(),
                   history=[],stage1_seconds=0.)
        n1=0; n2=1
        manifest.update(stage1_updates=0,stage2_network_updates=0,
            scratch_time_budget_s=budget,comparison_summary=str(a.scratch_control_summary.resolve()),
            comparison_summary_sha256=hashlib.sha256(a.scratch_control_summary.read_bytes()).hexdigest(),
            comparison_endpoint=target,pixel_budget='Total ABCD stage1 + 500 pixel updates; first crossing, at most one update overshoot',
            initialization='Complex object ones, probe ones, fresh object and probe Adam; no network')
        result=stage2(state,'scratch_pixel',pixel=True,budget=budget)
        manifest['results']=[result]
        assert result['stage2_seconds']>=budget
        assert manifest['scene_sha256']==hashlib.sha256(a.scene.read_bytes()).hexdigest()
        save_json(a.outdir/'manifest.json',{k:v for k,v in manifest.items() if k!='results'})
        save_json(a.outdir/'summary.json',manifest)
        print('COMPLETED scratch control',flush=True)
        return

    if a.resume_full_checkpoint:
        state=torch.load(a.resume_full_checkpoint,map_location=device,weights_only=False)
        mask=footprint_masks(sc.pos,[list(range(cfg.n_pat))],m,cfg.N,device)[0]
        old_obj=state['obj'].detach().clone()
        state['obj']=torch.where(mask,old_obj,torch.ones_like(old_obj))
        assert torch.equal(old_obj[mask],state['obj'][mask])
        assert len(state['history']) and state['history'][-1]['it']==n1
        manifest['resume_full_checkpoint']=str(a.resume_full_checkpoint.resolve())
        manifest['checkpoint_sha256']=hashlib.sha256(a.resume_full_checkpoint.read_bytes()).hexdigest()
        manifest['readout_correction']='Outside union of computational probe footprints set to ones, same as ABCD fusion; all illuminated values, probe and probe Adam unchanged'
        torch.save(state,a.outdir/'full_stage1_checkpoint.pt')
        result=stage2(state,'full_residual')
        manifest['results']=[result]
        save_json(a.outdir/'summary.json',manifest)
        print('COMPLETED matched-readout full control',flush=True)
        return

    abcd=stage1('abcd')
    residual=stage2(abcd,'abcd_residual')
    torch.cuda.empty_cache()
    pixel=stage2(abcd,'abcd_pixel',pixel=True,budget=None if a.equal_updates_only else residual['stage2_seconds'])
    del abcd
    torch.cuda.empty_cache()
    results=[residual,pixel]
    if not a.stage2_comparison_only:
        full=stage1('full')
        fullres=stage2(full,'full_residual')
        results.append(fullres)
    manifest['results']=results
    assert manifest['scene_sha256']==hashlib.sha256(a.scene.read_bytes()).hexdigest()
    assert manifest['reference_sha256']==hashlib.sha256(a.reference.read_bytes()).hexdigest()
    save_json(a.outdir/'summary.json',manifest)
    print('COMPLETED',str(a.outdir.resolve()),flush=True)


if __name__=='__main__':
    main()
