"""Continue saved stage-1 width ablations with a configurable residual U-Net.

Explicitly requested experiment; does not modify production code or rerun stage1.
"""
import argparse
import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulations.run_two_stage_ablation import (
    WindowCfg, load_scene, ProPtyUNet, cabs, _fwd, initialize_residual_heads,
    residual_object, reconstruction_channels, snapshot_readout,
    scaled_amplitude_loss, ObjectAmplitudeTGV, evaluate, probe_relerr, sync, save_json,
    poisson_profile_loss, Stage2SkipAblation)


def decode_residual(base, first, second, mode):
    """Change only output coordinates; polar matches the compact CNN experiment."""
    if mode == 'complex':
        return residual_object(base,first,second)
    if mode == 'polar':
        obj = base.detach()*torch.exp(torch.complex(first,second))
        return obj,cabs(obj)
    raise ValueError(mode)


class SixLayerDeepDecoder(nn.Module):
    """DD-inspired polar correction, not the original sigmoid-image formulation.

    Six bias-free pointwise mixers, fixed bilinear resize, ReLU, channel norm.
    Batch size is one; BN uses current spatial moments, never running moments.
    The final hidden block does not upsample. Two signed linear output heads.
    """
    def __init__(self,width,size=624):
        super().__init__()
        self.sizes = tuple((size+2**power-1)//2**power for power in (4,3,2,1,0,0))
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width,width,1,bias=False),
                          nn.Upsample(size=(size,size),mode='bilinear',align_corners=False)
                          if i < 5 else nn.Identity(),
                          nn.ReLU(),nn.BatchNorm2d(width,eps=1e-5,affine=True,track_running_stats=False))
            for i,size in enumerate(self.sizes)])
        self.head_amp = nn.Conv2d(width,1,1)
        self.head_phs = nn.Conv2d(width,1,1)

    def forward(self,x):
        for block in self.blocks: x = block(x)
        return self.head_amp(x)[0],self.head_phs(x)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Completed width-screen directory')
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--updates', type=int, default=500)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--stage1-names', nargs='+', choices=['u32','u24','u16'], default=['u32','u24','u16'])
    p.add_argument('--stage2-width', type=int, choices=[16,24,32,64], default=32)
    p.add_argument('--architecture', choices=['unet','deep-decoder'], default='unet')
    p.add_argument('--residual-mode', choices=['complex','polar'], default='complex',
                   help='Complex: O1 + dr + i*di; polar: O1 * exp(da + i*dphi), no clipping')
    p.add_argument('--remove-internal-concat', action='store_true',
                   help='Remove all three encoder-to-decoder concatenations; keep external residual')
    p.add_argument('--data-loss', choices=['auto','amplitude','poisson'], default='auto',
                   help='Auto matches previous compact controls: clean amplitude, noisy Poisson')
    a = p.parse_args()
    if a.outdir.exists() or a.updates < 1 or a.eval_every < 1:
        p.error('Positive update/evaluation counts and a new output directory required')
    if a.architecture == 'deep-decoder' and (a.stage2_width not in (32,64) or a.remove_internal_concat or a.residual_mode != 'polar'):
        p.error('Deep Decoder experiment requires width32/64, polar residual, and no concat flag (already absent)')
    if a.architecture == 'unet' and a.stage2_width == 64:
        p.error('Width64 is reserved for the Deep Decoder experiment')
    source = json.loads((a.source/'summary.json').read_text(encoding='utf-8'))
    cfg = WindowCfg(**{k:v for k,v in source['cfg'].items()
                       if k in {f.name for f in fields(WindowCfg) if f.init}})
    assert cfg.base_ch == 32 and cfg.skip_mode == 'concat'
    assert cfg.window_stage2_readout == 'complex-residual' and source['stage1_updates'] == 500
    assert cfg.window_stage2_lr_net == .0005 and cfg.window_stage2_lr_probe == .01
    assert cfg.window_stage2_tgv == .005
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert digest(source['scene']) == source['scene_sha256']
    device = torch.device('cuda')
    sc, scene_cfg, error = load_scene(Path(source['scene']), device)
    assert scene_cfg['noise'] == cfg.noise
    for key in ('N','grid','step_px','obj_size','wlength','det_pixel','z'):
        assert getattr(cfg,key) == scene_cfg[key]
    assert cfg.fft_size == scene_cfg.get('fft_size', 0)
    names = list(dict.fromkeys(a.stage1_names))
    loss_kind = ('amplitude' if cfg.noise == 'none' else 'poisson') if a.data_loss == 'auto' else a.data_loss
    counts = None; alpha = None
    if loss_kind == 'poisson':
        with h5py.File(source['scene'],'r') as f:
            counts = torch.as_tensor(f['measurement_noise/counts'][()],dtype=torch.float32,device=device)
            alpha = float(f['measurement_noise'].attrs['global_alpha'])
        assert torch.allclose(counts/alpha,sc.Imt,rtol=2e-6,atol=1e-7)
    hashes = {name:digest(a.source/name/'stage1_500.pt') for name in names}
    manifest = dict(source=str(a.source.resolve()), source_summary_sha256=digest(a.source/'summary.json'),
        checkpoint_sha256=hashes, scene_sha256=source['scene_sha256'], cfg=source['cfg'],
        stage1_updates=500, stage2_updates=a.updates, stage2_seed=0, stage1_names=names,
        stage2_width=a.stage2_width, architecture=a.architecture, data_loss=loss_kind, residual_mode=a.residual_mode,
        removed_internal_concat=a.remove_internal_concat,
        stage2=(f'Deep Decoder: width{a.stage2_width}, 6 layers, fixed random input, no internal skips; zero heads'
                if a.architecture=='deep-decoder' else
                f'ProPtyUNet: width{a.stage2_width}, internal_concat={not a.remove_internal_concat}, {a.residual_mode} residual; zero heads'),
        decoder_spec=(dict(input_shape=[1,a.stage2_width,(cfg.net_size+31)//32,(cfg.net_size+31)//32],input_seed=100000,
            input_distribution='Uniform[0,0.1), fixed, no gradients',
            sizes=[(cfg.net_size+2**power-1)//2**power for power in (4,3,2,1,0,0)],
            block='1x1 bias=False -> bilinear align_corners=False -> ReLU -> channel norm affine=True eps=1e-5',
            normalization='BatchNorm2d track_running_stats=False; batch size1; spatial per-channel moments',
            output='Two signed 1x1 heads with bias, zero initialized; no sigmoid')
            if a.architecture=='deep-decoder' else None),
        residual_formula='O1 + dr + i*di' if a.residual_mode=='complex' else 'O1 * exp(da + i*dphi); no clipping',
        loss=f'Full-data profiled {loss_kind} loss + energy-scaled TGV 0.005',
        poisson_scaling='Profiled KL/(2 alpha)' if counts is not None else None,
        probe='Inherits P1 AND Adam moments; jointly updated at lr0.01',
        timing='CUDA synchronized updates; excludes evaluation/readout/setup/save. Stage1 times reused.',
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, forward_relative_error=error,
        caveats=['One object, one initialization and one noise realization per dose; no universal superiority claim.',
                 'All fixed endpoints reported, no GT checkpoint selection.',
                 'Historical ROI evaluator retained unchanged.',
                 'Removing concatenations also reduces decoder input channels and parameter count; not a parameter-matched intervention.'])
    a.outdir.mkdir(parents=True)
    save_json(a.outdir/'manifest.json', manifest)
    shutil.copyfile(__file__, a.outdir/'runner_snapshot.py')
    torch.backends.cudnn.benchmark = True
    pad = cfg.net_size-cfg.obj_size
    crop = slice(pad//2,pad//2+cfg.obj_size)
    energy = sc.sqrtIm.square().mean().detach()

    def score(obj, probe):
        result = evaluate(obj.detach().cpu().numpy()[sc.roi], sc.obj[sc.roi])
        result['probe_relerr'] = probe_relerr(probe.detach().cpu().numpy(), sc.probe)
        with torch.no_grad():
            field = _fwd(cfg,obj.detach(),probe.detach(),sc.post,sc.Q)
            result['full_amplitude_loss'] = float(scaled_amplitude_loss(cabs(field),sc.sqrtIm)[0])
            if counts is not None:
                result['full_poisson_loss'] = float(poisson_profile_loss(field.abs().square(),counts,alpha))
        return result

    results = []
    for name in names:
        state = torch.load(a.source/name/'stage1_500.pt',map_location=device,weights_only=False)
        assert state['stage1_updates'] == 500
        base = state['obj'].detach().clone()
        assert tuple(base.shape) == tuple(sc.obj.shape)
        conditioning,_ = reconstruction_channels(base,None,cfg.net_size,include_probe=False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            net = (SixLayerDeepDecoder(a.stage2_width,cfg.net_size) if a.architecture=='deep-decoder' else
                   Stage2SkipAblation(a.stage2_width,remove='all') if a.remove_internal_concat else
                   ProPtyUNet(3,a.stage2_width,n_fields=1,ph_ch=1,skip_mode='concat',wavelet_threshold=cfg.wavelet_threshold))
            initialize_residual_heads(net)
        if a.architecture=='deep-decoder':
            rng = torch.Generator().manual_seed(100000)
            n0 = (cfg.net_size+31)//32
            conditioning = (.1*torch.rand((1,a.stage2_width,n0,n0),generator=rng)).to(device)
            assert not conditioning.requires_grad
            assert sum(p.numel() for p in net.parameters()) == 6*a.stage2_width**2+14*a.stage2_width+2
        net = net.to(device)
        opt = torch.optim.Adam(net.parameters(),lr=cfg.window_stage2_lr_net,weight_decay=cfg.weight_decay)
        pr = torch.nn.Parameter(state['probe'].real.clone())
        pi = torch.nn.Parameter(state['probe'].imag.clone())
        optp = torch.optim.Adam([pr,pi],lr=cfg.window_stage2_lr_probe)
        optp.load_state_dict(copy.deepcopy(state['probe_optimizer']))
        for group in optp.param_groups: group['lr'] = cfg.window_stage2_lr_probe
        tgv = ObjectAmplitudeTGV(cfg,sc.pos,device)

        def decode():
            first,second = net(conditioning)
            return decode_residual(base,first[0][crop,crop],second[0][crop,crop],a.residual_mode)

        def read():
            with snapshot_readout(net): return decode()[0].detach().clone()

        assert torch.equal(read(),base)
        history = [dict(update=0,seconds=0.,**score(base,state['probe']))]
        for key in ('psnr_amp','psnr_phs','relerr'):
            assert np.isclose(history[0][key],state['history'][-1][key],atol=1e-5,rtol=1e-5)
        seconds = 0.
        for i in range(1,a.updates+1):
            sync(); start = time.perf_counter()
            opt.zero_grad(set_to_none=True); optp.zero_grad(set_to_none=True)
            obj,amp = decode(); probe = torch.complex(pr,pi)
            field = _fwd(cfg,obj,probe,sc.post,sc.Q)
            data = (poisson_profile_loss(field.abs().square(),counts,alpha) if counts is not None else
                    scaled_amplitude_loss(cabs(field),sc.sqrtIm)[0])
            loss = data+energy*cfg.window_stage2_tgv*tgv(amp)[0]
            if not torch.isfinite(loss): raise FloatingPointError(name)
            loss.backward(); opt.step(); optp.step(); sync()
            seconds += time.perf_counter()-start
            del obj,amp,probe,field,data,loss
            if i % a.eval_every == 0 or i == a.updates:
                obj = read(); probe = torch.complex(pr,pi).detach().clone()
                history.append(dict(update=i,seconds=seconds,**score(obj,probe)))
                print(f'{cfg.noise} {name} -> {a.architecture}{a.stage2_width} {a.residual_mode} S2 {i}: {history[-1]["psnr_amp"]:.3f}/{history[-1]["psnr_phs"]:.3f}, {seconds:.2f}s',flush=True)
        dest = a.outdir/name; dest.mkdir()
        if a.architecture=='deep-decoder':
            torch.save(dict(network=net.state_dict(),fixed_input=conditioning.detach().cpu()),dest/'decoder_state.pt')
        save_json(dest/'history.json',history)
        rs,cs = sc.roi
        np.savez_compressed(dest/'result.npz',obj_rec=obj.cpu().numpy(),probe_rec=probe.cpu().numpy(),
            obj_gt=sc.obj,probe_gt=sc.probe,roi=[rs.start,rs.stop,cs.start,cs.stop],
            cfg=json.dumps(source['cfg']),history=json.dumps(history),run_metadata=json.dumps(manifest))
        with np.load(dest/'result.npz',allow_pickle=False) as z:
            check = evaluate(z['obj_rec'][sc.roi],z['obj_gt'][sc.roi])
        assert all(np.isclose(check[k],history[-1][k],atol=1e-5,rtol=1e-5) for k in ('psnr_amp','psnr_phs','relerr'))
        compact = next((b for r in source['results'] if r['name']==name
                        for b in r['stage2'] if b['stage1_updates']==500), None)
        results.append(dict(name=name,architecture=a.architecture,stage2_width=a.stage2_width,residual_mode=a.residual_mode,
            removed_internal_concat=a.remove_internal_concat,parameters=sum(p.numel() for p in net.parameters()),
            stage1=history[0],final=history[-1],stage1_seconds=state['stage1_seconds'],stage2_seconds=seconds,
            total_seconds=state['stage1_seconds']+seconds,
            compact_control_final=compact['history'][-1] if compact else None,
            result_file=str((dest/'result.npz').resolve())))
        save_json(a.outdir/'summary.json',dict(manifest,results=results))
        del net,opt,optp,pr,pi,state,obj,probe,base,conditioning,tgv
        torch.cuda.empty_cache()
    assert hashes == {name:digest(a.source/name/'stage1_500.pt') for name in names}
    assert digest(source['scene']) == source['scene_sha256']
    lines = [f'DATA {source["scene"]}: fixed stage1 500 + {a.architecture} stage2 '+str(a.updates)+' updates',
             f'Second-stage width{a.stage2_width}, {a.residual_mode} residual, seed0; loss={loss_kind}.',
             'Stage1 checkpoint and probe Adam states unchanged; all saved final fields rescored.',
             'PSNR amp/phase, dB; synchronized training seconds only; stage1 times reused.']
    for r in results:
        f = r['final']; c = r['compact_control_final']
        compact_text = f"compact control {c['psnr_amp']:.3f}/{c['psnr_phs']:.3f}; " if c else ''
        lines.append(f"{r['name']}: {a.architecture}{a.stage2_width} {f['psnr_amp']:.3f}/{f['psnr_phs']:.3f}; "
                     f"{compact_text}"
                     f"time {r['stage1_seconds']:.2f}+{r['stage2_seconds']:.2f}={r['total_seconds']:.2f}")
    (a.outdir/'results.txt').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
