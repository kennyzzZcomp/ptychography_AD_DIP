"""Explicitly requested stage-1 architecture screen, with an identical stage 2.

No production edits. All candidate endpoints are retained, not GT-selected.
The baseline additionally branches at update 300 to diagnose switch timing.
"""
import argparse
import copy
from dataclasses import fields, asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulations.run_two_stage_ablation import (
    WindowCfg, load_scene, ProPtyUNet, make_field, cabs, _fwd,
    _initialize_object_heads, measurement_groups, footprint_masks, fuse_fields,
    scaled_amplitude_loss, ObjectAmplitudeTGV, snapshot_readout,
    reconstruction_channels, evaluate, probe_relerr, poisson_profile_loss, sync, save_json)
from simulations.run_compact_stage2_search import CompactResidual
from functions.common.unet import DoubleConv

CANDIDATES = ['u32', 'u24', 'u16', 'u16_no_concat', 'u16_bilinear', 'shallow16']


class Stage1UNet(ProPtyUNet):
    def __init__(self, kind):
        width = {'u32': 32, 'u24': 24}.get(kind, 16)
        super().__init__(25, width, n_fields=1, ph_ch=2, skip_mode='concat')
        self.no_concat = kind == 'u16_no_concat'
        if self.no_concat:
            for block, size in ((self.d1, width), (self.d2, width*2), (self.d3, width*4)):
                block.f[0] = nn.Conv2d(size, size, 3, padding=1)
        if kind == 'u16_bilinear':
            self.u3 = bilinear_up(width*8, width*4)
            self.u2 = bilinear_up(width*4, width*2)
            self.u1 = bilinear_up(width*2, width)

    def forward_features(self, x):
        if not self.no_concat:
            return super().forward_features(x)
        x1 = self.e1(x); x2 = self.e2(self.pool(x1)); x3 = self.e3(self.pool(x2))
        y = self.d3(self.u3(self.bot(self.pool(x3))))
        return self.d1(self.u1(self.d2(self.u2(y))))


def bilinear_up(cin, cout):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                         nn.Conv2d(cin, cout, 3, padding=1))


class ShallowEncoderDecoder(nn.Module):
    """Two spatial scales, no concatenation, fixed bilinear upsampling.

    A combined compact-architecture candidate, not an isolated one-factor ablation.
    Still measurement-conditioned: does not replace the 25 diffraction channels.
    """
    def __init__(self):
        super().__init__()
        self.e1 = DoubleConv(25, 16); self.e2 = DoubleConv(16, 32)
        self.bot = DoubleConv(32, 64); self.pool = nn.MaxPool2d(2)
        self.u2 = bilinear_up(64, 32); self.d2 = DoubleConv(32, 32)
        self.u1 = bilinear_up(32, 16); self.d1 = DoubleConv(16, 16)
        self.head_amp = nn.Conv2d(16, 1, 3, padding=1)
        self.head_phs = nn.Conv2d(16, 2, 3, padding=1)

    def forward(self, x):
        x = self.e2(self.pool(self.e1(x)))
        x = self.d1(self.u1(self.d2(self.u2(self.bot(self.pool(x))))))
        return self.head_amp(x)[0], self.head_phs(x)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path,
                   help='Optional historical NPZ config; omitted: fixed paper U16→U32 settings, H5 geometry')
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--candidates', nargs='+', choices=CANDIDATES, default=CANDIDATES)
    p.add_argument('--updates', type=int, default=500)
    p.add_argument('--stage2-updates', type=int, default=500)
    p.add_argument('--stage1-only', action='store_true',
                   help='Save first-stage checkpoints without running the compact CNN second stage')
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--early-switch', type=int, default=300)
    a = p.parse_args()
    if a.outdir.exists(): p.error('Choose a new output directory, never overwrite results')
    if a.updates <= 0 or a.updates % 4 or a.stage2_updates <= 0 or a.eval_every <= 0:
        p.error('Positive budgets required, stage1 budget must be a multiple of 4')
    device = torch.device('cuda')
    sc, scene_cfg, forward_error = load_scene(a.scene, device)
    if a.reference:
        with np.load(a.reference, allow_pickle=False) as z:
            source = json.loads(z['cfg'].item()); gt = z['obj_gt'].copy(); pos = z['positions'].copy()
        assert np.array_equal(gt, sc.obj) and np.array_equal(pos, sc.pos)
        for key in ('N','grid','step_px','obj_size','wlength','det_pixel','z'):
            assert source[key] == scene_cfg[key], key
        assert source.get('fft_size', 0) == scene_cfg.get('fft_size', 0)
    else:
        source = dict(scene_cfg)
        source.update(base_ch=32, lr_net=.0005, lr_probe=.01, tgv_amp=.01,
            obj_init_alpha=0., skip_mode='concat', window_consistency=0.,
            window_loss_mode='independent', probe_mode='pixel', weight_decay=0.,
            window_layout='sparse', window_side=5, window_update='sequential',
            window_stage2_readout='complex-residual', window_stage2_lr_net=.0005,
            window_stage2_lr_probe=.01, window_stage2_tgv=.005)
    cfg = WindowCfg(**{k: v for k, v in source.items() if k in {f.name for f in fields(WindowCfg) if f.init}})
    assert cfg.base_ch == 32 and cfg.lr_net == .0005 and cfg.lr_probe == .01 and cfg.tgv_amp == .01
    assert cfg.obj_init_alpha == 0 and cfg.skip_mode == 'concat' and cfg.window_consistency == 0
    assert cfg.window_loss_mode == 'independent' and cfg.probe_mode == 'pixel'
    cfg.device = 'cuda'; device = torch.device('cuda')
    for key in ('noise','snr_db','noise_seed'): setattr(cfg, key, scene_cfg[key])
    cfg.quad_sign = scene_cfg.get('quad_sign', -1.)
    counts = None; alpha = None
    with h5py.File(a.scene, 'r') as f:
        if 'measurement_noise/counts' in f:
            counts = torch.as_tensor(f['measurement_noise/counts'][()], dtype=torch.float32, device=device)
            alpha = float(f['measurement_noise'].attrs['global_alpha'])
            assert torch.allclose(counts/alpha, sc.Imt, rtol=2e-6, atol=1e-7)
    groups = measurement_groups(10, 'sparse', 5)
    assert sorted(sum(groups, [])) == list(range(100)) and all(len(g) == 25 for g in groups)
    selections = [torch.tensor(g, device=device) for g in groups]
    m, ns = cfg.obj_size, cfg.net_size
    d, pad = m-sc.Imt.shape[-1], ns-m; crop = slice(pad//2, pad//2+m)
    stack = F.pad(sc.Imt[None], (d//2, d-d//2, d//2, d-d//2))
    stack = F.pad(stack, (pad//2, pad-pad//2, pad//2, pad-pad//2))
    stack = stack/stack.amax((2,3), keepdim=True).clamp_min(1e-12)
    inputs = [stack[:, g] for g in selections]
    masks = footprint_masks(sc.pos, groups, m, cfg.N, device)
    energy = sc.sqrtIm.square().mean().detach()
    a.outdir.mkdir(parents=True)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(reference=str(a.reference.resolve()) if a.reference else None, scene=str(a.scene.resolve()),
        reference_sha256=digest(a.reference) if a.reference else None, scene_sha256=digest(a.scene), cfg=asdict(cfg),
        scene_cfg=scene_cfg, forward_relative_error=forward_error, seed=a.seed,
        candidates=a.candidates, stage1_updates=a.updates, stage2_updates=a.stage2_updates,
        early_switch=a.early_switch, groups=groups, gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        stage1_only=a.stage1_only,
        stage1_loss='Original group-profiled amplitude MSE + energy * TGV 0.01, unchanged for architecture comparisons',
        stage2='Identical 5122-parameter CNN16 polar correction; fresh seed 0; O1/P1 + probe Adam moments inherited; fixed 500 default updates',
        stage2_loss='Poisson KL/(2 alpha) + energy * TGV 0.005 for noisy data; original amplitude MSE + same TGV for clean data',
        timing='CUDA-synchronized training updates only, excludes readout, scoring, setup and save',
        caveats=['Exploratory single object and one noise realization per dose; NOT guaranteed universal superiority.',
                 'All candidates kept; fixed 300/500 switch diagnostic, no GT-selected checkpoint or early stopping.',
                 'No changes to ABCD membership, order, 25 input channels, normalized diffraction inputs, or output field parameterization.',
                 'Do not conflate architecture size, convergence speed, and noise overfitting.',
                 'Second stage uses a common compact network; its clean-data ceiling need not match original full U-Net.'])
    if a.stage1_only:
        manifest.update(stage2_updates=0, stage2='Not run (--stage1-only); checkpoints only', stage2_loss=None)
    save_json(a.outdir/'manifest.json', manifest)
    shutil.copyfile(__file__, a.outdir/'runner_snapshot.py')
    torch.backends.cudnn.benchmark = True

    def score(obj, probe):
        metrics = evaluate(obj.detach().cpu().numpy()[sc.roi], sc.obj[sc.roi])
        metrics['probe_relerr'] = probe_relerr(probe.detach().cpu().numpy(), sc.probe)
        with torch.no_grad():
            intensity = _fwd(cfg, obj.detach(), probe.detach(), sc.post, sc.Q).abs().square()
            metrics['full_amplitude_loss'] = float(scaled_amplitude_loss(intensity.sqrt(), sc.sqrtIm)[0])
            if counts is not None:
                metrics['full_poisson_loss'] = float(poisson_profile_loss(intensity, counts, alpha))
        return metrics

    def save_field(dest, obj, probe, history):
        rs, cs = sc.roi
        np.savez_compressed(dest, obj_rec=obj.cpu().numpy(), probe_rec=probe.cpu().numpy(),
                            obj_gt=sc.obj, probe_gt=sc.probe, roi=[rs.start, rs.stop, cs.start, cs.stop],
                            history=json.dumps(history))

    def stage1(kind):
        torch.manual_seed(a.seed)
        net = (ShallowEncoderDecoder() if kind == 'shallow16' else Stage1UNet(kind)).to(device)
        _initialize_object_heads(net, cfg.obj_init_alpha)
        parameters = sum(p.numel() for p in net.parameters())
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr_net, weight_decay=cfg.weight_decay)
        pr = nn.Parameter(torch.from_numpy(sc.P0.real.copy()).to(device))
        pi = nn.Parameter(torch.from_numpy(sc.P0.imag.copy()).to(device))
        optp = torch.optim.Adam([pr, pi], lr=cfg.lr_probe)
        tgvs = [ObjectAmplitudeTGV(cfg, sc.pos[g], device) for g in groups]
        dest = a.outdir/kind; dest.mkdir()
        def decode(k):
            amp, phase = net(inputs[k])
            return make_field(amp[0], phase[:2])[crop,crop], F.softplus(amp[0])[crop,crop]
        def read():
            with snapshot_readout(net):
                return fuse_fields([decode(k)[0] for k in range(4)], masks).detach().clone()
        initial = read(); assert torch.allclose(initial, torch.ones_like(initial), atol=2e-7, rtol=0)
        history = [dict(update=0, seconds=0., **score(initial, torch.complex(pr,pi)))]
        states = {}; seconds = 0.
        for i in range(1, a.updates+1):
            k = (i-1)%4; sync(); start = time.perf_counter()
            opt.zero_grad(set_to_none=True); optp.zero_grad(set_to_none=True)
            obj, amp = decode(k); probe = torch.complex(pr,pi)
            data = scaled_amplitude_loss(cabs(_fwd(cfg,obj,probe,sc.post[selections[k]],sc.Q)), sc.sqrtIm[selections[k]])[0]
            regularizer = tgvs[k](amp)[0]
            loss = data+energy*cfg.tgv_amp*regularizer
            if not torch.isfinite(loss): raise FloatingPointError(kind)
            loss.backward(); opt.step(); optp.step(); sync(); seconds += time.perf_counter()-start
            del obj, amp, probe, data, regularizer, loss
            checkpoint = i == a.updates or (i == a.early_switch and kind == 'u32')
            if i%a.eval_every == 0 or checkpoint:
                obj = read(); probe = torch.complex(pr,pi).detach().clone()
                history.append(dict(update=i, seconds=seconds, **score(obj,probe)))
                print(f'{a.scene.stem} {kind} S1 {i}: {history[-1]["psnr_amp"]:.3f}/{history[-1]["psnr_phs"]:.3f} {seconds:.2f}s', flush=True)
                if checkpoint:
                    state = dict(obj=obj, probe=probe, probe_optimizer=copy.deepcopy(optp.state_dict()),
                                 history=copy.deepcopy(history), stage1_seconds=seconds, stage1_updates=i)
                    torch.save(state, dest/f'stage1_{i}.pt'); states[i] = state
                    save_field(dest/f'stage1_{i}.npz', obj, probe, history)
        save_json(dest/'stage1_history.json', history)
        return dict(name=kind, parameters=parameters, stage1_history=history, stage1_seconds=seconds), states

    def stage2(state, dest):
        base = state['obj'].detach().clone(); conditioning,_ = reconstruction_channels(base,None,ns,include_probe=False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            net = CompactResidual('cnn16_polar',ns)
        net = net.to(device)
        opt = torch.optim.Adam(net.parameters(),lr=cfg.window_stage2_lr_net,weight_decay=cfg.weight_decay)
        pr = nn.Parameter(state['probe'].real.clone()); pi = nn.Parameter(state['probe'].imag.clone())
        optp = torch.optim.Adam([pr,pi],lr=cfg.window_stage2_lr_probe)
        optp.load_state_dict(copy.deepcopy(state['probe_optimizer']))
        for group in optp.param_groups: group['lr'] = cfg.window_stage2_lr_probe
        tgv = ObjectAmplitudeTGV(cfg,sc.pos,device)
        def decode():
            raw = net(conditioning)
            return base*torch.exp(torch.complex(raw[0,crop,crop],raw[1,crop,crop]))
        def read():
            with snapshot_readout(net): return decode().detach().clone()
        assert torch.equal(read(),base)
        history = [dict(update=0,seconds=0.,**score(base,state['probe']))]; seconds = 0.
        for i in range(1,a.stage2_updates+1):
            sync(); start = time.perf_counter()
            opt.zero_grad(set_to_none=True); optp.zero_grad(set_to_none=True)
            obj = decode(); probe = torch.complex(pr,pi)
            field = _fwd(cfg,obj,probe,sc.post,sc.Q)
            data = (poisson_profile_loss(field.abs().square(),counts,alpha) if counts is not None else
                    scaled_amplitude_loss(cabs(field),sc.sqrtIm)[0])
            loss = data+energy*cfg.window_stage2_tgv*tgv(cabs(obj))[0]
            if not torch.isfinite(loss): raise FloatingPointError(str(dest))
            loss.backward(); opt.step(); optp.step(); sync(); seconds += time.perf_counter()-start
            del obj,probe,field,data,loss
            if i%a.eval_every == 0 or i == a.stage2_updates:
                obj = read(); probe = torch.complex(pr,pi).detach().clone()
                history.append(dict(update=i,seconds=seconds,**score(obj,probe)))
                print(f'{a.scene.stem} {dest.name} S2 {i}: {history[-1]["psnr_amp"]:.3f}/{history[-1]["psnr_phs"]:.3f} {seconds:.2f}s',flush=True)
        dest.mkdir(); save_json(dest/'history.json',history)
        save_field(dest/'result.npz',obj,probe,history)
        return dict(stage1_updates=state['stage1_updates'], stage1_seconds=state['stage1_seconds'],
                    stage2_seconds=seconds,total_seconds=seconds+state['stage1_seconds'],history=history,
                    result_file=str((dest/'result.npz').resolve()))

    results = []
    for kind in a.candidates:
        result, states = stage1(kind); result['stage2'] = []; results.append(result)
        save_json(a.outdir/'summary.json',dict(manifest,results=results))
        torch.cuda.empty_cache()
        if not a.stage1_only:
            for update, state in sorted(states.items()):
                result['stage2'].append(stage2(state,a.outdir/kind/f'stage2_from_{update}'))
                save_json(a.outdir/'summary.json',dict(manifest,results=results))
        del states; torch.cuda.empty_cache()
    assert digest(a.scene) == manifest['scene_sha256']
    print('COMPLETED stage1 architecture screen',flush=True)


if __name__ == '__main__':
    main()
