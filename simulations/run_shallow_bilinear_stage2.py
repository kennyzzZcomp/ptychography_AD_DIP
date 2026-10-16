"""Isolated, explicitly requested stage-2 experiment; no production changes.

Same saved U16 O1/P1/probe Adam states, 500 updates, and existing loss/TGV.
The optional HF branch subtracts a learned bounded fraction of O1-Gaussian(O1).
It is a hypothesis, not a validated denoiser or reproduction of the ICCV paper.
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

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulations.run_two_stage_ablation import (
    WindowCfg, load_scene, cabs, _fwd, reconstruction_channels,
    scaled_amplitude_loss, ObjectAmplitudeTGV, evaluate, probe_relerr,
    sync, save_json, poisson_profile_loss)


class ShallowBilinear(nn.Module):
    def __init__(self, highpass=False):
        super().__init__()
        self.highpass = highpass
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.LeakyReLU(.1))
        self.enc0 = block(3, 16)
        self.enc1 = block(16, 32)
        self.enc2 = block(32, 32)
        self.dec = block(32, 16)
        self.head = nn.Conv2d(16, 3 if highpass else 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if highpass:
            self.strength = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        shape = x.shape[-2:]
        x = self.enc0(x)
        x = self.enc1(F.avg_pool2d(x, 2))
        half = x.shape[-2:]
        x = self.enc2(F.avg_pool2d(x, 2))
        x = self.dec(F.interpolate(x, size=half, mode='bilinear', align_corners=False))
        x = F.interpolate(x, size=shape, mode='bilinear', align_corners=False)
        return self.head(x)[0]

    @torch.no_grad()
    def project(self):
        if self.highpass:
            self.strength.clamp_(0., 1.)


def complex_highpass(base):
    """Fixed sigma=1 pixel Gaussian, radius=3, reflect boundary; no angle blur."""
    q = torch.arange(-3, 4, device=base.device, dtype=base.real.dtype)
    k = torch.exp(-q.square()/2)
    k = k/k.sum()
    kernel = (k[:, None]*k[None, :])[None, None].repeat(2, 1, 1, 1)
    x = torch.stack((base.real, base.imag))[None]
    smooth = F.conv2d(F.pad(x, (3, 3, 3, 3), mode='reflect'), kernel, groups=2)[0]
    return (base-torch.complex(smooth[0], smooth[1])).detach()


def decode(net, inputs, base, hf, crop):
    raw = net(inputs)[:, crop, crop]
    gate = net.strength*torch.sigmoid(raw[2]) if net.highpass else torch.zeros_like(base.real)
    corrected = base-gate*hf if net.highpass else base
    obj = corrected*torch.exp(torch.complex(raw[0], raw[1]))
    return obj, cabs(obj), gate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def self_test():
    torch.manual_seed(0)
    base = torch.complex(torch.rand(28, 28)+.2, torch.rand(28, 28)*.1)
    inputs = torch.rand(1, 3, 32, 32)
    hf = complex_highpass(base)
    for flag, expected in [(False, 18994), (True, 19012)]:
        net = ShallowBilinear(flag)
        assert sum(p.numel() for p in net.parameters()) == expected
        obj, _, gate = decode(net, inputs, base, hf, slice(2, 30))
        assert torch.equal(obj, base) and torch.count_nonzero(gate) == 0
        loss = (obj.real-.6).square().mean()+obj.imag.square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        assert net.head.weight.grad.abs().sum() > 0
        if flag:
            assert net.strength.grad.abs() > 0
            with torch.no_grad(): net.strength.fill_(1.2)
            net.project()
            _, _, gate = decode(net, inputs, base, hf, slice(2, 30))
            assert gate.min() >= 0 and gate.max() <= 1
        print(f'self-test highpass={flag}: zero residual, finite gradients, params={expected}', flush=True)


def run_dose(args, dose):
    source_dir = args.source_root/dose
    source = json.loads((source_dir/'summary.json').read_text(encoding='utf-8'))
    cfg = WindowCfg(**{k: v for k, v in source['cfg'].items()
                      if k in {f.name for f in fields(WindowCfg) if f.init}})
    assert source['stage1_updates'] == 500
    assert cfg.window_stage2_lr_net == .0005 and cfg.window_stage2_lr_probe == .01
    assert cfg.window_stage2_tgv == .005
    checkpoint = source_dir/'u16/stage1_500.pt'
    scene = Path(source['scene'])
    original_hashes = (digest(checkpoint), digest(scene))
    assert original_hashes[1] == source['scene_sha256']
    device = torch.device('cuda')
    sc, scene_cfg, forward_error = load_scene(scene, device)
    assert sc.Imt.shape[0] == 100 and cfg.grid == 10
    assert scene_cfg['noise'] == cfg.noise
    for k in ('N', 'grid', 'step_px', 'obj_size', 'wlength', 'det_pixel', 'z'):
        assert getattr(cfg, k) == scene_cfg[k]
    counts = None
    alpha = None
    if cfg.noise != 'none':
        with h5py.File(scene, 'r') as f:
            counts = torch.as_tensor(f['measurement_noise/counts'][()], dtype=torch.float32, device=device)
            alpha = float(f['measurement_noise'].attrs['global_alpha'])
        assert torch.allclose(counts/alpha, sc.Imt, rtol=2e-6, atol=1e-7)
    pad = (cfg.net_size-cfg.obj_size)//2
    crop = slice(pad, pad+cfg.obj_size)
    energy = sc.sqrtIm.square().mean().detach()
    dest = args.outdir/dose
    dest.mkdir()
    manifest = dict(scene=str(scene), scene_sha256=original_hashes[1], checkpoint=str(checkpoint),
        checkpoint_sha256=original_hashes[0], source_summary_sha256=digest(source_dir/'summary.json'),
        cfg=source['cfg'], stage1_updates=500, stage2_updates=args.updates,
        seed=args.seed, variants=args.variants, gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__, forward_error=forward_error,
        loss='profiled amplitude' if counts is None else 'profiled Poisson KL/(2 alpha)',
        tgv=.005, network_lr=.0005, probe_lr=.01,
        probe='same P1 and inherited Adam moments; joint updates',
        timing='synchronized training only, includes gate projection; excludes setup/eval/save; S1 time reused',
        architecture='3x3 3->16 full; avgpool2; 3x3 16->32; avgpool2; 3x3 32->32; bilinear2; 3x3 32->16; bilinear to full; linear1x1 head; LeakyReLU .1; no BN or internal skip',
        formula='plain: O1*exp(da+i*dphi); hf: (O1-g*(O1-G_sigma1(O1)))*exp(da+i*dphi)',
        gate='g=s*sigmoid(coarse-upsampled gate head); s initialized0 and projected[0,1]; zero output heads; g initially0',
        caveats=['Exploratory single object/initialization and one noise realization per dose.',
                 'No GT used for training, gate, model selection, or stopping; fixed endpoints.',
                 'HF contains genuine detail as well as noise; suppression can destroy fine features.',
                 'This is a new hypothesis inspired by ICCV2023, not a paper reproduction.',
                 'Local GPU results must not be timing-compared with L4 runs.'])
    save_json(dest/'manifest.json', manifest)

    def score(obj, probe):
        values = evaluate(obj.detach().cpu().numpy()[sc.roi], sc.obj[sc.roi])
        values['probe_relerr'] = probe_relerr(probe.detach().cpu().numpy(), sc.probe)
        with torch.no_grad():
            field = _fwd(cfg, obj, probe, sc.post, sc.Q)
            values['full_amplitude_loss'] = float(scaled_amplitude_loss(cabs(field), sc.sqrtIm)[0])
            if counts is not None:
                values['full_poisson_loss'] = float(poisson_profile_loss(field.abs().square(), counts, alpha))
        return values

    results = []
    for variant in args.variants:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        assert state['stage1_updates'] == 500
        base = state['obj'].detach().clone()
        inputs, _ = reconstruction_channels(base, None, cfg.net_size, include_probe=False)
        hf = complex_highpass(base)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(args.seed)
            net = ShallowBilinear(variant == 'bilinear16_hf')
        net = net.to(device)
        opt = torch.optim.Adam(net.parameters(), lr=.0005, weight_decay=cfg.weight_decay)
        pr, pi = (nn.Parameter(state['probe'].real.clone()), nn.Parameter(state['probe'].imag.clone()))
        optp = torch.optim.Adam([pr, pi], lr=.01)
        optp.load_state_dict(copy.deepcopy(state['probe_optimizer']))
        for group in optp.param_groups: group['lr'] = .01
        tgv = ObjectAmplitudeTGV(cfg, sc.pos, device)
        def read():
            with torch.no_grad():
                obj, _, gate = decode(net, inputs, base, hf, crop)
                return obj.detach().clone(), torch.complex(pr, pi).detach().clone(), gate.detach().clone()
        obj, probe, gate = read()
        assert torch.equal(obj, base) and torch.equal(probe, state['probe'])
        history = [dict(update=0, seconds=0., **score(obj, probe))]
        for k in ('psnr_amp', 'psnr_phs', 'relerr'):
            assert np.isclose(history[0][k], state['history'][-1][k], atol=1e-5, rtol=1e-5)
        seconds = 0.
        for i in range(1, args.updates+1):
            sync(); start = time.perf_counter()
            opt.zero_grad(set_to_none=True); optp.zero_grad(set_to_none=True)
            obj, amp, gate = decode(net, inputs, base, hf, crop)
            field = _fwd(cfg, obj, torch.complex(pr, pi), sc.post, sc.Q)
            data = (poisson_profile_loss(field.abs().square(), counts, alpha) if counts is not None
                    else scaled_amplitude_loss(cabs(field), sc.sqrtIm)[0])
            loss = data+energy*.005*tgv(amp)[0]
            if not torch.isfinite(loss): raise FloatingPointError((dose, variant, i))
            loss.backward(); opt.step(); optp.step(); net.project(); sync()
            seconds += time.perf_counter()-start
            del obj, amp, gate, field, data, loss
            if i % args.eval_every == 0 or i == args.updates:
                obj, probe, gate = read()
                row = dict(update=i, seconds=seconds, **score(obj, probe),
                           gate_mean_roi=float(gate[sc.roi].mean()), gate_max_roi=float(gate[sc.roi].max()),
                           gate_strength=float(net.strength) if net.highpass else 0.)
                history.append(row)
                print(f'{dose} {variant} {i}: {row["psnr_amp"]:.3f}/{row["psnr_phs"]:.3f}, {seconds:.2f}s; gate={row["gate_mean_roi"]:.4f}', flush=True)
        variant_dir = dest/variant
        variant_dir.mkdir()
        rs, cs = sc.roi
        np.savez_compressed(variant_dir/'result.npz', obj_rec=obj.cpu().numpy(), probe_rec=probe.cpu().numpy(),
            obj_gt=sc.obj, probe_gt=sc.probe, obj_stage1=base.cpu().numpy(), gate=gate.cpu().numpy(),
            roi=[rs.start, rs.stop, cs.start, cs.stop], history=json.dumps(history), metadata=json.dumps(manifest))
        with np.load(variant_dir/'result.npz', allow_pickle=False) as z:
            check = evaluate(z['obj_rec'][sc.roi], z['obj_gt'][sc.roi])
        assert all(np.isclose(check[k], history[-1][k], atol=1e-5, rtol=1e-5) for k in ('psnr_amp','psnr_phs','relerr'))
        torch.save(dict(network=net.state_dict(), optimizer=opt.state_dict(), probe=torch.complex(pr,pi).detach(),
                        probe_optimizer=optp.state_dict()), variant_dir/'final_state.pt')
        save_json(variant_dir/'history.json', history)
        results.append(dict(variant=variant, parameters=sum(p.numel() for p in net.parameters()),
            initial=history[0], final=history[-1], stage1_seconds=state['stage1_seconds'], stage2_seconds=seconds,
            total_seconds=state['stage1_seconds']+seconds, result_file=str((variant_dir/'result.npz').resolve())))
        save_json(dest/'summary.json', dict(manifest, results=results))
        del net, opt, optp, pr, pi, state, base, inputs, hf, obj, probe, gate, tgv
        torch.cuda.empty_cache()
    assert original_hashes == (digest(checkpoint), digest(scene))
    return dict(dose=dose, results=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=ROOT/'output/light_stage1_search_20261012')
    p.add_argument('--outdir', type=Path, default=ROOT/'output/shallow_bilinear_stage2_20261013')
    p.add_argument('--doses', nargs='+', choices=['clean','1e5','1e4'], default=['clean','1e5','1e4'])
    p.add_argument('--variants', nargs='+', choices=['bilinear16','bilinear16_hf'], default=['bilinear16','bilinear16_hf'])
    p.add_argument('--updates', type=int, default=500)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test(); return
    if args.outdir.exists() or args.updates < 1 or args.eval_every < 1:
        p.error('Use a NEW output directory and positive update/eval counts')
    if len(set(args.doses)) != len(args.doses) or len(set(args.variants)) != len(args.variants):
        p.error('Duplicate conditions are not allowed')
    args.outdir.mkdir(parents=True)
    shutil.copyfile(__file__, args.outdir/'runner_snapshot.py')
    torch.backends.cudnn.benchmark = True
    results = []
    for dose in args.doses:
        results.append(run_dose(args, dose))
        save_json(args.outdir/'all_results.json', results)
    print('DONE', args.outdir, flush=True)


if __name__ == '__main__':
    main()
