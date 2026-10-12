"""Matched U16 checkpoint -> full-data pixel AD + TGV, without rerunning stage1.

An explicitly requested fixed-update diagnostic, not a tuned AD optimum.
Uses the same losses, probe Adam state and scoring as the recent U16 -> U32 runs.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulations.run_two_stage_ablation import (
    WindowCfg, load_scene, cabs, _fwd, scaled_amplitude_loss,
    ObjectAmplitudeTGV, evaluate, probe_relerr, sync, save_json,
    poisson_profile_loss,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(source_dir, control_dir, dest, updates, eval_every):
    source = json.loads((source_dir / 'summary.json').read_text(encoding='utf-8'))
    control = json.loads((control_dir / 'summary.json').read_text(encoding='utf-8'))
    checkpoint = source_dir / 'u16' / 'stage1_500.pt'
    checkpoint_hash = digest(checkpoint)
    assert checkpoint_hash == control['checkpoint_sha256']['u16']
    assert digest(source['scene']) == source['scene_sha256'] == control['scene_sha256']
    cfg = WindowCfg(**{k: v for k, v in source['cfg'].items()
                       if k in {f.name for f in fields(WindowCfg) if f.init}})
    assert source['stage1_updates'] == 500 and cfg.grid == 10
    assert cfg.window_stage2_lr_net == .0005 and cfg.window_stage2_lr_probe == .01
    assert cfg.window_stage2_tgv == .005
    device = torch.device('cuda')
    sc, scene_cfg, forward_error = load_scene(Path(source['scene']), device)
    for key in ('N', 'grid', 'step_px', 'obj_size', 'wlength', 'det_pixel', 'z', 'noise'):
        assert getattr(cfg, key) == scene_cfg[key], key
    assert len(sc.pos) == 100
    loss_kind = 'amplitude' if cfg.noise == 'none' else 'poisson'
    assert loss_kind == control['data_loss']
    counts = None
    alpha = None
    if loss_kind == 'poisson':
        with h5py.File(source['scene'], 'r') as h:
            counts = torch.as_tensor(h['measurement_noise/counts'][()], dtype=torch.float32, device=device)
            alpha = float(h['measurement_noise'].attrs['global_alpha'])
        assert torch.allclose(counts / alpha, sc.Imt, rtol=2e-6, atol=1e-7)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    assert state['stage1_updates'] == 500
    base = state['obj'].detach().clone()
    real = torch.nn.Parameter(base.real.clone())
    imag = torch.nn.Parameter(base.imag.clone())
    pr = torch.nn.Parameter(state['probe'].real.clone())
    pi = torch.nn.Parameter(state['probe'].imag.clone())
    opt = torch.optim.Adam([real, imag], lr=cfg.window_stage2_lr_net, weight_decay=cfg.weight_decay)
    optp = torch.optim.Adam([pr, pi], lr=cfg.window_stage2_lr_probe)
    optp.load_state_dict(copy.deepcopy(state['probe_optimizer']))
    for group in optp.param_groups:
        group['lr'] = cfg.window_stage2_lr_probe
    for original_group, copied_group in zip(state['probe_optimizer']['param_groups'], optp.state_dict()['param_groups']):
        for old_id, new_id in zip(original_group['params'], copied_group['params']):
            for key, expected in state['probe_optimizer']['state'][old_id].items():
                actual = optp.state_dict()['state'][new_id][key]
                assert torch.equal(expected.cpu(), actual.cpu()) if torch.is_tensor(expected) else expected == actual
    assert torch.equal(torch.complex(real, imag), base)
    assert torch.equal(torch.complex(pr, pi), state['probe'])
    tgv = ObjectAmplitudeTGV(cfg, sc.pos, device)
    energy = sc.sqrtIm.square().mean().detach()

    @torch.no_grad()
    def score(obj, probe):
        result = evaluate(obj.cpu().numpy()[sc.roi], sc.obj[sc.roi])
        result['probe_relerr'] = probe_relerr(probe.cpu().numpy(), sc.probe)
        field = _fwd(cfg, obj, probe, sc.post, sc.Q)
        result['full_amplitude_loss'] = float(scaled_amplitude_loss(cabs(field), sc.sqrtIm)[0])
        if counts is not None:
            result['full_poisson_loss'] = float(poisson_profile_loss(field.abs().square(), counts, alpha))
        return result

    manifest = dict(
        source=str(source_dir.resolve()), checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=checkpoint_hash, source_summary_sha256=digest(source_dir / 'summary.json'),
        scene=source['scene'], scene_sha256=source['scene_sha256'], cfg=source['cfg'],
        comparison_control=str(control_dir.resolve()), control_summary_sha256=digest(control_dir / 'summary.json'),
        stage1_architecture='U-Net16, ABCD', stage1_updates=500, stage2_updates=updates,
        stage2='Full-data direct real/imaginary pixel Adam, all 100 frames per update',
        pixel_lr=.0005, probe_lr=.01, tgv=.005, data_loss=loss_kind,
        loss='Same profiled amplitude loss (clean) or Poisson KL/(2 alpha) (noisy), plus energy-scaled amplitude TGV',
        optimizer='Fresh object Adam; exact inherited probe Adam moments; fresh TGV as network controls',
        timing='CUDA synchronized updates; excludes scoring/readout/setup/save; stage1 time reused',
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, forward_relative_error=forward_error,
        caveats=['Fixed endpoints, no GT stopping, no learning-rate or TGV sweep.',
                 'Same numeric learning rate does not mean matched effective step size across parameterizations.',
                 'Historical ROI evaluator retained for comparability.',
                 'One stage1 seed and one object/noise realization per dose; not a universal ranking.'])
    dest.mkdir()
    save_json(dest / 'manifest.json', manifest)
    history = [dict(update=0, seconds=0., **score(base, state['probe']))]
    net_control = next(r for r in control['results'] if r['name'] == 'u16')
    for key in ('psnr_amp', 'psnr_phs', 'relerr'):
        assert np.isclose(history[0][key], state['history'][-1][key], atol=1e-5, rtol=1e-5)
        assert np.isclose(history[0][key], net_control['stage1'][key], atol=1e-5, rtol=1e-5)
    seconds = 0.
    for i in range(1, updates + 1):
        sync()
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        optp.zero_grad(set_to_none=True)
        obj = torch.complex(real, imag)
        probe = torch.complex(pr, pi)
        field = _fwd(cfg, obj, probe, sc.post, sc.Q)
        data = (poisson_profile_loss(field.abs().square(), counts, alpha) if counts is not None else
                scaled_amplitude_loss(cabs(field), sc.sqrtIm)[0])
        loss = data + energy * cfg.window_stage2_tgv * tgv(cabs(obj))[0]
        if not torch.isfinite(loss):
            raise FloatingPointError(str(dest))
        loss.backward()
        opt.step()
        optp.step()
        sync()
        seconds += time.perf_counter() - start
        del obj, probe, field, data, loss
        if i % eval_every == 0 or i == updates:
            obj = torch.complex(real, imag).detach()
            probe = torch.complex(pr, pi).detach()
            history.append(dict(update=i, seconds=seconds, **score(obj, probe)))
            print(f'{dest.name} pixel S2 {i}: {history[-1]["psnr_amp"]:.3f}/{history[-1]["psnr_phs"]:.3f}, {seconds:.2f}s', flush=True)
    rs, cs = sc.roi
    save_json(dest / 'history.json', history)
    np.savez_compressed(dest / 'result.npz', obj_rec=obj.cpu().numpy(), probe_rec=probe.cpu().numpy(),
                        obj_gt=sc.obj, probe_gt=sc.probe, roi=[rs.start, rs.stop, cs.start, cs.stop],
                        cfg=json.dumps(source['cfg']), history=json.dumps(history), run_metadata=json.dumps(manifest))
    with np.load(dest / 'result.npz', allow_pickle=False) as z:
        rescored = evaluate(z['obj_rec'][sc.roi], z['obj_gt'][sc.roi])
    for key in ('psnr_amp', 'psnr_phs', 'relerr'):
        assert np.isclose(rescored[key], history[-1][key], atol=1e-5, rtol=1e-5)
    assert checkpoint_hash == digest(checkpoint) and source['scene_sha256'] == digest(source['scene'])
    summary = dict(manifest, stage1=history[0], final=history[-1],
                   stage1_seconds=state['stage1_seconds'], stage2_seconds=seconds,
                   total_seconds=state['stage1_seconds'] + seconds,
                   unet32_polar_control=net_control['final'], cnn16_polar_control=net_control['compact_control_final'],
                   saved_fields_rescored=True, input_files_unchanged=True)
    save_json(dest / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path('output/light_stage1_search_20261012'))
    p.add_argument('--control-root', type=Path, default=Path('output/unet16_polar32_20261012'))
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--doses', nargs='+', choices=['clean', '1e5', '1e4'], default=['clean', '1e5', '1e4'])
    p.add_argument('--updates', type=int, default=500)
    p.add_argument('--eval-every', type=int, default=50)
    a = p.parse_args()
    if a.outdir.exists() or a.updates < 1 or a.eval_every < 1:
        p.error('New output directory and positive budgets required')
    a.outdir.mkdir(parents=True)
    shutil.copyfile(__file__, a.outdir / 'runner_snapshot.py')
    torch.backends.cudnn.benchmark = True
    summaries = {}
    for dose in dict.fromkeys(a.doses):
        summaries[dose] = run(a.source_root / dose, a.control_root / dose,
                              a.outdir / dose, a.updates, a.eval_every)
        save_json(a.outdir / 'comparison.json', summaries)
        torch.cuda.empty_cache()
    lines = ['U16 ABCD stage1 (500) -> pixel AD + TGV stage2 (' + str(a.updates) + ').',
             'Amplitude / phase PSNR (dB), fixed endpoints. Training seconds exclude evaluation/setup/save.',
             'Same checkpoint hash, loss, TGV, probe Adam and evaluation as network controls.']
    for dose, r in summaries.items():
        v = r['final']
        lines.append(f"{dose}: AD {v['psnr_amp']:.3f}/{v['psnr_phs']:.3f}; "
                     f"time {r['stage1_seconds']:.2f}+{r['stage2_seconds']:.2f}={r['total_seconds']:.2f}")
    (a.outdir / 'results.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
