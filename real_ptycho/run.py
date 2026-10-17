"""python real_ptycho/run.py --help; all new artifacts stay inside real_ptycho/."""
import argparse
from datetime import datetime
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core import (UNet, TGV, NLMPrior, Propagator, chunked_data, geometry,
                  preserve_buffers, select_center_grid, center_crop_field, center_pad_probe)

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def set_optimizer_lr(optimizer, lr):
    """Change the step size without resetting Adam moments or step counts."""
    for group in optimizer.param_groups:
        group['lr'] = lr


def network_input(frames, side, channels, device):
    """Pad conditioning channels only; these are never synthetic measurements."""
    x = torch.from_numpy(np.array(frames, copy=True)).to(device)[None]
    x = x/x.amax((-2,-1), keepdim=True).clamp_min(1e-12)
    pad = side-x.shape[-1]
    if pad < 0 or channels < x.shape[1]:
        raise ValueError('Network canvas/channel count is smaller than the input')
    return F.pad(x, (pad//2, pad-pad//2, pad//2, pad-pad//2, 0, channels-x.shape[1]))


def arguments(argv=None):
    p = argparse.ArgumentParser(description='Real ptydata_33: U16+TGV -> residual U32+NLM (no stage-2 TGV).')
    p.add_argument('--data', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--num-frames', type=int, choices=[100, 200, 225], default=225, help='225 full 15x15 (default); 200 nearest center; 100 central 10x10')
    p.add_argument('--prepare-only', action='store_true', help='Select/cache requested frames and inspect geometry; no training')
    p.add_argument('--smoke', action='store_true', help='Explicit diagnostic: crop max(256, probe-size), 4+2 updates; NOT final reconstruction')
    p.add_argument('--detector-crop', type=int, default=0, help='0 retains all 2000 pixels; positive value is an explicit center crop, NOT resize/binning')
    p.add_argument('--probe-size', type=int, default=512, help='Optimized central probe side, zero-padded to detector size; 0 restores full-size probe')
    p.add_argument('--wavelength-nm', type=float, default=632.8)
    p.add_argument('--distance-mm', type=float, default=6.)
    p.add_argument('--pixel-um', type=float, default=3.76)
    p.add_argument('--position-order', choices=['yx', 'xy'], default='yx', help='Assumed mapping of the two stored position coordinates to detector axes')
    p.add_argument('--flip-y', action='store_true')
    p.add_argument('--flip-x', action='store_true')
    p.add_argument('--stage1-steps', type=int, default=500)
    p.add_argument('--stage2-steps', type=int, default=500)
    p.add_argument('--lr-net', type=float, default=.0005, help='Legacy fallback for BOTH stages; stage-specific options override it')
    p.add_argument('--lr-probe', type=float, default=.01, help='Legacy fallback for BOTH stages; stage-specific options override it')
    p.add_argument('--stage1-lr-net', type=float, default=None, help='Stage-1 network LR (default: --lr-net, 0.0005)')
    p.add_argument('--stage1-lr-probe', type=float, default=None, help='Stage-1 probe LR (default: --lr-probe, 0.01)')
    p.add_argument('--stage2-lr-net', type=float, default=None, help='Stage-2 network LR (default: --lr-net, 0.0005)')
    p.add_argument('--stage2-lr-probe', type=float, default=None, help='Stage-2 probe LR (default: --lr-probe, 0.01); retains probe Adam state')
    p.add_argument('--tgv', type=float, default=.01, help='Stage 1 ONLY; stage 2 has NO TGV')
    p.add_argument('--nlm-ratio', type=float, default=.5, help='Initial prior/data gradient-norm ratio, NOT literal penalty coefficient')
    p.add_argument('--nlm-refresh', type=int, default=25)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--chunk', type=int, default=1, help='Frames per FFT; full stage-2 gradient still uses ALL selected frames')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-checkpoint', action='store_true', help='Disable activation checkpointing; increases GPU memory')
    a = p.parse_args(argv)
    for stage in (1, 2):
        for kind in ('net', 'probe'):
            key = f'stage{stage}_lr_{kind}'
            if getattr(a, key) is None:
                setattr(a, key, getattr(a, f'lr_{kind}'))
    for key in ('lr_net', 'lr_probe', 'stage1_lr_net', 'stage1_lr_probe', 'stage2_lr_net', 'stage2_lr_probe'):
        if not math.isfinite(getattr(a, key)) or getattr(a, key) <= 0:
            p.error(f'--{key.replace("_", "-")} must be finite and positive')
    if a.data is None:
        candidates = [HERE/'data/ptydata_33.h5', HERE.parent/'experiment/ptydata_33.h5']
        a.data = next((x for x in candidates if x.is_file()), candidates[0])
    if not a.data.is_file():
        p.error(f'Data missing: {a.data}\nPut ptydata_33.h5 in {HERE / "data"}, or pass --data PATH.')
    if a.smoke:
        a.detector_crop = a.detector_crop or max(256, a.probe_size)
        a.stage1_steps, a.stage2_steps, a.eval_every = 4, 2, 1
    if a.stage1_steps < 4 or a.stage1_steps % 4 or a.stage2_steps < 1:
        p.error('Stage 1 must be a positive multiple of 4; stage 2 must be positive')
    if min(a.chunk, a.eval_every, a.nlm_refresh) < 1 or min(a.tgv, a.nlm_ratio) < 0:
        p.error('Invalid chunk/interval/regularizer strength')
    if min(a.wavelength_nm, a.distance_mm, a.pixel_um, a.lr_net, a.lr_probe) <= 0:
        p.error('Physical scales and learning rates must be positive')
    if a.detector_crop < 0:
        p.error('--detector-crop must be zero or positive')
    if a.probe_size < 0 or a.probe_size % 2 or a.probe_size > (a.detector_crop or 2000):
        p.error('--probe-size must be 0 (full size) or a positive even size no larger than the detector')
    a.out = a.out or HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if a.out.exists():
        p.error('Output already exists; choose a fresh --out (no overwriting)')
    if not a.prepare_only and a.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA unavailable. In Colab select a GPU runtime; CPU is only recommended for --smoke.')
    return a


def prepare(a):
    a.out.mkdir(parents=True)
    with h5py.File(a.data, 'r') as f:
        frames = f['ptychogramx']
        if frames.shape != (225, 2000, 2000):
            raise ValueError(f'Expected (225,2000,2000) intensities, got {frames.shape}')
        # 原始坐标单位就是 mm；不乘10，也不改写原H5。
        positions_mm = f['positionx'][()].astype(np.float64)
        ids, coords, groups, selection = select_center_grid(positions_mm, a.num_frames)
        n = a.detector_crop or frames.shape[-1]
        if n < 32 or n > frames.shape[-1] or n % 2:
            raise ValueError('Detector crop must be even, between 32 and 2000')
        low = (frames.shape[-1]-n)//2
        cache = np.lib.format.open_memmap(a.out/'intensity_cache.npy', mode='w+', dtype=np.float32, shape=(len(ids),n,n))
        source_hash = hashlib.sha256()
        source_hash.update(coords.tobytes()); source_hash.update(ids.tobytes())
        total, maximum = 0., 0.
        for j, original in enumerate(ids):
            data = np.asarray(frames[int(original), low:low+n, low:low+n], dtype=np.float32)
            if not np.isfinite(data).all() or (data < 0).any():
                raise ValueError(f'Frame {original} contains negative/nonfinite intensity; preprocessing must be clarified')
            if not data.max() > 0:
                raise ValueError(f'Frame {original} is all zero')
            cache[j] = data
            source_hash.update(data.tobytes())
            total += data.sum(dtype=np.float64)
            maximum = max(maximum, float(data.max()))
        scale = total/cache.size
        for j in range(len(ids)):
            cache[j] /= scale  # ONE common intensity scale, never individual exposure normalization.
        cache.flush()
    pos, side, dx = geometry(coords, n, a.wavelength_nm, a.distance_mm, a.pixel_um,
                             a.position_order, a.flip_y, a.flip_x)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(selection=selection, selected_frame_ids_zero_based=ids.tolist(),
                  positions_mm=coords.tolist(), positions_object_pixels=pos.tolist(),
                  groups=[g.tolist() for g in groups], stage1_input_channels=max(map(len, groups)),
                  detector_size=n, object_size=side, position_input_unit='mm', position_scale_to_mm=1.0,
                  probe_parameter_size=a.probe_size or n, probe_padding_each_side=(n-(a.probe_size or n))//2,
                  object_pixel_um=dx*1e6, original_intensity_mean=scale, original_intensity_max=maximum,
                  selected_input_sha256=source_hash.hexdigest(), torch_version=torch.__version__,
                  stage1='Shared U-Net16, sequential ABCD (one group/update), amplitude TGV',
                  stage2=f'U-Net32 complex additive residual, all {len(ids)} frames/update, NLM only, TGV=0',
                  probe='Central complex pixels only; same-pitch zero pad before multiplication/FFT; normalized full inverse-FFT initialization then center crop; Adam state continued',
                  loss=f'Global gain-profiled amplitude MSE per group (S1) / all {len(ids)} (S2)',
                  propagation='Centered unitary Fraunhofer FFT, bilinear object sampling at measured subpixel coordinates',
                  regularizer_domain='Union of computational probe footprints, NOT a known illuminated support',
                  caveats=['Requested Fraunhofer model; physical validity at 6 mm has NOT been established.',
                           'Position-axis mapping yx is an assumption unless independently verified.',
                           '3.76 um pitch assumes center crop only, without binning or resize.',
                           'No ground truth, no PSNR; lower data loss does not establish better reconstruction.',
                           'NLM is an exploratory prior transferred from simulation, not validated for this dataset.',
                           'Training timing includes two-pass chunking, input transfers and NLM refresh; excludes evaluation/setup/save.'])
    config['source_sha256'] = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                               for name in ('run.py', 'core.py')}
    write_json(a.out/'config.json', config)
    with (a.out/'selected_frames.csv').open('w', encoding='utf-8') as out:
        out.write('selected_index,original_frame_zero_based,row,col,group,coordinate0_mm,coordinate1_mm\n')
        for j, original in enumerate(ids):
            r, c = selection['grid_rows_zero_based'][j], selection['grid_cols_zero_based'][j]
            group = next(k for k, g in enumerate(groups) if j in g)
            out.write(f'{j},{original},{r},{c},{"ABCD"[group]},{coords[j,0]:.9f},{coords[j,1]:.9f}\n')
    print(f'[data] selected {len(ids)} frames, each {n}x{n}; object {side}x{side}; dx={dx*1e6:.6f} um', flush=True)
    print(f'[probe] optimize {config["probe_parameter_size"]}x{config["probe_parameter_size"]}; '
          f'zero-pad {config["probe_padding_each_side"]} pixels each side to {n}x{n}; coordinates read as mm', flush=True)
    print(f'[selection] rule={selection["selection_rule"]}; ABCD sizes={selection["group_sizes"]}; '
          f'network input channels={config["stage1_input_channels"]} (zero-pad missing channels only)', flush=True)
    print('[warning] Fraunhofer and coordinate-axis assumptions require experimental validation.', flush=True)
    if a.smoke:
        print('[SMOKE ONLY] Cropped detector and 4+2 updates: pipeline check, NOT reconstruction quality.', flush=True)
    plot_setup(a.out, cache, coords, groups)
    return cache, pos, side, groups, config


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def plot_setup(dest, data, coords, groups):
    plt = plotting()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
    for k, group in enumerate(groups):
        axs[0].scatter(coords[group,1], coords[group,0], s=25, c=colors[k], label='ABCD'[k])
    axs[0].set(xlabel='Stored coordinate 1 (mm)', ylabel='Stored coordinate 0 (mm)', title=f'Selected {len(coords)} scan positions')
    axs[0].set_aspect('equal'); axs[0].legend(ncol=4, fontsize=8, loc='upper center', bbox_to_anchor=(.5,-.18), frameon=False)
    mean = np.zeros(data.shape[1:], dtype=np.float64)
    for frame in data:
        mean += frame/len(data)
    im = axs[1].imshow(np.log1p(mean), cmap='magma')
    axs[1].set_title('Mean measurement: log(1 + normalized intensity)')
    axs[1].set_axis_off(); fig.colorbar(im, ax=axs[1], shrink=.8)
    fig.savefig(dest/'input_preview.png', dpi=300); plt.close(fig)


def plot_results(dest, s1, s2, probe, history, pixel_um, smoke=False):
    plt = plotting()
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
    for col, (name, field) in enumerate((('Stage 1', s1), ('Stage 2', s2), ('Probe (stage 2)', probe))):
        extent = [0, field.shape[1]*pixel_um, field.shape[0]*pixel_um, 0]
        amp = np.abs(field)
        # Display scale only; raw arrays are saved without this percentile clipping.
        im = axs[0,col].imshow(amp, cmap='gray', vmin=0, vmax=max(float(np.percentile(amp, 99.5)), 1e-12), extent=extent)
        axs[0,col].set_title(name+' amplitude'); fig.colorbar(im, ax=axs[0,col], shrink=.7)
        im = axs[1,col].imshow(np.angle(field), cmap='twilight', vmin=-np.pi, vmax=np.pi, extent=extent)
        axs[1,col].set_title(name+' phase (rad)'); fig.colorbar(im, ax=axs[1,col], shrink=.7)
        for row in range(2):
            axs[row,col].set(xlabel='x (um)', ylabel='y (um)')
    fig.suptitle('SMOKE TEST ONLY: cropped detector, 4 + 2 updates' if smoke else 'Real-data reconstruction (no ground truth)')
    fig.savefig(dest/'reconstruction.png', dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,4), layout='constrained')
    for stage, color in ((1, '#0072B2'), (2, '#D55E00')):
        rows = [h for h in history if h['stage']==stage]
        ax.plot([h['total_update'] for h in rows], [h['relative_amplitude_rmse'] for h in rows],
                'o-', color=color, label=f'Stage {stage}', markersize=3)
    ax.set(xlabel='Optimizer update (iteration)', ylabel='Full-data relative amplitude RMSE',
           title='Measurement fit, not image quality'); ax.legend(); ax.grid(alpha=.15)
    fig.savefig(dest/'loss_curve.png', dpi=300); fig.savefig(dest/'loss_curve.pdf'); plt.close(fig)


def train(a, data, positions, side, groups, config):
    device = torch.device(a.device)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    n = data.shape[-1]
    prop = Propagator(positions, n)
    masks = [prop.mask(g, side, device) for g in groups]
    count = sum(m.float() for m in masks)
    union = count > 0
    mean = np.zeros((n,n), dtype=np.float64)
    for frame in data:
        mean += frame/len(data)
    spectrum = torch.from_numpy(np.sqrt(mean).astype(np.float32)).to(device)
    initial_probe = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(spectrum), norm='ortho'))
    initial_probe /= initial_probe.abs().square().mean().sqrt().clamp_min(1e-12)
    # Preserve the previous initialization values inside the retained region.
    # Do not renormalize using the smaller window (which would change its scale).
    probe_size = a.probe_size or n
    initial_probe = center_crop_field(initial_probe, probe_size)
    pr, pi = nn.Parameter(initial_probe.real.clone()), nn.Parameter(initial_probe.imag.clone())
    optp = torch.optim.Adam([pr, pi], lr=a.stage1_lr_probe)
    del initial_probe, spectrum

    def full_probe():
        return center_pad_probe(torch.complex(pr, pi), n)

    input_channels = max(map(len, groups))
    net = UNet(input_channels, 16, checkpoint_blocks=not a.no_checkpoint).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=a.stage1_lr_net)
    print(f'[S1 LR] network={a.stage1_lr_net:g}, probe={a.stage1_lr_probe:g}', flush=True)
    tgvs = [TGV(m) for m in masks] if a.tgv else None
    def input_group(k):
        return network_input(data[groups[k]], side, input_channels, device)

    def read_stage1():
        with preserve_buffers(net), torch.no_grad():
            total = torch.zeros((side,side), dtype=torch.complex64, device=device)
            for k in range(4):
                total += net(input_group(k))*masks[k]
            return torch.where(union, total/count.clamp_min(1), torch.ones_like(total))

    history = []
    all_ids = np.arange(len(data))
    config['hardware'] = torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'
    write_json(a.out/'config.json', config)
    stage_seconds = {1: 0., 2: 0.}

    def evaluate(obj, stage, step):
        _, _, stats = chunked_data(obj, full_probe(), all_ids, data, prop, a.chunk, False)
        item = dict(stage=stage, stage_update=step, total_update=step+(a.stage1_steps if stage==2 else 0),
                    stage_training_seconds=stage_seconds[stage], **stats)
        history.append(item); write_json(a.out/'history.json', history)
        print(f'S{stage} {step}: full amplitude rRMSE={stats["relative_amplitude_rmse"]:.6f}, '
              f'train={stage_seconds[stage]:.2f}s', flush=True)

    def update(obj, go, gp, prior):
        if prior is not None:
            go = go+torch.autograd.grad(prior, obj, retain_graph=True)[0]
        if not torch.isfinite(go).all():
            raise FloatingPointError('Non-finite regularized object gradient')
        obj.backward(go)
        # chunked_data differentiates a detached full-size probe: apply pad's adjoint.
        compact_gp = center_crop_field(gp, probe_size)
        pr.grad, pi.grad = compact_gp.real.contiguous(), compact_gp.imag.contiguous()
        opt.step(); optp.step()

    initial = read_stage1()
    evaluate(initial, 1, 0); del initial
    for step in range(1, a.stage1_steps+1):
        k = (step-1)%4
        sync(device); start = time.perf_counter()
        opt.zero_grad(set_to_none=True); optp.zero_grad(set_to_none=True)
        obj = net(input_group(k))
        go, gp, _ = chunked_data(obj, full_probe(), groups[k], data, prop, a.chunk)
        reg = a.tgv*tgvs[k](obj) if tgvs else None  # Dataset intensity mean was normalized to one.
        update(obj, go, gp, reg)
        del obj, go, gp, reg
        sync(device); stage_seconds[1] += time.perf_counter()-start
        if step % a.eval_every == 0 or step == a.stage1_steps:
            evaluate(read_stage1(), 1, step)
    base = read_stage1().detach()
    s1_cpu = base.cpu().numpy().copy()
    np.savez(a.out/'stage1.npz', obj_rec=s1_cpu, probe_rec=full_probe().detach().cpu().numpy(),
             probe_compact=torch.complex(pr,pi).detach().cpu().numpy(),
             positions_px=positions, original_frame_ids=config['selected_frame_ids_zero_based'])
    torch.save(dict(network=net.state_dict(), probe_real=pr.detach(), probe_imag=pi.detach(),
                    probe_optimizer=optp.state_dict(), config=config), a.out/'stage1_weights.pt')
    del tgvs, opt, net
    gc.collect()
    if device.type=='cuda':
        torch.cuda.empty_cache()

    setup_start = time.perf_counter()
    amp = base.abs()
    conditioning = torch.stack((amp/amp.max().clamp_min(1e-12), base.real/amp.clamp_min(1e-12),
                                base.imag/amp.clamp_min(1e-12)))[None].detach()
    del amp
    torch.manual_seed(a.seed)
    net = UNet(3, 32, residual=True, checkpoint_blocks=not a.no_checkpoint).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=a.stage2_lr_net)
    set_optimizer_lr(optp, a.stage2_lr_probe)
    print(f'[S2 LR] network={a.stage2_lr_net:g}, probe={a.stage2_lr_probe:g}; probe Adam state retained', flush=True)
    coverage = prop.coverage(full_probe().detach(), side)
    prior = NLMPrior(base, union, coverage) if a.nlm_ratio else None
    coefficient, sigma = 0., None
    if prior:
        trial = base.detach().requires_grad_(True)
        gd, _, _ = chunked_data(trial, full_probe(), all_ids, data, prop, a.chunk)
        gr = torch.autograd.grad(prior(trial), trial)[0]
        dn, pn = gd[union].norm().item(), gr[union].norm().item()
        coefficient = a.nlm_ratio*dn/pn if pn > 1e-12 else 0.
        config.update(nlm_initial_data_gradient_norm=dn, nlm_initial_prior_gradient_norm=pn)
        sigma = prior.sigma
        print(f'[NLM] sigma={sigma:.6g}, calibrated lambda={coefficient:.6g}; stage-2 TGV=0', flush=True)
        if coefficient == 0:
            print('[warning] NLM calibration gave zero weight; this is recorded, not silently changed.', flush=True)
        del trial, gd, gr
    sync(device)
    config.update(nlm_sigma=sigma, nlm_coefficient=coefficient, stage2_setup_seconds=time.perf_counter()-setup_start)
    write_json(a.out/'config.json', config)
    evaluate(base, 2, 0)
    for step in range(1, a.stage2_steps+1):
        sync(device); start = time.perf_counter()
        opt.zero_grad(set_to_none=True); optp.zero_grad(set_to_none=True)
        obj = base+net(conditioning)
        if prior and step > 1 and (step-1) % a.nlm_refresh == 0:
            prior.refresh(obj)
        go, gp, _ = chunked_data(obj, full_probe(), all_ids, data, prop, a.chunk)
        reg = coefficient*prior(obj) if prior else None
        update(obj, go, gp, reg)
        del obj, go, gp, reg
        sync(device); stage_seconds[2] += time.perf_counter()-start
        if step % a.eval_every == 0 or step == a.stage2_steps:
            with preserve_buffers(net), torch.no_grad():
                evaluate(base+net(conditioning), 2, step)
    with preserve_buffers(net), torch.no_grad():
        final = (base+net(conditioning)).cpu().numpy()
    probe = full_probe().detach().cpu().numpy()
    np.savez(a.out/'reconstruction.npz', obj_rec=final, obj_stage1=s1_cpu, probe_rec=probe,
             probe_compact=torch.complex(pr,pi).detach().cpu().numpy(),
             positions_px=positions, original_frame_ids=config['selected_frame_ids_zero_based'],
             object_pixel_um=config['object_pixel_um'], config=json.dumps(config))
    torch.save(dict(network=net.state_dict(), probe_real=pr.detach(), probe_imag=pi.detach(),
                    probe_optimizer=optp.state_dict(), config=config), a.out/'stage2_weights.pt')
    plot_results(a.out, s1_cpu, final, probe, history, config['object_pixel_um'], a.smoke)
    write_json(a.out/'summary.json', dict(stage1_seconds=stage_seconds[1], stage2_seconds=stage_seconds[2],
        final_relative_amplitude_rmse=history[-1]['relative_amplitude_rmse'], smoke_only=a.smoke,
        warning='No ground truth; measurement fit is NOT PSNR or proof of model correctness.'))


def main():
    a = arguments()
    start = time.perf_counter()
    try:
        data, positions, side, groups, config = prepare(a)
        if not a.prepare_only:
            train(a, data, positions, side, groups, config)
        print(f'Done: {a.out.resolve()} | elapsed {time.perf_counter()-start:.2f}s', flush=True)
    except Exception as exc:
        if a.out.is_dir():
            write_json(a.out/'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
