"""Offline resolution audit. Does not modify production geometry or defaults.

python -m multiwave.diagnose_resolution --stage audit --outdir ...
python -m multiwave.diagnose_resolution --stage fit --detector 192 --parameter direct --init flat --steps 300 --outdir ...
All comparisons retain the original 384 object / 192 probe, 1 um sampling,
scan positions, truth and physical probe. Only the recorded detector crop varies.
"""
import argparse
from dataclasses import replace, asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .config import Config
from .scene import simulate
from .physics import asm_transfer, amplitude_loss


class DiagnosticOperator:
    def __init__(self, scene, cfg, fft_size=1024, dtype=torch.float32, mode=0):
        self.scene, self.cfg, self.fft_size = scene, cfg, fft_size
        self.probe = scene.operator.probes[mode].to(torch.complex128 if dtype == torch.float64 else torch.complex64)
        self.h = asm_transfer(fft_size, cfg.pixel_um*1e-6, cfg.wavelengths_nm[mode]*1e-9,
                              cfg.distance_mm*1e-3, dtype=self.probe.dtype)

    def wave(self, amp, ids):
        r, c = self.scene.operator.rows[ids], self.scene.operator.cols[ids]
        exit_wave = amp[r[:, :, None], c[:, None, :]] * self.probe
        p = (self.fft_size-self.cfg.patch_size)//2
        return torch.fft.ifft2(torch.fft.fft2(F.pad(exit_wave, (p, p, p, p)))*self.h)

    def __call__(self, amp, ids, detector):
        wave = self.wave(amp, ids)
        p = (self.fft_size-detector)//2
        return wave[:, p:p+detector, p:p+detector].abs().square()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


class MixedDiagnosticOperator:
    def __init__(self, scene, cfg, fft_size):
        self.ops = [DiagnosticOperator(scene, cfg, fft_size=fft_size, mode=i)
                    for i in range(len(cfg.wavelengths_nm))]

    def __call__(self, amp, ids, detector):
        return sum(op(amp, ids, detector) for op in self.ops)


def setup(wavelengths=(515.,)):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = replace(Config.preset('highres'), wavelengths_nm=tuple(wavelengths), probe_mode='known', pad_factor=4)
    return cfg, simulate(cfg)


def metrics(a, gt, roi):
    e = a-gt
    center = (slice(170, 214), slice(170, 234))
    # A fixed window + identical spectral mask for every variant; no inferred resolution cutoff.
    win = torch.hann_window(116,device=a.device)[:, None]*torch.hann_window(116,device=a.device)[None, :]
    af = torch.fft.fft2(a[134:250, 134:250]*win)
    gf = torch.fft.fft2(gt[134:250, 134:250]*win)
    f = torch.fft.fftfreq(116,device=a.device)
    hf = torch.sqrt(f[:, None]**2+f[None, :]**2) >= .15
    return dict(object_error=float(e[roi].norm()/gt[roi].norm()),
                center_rmse=float(e[center].square().mean().sqrt()),
                high_frequency_error=float((af-gf)[hf].norm()/gf[hf].norm()))


def audit(out):
    cfg, s = setup()
    gt = s.objects[0].real
    op = DiagnosticOperator(s, cfg)
    result = {'config': asdict(cfg), 'fft_size': 1024,
              'transfer_retained_fraction': float((op.h.abs()>0).float().mean())}
    with torch.no_grad():
        result['truth_loss'] = {label: float(amplitude_loss(s.operator(s.objects, ids), s.measured[ids]))
                                for label, ids in [('train', s.train), ('holdout', s.holdout)]}
        same = DiagnosticOperator(s, cfg, fft_size=768)
        result['diagnostic_production_relative_difference'] = float((same(gt, s.train[:2], 192)-s.clean[s.train[:2]]).norm()/s.clean[s.train[:2]].norm())
    # Nonzero residual, double precision, local directions and multiple finite difference steps.
    od = DiagnosticOperator(s, cfg, dtype=torch.float64)
    g = gt.double()
    ids = s.train[:2]
    target = od(g, ids, 192).detach()
    x = (.8*g+.1).requires_grad_()
    loss = amplitude_loss(od(x, ids, 192), target)
    grad = torch.autograd.grad(loss, x)[0]
    direction = torch.randn_like(x)
    direction /= direction.norm()
    ad = float((grad*direction).sum())
    checks = []
    with torch.no_grad():
        for eps in (.01, .001, .0001):
            fd = float((amplitude_loss(od(x+eps*direction, ids, 192), target)-amplitude_loss(od(x-eps*direction, ids, 192), target))/(2*eps))
            checks.append(dict(epsilon=eps, autodiff=ad, finite_difference=fd, relative_error=abs(fd-ad)/max(abs(fd), abs(ad), 1e-20)))
    result['gradient_checks'] = checks
    # Independent direct Rayleigh-Sommerfeld quadrature (9 points, full original exit patch).
    oo = DiagnosticOperator(s, cfg, fft_size=1536, dtype=torch.float64)
    row, col = s.operator.positions[12].tolist()
    field = g[row:row+192, col:col+192]*oo.probe
    coords = torch.arange(192, dtype=torch.float64)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    k, z = 2*math.pi/.515, 1500.
    wave = oo.wave(g, torch.tensor([12]))[0]
    references, numerical = [], []
    for y in (48, 96, 144):
        for x0 in (48, 96, 144):
            radius = torch.sqrt((xx-x0)**2+(yy-y)**2+z*z)
            kernel = z/(2*math.pi*radius**2)*(1/radius-1j*k)*torch.exp(1j*k*(radius-z))
            references.append((kernel*field).sum())
            numerical.append(wave[y+672, x0+672])
    ref, num = torch.stack(references), torch.stack(numerical)
    result['rayleigh_sommerfeld_9point_relative_field_error'] = float((num-ref).norm()/ref.norm())
    # Equal-L2 localized sinusoidal perturbations. Centered derivative at actual truth;
    # +/- perturbations can leave [0,1], deliberately testing the forward Jacobian only.
    a = torch.arange(384)-191.5
    yy, xx = torch.meshgrid(a, a, indexing='ij')
    envelope = torch.exp(-(xx.square()+yy.square())/(2*24**2))
    entries = []
    with torch.no_grad():
        for freq in (.02, .05, .10, .15, .20, .30, .40):
            for axis, coord in [('x', xx), ('y', yy)]:
                d = envelope*torch.cos(2*math.pi*freq*coord)
                d /= d.norm()
                sums = {n: [0., 0., 0.] for n in (192, 384, 768)}
                for ids in s.train.split(2):
                    wp, wm = op.wave(gt+.01*d, ids), op.wave(gt-.01*d, ids)
                    for n in sums:
                        q = (1024-n)//2
                        ip = wp[:, q:q+n, q:q+n].abs().square()
                        im = wm[:, q:q+n, q:q+n].abs().square()
                        di = (ip-im)/.02
                        da = ((ip+1e-12).sqrt()-(im+1e-12).sqrt())/.02
                        sums[n][0] += float(di.square().sum())
                        sums[n][1] += float(da.square().sum())
                        sums[n][2] += float(((ip+im)/2).sum())
                for n, (di, da, power) in sums.items():
                    entries.append(dict(frequency=freq, axis=axis, detector=n, intensity_jacobian_norm=di**.5,
                                        amplitude_jacobian_norm=da**.5, normalized_amplitude_sensitivity=(da/power)**.5))
            print(f'sensitivity frequency {freq} done', flush=True)
    result['sensitivity'] = entries
    save_json(out/'audit.json', result)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for n in (192, 384, 768):
        rows = [r for r in entries if r['detector']==n and r['axis']=='x']
        ax.semilogy([r['frequency'] for r in rows], [r['normalized_amplitude_sensitivity'] for r in rows], 'o-', label=f'{n} um detector')
    ax.set(xlabel='Object frequency (cycles/um)', ylabel='Normalized amplitude sensitivity', title='Same probe / object / FFT / scans; detector crop only')
    ax.legend(); ax.grid(alpha=.3)
    fig.savefig(out/'sensitivity.png', dpi=160); plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ('config','sensitivity')}, indent=2), flush=True)


def fit(args, out):
    cfg, s = setup(args.wavelengths_nm)
    gt = s.objects[0].real
    op = MixedDiagnosticOperator(s, cfg, fft_size=args.fft_size)
    with torch.no_grad():
        targets = torch.cat([op(gt, ids, args.detector) for ids in torch.arange(25).split(2)])
    if args.init == 'flat':
        initial = torch.full_like(gt, math.exp(-.1))
    else:
        generator = torch.Generator().manual_seed(71)
        initial = (gt+.03*torch.randn(gt.shape, generator=generator)).clamp(.001, .999)
    if args.parameter == 'sigmoid':
        p = torch.nn.Parameter(torch.logit(initial))
        field = lambda: torch.sigmoid(p)
    else:
        p = torch.nn.Parameter(initial.clone())
        field = lambda: p
    optimizer = torch.optim.Adam([p], lr=args.lr)
    norm = targets[s.train].sum().clamp_min(1e-12)
    history = []
    start = time.perf_counter()

    def record(step):
        with torch.no_grad():
            amp = field()
            m = metrics(amp, gt, s.roi)
            for label, ids in [('train', s.train), ('holdout', s.holdout)]:
                total = 0.
                for batch in ids.split(2):
                    total += float(((op(amp, batch, args.detector)+1e-12).sqrt()-(targets[batch]+1e-12).sqrt()).square().sum())
                m[label] = (total/float(targets[ids].sum()))**.5
            history.append(dict(step=step, elapsed=time.perf_counter()-start, **m))
            print(f'{args.parameter}/{args.init} det={args.detector} {step}/{args.steps} '+str(m), flush=True)
            save_json(out/'fit.json', dict(arguments=vars(args), config=asdict(cfg), history=history))
            np.savez_compressed(out/'fields.npz', truth=gt.numpy(), reconstruction=amp.detach().numpy(), initial=initial.numpy(), roi=s.roi.numpy())
    record(0)
    for step in range(1, args.steps+1):
        optimizer.zero_grad(set_to_none=True)
        for ids in s.train.split(2):
            prediction = op(field(), ids, args.detector)
            loss = ((prediction+1e-12).sqrt()-(targets[ids]+1e-12).sqrt()).square().sum()/norm
            loss.backward()
        optimizer.step()
        if args.parameter == 'direct':
            with torch.no_grad():
                p.clamp_(0, 1)
        if step % 50 == 0 or step == args.steps:
            record(step)
    from .report import plot_usaf_detail
    plot_usaf_detail({'method':f'{args.parameter}_{args.init}_det{args.detector}', 'objects':field().detach().numpy()[None].astype(np.complex64)}, s, cfg, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['audit','fit'], required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--detector', type=int, default=192, choices=[192,384,768])
    parser.add_argument('--fft-size', type=int, default=1024)
    parser.add_argument('--parameter', choices=['sigmoid','direct'], default='sigmoid')
    parser.add_argument('--init', choices=['flat','near'], default='flat')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=.03)
    parser.add_argument('--wavelengths-nm', type=float, nargs='+', default=[515.])
    args = parser.parse_args()
    if args.fft_size < args.detector or args.fft_size % 2:
        parser.error('FFT must be even and at least detector size')
    out = Path(args.outdir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'audit':
        audit(out)
    else:
        fit(args, out)


if __name__ == '__main__':
    main()
