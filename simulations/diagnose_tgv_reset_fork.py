"""One stage-1 run, then two identical-state stage-2 branches; no GPU RNG reruns.

This intentionally uses the SAME continuation loop for both branches. Amplitude
TGV state (v and Adam) is the sole difference. No claimed performance benchmark.
"""
import argparse
import copy
import contextlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from simulations.ProPtyNet_paper import Cfg
from simulations.diagnose_tgv_reset_pair import Tee
from functions.paperrepro.solvers_addip import run_net, _fwd
from functions.paperrepro.tgv import ObjectAmplitudeTGV
from functions.paperrepro.evaluate import evaluate, probe_relerr
from functions.paperrepro.report import _save
from functions.addip.model import make_field
from functions.addip.losses import cabs


class Captured(Exception):
    pass


def continuation(cfg, state, reset, updates=1000):
    # Copy the tuple together to preserve optimizer -> parameter references.
    net, opt, pr, pi, opt_p, tgv = copy.deepcopy(tuple(state[k] for k in
        ('net', 'opt_net', 'probe_real', 'probe_imag', 'opt_probe', 'tgv')))
    x, sc = state['network_input'], state['scene']
    for group in opt.param_groups:
        group['lr'] = .002
    for group in opt_p.param_groups:
        group['lr'] = .01
    if reset:
        tgv = ObjectAmplitudeTGV(cfg, sc.pos, x.device)
    audit = dict(tgv_norm=float(tgv.v.detach().norm()),
                 tgv_adam_step=float(tgv.opt.state.get(tgv.v, {}).get('step', 0)),
                 probe_adam_steps=[float(v['step']) for v in opt_p.state.values()])
    target, post = sc.sqrtIm, sc.post
    energy = target.square().mean().detach()
    pad = (cfg.net_size-cfg.obj_size)//2
    crop = slice(pad, pad+cfg.obj_size)
    hist = []
    start = time.perf_counter()
    for step in range(updates):
        a, ph = net(x)
        obj = make_field(a[0], ph[:2])[crop, crop]
        amp = F.softplus(a[0])[crop, crop]
        probe = torch.complex(pr, pi)
        ua = cabs(_fwd(cfg, obj, probe, post, sc.Q))
        ua = ua*((ua*target).sum()/(ua*ua).sum().clamp_min(1e-20))
        data = F.mse_loss(ua, target)
        reg, _, _ = tgv(amp)
        loss = data + .0001*energy*reg
        opt.zero_grad(set_to_none=True)
        opt_p.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite continuation loss')
        opt.step()
        opt_p.step()
        if step == 0 or (step+1)%25 == 0 or step+1 == updates:
            rec, pc = obj.detach().cpu().numpy(), probe.detach().cpu().numpy()
            rs, cs = sc.roi
            row = dict(it=1000+step+1, **evaluate(rec[rs, cs], sc.obj[rs, cs]),
                       relerr_p=probe_relerr(pc, sc.probe), loss=float(loss.detach()),
                       data_loss=float(data.detach()), tgv_amp=float(reg.detach()),
                       elapsed_s=time.perf_counter()-start)
            hist.append(row)
            if (step+1)%100 == 0 or step == 0:
                print(('reset' if reset else 'retain'), json.dumps(row), flush=True)
        del obj, amp, probe, ua, data, reg, loss, a, ph
    _save(cfg, rec, pc, sc.obj, sc.probe, hist, sc.roi, sc.pos, tag='net',
          train_elapsed_s=time.perf_counter()-start)
    return dict(audit=audit, final=hist[-1], history=hist)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=False)
    with np.load(args.reference) as d:
        config = json.loads(d['cfg'].item())
    cfg = Cfg(**(config | {'outdir': str(root/'stage1'), 'device': 'cuda'}))
    Path(cfg.outdir).mkdir()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    captured = {}

    def capture(state):
        captured.update(state)
        raise Captured()

    with (root/'run.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        try:
            run_net(cfg, stage2_observer=capture)
        except Captured:
            pass
        if not captured:
            raise RuntimeError('Transition not reached')
        # Avoid benchmark selecting different deterministic convolution algorithms
        # between branches. Both start from byte-identical copied state.
        torch.backends.cudnn.benchmark = False
        torch.save({k: captured[k].state_dict() for k in ('net','opt_net','opt_probe')}
                   | {'probe_real': captured['probe_real'].detach(),
                      'probe_imag': captured['probe_imag'].detach(),
                      'tgv_vector': captured['tgv'].v.detach(),
                      'tgv_optimizer': captured['tgv'].opt.state_dict(),
                      'input': captured['network_input'], 'cfg': config}, root/'switch_checkpoint.pt')
        summary = {'shared_stage1': True, 'gpu': torch.cuda.get_device_name(0),
                   'cfg': config, 'runs': {}, 'stage1_final': captured['history'][-1]}
        for reset in (False, True):
            name = 'reset' if reset else 'retain'
            out = root/name
            out.mkdir()
            branch_cfg = Cfg(**(config | {'outdir': str(out), 'reset_tgv_at_switch': reset}))
            summary['runs'][name] = continuation(branch_cfg, captured, reset)
            (root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        summary['reset_minus_retain_psnr'] = (summary['runs']['reset']['final']['psnr_amp']
                                             - summary['runs']['retain']['final']['psnr_amp'])
        (root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print('FINAL paired delta dB:', summary['reset_minus_retain_psnr'], flush=True)


if __name__ == '__main__':
    main()
