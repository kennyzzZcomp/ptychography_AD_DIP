"""Bounded paired test using the historical NPZ config, not launcher defaults.

Only the amplitude TGV auxiliary and its Adam are reset. Never reset probe Adam.
Both runs use this GPU and deterministic algorithms; historical Colab scores are
context, NOT the local paired control. Output must be a new directory.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from simulations.ProPtyNet_paper import Cfg
from functions.paperrepro.solvers_addip import run_net


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=False)
    with np.load(args.reference, allow_pickle=False) as data:
        config = json.loads(data['cfg'].item())
    if (config['stage2_input'] != 'object' or config['stage2_network'] != 'fresh'
            or config['measurement_schedule'] != '0:3,1000:1' or config['iters'] != 2000):
        raise ValueError('Expected historical 1000+1000 fresh-object progressive configuration')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    summary = {'reference': args.reference, 'gpu': torch.cuda.get_device_name(0),
               'deterministic_algorithms': True, 'runs': {}}
    for reset in (False, True):
        name = 'reset' if reset else 'retain'
        out = root / name
        out.mkdir()
        cfg = Cfg(**(config | {'outdir': str(out), 'device': 'cuda', 'reset_tgv_at_switch': reset}))
        with (out/'run.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
            hist = run_net(cfg)
        summary['runs'][name] = {'final': hist[-1], 'stage1': next(r for r in hist if r['it'] == 1000)}
        (root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        torch.cuda.empty_cache()
    with np.load(root/'retain'/'stage1_conditioning.npz') as a, np.load(root/'reset'/'stage1_conditioning.npz') as b:
        summary['identical_stage1_snapshot'] = {k: bool(np.array_equal(a[k], b[k])) for k in a.files}
    summary['reset_minus_retain_psnr'] = (summary['runs']['reset']['final']['psnr_amp']
                                         - summary['runs']['retain']['final']['psnr_amp'])
    (root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
