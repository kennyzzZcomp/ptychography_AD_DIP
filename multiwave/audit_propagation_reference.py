"""Independent direct diffraction and FFT window convergence at unchanged geometry."""
import argparse
import json
import math
from pathlib import Path
import torch
from .diagnose_resolution import setup, DiagnosticOperator, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    cfg, s = setup()
    gt = s.objects[0].real.double()
    records = {}
    ids = torch.tensor([0, 12, 24])
    with torch.no_grad():
        previous = None
        for fft in (1024, 1536, 2048):
            op = DiagnosticOperator(s, cfg, fft_size=fft, dtype=torch.float64)
            pred = op(gt, ids, 768)
            rows = {}
            if previous is not None:
                for n in (192,384,768):
                    p = (768-n)//2
                    a, b = previous[:,p:p+n,p:p+n], pred[:,p:p+n,p:p+n]
                    rows[str(n)] = dict(relative_intensity_difference=float((a-b).norm()/b.norm()),
                                        relative_amplitude_difference=float((a.sqrt()-b.sqrt()).norm()/b.sqrt().norm()))
            # Direct R-S integral on input samples, no FFT or transfer kernel reused.
            row, col = s.operator.positions[12].tolist()
            field = gt[row:row+192,col:col+192]*op.probe
            yy, xx = torch.meshgrid(torch.arange(192,dtype=torch.float64),torch.arange(192,dtype=torch.float64),indexing='ij')
            wave = op.wave(gt,torch.tensor([12]))[0]
            refs, nums = [], []
            for y in (-160.,-32.,96.,224.,352.):
                for x in (-160.,-32.,96.,224.,352.):
                    z, k = 1500.,2*math.pi/.515
                    radius = torch.sqrt((xx-x)**2+(yy-y)**2+z*z)
                    kernel = z/(2*math.pi*radius**2)*(1/radius-1j*k)*torch.exp(1j*k*(radius-z))
                    refs.append((kernel*field).sum())
                    nums.append(wave[int(y)+(fft-192)//2,int(x)+(fft-192)//2])
            ref, num = torch.stack(refs), torch.stack(nums)
            records[str(fft)] = dict(previous_fft_comparison=rows, rs_25point_relative_field_error=float((num-ref).norm()/ref.norm()))
            previous = pred
            print(fft, records[str(fft)], flush=True)
    save_json(out/'reference.json', records)


if __name__ == '__main__':
    main()
