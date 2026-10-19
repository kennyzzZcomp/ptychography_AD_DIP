"""Verify the installed official PtyRAD and its single-slice optical limit.

This is a small synthetic runtime/parity check, NOT a reconstruction runner.
There is no replacement implementation or fallback when PtyRAD is absent.
"""
import hashlib
import importlib.metadata
import inspect
import json
from datetime import datetime
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent


def relative_error(value, reference):
    return float((value-reference).norm()/reference.norm().clamp_min(1e-30))


def main():
    try:
        import ptyrad
        from ptyrad.core.forward import multislice_forward
        from ptyrad.core.losses import CombinedLoss
        from ptyrad.core.models.ptycho import PtychoModel
    except ImportError as exc:
        raise SystemExit('Official PtyRAD or its dependencies are missing. Install ptyrad==1.0.0 '
                         'inside real_ptyrad_ad/.venv. No reconstruction was run.\n'+str(exc)) from exc
    version = importlib.metadata.version('ptyrad')
    if version != '1.0.0':
        raise SystemExit(f'Expected the recorded official release 1.0.0; found {version}')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; no silent CPU fallback')
    torch.manual_seed(0)
    n, batch = 32, 3
    amp = (.5+torch.rand(batch,1,1,n,n,device='cuda')).requires_grad_()
    phase = torch.randn_like(amp).requires_grad_()
    probe = torch.randn(batch,1,n,n,device='cuda',dtype=torch.complex64).requires_grad_()
    transfer = torch.ones(batch,n,n,device='cuda',dtype=torch.complex64)
    predicted = multislice_forward(amp, phase, probe, transfer)
    field = torch.polar(amp[:,0,0],phase[:,0,0])*probe[:,0]
    expected = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field,dim=(-2,-1)),
                                               norm='ortho'),dim=(-2,-1)).abs().square()+1e-10
    target = torch.rand_like(predicted)+.1
    actual_loss = (predicted.sqrt()-target.sqrt()).square().mean()
    expected_loss = (expected.sqrt()-target.sqrt()).square().mean()
    actual_grad = torch.autograd.grad(actual_loss,(amp,phase,probe),retain_graph=True)
    expected_grad = torch.autograd.grad(expected_loss,(amp,phase,probe))
    grad_errors = [relative_error(a,b) for a,b in zip(actual_grad,expected_grad)]
    # This explicitly checks the official normalized amplitude RMSE. It is not
    # relabeled as the MSE used in real_ptycho, and no fitted gain is introduced.
    criterion = CombinedLoss({'loss_single':dict(state=True,weight=1.,dp_pow=.5)},device='cuda')
    official_loss = criterion.get_loss_single(predicted.detach(),target)
    formula_loss = actual_loss.detach().sqrt()/target.sqrt().mean()
    intensity_error = relative_error(predicted,expected)
    normalized_loss_error = float((official_loss-formula_loss).abs())
    if intensity_error > 3e-6 or max(grad_errors) > 3e-5 or normalized_loss_error > 1e-6:
        raise AssertionError(dict(intensity_error=intensity_error,gradient_errors=grad_errors,
                                  normalized_loss_error=normalized_loss_error))
    sources = {}
    for function in (multislice_forward,CombinedLoss,PtychoModel):
        path = Path(inspect.getfile(function)).resolve()
        sources[function.__name__] = dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report = dict(python=sys.version,executable=sys.executable,prefix=sys.prefix,
                  ptyrad_version=version,ptyrad_file=str(Path(ptyrad.__file__).resolve()),
                  torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
                  single_slice_intensity_relative_error=intensity_error,
                  amplitude_phase_probe_gradient_relative_errors=grad_errors,
                  official_normalized_rmse_formula_absolute_error=normalized_loss_error,
                  official_source_files=sources,
                  scope='32x32 random-fields parity check ONLY; experimental reconstruction not run')
    out = HERE/'results'/datetime.now().strftime('runtime_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    (out/'runtime.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('OUTPUT',out)


if __name__ == '__main__':
    main()

