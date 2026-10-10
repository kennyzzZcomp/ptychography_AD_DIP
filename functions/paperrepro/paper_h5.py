"""Load exported simulation data without regenerating measurements or training on truth."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from functions.paperrepro.optics import forward_ptycho, make_quad_phase


SCENE_FIELDS = (
    'wlength', 'N', 'det_pixel', 'z', 'grid', 'step_px', 'probe_diam_um',
    'obj_size', 'amp_image', 'phs_image', 'obj_phase_rad', 'probe_amp_image',
    'probe_phase_mode', 'probe_phase_rad', 'obj_amp_binary_invert', 'obj_amp_floor',
    'noise', 'snr_db', 'noise_seed', 'eval_size',
)


def read_config(path):
    with h5py.File(path, 'r') as f:
        if f.attrs.get('schema') != 'proptynet-ptyinr-v1':
            raise ValueError('Expected exported proptynet-ptyinr-v1 H5 scene')
        if f.attrs.get('fft_convention') != 'ifftshift -> fft2(norm=ortho) -> fftshift':
            raise ValueError('Unsupported FFT convention')
        if f.attrs.get('position_convention') != 'integer [row,column] patch top-left; no transpose':
            raise ValueError('Unsupported scan-coordinate convention')
        if float(f.attrs.get('diffraction_scale', 1)) != 1:
            raise ValueError('Unsupported additional diffraction scaling')
        config = json.loads(f.attrs['scene_config_json'])
        required = ('wlength', 'N', 'det_pixel', 'z', 'grid', 'step_px',
                    'probe_diam_um', 'obj_size', 'quad_sign')
        if any(k not in config for k in required):
            raise ValueError('Missing required physical scene configuration')
        if config['quad_sign'] not in (-1, 1):
            raise ValueError('Invalid quadratic-phase sign')
        return config


def load_scene(path, cfg):
    """Validate all clean intensities with the actual paper forward, then load verbatim."""
    source = read_config(path)
    device = cfg.dev()
    with h5py.File(path, 'r') as f:
        arrays = {k: f[k][()] for k in (
            'intensity', 'diffamp', 'positions_px', 'quadratic_phase', 'probe_init',
            'roi', 'diagnostic_truth/object', 'diagnostic_truth/probe',
            'diagnostic_truth/probe_normalized', 'diagnostic_truth/clean_intensity')}
        fp = str(f.attrs['scene_fingerprint'])
        peak = float(f.attrs['raw_intensity_global_max'])
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError('Invalid clean global intensity normalization')
        scalars = {'lambda_nm': cfg.wlength * 1e9, 'z_m': cfg.z,
                   'ccd_pixel_um': cfg.det_pixel * 1e6, 'sample_pixel_m': cfg.dx1}
        for key, expected in scalars.items():
            if not np.isclose(float(f[key][()]), expected, rtol=1e-7, atol=0):
                raise ValueError(f'{key} disagrees with scene configuration')
        if not np.array_equal(f['object_shape'][()], [cfg.obj_size] * 2):
            raise ValueError('Object shape disagrees with configuration')
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite H5 scene values')
    im, amp = arrays['intensity'], arrays['diffamp']
    clean = arrays['diagnostic_truth/clean_intensity']
    obj, probe = arrays['diagnostic_truth/object'], arrays['diagnostic_truth/probe']
    pn = arrays['diagnostic_truth/probe_normalized']
    pos, q, p0 = (arrays[k] for k in ('positions_px', 'quadratic_phase', 'probe_init'))
    n, m = cfg.N, cfg.obj_size
    if (im.shape != (cfg.n_pat, n, n) or amp.shape != im.shape or clean.shape != im.shape
            or obj.shape != (m, m) or any(v.shape != (n, n) for v in (probe, pn, q, p0))):
        raise ValueError('Scene array shapes disagree with configuration')
    if np.any(im < 0) or np.any(amp < 0) or np.any(clean < 0):
        raise ValueError('Negative measured or clean intensity/amplitude')
    if not np.allclose(amp ** 2, im, rtol=2e-5, atol=1e-7):
        raise ValueError('diffamp squared disagrees with intensity')
    if (pos.shape != (cfg.n_pat, 2) or not np.equal(pos, np.round(pos)).all()
            or pos.min() < 0 or pos.max() + n > m):
        raise ValueError('Invalid integer patch positions')
    pos = pos.astype(np.int64)
    roi = arrays['roi']
    if roi.shape != (4,) or not np.equal(roi, np.round(roi)).all():
        raise ValueError('Invalid ROI')
    a, b, c, d = map(int, roi)
    if not (0 <= a < b <= m and 0 <= c < d <= m):
        raise ValueError('ROI exceeds canvas')
    if not np.allclose(p0, 1):
        raise ValueError('Paper solver uses constant-one probe initialization; H5 disagrees')
    if not np.allclose(pn, probe / np.sqrt(peak), rtol=2e-5, atol=1e-7):
        raise ValueError('Normalized diagnostic probe disagrees with global scale')
    tensor = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(device)
    qt = tensor(q.astype(np.complex64))
    if not torch.allclose(qt, make_quad_phase(cfg, device), rtol=1e-5, atol=1e-5):
        raise ValueError('Stored quadratic phase disagrees with physical configuration')
    sc = SimpleNamespace(obj=obj, probe=probe, pos=pos, roi=(slice(a, b), slice(c, d)),
                         Q=qt, post=tensor(pos), Im=im, Icl=clean,
                         Imt=tensor(im.astype(np.float32)), Iclt=tensor(clean.astype(np.float32)),
                         fp=fp)
    error = denominator = 0.
    with torch.no_grad():
        ot, pt = tensor(obj.astype(np.complex64)), tensor(pn.astype(np.complex64))
        for start in range(0, len(pos), 8):
            pred = forward_ptycho(ot, pt, sc.post[start:start+8], qt, n)
            target = sc.Iclt[start:start+8]
            error += (pred - target).double().square().sum().item()
            denominator += target.double().square().sum().item()
    relative = (error / max(denominator, 1e-30)) ** .5
    if denominator <= 0 or not np.isfinite(relative) or relative > 1e-5:
        raise ValueError(f'Paper forward disagrees with clean H5 intensities: {relative}')
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    report = dict(scene=str(Path(path).resolve()), scene_sha256=digest.hexdigest(),
                  scene_fingerprint=fp, source_scene_config=source,
                  relative_clean_intensity_l2=relative, patterns=list(im.shape),
                  object_shape=list(obj.shape), roi=roi.tolist(),
                  coordinate_convention='integer row,column patch top-left; unchanged',
                  measurements='Stored intensity, no resimulation or renormalization',
                  truth_usage='Forward validation and evaluation only in blind run; '
                              '--probe-mode truth explicitly enables non-blind diagnostic',
                  algorithm='Existing paper run: four-head U-Net and original paper loss')
    return sc, report
