"""Export a shared ProPtyNet scene for the separate PtyINR adaptation."""
import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulations.ProPtyNet_paper import Cfg, PRESETS
from functions.paperrepro.scene import build_scene
from functions.paperrepro.optics import forward_field
from functions.paperrepro.sample import make_positions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-config', type=Path, help='Baseline result NPZ, config JSON or manifest')
    p.add_argument('--preset', choices=PRESETS, default=None)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scan-shift-y', type=int, default=None,
                   help='Row shift from centred raster (pixels); defaults to scene config or zero')
    p.add_argument('--scan-shift-x', type=int, default=None,
                   help='Column shift from centred raster (pixels); positive is right')
    args = p.parse_args()
    loaded = {}
    if args.scene_config:
        if args.scene_config.suffix == '.npz':
            with np.load(args.scene_config, allow_pickle=False) as data:
                loaded = json.loads(str(data['cfg'].item()))
        else:
            loaded = json.loads(args.scene_config.read_text(encoding='utf-8-sig'))
            loaded = loaded.get('scene_config', loaded)
    names = {f.name for f in fields(Cfg)}
    kw = {k:v for k,v in loaded.items() if k in names}
    if args.preset:
        kw.update(PRESETS[args.preset], preset=args.preset)
    cfg = Cfg(**{**kw, 'device':args.device})
    cfg.quad_sign = loaded.get('quad_sign', -1.)
    if args.output.exists():
        raise FileExistsError(args.output)
    shift = [args.scan_shift_y if args.scan_shift_y is not None else loaded.get('scan_shift_y', 0),
             args.scan_shift_x if args.scan_shift_x is not None else loaded.get('scan_shift_x', 0)]
    scene = build_scene(cfg, cfg.dev(), positions=make_positions(cfg) + np.asarray(shift))
    # The simulator divides all intensities by ONE global maximum.
    # Store the corresponding probe scale for diagnostic truth checks only.
    with torch.no_grad():
        obj = torch.from_numpy(scene.obj).to(scene.Q.device)
        probe = torch.from_numpy(scene.probe).to(scene.Q.device)
        peak = 0.
        for start in range(0, len(scene.post), 4):
            u = forward_field(obj, probe, scene.post[start:start+4], scene.Q, cfg.N)
            peak = max(peak, (u.real.square()+u.imag.square()).max().item())
    if peak <= 0:
        raise ValueError('Zero truth intensity')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rs,cs = scene.roi
    with h5py.File(args.output, 'x') as f:
        f.attrs['schema'] = 'proptynet-ptyinr-v1'
        f.attrs['scene_fingerprint'] = scene.fp
        f.attrs['scene_config_json'] = json.dumps({**asdict(cfg),'quad_sign':cfg.quad_sign,
                                                  'scan_shift_y':int(shift[0]),'scan_shift_x':int(shift[1])})
        f.attrs['position_convention'] = 'integer [row,column] patch top-left; no transpose'
        f.attrs['fft_convention'] = 'ifftshift -> fft2(norm=ortho) -> fftshift'
        f.attrs['diffraction_scale'] = 1.
        f.attrs['raw_intensity_global_max'] = peak
        arrays = dict(diffamp=np.sqrt(np.maximum(scene.Im,0)), intensity=scene.Im,
                      positions_px=scene.pos, quadratic_phase=scene.Q.cpu().numpy(),
                      probe_init=scene.P0, roi=np.array([rs.start,rs.stop,cs.start,cs.stop]),
                      lambda_nm=cfg.wlength*1e9, z_m=cfg.z, ccd_pixel_um=cfg.det_pixel*1e6,
                      sample_pixel_m=cfg.dx1, object_shape=np.array(scene.obj.shape))
        for key,value in arrays.items():
            f.create_dataset(key,data=value)
        truth=f.create_group('diagnostic_truth')
        truth.attrs['usage']='Ground truth for verification/evaluation only; not blind optimization'
        truth.create_dataset('object',data=scene.obj)
        truth.create_dataset('probe',data=scene.probe)
        truth.create_dataset('probe_normalized',data=scene.probe/np.sqrt(peak))
        truth.create_dataset('clean_intensity',data=scene.Icl)
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps(dict(file=str(args.output.resolve()),sha256=digest,
                          scene_fingerprint=scene.fp,patterns=list(scene.Im.shape)),indent=2))
    print('Export only. This schema requires the aligned adapter, not official main.py directly.')


if __name__ == '__main__':
    main()
