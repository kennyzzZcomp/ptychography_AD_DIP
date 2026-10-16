"""Controlled 64/128 FFT comparison. One entrypoint, no training on import.

prepare: regenerate clean measurements from the SAME saved object/probe, then
         independent Poisson counts at fixed expected photons/frame.
ours/deepie: run existing solvers on the selected saved H5, never resize intensity.
N remains the 64-pixel probe/patch parameter; fft_size is detector/FFT size.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from functions.paperrepro.optics import forward_ptycho
from simulations.run_progressive_h5 import load_scene
from simulations.make_poisson_h5 import generate, file_sha256, scene_fingerprint

DEFAULT_SOURCE = ROOT / 'results_paper/ptyinr_obj612_grid10_d59_s10_center/scene_obj612_grid10_d59_s10_center.h5'
DEFAULT_ROOT = ROOT / 'results_paper/sampling128'


def make_clean(source, output, size=128):
    """Keep O/P/Q/positions/ROI/init and wave normalization exactly unchanged.

    Orthonormal padding conserves each frame's integrated intensity. Detector
    pitch changes by old_fft/new_fft; detector extent and object pixel pitch do
    not change. This is ideal point-intensity sampling, not finite pixel binning.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Will not overwrite {output}')
    before = file_sha256(source)
    sc, config, old_error = load_scene(source, torch.device('cpu'))
    n = config['N']; old_size = config.get('fft_size', 0) or n
    if size < old_size or (size-n) % 2:
        raise ValueError('New FFT must be >= old FFT and use even symmetric padding')
    with h5py.File(source, 'r') as f:
        if config['noise'] != 'none' or 'measurement_noise' in f:
            raise ValueError('Use the original CLEAN H5; do not interpolate or stack noise')
        np.testing.assert_array_equal(f['intensity'][()], f['diagnostic_truth/clean_intensity'][()])
    with torch.no_grad():
        clean = forward_ptycho(torch.from_numpy(sc.obj), torch.from_numpy(sc.probe),
            sc.post, sc.Q, n, chunk=16, fft_size=size).numpy().astype(np.float32)
    np.testing.assert_allclose(clean.sum((1,2)), sc.Icl.sum((1,2)), rtol=2e-6)
    updated = dict(config, fft_size=int(size), det_pixel=config['det_pixel']*old_size/size,
        detector_sampling='centered zero-padded exit wave; ideal point samples; not pixel integration',
        intensity_normalization='Inherited source wave scale; NOT renormalized to new peak')
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, 'r') as src, h5py.File(output, 'x') as dst:
        for key, value in src.attrs.items(): dst.attrs[key] = value
        for key in src: src.copy(key, dst)
        for key, array in [('intensity',clean), ('diffamp',np.sqrt(clean)),
                           ('diagnostic_truth/clean_intensity',clean)]:
            del dst[key]
            dst.create_dataset(key, data=array, compression='gzip', shuffle=True)
        dst['ccd_pixel_um'][...] = updated['det_pixel']*1e6
        dst.attrs['scene_config_json'] = json.dumps(updated)
        dst.attrs['source_scene_sha256'] = before
        dst.attrs['source_scene_path'] = str(source)
        dst.attrs['sampling_note'] = updated['detector_sampling']
        dst.attrs['scene_fingerprint'] = scene_fingerprint(dst)
        dst.attrs['scene_fingerprint_algorithm'] = 'sha256-named-arrays-excluding-measurement_noise-v1-first12'
    _, _, new_error = load_scene(output, torch.device('cpu'))
    if file_sha256(source) != before:
        raise AssertionError('Source changed')
    return dict(source=str(source), source_sha256=before, output=str(output),
        fft_size=size, patch_size=n, shape=list(clean.shape),
        old_forward_error=old_error, new_forward_error=new_error,
        preserved='Object, both truth probes, Q, positions, ROI, initial probe, object pixel pitch, integrated clean intensity',
        detector_pitch_um=updated['det_pixel']*1e6,
        point_sampling_not_pixel_integration=True)


def commands(method, scene, out):
    """Frozen paper method; historical DeePIE settings with requested 500-step decay."""
    python = [sys.executable, '-u']
    scripts = ROOT/'simulations'
    if method == 'ours':
        return [python+[str(scripts/'run_light_stage1_search.py'), '--scene',str(scene),
            '--candidates','u16','--updates','500','--stage1-only','--eval-every','50',
            '--seed','0','--outdir',str(out/'stage1')],
            python+[str(scripts/'run_original_stage2_from_widths.py'),'--source',str(out/'stage1'),
            '--stage1-names','u16','--stage2-width','32','--architecture','unet',
            '--residual-mode','complex','--data-loss','auto','--updates','500',
            '--eval-every','50','--outdir',str(out/'stage2')]]
    return [python+[str(scripts/'run_deepie_h5.py'),'run','--scene',str(scene),
        '--device','cuda','--iters','1000','--eval-every','100',
        '--lr','1e-6','--probe-lr','.001','--decay-every','500','--decay-factor','.5',
        '--scan-chunk','100','--outdir',str(out)]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare','ours','deepie'])
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT, help='All data/results in this directory')
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE, help='Original clean 64 H5; prepare only')
    p.add_argument('--fft-size', type=int, choices=[64,128], default=128)
    p.add_argument('--dose', choices=['clean','1e5','1e4'], default='clean')
    p.add_argument('--dry-run', action='store_true', help='Print training commands only')
    a = p.parse_args()
    dest = a.root.resolve()/f'fft{a.fft_size}'
    if a.mode == 'prepare':
        if a.dry_run: p.error('--dry-run is for training commands')
        if dest.exists(): p.error(f'Choose a new --root; refusing to overwrite {dest}')
        if not a.source.is_file(): p.error(f'Clean H5 not found: {a.source}; pass --source YOUR_CLEAN_H5')
        clean = dest/'data/clean.h5'
        report = make_clean(a.source, clean, a.fft_size)
        for dose in ('1e5','1e4'):
            noise = generate(clean, clean.with_name(f'{dose}.h5'), float(dose), 42)
            report[dose] = noise['noise']
        (dest/'generation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f'Created {clean}: {report["shape"]}; probe/patch {report["patch_size"]} unchanged')
        print(f'Forward relative amplitude error: {report["new_forward_error"]:.3g}')
        for dose in ('1e5','1e4'):
            print(f'{dose}: mean expected photons/frame={float(dose):g}; '
                  f'intensity-array SNR={report[dose]["measured_intensity_snr_db"]:.3f} dB')
        print(f'Full audit: {dest / "generation.json"}; no reconstruction run')
        return
    scene = dest/'data'/f'{a.dose}.h5'
    out = dest/a.dose/a.mode
    if not scene.is_file(): p.error(f'Missing {scene}; run prepare first with the same --root/--fft-size')
    if out.exists(): p.error(f'Results already exist: {out}; no overwrite or implicit resume')
    for command in commands(a.mode, scene, out):
        print(subprocess.list2cmdline(command), flush=True)
        if not a.dry_run: subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
