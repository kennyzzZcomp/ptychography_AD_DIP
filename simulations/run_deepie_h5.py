"""Run the existing independent DeePIE implementation on exact shared H5 data."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulations.DeePIE import parser, configurations, SCENE_FIELDS
from simulations.ProPtyNet_paper import Cfg
from simulations.run_progressive_h5 import load_scene
from functions.paperrepro.deepie_solver import run_deepie


def main():
    p = parser()
    p.description = __doc__
    p.add_argument('--scene', type=Path, required=True)
    args = p.parse_args()
    # Geometry and initialization are exclusively owned by the imported file.
    forbidden = [k for k in (*SCENE_FIELDS, 'obj_amp_binary_invert', 'quad_sign',
                             'sample_pixel_um', 'preset', 'scene_config')
                 if getattr(args, k) is not None]
    if forbidden:
        p.error('H5 owns scene parameters; remove overrides: ' + ', '.join(forbidden))
    _, mc, tc = configurations(args)
    cfg = Cfg(device=args.device)
    sc, source, relative = load_scene(args.scene, cfg.dev())
    kw = {k:v for k,v in source.items() if k in {*SCENE_FIELDS, 'obj_amp_binary_invert'}}
    kw.update(device=args.device, iters=args.iters, eval_every=args.eval_every,
              outdir=args.outdir, probe_mode='pixel', eval_size=0)
    cfg = Cfg(**kw)
    cfg.quad_sign = source.get('quad_sign', -1.)
    report = dict(scene=str(args.scene.resolve()),
        scene_sha256=hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        scene_fingerprint=sc.fp, relative_clean_amplitude_l2=relative,
        patterns=list(sc.Im.shape), object_shape=list(sc.obj.shape),
        roi=[sc.roi[0].start,sc.roi[0].stop,sc.roi[1].start,sc.roi[1].stop],
        source_scene_config=source,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        loader_sha256=hashlib.sha256(Path(sys.modules[load_scene.__module__].__file__).read_bytes()).hexdigest(),
        truth_usage='Forward verification, evaluation and saved outputs only',
        probe_initialization='H5 probe_init; optional one-time measured RMS calibration',
        data_usage='Verbatim measurements, positions, quadratic phase and ROI; no scene regeneration')
    out = Path(args.outdir)
    if out.exists():
        p.error('Choose a new output directory')
    out.mkdir(parents=True)
    (out/'scene_adapter.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_scene_config'},indent=2),flush=True)
    print(json.dumps(dict(model=asdict(mc),training=asdict(tc)),indent=2),flush=True)
    if args.mode == 'run':
        run_deepie(cfg, mc, tc, scene=sc, scene_metadata=report)


if __name__ == '__main__':
    main()
