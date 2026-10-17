"""Independent PtyLab mPIE adapter. Does not import or edit the two-stage code."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

# Headless in Colab; no Qt window, even though PtyLab imports visualization modules.
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
DISABLED = (
    'objectTVregSwitch', 'l2reg', 'objectSmoothenessSwitch', 'probeSmoothenessSwitch',
    'absObjectSwitch', 'absProbeSwitch', 'objectContrastSwitch', 'probeBoundary',
    'absorbingProbeBoundary', 'modulusEnforcedProbeSwitch', 'binaryProbeSwitch',
    'positionCorrectionSwitch', 'TV_autofocus', 'comStabilizationSwitch',
    'backgroundModeSwitch', 'adaptiveDenoisingSwitch', 'FourierMaskSwitch',
    'CPSCswitch', 'OPRP', 'couplingSwitch', 'orthogonalizationSwitch',
    'probePowerCorrectionSwitch', 'probeSpectralPowerCorrectionSwitch',
    'PSDestimationSwitch', 'map_position_to_z_change',
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def arguments(argv=None):
    p = argparse.ArgumentParser(description='225-frame PtyLab mPIE, mm coordinates, Fraunhofer, no TGV/NLM.')
    p.add_argument('--data', type=Path, default=HERE/'data/ptydata_33.h5')
    p.add_argument('--out', type=Path)
    p.add_argument('--detector-crop', type=int, default=1024, help='Centered crop, not binning/resize; 0 retains original size')
    p.add_argument('--iters', type=int, default=50, help='Full sweeps: each iteration visits ALL 225 positions')
    p.add_argument('--wavelength-nm', type=float, default=632.8)
    p.add_argument('--distance-mm', type=float, default=6.)
    p.add_argument('--pixel-um', type=float, default=3.76)
    p.add_argument('--position-order', choices=['yx','xy'], default='yx')
    p.add_argument('--flip-y', action='store_true')
    p.add_argument('--flip-x', action='store_true')
    p.add_argument('--probe-diameter-um', type=float, help='Initial circular probe diameter only; default PtyLab field-of-view/3, NOT measured truth')
    p.add_argument('--beta-object', type=float, default=.25)
    p.add_argument('--beta-probe', type=float, default=.25)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpu', action='store_true', help='Explicit CPU mode for testing; default GPU, no silent fallback')
    p.add_argument('--prepare-only', action='store_true', help='Convert H5 and record geometry only, no PtyLab import or reconstruction')
    p.add_argument('--check-only', action='store_true', help='Convert and initialize PtyLab, but do not reconstruct')
    a = p.parse_args(argv)
    if not a.data.is_file():
        p.error(f'Data missing: {a.data}. Pass --data with the actual H5 path (mount Drive first).')
    if a.iters < 1 or a.seed < 0:
        p.error('--iters must be positive and --seed nonnegative')
    if a.detector_crop < 0 or a.detector_crop % 2:
        p.error('--detector-crop must be 0 or a positive even size')
    for key in ('wavelength_nm','distance_mm','pixel_um','beta_object','beta_probe','probe_diameter_um'):
        value = getattr(a,key)
        if value is not None and (not np.isfinite(value) or value <= 0):
            p.error(f'--{key.replace("_","-")} must be finite and positive')
    a.out = a.out or HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if a.out.exists():
        p.error(f'Output already exists (not overwritten): {a.out}')
    return a


def prepare(a):
    """Stream one frame at a time. Preserve source-frame/position correspondence."""
    a.out.mkdir(parents=True, exist_ok=False)
    converted = a.out/'ptylab_input.h5'
    with h5py.File(a.data,'r') as source:
        frames = source['ptychogramx']
        if frames.ndim != 3 or frames.shape[0] != 225 or frames.shape[1] != frames.shape[2]:
            raise ValueError(f'Expected 225 square intensity frames, got {frames.shape}')
        coords_mm = np.asarray(source['positionx'][()],dtype=np.float64)
        if coords_mm.shape != (2,225) or not np.isfinite(coords_mm).all():
            raise ValueError('positionx must be finite with shape (2,225), in mm')
        n = a.detector_crop or frames.shape[-1]
        if n < 16 or n > frames.shape[-1] or n%2 or (frames.shape[-1]-n)%2:
            raise ValueError('Crop must be an even size between 16 and the source side, with integer center alignment')
        low = (frames.shape[-1]-n)//2
        encoder = coords_mm.T[:,[0,1] if a.position_order == 'yx' else [1,0]].copy()*1e-3
        encoder[:,0] *= -1 if a.flip_y else 1
        encoder[:,1] *= -1 if a.flip_x else 1
        origin = encoder.mean(axis=0)
        encoder -= origin
        wave, z, pitch = a.wavelength_nm*1e-9, a.distance_mm*1e-3, a.pixel_um*1e-6
        digest = hashlib.sha256(coords_mm.tobytes())
        total = 0.
        with h5py.File(converted,'x') as target:
            stack = target.create_dataset('ptychogram',(225,n,n),dtype='float32',chunks=(1,n,n))
            for j in range(225):
                frame = np.asarray(frames[j,low:low+n,low:low+n],dtype=np.float32)
                if not np.isfinite(frame).all() or np.min(frame)<0 or np.max(frame)<=0:
                    raise ValueError(f'Frame {j} contains invalid/negative/all-zero intensities; clarify preprocessing')
                digest.update(frame.tobytes())
                stack[j] = frame
                total += frame.sum(dtype=np.float64)
            intensity_mean = total/(225*n*n)
            # Same common scaling for all frames, no per-frame exposure normalization.
            for j in range(225):
                stack[j] = stack[j]/intensity_mean
            for key,value in dict(encoder=encoder,wavelength=wave,zo=z,dxd=pitch).items():
                target.create_dataset(key,data=value)
            if a.probe_diameter_um is not None:
                target.create_dataset('entrancePupilDiameter',data=a.probe_diameter_um*1e-6)
            target.create_dataset('original_frame_ids',data=np.arange(225))
            target.attrs['source_position_unit'] = 'mm'
            target.attrs['description'] = 'All 225 frames, centered crop only; common intensity mean normalization'
        source_shape = list(frames.shape)
    config = {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(source_shape=source_shape, selected_frames=225, crop_side=n,
        crop_slice_zero_based=[low,low+n], position_input_unit='mm', position_scale_to_m=1e-3,
        encoder_origin_m=origin.tolist(), intensity_mean_before_scaling=intensity_mean,
        input_sha256=digest.hexdigest(), input_sha256_scope='raw positionx and selected raw cropped float32 frames in original order',
        object_pixel_um=wave*z/(n*pitch)*1e6, propagator='Fraunhofer',
        intensity_constraint='standard', initialization='PtyLab circ probe and ones object (small library initialization perturbation)',
        initial_probe_diameter_rule='user supplied' if a.probe_diameter_um else 'PtyLab default: probe field-of-view / 3',
        optimized_probe='full detector-sized complex array; no 512 hard support',
        disabled_constraints=list(DISABLED), TGV=False, NLM=False,
        iteration_definition='One sweep visits all 225 frames; not one neural-network optimizer update',
        caveats=['yx axis mapping and signs still require experimental verification',
                 'Far-field propagation selected per user; no automatic optical-geometry validation',
                 'No dark subtraction, background estimate, saturation mask, binning, or image resizing',
                 'PtyLab 0.3.1 mPIE rounds encoder/dxo to integer positions; unlike the two-stage bilinear sampler',
                 'No ground truth: no PSNR, internal error is not the two-stage amplitude rRMSE'])
    config['entry_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(a.out/'config.json',config)
    print(f'[data] 225 frames; crop [{low}:{low+n}] -> {n}x{n}; position mm -> m (x0.001)',flush=True)
    print(f'[geometry] lambda={a.wavelength_nm:g} nm, z={a.distance_mm:g} mm, detector pitch={a.pixel_um:g} um; dx={config["object_pixel_um"]:.6f} um',flush=True)
    return converted,config


def build_engine(converted,a):
    import PtyLab
    from PtyLab import Engines
    data = PtyLab.ExperimentalData(converted,operationMode='CPM')
    params = PtyLab.Params()
    params.gpuSwitch = not a.cpu
    params.propagatorType = 'Fraunhofer'
    params.intensityConstraint = 'standard'
    params.positionOrder = 'random'
    params.fftshiftSwitch = False
    params.saveMemory = True
    for key in DISABLED:
        setattr(params,key,False)
    recon = PtyLab.Reconstruction(data,params)
    recon.npsm = recon.nosm = recon.nlambda = recon.nslice = 1
    recon.initialProbe, recon.initialObject = 'circ','ones'
    recon.initializeObjectProbe()
    monitor = PtyLab.DummyMonitor()
    engine = Engines.mPIE(recon,data,params,monitor)
    engine.betaObject,engine.betaProbe = a.beta_object,a.beta_probe
    engine.numIterations = a.iters
    return data,recon,params,engine


def require_runtime(cpu):
    try:
        version = importlib.metadata.version('ptylab')
        import PtyLab
    except ImportError as exc:
        raise RuntimeError('Install requirements-colab.txt first; the desktop installation is not available inside Colab.') from exc
    if version != '0.3.1':
        raise RuntimeError(f'Tested PtyLab version is 0.3.1, found {version}; install the pinned requirements.')
    runtime = dict(ptylab_version=version,ptylab_module=PtyLab.__file__,device='CPU')
    if not cpu:
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount()<1:
                raise RuntimeError('No GPU detected')
            # Exercise allocation/FFT now, so missing CUDA libraries fail before reading 1GB.
            cp.fft.fft2(cp.ones((8,8),dtype=cp.complex64))
            cp.cuda.Stream.null.synchronize()
            name = cp.cuda.runtime.getDeviceProperties(0)['name']
            runtime.update(device=name.decode() if isinstance(name,bytes) else str(name),cupy_version=cp.__version__)
        except Exception as exc:
            raise RuntimeError('GPU unavailable. Select a Colab GPU runtime and install requirements-colab.txt. No CPU fallback was used.') from exc
    return runtime


def synchronize(cpu):
    if not cpu:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()


def plot_results(out,obj,probe,error,dx_um):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,
        'axes.spines.right':False,'legend.frameon':False,'svg.fonttype':'none'})
    fig,axes = plt.subplots(2,2,figsize=(10,9),layout='constrained')
    for col,(name,field) in enumerate((('Object',obj),('Probe',probe))):
        extent = [0,field.shape[1]*dx_um,field.shape[0]*dx_um,0]
        vmax = max(float(np.percentile(np.abs(field),99.5)),1e-12)
        im = axes[0,col].imshow(np.abs(field),cmap='gray',vmin=0,vmax=vmax,extent=extent)
        axes[0,col].set_title(f'{name} amplitude'); fig.colorbar(im,ax=axes[0,col],shrink=.8)
        im = axes[1,col].imshow(np.angle(field),cmap='twilight',vmin=-np.pi,vmax=np.pi,extent=extent)
        axes[1,col].set_title(f'{name} phase'); fig.colorbar(im,ax=axes[1,col],shrink=.8,label='rad')
        for row in range(2):
            axes[row,col].set(xlabel='x (um)',ylabel='y (um)')
    fig.suptitle('PtyLab mPIE | Fraunhofer | no TGV / NLM | no ground truth')
    fig.savefig(out/'reconstruction.png',dpi=300)
    fig.savefig(out/'reconstruction.pdf'); plt.close(fig)
    fig,ax = plt.subplots(figsize=(6,4),layout='constrained')
    ax.plot(np.arange(1,len(error)+1),error,color='#3775BA',linewidth=2)
    ax.set(xlabel='Iteration (full sweep over 225 frames)',ylabel='PtyLab internal error',title='Measurement fit, not PSNR')
    ax.grid(alpha=.15)
    fig.savefig(out/'error_curve.png',dpi=300); fig.savefig(out/'error_curve.pdf'); plt.close(fig)


def reconstruct(converted,a,config,runtime):
    np.random.seed(a.seed)
    if not a.cpu:
        import cupy as cp
        cp.random.seed(a.seed)
    data,recon,params,engine = build_engine(converted,a)
    config.update(runtime=runtime,object_side=int(recon.No),probe_side=int(recon.Np),
        actual_initial_probe_diameter_um=float(data.entrancePupilDiameter)*1e6,
        beta_object=float(engine.betaObject),beta_probe=float(engine.betaProbe),
        alpha_object=float(engine.alphaObject),alpha_probe=float(engine.alphaProbe),
        feedback_m=float(engine.feedbackM),friction_m=float(engine.frictionM),
        save_memory=bool(params.saveMemory),momentum_acceleration=bool(params.momentumAcceleration))
    write_json(a.out/'config.json',config)
    print(f'[mPIE] {runtime["device"]}; {a.iters} sweeps x 225 frames; object={recon.No}; probe={recon.Np}; no TGV/NLM',flush=True)
    print(f'[init] circular probe diameter={config["actual_initial_probe_diameter_um"]:.3f} um (initial guess only); full probe remains optimizable',flush=True)
    if a.check_only:
        print('[check-only] Initialized successfully; no reconstruction was run.',flush=True)
        return
    synchronize(a.cpu)
    start = time.perf_counter()
    engine.reconstruct()
    synchronize(a.cpu)
    elapsed = time.perf_counter()-start
    from PtyLab.utils.gpuUtils import asNumpyArray
    obj = asNumpyArray(recon.object).squeeze().copy()
    probe = asNumpyArray(recon.probe).squeeze().copy()
    error = np.asarray(asNumpyArray(recon.error),dtype=np.float64).ravel()
    if not all(np.isfinite(x).all() for x in (obj,probe,error)):
        raise FloatingPointError('Nonfinite reconstruction/error; no successful result reported')
    np.savez(a.out/'reconstruction.npz',obj_rec=obj,probe_rec=probe,error=error,
        object_pixel_um=float(recon.dxo)*1e6,encoder_m=asNumpyArray(recon.encoder_corrected),
        positions_px=asNumpyArray(recon.positions),original_frame_ids=np.arange(225),config=json.dumps(config))
    write_json(a.out/'summary.json',{'sweeps':len(error),'frame_visits':225*len(error),
        'reconstruct_call_seconds':elapsed,'final_ptylab_error':float(error[-1]),
        'timing_scope':'synchronized reconstruct() call incl library setup/transfers; excludes H5 preparation, object initialization, saves and figures',
        'warning':'No ground truth; internal error is not PSNR or two-stage amplitude rRMSE'})
    plot_results(a.out,obj,probe,error,float(recon.dxo)*1e6)
    print(f'[done] {len(error)} sweeps, reconstruct()={elapsed:.2f}s; results: {a.out.resolve()}',flush=True)


def main():
    a = arguments()
    try:
        runtime = None if a.prepare_only else require_runtime(a.cpu)
        converted,config = prepare(a)
        if a.prepare_only:
            print(f'[prepared] {converted.resolve()} (no reconstruction)',flush=True)
        else:
            reconstruct(converted,a,config,runtime)
    except Exception as exc:
        if a.out.is_dir():
            write_json(a.out/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise


if __name__ == '__main__':
    main()
