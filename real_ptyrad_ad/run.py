"""Optical data adapter calling installed OFFICIAL PtyRAD model/loss/recon_step.

No implementation of the user's solvers or the installed package is edited.
The small outer loop adds budget limits, common evaluation and NPZ snapshots.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import inspect
import json
import logging
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'real_ptycho'))
from core import Propagator, mpie_geometry, select_center_grid
from initialization import initial_fields

from ptyrad.core.models.ptycho import PtychoModel
from ptyrad.core.losses import CombinedLoss
from ptyrad.core.constraints import CombinedConstraint
from ptyrad.params.model_params import ModelParams
from ptyrad.params.loss_params import LossParams
from ptyrad.params.constraint_params import ConstraintParams
from official_runtime import load_official_updates
_updates, _update_loader = load_official_updates()
create_optimizer, make_batches = _updates.create_optimizer, _updates.make_batches
recon_step, toggle_grad_requires = _updates.recon_step, _updates.toggle_grad_requires


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(data_path, n, seed):
    """Cache only selected raw pixels, with the original adapter's common scale."""
    cache_dir = HERE/'results'/datetime.now().strftime('prepared_%Y%m%d_%H%M%S_%f')
    cache_dir.mkdir(parents=True, exist_ok=False)
    with h5py.File(data_path, 'r') as f:
        if f['ptychogramx'].shape != (225,2000,2000):
            raise ValueError('Expected 225 measured 2000x2000 intensities')
        ids, coords, groups, selection = select_center_grid(f['positionx'][()], 225)
        positions, side, dx, legacy, alignment = mpie_geometry(coords,n,632.8,6.,3.76)
        low = (2000-n)//2
        arr = np.lib.format.open_memmap(cache_dir/'intensity.npy', mode='w+',
                                        dtype=np.float32, shape=(225,n,n))
        digest = hashlib.sha256()
        digest.update(coords.tobytes()); digest.update(ids.tobytes())
        total = 0.
        for j, original in enumerate(ids):
            frame = f['ptychogramx'][int(original),low:low+n,low:low+n].astype(np.float32)
            if not np.isfinite(frame).all() or frame.min() < 0:
                raise ValueError('Nonfinite/negative measurement; no implicit correction')
            arr[j] = frame
            total += frame.sum(dtype=np.float64)
            digest.update(frame.tobytes())
        scale = total/arr.size
        if scale <= 0:
            raise ValueError('Invalid common intensity scale')
        for j in range(225):
            arr[j] /= scale
        arr.flush()
    obj, probe, init_meta = initial_fields('mpie',n,side,legacy,dx*1e6,seed)
    meta = dict(data=str(data_path.resolve()),n=n,side=side,object_pixel_um=dx*1e6,
                wavelength_nm=632.8,distance_mm=6.,pixel_um=3.76,position_input_unit='mm',
                position_scale_to_m=.001,crop_slice=[low,low+n],
                common_intensity_mean=scale,selected_input_sha256=digest.hexdigest(),
                original_frame_ids=ids.tolist(),positions_px=positions.tolist(),
                positions_mm=coords.tolist(),selection=selection,alignment=alignment,
                initialization=init_meta,seed=seed)
    write_json(cache_dir/'prepared.json',meta)
    np.savez(cache_dir/'fields.npz',obj=obj,probe=probe,positions=positions,ids=ids,
             groups=np.array(groups,dtype=object))
    return cache_dir


def load_prepared(folder):
    meta = json.loads((folder/'prepared.json').read_text(encoding='utf-8'))
    data = np.load(folder/'intensity.npy', mmap_mode='r')
    # No object arrays are needed for baseline reconstruction.
    with np.load(folder/'fields.npz',allow_pickle=False) as z:
        obj,probe,pos,ids = [z[k].copy() for k in ('obj','probe','positions','ids')]
    return data,obj,probe,pos,ids,meta


def create_model(data, obj, probe, positions, meta, args):
    n = probe.shape[-1]
    model_params = ModelParams().model_dump()
    model_params['preload_data'] = False
    for key in model_params['update_params']:
        model_params['update_params'][key] = dict(lr=0.,start_iter=None)
    for key,lr in [('obja',args.lr_object),('objp',args.lr_object),('probe',args.lr_probe)]:
        if key=='probe' and getattr(args,'fixed_probe',False):
            model_params['update_params'][key] = dict(lr=0.,start_iter=None)
        else:
            model_params['update_params'][key] = dict(lr=lr,start_iter=1)
    variables = dict(measurements=data,obj=obj[None,None],probe=probe[None],
                     obj_tilts=np.zeros((1,2),np.float32),slice_thickness=0.,
                     probe_pos_shifts=np.zeros((225,2),np.float32),
                     omode_occu=np.ones(1,np.float32),H=np.ones((n,n),np.complex64),
                     N_scan_slow=15,N_scan_fast=15,crop_pos=positions,
                     dx=meta['object_pixel_um'],dk=1/(n*meta['object_pixel_um']),
                     lambd=.6328,random_seed=args.seed,length_unit='um',
                     scan_affine=[1.,0.,0.,1.],meas_Npix=n,simu_Npix=n,
                     simu_match_mode=None,recon_provenance={'adapter':'real_ptyrad_ad',
                                                            'single_slice_optical':True})
    model = PtychoModel(variables,model_params,device='cuda')
    loss_params = LossParams().model_dump()
    for key in loss_params:
        loss_params[key]['state'] = key == 'loss_single'
    loss = CombinedLoss(loss_params,device='cuda')
    if args.loss == 'matched-mse':
        loss = MatchedAmplitudeMSE()
    constraints = ConstraintParams().model_dump()
    for value in constraints.values():
        value['start_iter'] = None
    floor = getattr(args,'positive_amplitude_floor',0.)
    if floor > 0:
        constraints['obja_thresh'].update(start_iter=1,relax=0.,thresh=[floor,1e10])
    constraint = CombinedConstraint(constraints,device='cuda')
    optimizer = create_optimizer(model_params['optimizer_params'],model.optimizable_params)
    if floor > 0:
        def enforce_physical_amplitude(optimizer,step_args,step_kwargs):
            with torch.no_grad():
                constraint(model,1)
        optimizer.register_step_post_hook(enforce_physical_amplitude)
    return model,optimizer,loss,constraint,dict(model=model_params,loss=loss.loss_params,
                                               official_loss_params_reference=loss_params,
                                               constraint=constraints,
                                               amplitude_domain_projection=('official obja_thresh after EACH Adam step and sweep'
                                                                            if floor>0 else None),
                                               actual_loss=args.loss)


class MatchedAmplitudeMSE(torch.nn.Module):
    """Explicitly labeled diagnostic, NOT the official CombinedLoss."""
    def __init__(self):
        super().__init__()
        self.loss_params = {'matched_amplitude_mse':dict(state=True,weight=1.)}
        self.accumulation_chunks = None
        self.group_frames = None

    def forward(self,predicted,measured,*unused):
        value = (predicted.sqrt()-measured.sqrt()).square().mean()
        if self.group_frames is not None:
            # Official recon_step divides by grad_accumulation. Compensate
            # unequal last micro-batches to get the EXACT full-group MSE.
            value = value*self.accumulation_chunks*len(predicted)/self.group_frames
        return value,[value]


def fields(model):
    obj = torch.polar(model.opt_obja,model.opt_objp)[0,0]
    probe = model.get_complex_probe_view()[0]
    return obj.detach().cpu().numpy(),probe.detach().cpu().numpy()


@torch.no_grad()
def evaluate(model,batch=4):
    squared_error = target_energy = 0.
    for start in range(0,225,batch):
        part = np.arange(start,min(start+batch,225))
        predicted = model(part)
        measured = model.get_measurements(part)
        squared_error += float((predicted.sqrt()-measured.sqrt()).square().sum(dtype=torch.float64))
        target_energy += float(measured.sum(dtype=torch.float64))
        model.clear_cache()
    obj,probe = fields(model)
    n = probe.shape[-1]
    yy,xx = np.indices((n,n))
    energy = np.abs(probe)**2
    return dict(amplitude_mse=squared_error/(225*n*n),
                relative_amplitude_rmse=float(np.sqrt(squared_error/target_energy)),
                object_amp_max=float(np.abs(obj).max()),object_amp_min=float(np.abs(obj).min()),
                object_amp_mean=float(np.abs(obj).mean()),probe_amp_max=float(np.abs(probe).max()),
                signed_object_amplitude_min=float(model.opt_obja.min()),
                negative_object_amplitude_fraction=float((model.opt_obja<0).float().mean()),
                probe_power=float(energy.sum()),
                probe_intensity_centroid_yx=[float((yy*energy).sum()/energy.sum()),
                                             float((xx*energy).sum()/energy.sum())])


def preflight(model,obj,probe,positions):
    indices = np.array([0,14,112,210,224])
    with torch.no_grad():
        prediction = model(indices)
        actual_obj,actual_probe = fields(model)
        prop = Propagator(positions,probe.shape[-1],sampling='mpie-integer')
        expected = prop.amplitude(torch.from_numpy(obj).cuda(),torch.from_numpy(probe).cuda(),indices).square()+1e-10
        relative = float((prediction-expected).norm()/expected.norm())
        zero_shift = model.get_probes(indices)
        shift_error = float((zero_shift-torch.from_numpy(probe).cuda()[None,None]).norm()
                            /(np.sqrt(len(indices))*torch.from_numpy(probe).cuda().norm()))
    model.clear_cache()
    report = dict(full_size_intensity_relative_error=relative,
                  zero_probe_shift_relative_error=shift_error,
                  actual_object_initial_max_error=float(np.abs(actual_obj-obj).max()),
                  actual_probe_initial_max_error=float(np.abs(actual_probe-probe).max()),
                  active_parameters=[k for k,p in model.optimizable_tensors.items() if p.requires_grad],
                  detector_size=probe.shape[-1],frames=225)
    if relative > 3e-6 or report['actual_object_initial_max_error'] > 3e-7 or report['actual_probe_initial_max_error'] != 0:
        raise AssertionError(report)
    return report


def plot_snapshot(path,obj,probe,history,dx,label='Official PtyRAD amplitude loss'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(2,2,figsize=(9,8),layout='constrained')
    for ax,field,title in [(axs[0,0],obj,'Object amplitude'),(axs[0,1],probe,'Probe amplitude')]:
        amp = np.abs(field)
        im = ax.imshow(amp,cmap='gray',vmin=0,vmax=np.percentile(amp,99.8),
                       extent=[0,field.shape[1]*dx,field.shape[0]*dx,0])
        ax.set(title=title,xlabel='x (um)',ylabel='y (um)')
        fig.colorbar(im,ax=ax,shrink=.8)
    for ax,field,title in [(axs[1,0],obj,'Object phase'),(axs[1,1],probe,'Probe phase')]:
        im = ax.imshow(np.angle(field),cmap='twilight',vmin=-np.pi,vmax=np.pi,
                       extent=[0,field.shape[1]*dx,field.shape[0]*dx,0])
        ax.set(title=title,xlabel='x (um)',ylabel='y (um)')
        fig.colorbar(im,ax=ax,shrink=.8,label='rad')
    import textwrap
    fig.suptitle(textwrap.fill(label,width=88)+f'\nSweep {history[-1]["sweep"]} | '
                 f'Common amplitude rRMSE={history[-1]["relative_amplitude_rmse"]:.4f}; no ground truth')
    fig.savefig(path,dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'experiment/ptydata_33.h5')
    p.add_argument('--prepared',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--crop',type=int,default=1024)
    p.add_argument('--sweeps',type=int,default=100)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--lr-object',type=float,default=.0005)
    p.add_argument('--lr-probe',type=float,default=.005)
    p.add_argument('--loss',choices=['official','matched-mse'],default='official')
    p.add_argument('--grouping',choices=['random','abcd'],default='random',
                   help='abcd requires matched-mse; four exact accumulated group updates per sweep')
    p.add_argument('--reference-probe',type=Path,help='DIAGNOSTIC ONLY: load a reference reconstructed probe')
    p.add_argument('--fixed-probe',action='store_true',help='DIAGNOSTIC ONLY: do not optimize probe')
    p.add_argument('--positive-amplitude-floor',type=float,default=0.,
                   help='DIAGNOSTIC ONLY: official amplitude threshold after each Adam step; default off')
    p.add_argument('--eval-every',type=int,default=10)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-seconds',type=float,default=1200.)
    args = p.parse_args()
    if min(args.sweeps,args.batch,args.lr_object,args.lr_probe,args.eval_every,args.max_seconds) <= 0:
        p.error('Counts, rates and wall budget must be positive')
    if args.crop < 32 or args.crop > 2000 or args.crop%2:
        p.error('Crop must be even and between 32 and 2000')
    if args.grouping=='abcd' and args.loss!='matched-mse':
        p.error('ABCD exact group accumulation requires the explicitly labeled MSE diagnostic')
    if args.reference_probe is not None and not args.fixed_probe:
        p.error('Reference probe is only supported as a fixed-probe diagnostic')
    if args.positive_amplitude_floor<0 or args.positive_amplitude_floor>=1:
        p.error('Amplitude floor must be in [0,1); zero disables the projection')
    if importlib.metadata.version('ptyrad') != '1.0.0':
        p.error('This adapter is audited for official PtyRAD 1.0.0')
    if not torch.cuda.is_available():
        p.error('CUDA required; no silent fallback')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    prepared = args.prepared or prepare(args.data,args.crop,args.seed)
    print('[prepared]',prepared,flush=True)
    if args.prepare_only:
        return
    data,obj,probe,positions,ids,meta = load_prepared(prepared)
    if meta['seed'] != args.seed or meta['n'] != args.crop:
        p.error('Prepared seed/crop does not match requested configuration')
    if args.reference_probe is not None:
        with np.load(args.reference_probe,allow_pickle=False) as z:
            reference = z['probe_rec'].copy()
            reference_config = json.loads(str(z['config']))
        if reference.shape != probe.shape:
            p.error('Reference probe dimensions do not match detector crop')
        reference_mean = reference_config.get('intensity_mean_before_scaling',reference_config.get('original_intensity_mean'))
        if reference_mean is None or abs(reference_mean/meta['common_intensity_mean']-1)>1e-6:
            p.error('Reference probe is not calibrated to the SAME common intensity normalization')
        probe = reference
        meta = dict(meta,reference_probe_diagnostic=str(args.reference_probe.resolve()),
                    caveat='Probe supplied by a previous reconstruction; NOT blind reconstruction from circular P0')
    out = args.out or HERE/'results'/datetime.now().strftime('ad_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    file_handler = logging.FileHandler(out/'official.log',encoding='utf-8')
    logging.basicConfig(level=logging.INFO,handlers=[file_handler],
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    model,optimizer,loss,constraint,params = create_model(data,obj,probe,positions,meta,args)
    config = dict(adapter='Official PtyRAD model + official recon_step; optical import and saving wrapper',
                  ptyrad_version=importlib.metadata.version('ptyrad'),torch=torch.__version__,
                  device=torch.cuda.get_device_name(0),prepared=str(prepared.resolve()),
                  args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  input=meta,params=params,gain=1,TGV=0,NLM=0,
                  iteration_definition=('One sweep visits all 225; four exact ABCD updates' if args.grouping=='abcd'
                                        else 'One sweep visits all 225; optimizer steps per sweep=len(batches)'),
                  all_added_constraints_disabled=(args.positive_amplitude_floor==0),source_sha256={})
    config['official_update_loader'] = _update_loader
    for name,source in [('adapter',Path(__file__)),('user_core',ROOT/'real_ptycho/core.py'),
                        ('initialization',ROOT/'real_ptycho/initialization.py'),
                        ('official_model',Path(inspect.getfile(PtychoModel))),
                        ('official_loss',Path(inspect.getfile(CombinedLoss))),
                        ('official_step',Path(inspect.getfile(recon_step)))]:
        config['source_sha256'][name] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(out/'config.json',config)
    check = preflight(model,obj,probe,positions)
    write_json(out/'preflight.json',check)
    np.savez(out/'initialization.npz',obj_rec=obj,probe_rec=probe,positions_px=positions,
             original_frame_ids=ids,object_pixel_um=meta['object_pixel_um'])
    print('[preflight]',json.dumps(check),flush=True)
    if args.check_only:
        print('CHECK_ONLY',out,flush=True)
        return
    if args.grouping=='random':
        batches = make_batches(np.arange(225),positions,args.batch,mode='random',seed=args.seed)
        group_batches = [batches]
    else:
        rr = np.array(meta['selection']['grid_rows_zero_based'])
        cc = np.array(meta['selection']['grid_cols_zero_based'])
        groups = [np.flatnonzero((rr%2==a)&(cc%2==b)) for a,b in ((0,0),(0,1),(1,0),(1,1))]
        group_batches = [[g[start:start+args.batch] for start in range(0,len(g),args.batch)] for g in groups]
    write_json(out/'batches.json',[[np.asarray(b).tolist() for b in group] for group in group_batches])
    history = []
    cumulative = 0.
    start = time.perf_counter()
    status = 'complete'
    completed = 0
    updates_done = 0

    def save(sweep):
        stats = evaluate(model,args.batch)
        row = dict(sweep=sweep,optimizer_updates=updates_done,
                   training_seconds=cumulative,wall_seconds=time.perf_counter()-start,**stats)
        history.append(row)
        write_json(out/'history.json',history)
        current_obj,current_probe = fields(model)
        np.savez(out/f'sweep_{sweep:04d}.npz',obj_rec=current_obj,probe_rec=current_probe,
                 positions_px=positions,original_frame_ids=ids,object_pixel_um=meta['object_pixel_um'])
        label = 'PtyRAD model; matched-MSE diagnostic' if args.loss=='matched-mse' else 'Official PtyRAD amplitude loss'
        if args.fixed_probe:
            label += '; FIXED probe diagnostic, not blind baseline'
        if args.positive_amplitude_floor>0:
            label += '; positive-amplitude domain guard'
        plot_snapshot(out/f'sweep_{sweep:04d}.png',current_obj,current_probe,history,meta['object_pixel_um'],label)
        print('[AD]',json.dumps(row),flush=True)

    save(0)
    try:
        for sweep in range(1,args.sweeps+1):
            if time.perf_counter()-start > args.max_seconds:
                status = 'wall_budget_reached'
                break
            if args.grouping=='random':
                toggle_grad_requires(model,sweep)
                recon_step(batches,1,model,optimizer,None,loss,constraint,sweep,
                           compute_loss_fn=_updates.compute_loss)
                updates_done += len(batches)
            else:
                for microbatches in group_batches:
                    loss.accumulation_chunks = len(microbatches)
                    loss.group_frames = sum(len(b) for b in microbatches)
                    toggle_grad_requires(model,updates_done+1)
                    recon_step(microbatches,len(microbatches),model,optimizer,None,loss,constraint,
                               updates_done+1,compute_loss_fn=_updates.compute_loss)
                    updates_done += 1
            completed = sweep
            cumulative = float(np.sum(model.iter_times))
            if not all(torch.isfinite(t).all() for t in (model.opt_obja,model.opt_objp,model.opt_probe)):
                raise FloatingPointError('Nonfinite official AD fields')
            if sweep%args.eval_every==0 or sweep==args.sweeps:
                save(sweep)
    except Exception as exc:
        status = f'failed:{type(exc).__name__}:{exc}'
        write_json(out/'failure.json',dict(status=status,completed_sweeps=completed))
        raise
    finally:
        if status.startswith('failed'):
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                            completed_sweeps=completed,config=config),out/'failed_checkpoint.pt')
        else:
            if history[-1]['sweep'] != completed:
                save(completed)
            final_obj,final_probe = fields(model)
            np.savez(out/'reconstruction.npz',obj_rec=final_obj,probe_rec=final_probe,
                     positions_px=positions,original_frame_ids=ids,object_pixel_um=meta['object_pixel_um'],
                     config=json.dumps(config))
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                            completed_sweeps=completed,config=config),out/'checkpoint.pt')
        write_json(out/'summary.json',dict(status=status,completed_sweeps=completed,
                   optimizer_updates=updates_done,training_seconds=cumulative,
                   wall_seconds=time.perf_counter()-start,final=history[-1],
                   max_cuda_memory_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    print('OUTPUT',out.resolve(),flush=True)


if __name__=='__main__':
    main()
