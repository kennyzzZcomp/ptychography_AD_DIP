"""Local reference-probe diagnostic using the UNMODIFIED real_ptycho.train.

The process-local adapter supplies the calibrated reference P and suppresses
only its Adam step. Network constructors, input encoding, field derivatives,
ABCD fusion, residual stage and network Adam steps remain the original code.
This is NOT a blind reconstruction baseline or a fix to the original method.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import weakref

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/'real_ptycho'))
_spec = importlib.util.spec_from_file_location('unchanged_real_ptycho_entry',ROOT/'real_ptycho/run.py')
original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(original)
RealAdam = torch.optim.Adam


class FixedProbeAdam(RealAdam):
    """Keep the original probe optimizer interface, but never update P."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.suppressed_steps = 0
        self.on_suppressed_step = None

    def step(self,closure=None):
        if closure is not None:
            raise ValueError('This fixed-probe diagnostic does not support a closure')
        self.suppressed_steps += 1
        if self.on_suppressed_step is not None:
            self.on_suppressed_step()
        return None


def load_data(prepared,reference,seed):
    meta = json.loads((prepared/'prepared.json').read_text(encoding='utf-8'))
    if meta['seed']!=seed or meta['n']!=1024:
        raise ValueError('This diagnostic requires the audited seed-0 1024 cache')
    data = np.load(prepared/'intensity.npy',mmap_mode='r')
    with np.load(prepared/'fields.npz',allow_pickle=False) as z:
        requested_obj,positions,ids = [z[k].copy() for k in ('obj','positions','ids')]
    with np.load(reference,allow_pickle=False) as z:
        probe = z['probe_rec'].copy()
        reference_config = json.loads(str(z['config']))
    scale = reference_config.get('intensity_mean_before_scaling',reference_config.get('original_intensity_mean'))
    if scale is None or abs(scale/meta['common_intensity_mean']-1)>1e-6:
        raise ValueError('Reference P must use the same common intensity scale')
    if probe.shape!=(1024,1024) or not np.isfinite(probe).all():
        raise ValueError('Reference must contain a finite full 1024-square P')
    coords = np.asarray(meta['positions_mm'])
    pos,side,dx,legacy,alignment = original.mpie_geometry(coords,1024,632.8,6.,3.76)
    if not np.array_equal(pos,positions) or requested_obj.shape!=(side,side):
        raise ValueError('Cached integer slices or object canvas do not match the original adapter')
    rows = np.asarray(meta['selection']['grid_rows_zero_based'])
    cols = np.asarray(meta['selection']['grid_cols_zero_based'])
    groups = [np.flatnonzero((rows%2==a)&(cols%2==b)) for a,b in ((0,0),(0,1),(1,0),(1,1))]
    return data,requested_obj,probe,positions,ids,groups,legacy,alignment,meta


@contextmanager
def adapt_training(reference_probe,requested_obj,out,positions,ids,max_seconds):
    """Patch only this process; restore all bindings even after OOM/time limit."""
    old_initial,old_adam = original.initial_fields,torch.optim.Adam
    old_unet,old_chunked = original.UNet,original.chunked_data
    state = dict(stage=1,updates={1:0,2:0},network=None,optimizer=None,
                 probe_optimizer=None,suppressed_probe_steps=0,evaluations=0,first_steps=[],start=time.perf_counter())
    fixed_tensor = torch.from_numpy(reference_probe).cuda()
    (out/'snapshots').mkdir()

    def check_budget():
        if time.perf_counter()-state['start']>max_seconds:
            raise TimeoutError('Explicit diagnostic wall budget reached')

    def supplied_initial(*args,**kwargs):
        obj,unused_p,metadata = old_initial(*args,**kwargs)
        if not np.array_equal(obj,requested_obj):
            raise ValueError('Original O0 changed unexpectedly')
        metadata = dict(metadata,original_initialization_rule=metadata['rule'],
                        rule='Original flat O0; successful mPIE final P FIXED, not the circular initial P',
                        probe_rule='FIXED successful mPIE final P, diagnostic only',
                        original_circle_replaced=True,probe_update_disabled=True)
        return obj,reference_probe.copy(),metadata

    def unchanged_network(*args,**kwargs):
        module = old_unet(*args,**kwargs)
        state['stage'] = 2 if module.residual else 1
        state['network'] = weakref.ref(module)
        return module

    def optimizer_factory(params,*args,**kwargs):
        values = list(params)
        is_probe = len(values)==2 and all(tuple(p.shape)==tuple(reference_probe.shape) for p in values)
        if is_probe:
            opt = FixedProbeAdam(values,*args,**kwargs)
            def suppressed():
                state['suppressed_probe_steps'] += 1
            opt.on_suppressed_step = suppressed
            state['probe_optimizer'] = weakref.ref(opt)
            return opt
        opt = RealAdam(values,*args,**kwargs)
        state['optimizer'] = weakref.ref(opt)
        stage = state['stage']
        def counted_step(optimizer,step_args,step_kwargs):
            state['updates'][stage] += 1
        opt.register_step_post_hook(counted_step)
        return opt

    def monitored_data(obj,probe,frame_ids,measurements,propagator,chunk=1,gradient=True):
        check_budget()
        values = old_chunked(obj,probe,frame_ids,measurements,propagator,chunk,gradient)
        stage = state['stage']
        step = state['updates'][stage]
        if not torch.equal(probe.detach(),fixed_tensor):
            raise AssertionError('Reference probe changed in the supposedly frozen diagnostic')
        if gradient and step<4:
            go,gp,stats = values
            a = obj.detach().abs()
            row = dict(stage=stage,before_update=step+1,object_amp_max=float(a.max()),
                       object_amp_mean=float(a.mean()),field_gradient_l2=float(go.norm()),
                       probe_gradient_l2_computed_but_unused=float(gp.norm()),**stats)
            state['first_steps'].append(row)
            original.write_json(out/'first_step_diagnostics.json',state['first_steps'])
            print('[NN first steps]',json.dumps(row),flush=True)
        if not gradient and len(frame_ids)==len(ids):
            state['evaluations'] += 1
            np.savez(out/'snapshots'/f'stage{stage}_update{step:04d}.npz',
                     obj_rec=obj.detach().cpu().numpy(),probe_rec=probe.detach().cpu().numpy(),
                     positions_px=positions,original_frame_ids=ids)
            print(f'[NN checkpoint] S{stage} update {step}; P bitwise unchanged',flush=True)
        return values

    original.initial_fields = supplied_initial
    original.UNet = unchanged_network
    original.chunked_data = monitored_data
    torch.optim.Adam = optimizer_factory
    try:
        yield state
    except Exception as exc:
        # Preserve the last in-memory model without pretending it is a completed
        # post-update object field. The normal entry saves S1/S2 at completion.
        net = state['network']() if state['network'] is not None else None
        opt = state['optimizer']() if state['optimizer'] is not None else None
        torch.save(dict(network=net.state_dict() if net is not None else None,
                        optimizer=opt.state_dict() if opt is not None else None,
                        stage=state['stage'],updates=state['updates'],
                        error=f'{type(exc).__name__}: {exc}'),out/'interrupted_weights.pt')
        raise
    finally:
        original.initial_fields,original.UNet,original.chunked_data = old_initial,old_unet,old_chunked
        torch.optim.Adam = old_adam


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=HERE/'results/prepared_20261019_182658_419540')
    p.add_argument('--reference',type=Path,default=Path('C:/Users/kennyzz/Downloads/reconstruction_mpie_100ITER.npz'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage1-steps',type=int,default=500)
    p.add_argument('--stage2-steps',type=int,default=200)
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--max-seconds',type=float,default=1800.)
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error('CUDA required; no CPU or lower-resolution fallback')
    torch.set_num_threads(2)
    data,obj0,probe,positions,ids,groups,legacy,alignment,meta = load_data(args.prepared,args.reference,0)
    a = original.arguments(['--data',meta['data'],'--out',str(args.out),'--num-frames','225',
        '--detector-crop','1024','--probe-size','0','--initialization','mpie',
        '--object-sampling','mpie-integer','--stage1-feature-norm','none',
        '--stage1-steps',str(args.stage1_steps),'--stage2-steps',str(args.stage2_steps),
        '--stage1-lr-net','0.0005','--stage2-lr-net','0.0005',
        '--stage1-lr-probe','0.005','--stage2-lr-probe','0.005',
        '--tgv','0','--nlm-ratio','0','--eval-every',str(args.eval_every),'--chunk','1'])
    a.out.mkdir(parents=True,exist_ok=False)
    config = {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(object_size=len(obj0),detector_size=1024,object_pixel_um=meta['object_pixel_um'],
        original_intensity_mean=meta['common_intensity_mean'],
        selected_input_sha256=meta['selected_input_sha256'],selected_frame_ids_zero_based=ids.tolist(),
        positions_object_pixels=positions.tolist(),position_input_unit='mm',position_scale_to_mm=1.,
        initialization_positions_px=legacy.tolist(),sampling_alignment=alignment,
        groups=[g.tolist() for g in groups],selection=meta['selection'],stage1_input_channels=64,
        amplitude_gain=1.,amplitude_gain_mode='fixed',probe_parameter_size=1024,probe_padding_each_side=0,
        propagation='Original centered unitary Fraunhofer; unchanged mPIE integer slices',
        diagnostic_only=True,blind_reconstruction=False,
        reference_probe=str(args.reference.resolve()),probe_updates='Explicitly suppressed, fixed reference',
        seed=0,torch_version=torch.__version__,
        caveats=['Reference reconstructed P is supplied information, NOT a blind baseline.',
                 'Compared with uploaded failure, probe starting value AND its update status differ.',
                 'Current integer sampler is retained; uploaded old NPZ used bilinear sampling.',
                 'No experimental ground truth; use fit and structural diagnostics, not PSNR.'])
    sources = [Path(__file__),ROOT/'real_ptycho/run.py',ROOT/'real_ptycho/core.py',ROOT/'real_ptycho/initialization.py']
    config['source_sha256'] = {str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}
    original.write_json(a.out/'diagnostic_plan.json',config)
    # Freeze the exact sources used by this new diagnostic; never overwrite
    # originals or rely solely on a moving adapter's recorded checksum.
    source_folder = a.out/'source_snapshot'
    source_folder.mkdir()
    for source in sources:
        (source_folder/(source.parent.name+'_'+source.name)).write_bytes(source.read_bytes())
    started = time.perf_counter()
    print('[NN diagnostic] Original train() / U16 -> U32; reference P fixed; no TGV/NLM',flush=True)
    try:
        with adapt_training(probe,obj0,a.out,positions,ids,args.max_seconds) as state:
            original.train(a,data,positions,len(obj0),groups,config)
            with np.load(a.out/'reconstruction.npz',allow_pickle=False) as z:
                unchanged = np.array_equal(z['probe_rec'],probe)
            if not unchanged:
                raise AssertionError('Saved probe differs from the reference')
            info = dict(status='complete',network_updates=state['updates'],probe_bitwise_unchanged=unchanged,
                        suppressed_probe_steps=state['suppressed_probe_steps'],
                        wall_seconds=time.perf_counter()-started,
                        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        original_sources_unchanged=all(hashlib.sha256(s.read_bytes()).hexdigest()==config['source_sha256'][str(s.relative_to(ROOT))] for s in sources))
            original.write_json(a.out/'diagnostic_summary.json',info)
            print('[NN done]',json.dumps(info),flush=True)
    except Exception as exc:
        original.write_json(a.out/'diagnostic_failure.json',dict(type=type(exc).__name__,message=str(exc),
                            wall_seconds=time.perf_counter()-started))
        raise
    print('OUTPUT',a.out.resolve(),flush=True)


if __name__=='__main__':
    main()
