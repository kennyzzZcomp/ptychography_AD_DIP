"""Bounded diagnostic: ONE first update per case, then frozen-weight readouts.

Does not edit solver parameters, saved measurements, or production code. The
rolled/occluded inputs are readout interventions only; no training uses them.
"""
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from core import UNet, Propagator, chunked_data, mpie_geometry, select_center_grid, preserve_buffers
from initialization import initial_fields
from run import network_input
from functions.addip.model import ProPtyUNet, make_field
from functions.paperrepro.solvers_addip import _initialize_object_heads
from functions.paperrepro.optics import forward_field
from functions.paperrepro.window_training import measurement_groups, scaled_amplitude_loss


def peak(array):
    return dict(value=float(array.max()), yx=list(map(int, np.unravel_index(array.argmax(), array.shape))))


def input_stats(x):
    # Each conditioning channel already has maximum 1; quantify spatial sparsity.
    a = x.detach().float().cpu().numpy()[0]
    mean = a.mean(0)
    return dict(shape=list(a.shape), mean=float(a.mean()), rms=float(np.sqrt(np.mean(a*a))),
                fraction_above_001=float(np.mean(a > .01)),
                fraction_above_01=float(np.mean(a > .1)), mean_image_peak=peak(mean))


def feature_stats(feature):
    f = feature.detach()
    return dict(max_abs=float(f.abs().max()), rms=float(f.square().mean().sqrt()),
                max_rms_across_channels=float(f.square().mean(1).sqrt().max()))


def neutral_head_parameters(net):
    return [p.detach().clone().requires_grad_(True) for p in
            (net.head_amp.weight, net.head_amp.bias, net.head_phs.weight, net.head_phs.bias)]


def decode_heads(phi, parameters, offset=None):
    w,b,v,c=parameters
    r=F.conv2d(phi,w,b,padding=1)[0,0]
    q=F.conv2d(phi,v,c,padding=1)[0]
    a=F.softplus(r if offset is None else r+offset)
    den=torch.sqrt(q[0].square()+q[1].square()+1e-8)
    return torch.complex(a*q[0]/den,a*q[1]/den)


def first_adam_head_field(phi, parameters, gradients, offset=None):
    # Closed-form FIRST Adam step, beta corrections cancel; no optimizer state
    # or model parameter is modified by these counterfactual calculations.
    with torch.no_grad():
        virtual=[p-.0005*g/(g.abs()+1e-8) for p,g in zip(parameters,gradients)]
        return decode_heads(phi,virtual,offset)


def real_case(args):
    torch.manual_seed(0)
    with np.load(args.initialization) as z:
        old_config = json.loads(str(z['config']))
    with h5py.File(args.data, 'r') as f:
        ids, coords, groups, _ = select_center_grid(f['positionx'][()], 225)
        pos, side, dx, old, _ = mpie_geometry(coords, 1024, 632.8, 6., 3.76)
        g = groups[0]
        data = np.stack([f['ptychogramx'][int(ids[i]), 488:1512, 488:1512].astype('float32')
                         / old_config['original_intensity_mean'] for i in g])
    o0, p0, _ = initial_fields('mpie', 1024, side, old, dx*1e6, 0)
    net = UNet(64, 16, checkpoint_blocks=True).cuda()
    net.set_initial_object(torch.from_numpy(o0).cuda())
    pr = torch.nn.Parameter(torch.from_numpy(p0.real).cuda())
    pi = torch.nn.Parameter(torch.from_numpy(p0.imag).cuda())
    opt = torch.optim.Adam(net.parameters(), lr=.0005)
    optp = torch.optim.Adam([pr, pi], lr=.005)
    x = network_input(data, side, 64, 'cuda')
    prop = Propagator(pos[g], 1024, sampling='mpie-integer')
    features = {}
    # Capture feature maps only during a no-grad readout to avoid retaining an
    # extra autograd graph and to leave checkpoint recomputation untouched.
    with preserve_buffers(net), torch.no_grad():
        hook = net.d1.register_forward_hook(lambda module, inp, out: features.update(d1=out.detach().clone()))
        initial = net(x)
        hook.remove()
    phi = features.pop('d1')
    neutral = neutral_head_parameters(net)
    metrics = dict(input=input_stats(x), last_features=feature_stats(phi),
                   initial_object_peak=peak(initial.abs().cpu().numpy()),
                   learning_rates=dict(network=.0005, probe=.005), gain=1,
                   tgv=0, nlm=0, optimizer_updates=1)
    opt.zero_grad(set_to_none=True); optp.zero_grad(set_to_none=True)
    obj = net(x)
    go, gp, before = chunked_data(obj, torch.complex(pr, pi), range(len(g)), data, prop, 1)
    fixed_go=go.detach().clone()
    obj.backward(go)
    pr.grad, pi.grad = gp.real.contiguous(), gp.imag.contiguous()
    opt.step(); optp.step()
    del obj, go, gp, initial
    maps = {}
    with preserve_buffers(net), torch.no_grad():
        after = net(x)
        maps['real_input_mean'] = x.mean(1)[0].cpu().numpy()
        maps['real_feature_rms'] = phi.square().mean(1).sqrt()[0].cpu().numpy()
        maps['real_original_output'] = after.abs().cpu().numpy()
        metrics['output_after_update'] = peak(maps['real_original_output'])
        # On step 1 only heads change: replay the new heads on the OLD feature
        # tensor to determine whether backbone training is needed for this peak.
        r = net.head_amp(phi)[0, 0]
        q = net.head_phs(phi)[0]
        amp = F.softplus(r+net.initial_amp_offset)
        den = torch.sqrt(q[0].square()+q[1].square()+1e-8)
        from_heads = torch.complex(amp*q[0]/den, amp*q[1]/den)
        metrics['old_features_new_heads_max_field_error'] = float((from_heads-after).abs().max())
        for name, shift in [('roll_right128', (0,128)), ('roll_up128', (-128,0))]:
            alternative = net(torch.roll(x, shift, (-2,-1)))
            a = alternative.abs().cpu().numpy()
            metrics[name] = peak(a)
            maps['real_'+name] = a
            aligned = torch.roll(alternative, (-shift[0],-shift[1]), (-2,-1))
            metrics[name]['aligned_amplitude_relative_error'] = float(
                (aligned.abs()-after.abs()).norm()/after.abs().norm())
            del alternative, aligned
        cy, cx = metrics['input']['mean_image_peak']['yx']
        for half in (8,16,32):
            occluded = x.clone()
            occluded[...,cy-half:cy+half,cx-half:cx+half] = 0
            a = net(occluded).abs().cpu().numpy()
            metrics[f'occlude_center_{half*2}'] = peak(a)
            if half == 16:
                maps['real_occluded_center32'] = a
            del occluded
    metrics['original_group_mse_before'] = before['amplitude_mse']
    metrics['original_group_mse_after'] = chunked_data(after, torch.complex(pr,pi),range(len(g)),data,prop,1,False)[2]['amplitude_mse']
    initial_probe=torch.from_numpy(p0).cuda()
    initial_object=torch.from_numpy(o0).cuda()
    ab=aa=0.
    with torch.no_grad():
        for i in range(len(g)):
            a=prop.amplitude(initial_object,initial_probe,[i]);b=torch.from_numpy(data[i:i+1]).cuda().sqrt()
            ab+=float((a*b).sum(dtype=torch.float64));aa+=float(a.square().sum(dtype=torch.float64))
    gain=ab/aa
    virtual_obj=decode_heads(phi,neutral,net.initial_amp_offset)
    fixed_grads=torch.autograd.grad(virtual_obj,neutral,grad_outputs=fixed_go)
    replay=first_adam_head_field(phi,neutral,fixed_grads,net.initial_amp_offset)
    metrics['virtual_fixed_gain_replay_max_error']=float((replay-after).abs().max())
    virtual_obj=decode_heads(phi,neutral,net.initial_amp_offset)
    prof_go,_,_=chunked_data(virtual_obj,initial_probe*gain,range(len(g)),data,prop,1)
    prof_grads=torch.autograd.grad(virtual_obj,neutral,grad_outputs=prof_go)
    virtual=first_adam_head_field(phi,neutral,prof_grads,net.initial_amp_offset)
    metrics['profiled_gain_first_step_counterfactual']=dict(gain=gain,output_peak=peak(virtual.abs().cpu().numpy()),
        note='Algebraic first Adam head step only; production parameters/optimizer unchanged')
    maps['real_profiled_gain_virtual_output']=virtual.abs().cpu().numpy()
    del net, pr, pi, x, after, phi, opt, optp
    gc.collect(); torch.cuda.empty_cache()
    return metrics, maps


def simulation_case(path):
    """Original simulation network/physics, one original first update only.

    First-step amplitude TGV and its gradient are zero on the constant O0. This
    diagnostic therefore evaluates only its nonzero data-gradient contribution.
    """
    torch.manual_seed(0)
    with h5py.File(path,'r') as f:
        cfg = json.loads(f.attrs['scene_config_json'])
        data = torch.from_numpy(f['intensity'][()].astype('float32')).cuda()
        target = torch.from_numpy(f['diffamp'][()].astype('float32')).cuda()
        positions = torch.from_numpy(f['positions_px'][()].astype('int64')).cuda()
        p0 = torch.from_numpy(f['probe_init'][()].astype('complex64')).cuda()
        quad = torch.from_numpy(f['quadratic_phase'][()].astype('complex64')).cuda()
    group = measurement_groups(10,'sparse',5)[0]
    n, m = cfg['N'], cfg['obj_size']
    side = (m+7)//8*8
    low = (side-m)//2
    d, extra = m-data.shape[-1], side-m
    x = F.pad(data[None,group],(d//2,d-d//2,d//2,d-d//2))
    x = F.pad(x,(extra//2,extra-extra//2,extra//2,extra-extra//2))
    x = x/x.amax((-2,-1),keepdim=True).clamp_min(1e-12)
    net = ProPtyUNet(25,16,n_fields=1,ph_ch=2,skip_mode='concat').cuda()
    _initialize_object_heads(net,0.)
    pr = torch.nn.Parameter(p0.real.clone());pi = torch.nn.Parameter(p0.imag.clone())
    opt = torch.optim.Adam(net.parameters(),lr=.0005)
    optp = torch.optim.Adam([pr,pi],lr=.01)
    features={}
    with preserve_buffers(net),torch.no_grad():
        hook=net.d1.register_forward_hook(lambda mod,inp,out: features.update(d1=out.detach().clone()))
        net(x);hook.remove()
    phi=features.pop('d1')
    neutral=neutral_head_parameters(net)
    metrics=dict(input=input_stats(x),last_features=feature_stats(phi),
                 patch=n,fft_size=cfg.get('fft_size',0) or n,network_side=side,object_side=m,
                 initial_probe='ones',gain='profiled',network_width=16,
                 learning_rates=dict(network=.0005,probe=.01),optimizer_updates=1)
    amp,ph=net(x);obj=make_field(amp[0],ph[:2])[low:low+m,low:low+m]
    prediction=forward_field(obj,torch.complex(pr,pi),positions[group],quad,n,fft_size=cfg.get('fft_size',0)).abs()
    loss,_=scaled_amplitude_loss(prediction,target[group]);metrics['profiled_loss_before']=float(loss)
    metrics['initial_fitted_gain']=float((prediction*target[group]).sum()/prediction.square().sum())
    loss.backward();opt.step();optp.step()
    with preserve_buffers(net),torch.no_grad():
        amp,ph=net(x);after=make_field(amp[0],ph[:2])[low:low+m,low:low+m].abs().cpu().numpy()
    metrics['output_after_update']=peak(after)
    virtual_obj=decode_heads(phi,neutral)[low:low+m,low:low+m]
    pred=forward_field(virtual_obj,p0,positions[group],quad,n,fft_size=cfg.get('fft_size',0)).abs()
    fixed_loss=(pred-target[group]).square().mean()
    fixed_grads=torch.autograd.grad(fixed_loss,neutral)
    virtual=first_adam_head_field(phi,neutral,fixed_grads)[low:low+m,low:low+m]
    metrics['fixed_gain_first_step_counterfactual']=dict(output_peak=peak(virtual.abs().cpu().numpy()),
        min_amplitude=float(virtual.abs().min()),note='Algebraic first head step with g=1, unchanged inputs/initial fields')
    maps=dict(input_mean=x.mean(1)[0].cpu().numpy(),feature_rms=phi.square().mean(1).sqrt()[0].cpu().numpy(),output=after)
    del net,x,phi,obj,amp,ph,prediction,loss,data,target,pr,pi,opt,optp
    gc.collect();torch.cuda.empty_cache()
    return metrics,maps


def plot(maps,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    s=maps['real_original_output'].shape[0];lo,hi=s//2-200,s//2+200
    keys=['real_original_output','real_roll_right128','real_roll_up128',
          'real_occluded_center32','sim64_output','sim128_output']
    titles=['Real: first update','Same weights: input shifted right 128 px',
            'Same weights: input shifted up 128 px','Same weights: input center 32x32 masked',
            'Simulation FFT64: first update','Simulation FFT128: first update']
    vmax=max(float(maps[k].max()) for k in keys)
    for ax,key,title in zip(axs.flat,keys,titles):
        a=maps[key]
        if key.startswith('real_'):
            a=a[lo:hi,lo:hi];extent=[lo,hi,hi,lo]
        else:extent=[0,a.shape[1],a.shape[0],0]
        im=ax.imshow(a,cmap='gray',vmin=0,vmax=vmax,extent=extent)
        ax.set(title=title,xlabel='x (px)',ylabel='y (px)')
    fig.colorbar(im,ax=list(axs.flat),label='Object amplitude (common display scale)',shrink=.8)
    fig.suptitle('Bounded diagnostics; shifted/masked inputs are frozen-weight readouts only')
    fig.savefig(dest/'center_peak_readouts.png',dpi=180)
    plt.close(fig)


def route_diagnostic(args):
    """First-step counterfactuals with identical fields and physical gradients.

    No model parameter or optimizer is updated. Hooks temporarily zero feature
    routes for diagnosis; they do not implement a newly trained architecture.
    """
    torch.manual_seed(0)
    with np.load(args.initialization) as z:
        intensity_mean=json.loads(str(z['config']))['original_intensity_mean']
    with h5py.File(args.data,'r') as f:
        ids,coords,groups,_=select_center_grid(f['positionx'][()],225)
        pos,side,dx,old,_=mpie_geometry(coords,1024,632.8,6.,3.76)
        group=groups[0]
        data=np.stack([f['ptychogramx'][int(ids[i]),488:1512,488:1512].astype('float32')/intensity_mean for i in group])
    o0,p0,_=initial_fields('mpie',1024,side,old,dx*1e6,0)
    net=UNet(64,16,checkpoint_blocks=True).cuda()
    net.set_initial_object(torch.from_numpy(o0).cuda())
    x=network_input(data,side,64,'cuda')
    prop=Propagator(pos[group],1024,sampling='mpie-integer')
    probe=torch.from_numpy(p0).cuda()

    def extract(conditioning,paths=()):
        saved={};hooks=[]
        def mask_half(module,inputs):
            a=inputs[0];h=a.shape[1]//2
            if paths==('shallow_only',):
                return (torch.cat((torch.zeros_like(a[:,:h]),a[:,h:]),1),)
            return (torch.cat((a[:,:h],torch.zeros_like(a[:,h:])),1),)
        if paths==('shallow_only',):
            hooks.append(net.d1.register_forward_pre_hook(mask_half))
        else:
            for name in paths:hooks.append(getattr(net,name).register_forward_pre_hook(mask_half))
        hooks.append(net.d1.register_forward_hook(lambda mod,inp,out:saved.update(phi=out.detach().clone())))
        try:
            with preserve_buffers(net),torch.no_grad():field=net(conditioning).detach()
        finally:
            for hook in hooks:hook.remove()
        return saved['phi'],field

    base_phi,base=extract(x)
    go,gp,before=chunked_data(base,probe,range(len(group)),data,prop,1)
    with torch.no_grad():
        pp=probe.real-.005*gp.real/(gp.real.abs()+1e-8)
        pq=probe.imag-.005*gp.imag/(gp.imag.abs()+1e-8)
        probe_after=torch.complex(pp,pq)
    neutral=neutral_head_parameters(net)
    rows=[];maps={}
    variants=['raw','no_shallow_skip','no_all_skips','shallow_route_only',
              'sqrt_input','fixed_noise','fixed_noise_matched_rms','pixel_rms_features']
    for name in variants:
        if name=='raw':phi=base_phi
        elif name=='no_shallow_skip':phi,_=extract(x,('d1',))
        elif name=='no_all_skips':phi,_=extract(x,('d3','d2','d1'))
        elif name=='shallow_route_only':phi,_=extract(x,('shallow_only',))
        elif name=='sqrt_input':
            alternative=x.sqrt();phi,_=extract(alternative);del alternative
        elif name in ('fixed_noise','fixed_noise_matched_rms'):
            rng=torch.Generator(device='cuda').manual_seed(20261019)
            alternative=torch.rand(x.shape,device='cuda',generator=rng)
            if name=='fixed_noise':alternative*=.1
            else:alternative*=x.square().mean().sqrt()*np.sqrt(3.)
            phi,_=extract(alternative);del alternative
        else:
            phi=base_phi/base_phi.square().mean(1,keepdim=True).sqrt().clamp_min(1e-6)
        obj=decode_heads(phi,neutral,net.initial_amp_offset)
        initial_error=float((obj-base).abs().max())
        if initial_error>3e-7:raise RuntimeError('Counterfactual changed initial field')
        gradients=torch.autograd.grad(obj,neutral,grad_outputs=go)
        virtual=first_adam_head_field(phi,neutral,gradients,net.initial_amp_offset)
        only_o=chunked_data(virtual,probe,range(len(group)),data,prop,1,False)[2]['amplitude_mse']
        both=chunked_data(virtual,probe_after,range(len(group)),data,prop,1,False)[2]['amplitude_mse']
        a=virtual.abs().cpu().numpy()
        cy=cx=side//2
        row=dict(name=name,initial_field_max_error=initial_error,features=feature_stats(phi),
                 output_peak=peak(a),center_max=float(a[cy-64:cy+64,cx-64:cx+64].max()),
                 min_amplitude=float(a.min()),relative_object_change=float((virtual-base).norm()/base.norm()),
                 original_data_loss_before=before['amplitude_mse'],loss_after_object_only=only_o,
                 loss_after_both=both)
        rows.append(row);maps[name]=a
        print('ROUTE',json.dumps(row),flush=True)
        if name!='raw':del phi
        del obj,virtual,gradients
    report=dict(seed=0,independent_noise_seed=20261019,device=torch.cuda.get_device_name(),torch=torch.__version__,
                actual_optimizer_updates=0,virtual_first_step_lr=dict(network=.0005,probe=.005),gain=1,
                fields_measurements_positions_and_physical_gradients_identical=True,
                caveat='First-step feature interventions only; not full reconstructions or retrained no-skip networks',results=rows)
    (args.out/'route_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(args.out/'route_maps.npz',**maps)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,4,figsize=(15,8),layout='constrained')
    lo,hi=side//2-200,side//2+200
    for ax,name,row in zip(axs.flat,variants,rows):
        im=ax.imshow(maps[name][lo:hi,lo:hi],cmap='gray',vmin=0,vmax=2.5,extent=[lo,hi,hi,lo])
        ax.set(title=f'{name}\nGlobal max {row["output_peak"]["value"]:.3f}',xlabel='x (px)',ylabel='y (px)')
    fig.colorbar(im,ax=list(axs.flat),label='Object amplitude',shrink=.75)
    fig.suptitle('Identical first physical gradient; virtual first head update only')
    fig.savefig(args.out/'route_comparison.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'experiment/ptydata_33.h5')
    p.add_argument('--initialization',type=Path,default=Path('C:/Users/kennyzz/Downloads/initialization.npz'))
    p.add_argument('--out',type=Path,default=HERE/'results'/datetime.now().strftime('center_peak_diagnostic_%Y%m%d_%H%M%S_%f'))
    p.add_argument('--routes-only',action='store_true',help='Identical-gradient feature/input counterfactuals; no optimizer steps')
    args=p.parse_args()
    if args.out.exists():p.error('Choose a fresh output directory')
    if not torch.cuda.is_available():p.error('This bounded full-size diagnostic requires CUDA')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if args.routes_only:
        route_diagnostic(args)
        print('OUTPUT',args.out.resolve(),flush=True)
        return
    report=dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,seed=0)
    report['real'],maps=real_case(args)
    print('REAL',json.dumps(report['real']),flush=True)
    paths={'sim64':ROOT/'results_paper/ptyinr_obj612_grid10_d59_s10_center/scene_obj612_grid10_d59_s10_center.h5',
           'sim128':ROOT/'results_paper/sampling128/fft128/data/clean.h5'}
    for name,path in paths.items():
        report[name],extra=simulation_case(path)
        maps.update({name+'_'+k:v for k,v in extra.items()})
        print(name,json.dumps(report[name]),flush=True)
    (args.out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(args.out/'maps.npz',**maps)
    plot(maps,args.out)
    print('OUTPUT',args.out.resolve(),flush=True)


if __name__=='__main__':main()
