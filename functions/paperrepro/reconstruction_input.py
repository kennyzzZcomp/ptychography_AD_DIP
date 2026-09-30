"""One-shot reconstructed-field conditioning for the second progressive stage.

Only predictions are accepted here; neither truth fields nor measured intensities
are used to build the input channels. The physical objective is unchanged.
"""
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from functions.paperrepro.input_channels import SelectedInputConv

CHANNELS = ['object_amplitude', 'object_cos_phase', 'object_sin_phase',
            'probe_amplitude', 'probe_cos_phase', 'probe_sin_phase']


@contextmanager
def snapshot_readout(net):
    """Training-mode readout without a second persistent BN statistics update."""
    saved = [(b, b.detach().clone()) for b in net.buffers()]
    try:
        with torch.no_grad():
            yield
    finally:
        with torch.no_grad():
            for buffer, value in saved:
                buffer.copy_(value)


def reconstruction_channels(obj, probe, net_size, include_probe=True):
    """Encode amplitude + unit-circle phase; center-pad, never resize probe.

Each amplitude is divided by its own snapshot maximum. Phase at zero amplitude
is represented by (0,0). Padding is zero for all channels. Returned input is a
detached copy, fixed throughout stage 2; normalization is not applied to fields
used in propagation.
"""
    if (obj.ndim != 2 or not obj.is_complex() or obj.shape[0] != obj.shape[1]
            or not 0 < obj.shape[0] <= net_size):
        raise ValueError('Expected square complex object and object <= net_size')
    if include_probe and (probe is None or probe.ndim != 2 or not probe.is_complex()
            or probe.shape[0] != probe.shape[1] or not 0 < probe.shape[0] <= obj.shape[0]):
        raise ValueError('Expected square complex probe and probe <= object')

    def encode(field):
        field = field.detach()
        amp = field.abs()
        scale = amp.max().clamp_min(1e-12)
        denominator = amp.clamp_min(1e-12)
        return torch.stack((amp / scale, field.real / denominator, field.imag / denominator)), scale.item()

    def pad_to(field, size):
        d = size - field.shape[-1]
        return F.pad(field, (d//2, d-d//2, d//2, d-d//2))

    o, oscale = encode(obj)
    blocks = [pad_to(o, net_size)]
    metadata = {'channels': CHANNELS[:3], 'object_amplitude_scale': oscale}
    if include_probe:
        p, pscale = encode(probe)
        # Match shared_probe_field's two-stage coordinate convention even for odd pads.
        p = pad_to(p, obj.shape[0])
        blocks.append(pad_to(p, net_size))
        metadata.update(channels=CHANNELS, probe_amplitude_scale=pscale)
    result = torch.cat(blocks)[None].detach().clone()
    if not torch.isfinite(result).all():
        raise ValueError('Non-finite reconstructed conditioning input')
    return result, {**metadata, 'input_shape': list(result.shape), 'probe_in_input': include_probe}


def replace_input_stem(net, optimizer, seed, in_channels=6):
    """Replace diffraction stem by a reconstructed-field stem, retaining other Adam state.

Diffraction-channel kernels have no one-to-one meaning for object/probe channels.
Use standard Conv2d initialization from an isolated CPU RNG; do not duplicate or
average diffraction kernels. The rest of the same U-Net is left untouched.
"""
    old = net.e1.f[0]
    conv = old.conv if isinstance(old, SelectedInputConv) else old
    if not isinstance(conv, nn.Conv2d) or conv.groups != 1:
        raise ValueError('Expected ungrouped first convolution')
    old_params = list(old.parameters())
    old_ids = {id(p) for p in old_params}
    owners = [g for g in optimizer.param_groups if any(id(p) in old_ids for p in g['params'])]
    if len(owners) != 1 or not old_ids <= {id(p) for p in owners[0]['params']}:
        raise ValueError('Input stem parameters must belong to one optimizer group')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                        conv.padding, conv.dilation, bias=conv.bias is not None,
                        padding_mode=conv.padding_mode)
    new = new.to(device=conv.weight.device, dtype=conv.weight.dtype)
    new.train(old.training)
    group = owners[0]
    group['params'] = [p for p in group['params'] if id(p) not in old_ids] + list(new.parameters())
    for p in old_params:
        optimizer.state.pop(p, None)
    net.e1.f[0] = new
    return {'old_stem_parameters': sum(p.numel() for p in old_params),
            'new_stem_parameters': sum(p.numel() for p in new.parameters()),
            'stem_initialization': 'fresh torch Conv2d, isolated CPU RNG', 'stem_seed': int(seed),
            'optimizer_reset': 'first convolution only; other states retained'}
