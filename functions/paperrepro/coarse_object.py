"""Coarse network representation; propagation and detector sampling unchanged."""
import torch
import torch.nn.functional as F


def half_input(x):
    """Area-resize fixed network input, then pad for three U-Net pool levels."""
    shape = tuple((s + 1) // 2 for s in x.shape[-2:])
    small = F.interpolate(x, size=shape, mode='area')
    dh, dw = (-shape[0]) % 8, (-shape[1]) % 8
    top, left = dh // 2, dw // 2
    return F.pad(small, (left, dw-left, top, dh-top)), (top, left, *shape)


def lift_field(field, crop, size):
    """Interpolate complex Cartesian components, never wrapped phase angles."""
    top, left, h, w = crop
    field = field[top:top+h, left:left+w]
    parts = torch.stack((field.real, field.imag))[None]
    parts = F.interpolate(parts, size=size, mode='bilinear', align_corners=False)[0]
    return torch.complex(parts[0], parts[1])
