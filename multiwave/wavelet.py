"""All-band learned Haar skip fusion; no thresholds or discarded subbands."""
import torch
from torch import nn
from functions.common.wavelet_skip import haar_dwt, haar_idwt


class DWTConcatSkip(nn.Module):
    """Mix concatenated LL/LH/HL/HH at half resolution, then invert Haar.

    A zero-initialized residual 1x1 convolution starts as identity. This changes
    encoder skips only; the original pooling and decoder remain in place.
    """
    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Conv2d(4*channels, 4*channels, 1)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, x):
        bands, shape = haar_dwt(x)  # B,C,4,H/2,W/2
        packed = bands.transpose(1, 2).flatten(1, 2)  # B,4C,H/2,W/2
        fused = packed + self.mix(packed)
        bands = fused.reshape(x.shape[0], 4, x.shape[1], *fused.shape[-2:]).transpose(1, 2)
        return haar_idwt(bands, shape)
