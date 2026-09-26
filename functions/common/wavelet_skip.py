"""One-level orthonormal Haar filtering of encoder skips (not decoder upsampling)."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def haar_dwt(x):
    """Return LL/LH/HL/HH and original size; replicate-pad odd edges."""
    shape = x.shape[-2:]
    x = F.pad(x, (0, shape[1] % 2, 0, shape[0] % 2), mode="replicate")
    a, b, c, d = x[..., ::2, ::2], x[..., ::2, 1::2], x[..., 1::2, ::2], x[..., 1::2, 1::2]
    return torch.stack(((a+b+c+d)/2, (a-b+c-d)/2,
                        (a+b-c-d)/2, (a-b-c+d)/2), dim=2), shape


def haar_idwt(bands, shape):
    ll, lh, hl, hh = bands.unbind(2)
    a, b = (ll+lh+hl+hh)/2, (ll-lh+hl-hh)/2
    c, d = (ll+lh-hl-hh)/2, (ll-lh-hl+hh)/2
    top = torch.stack((a, b), dim=-1).flatten(-2)
    bottom = torch.stack((c, d), dim=-1).flatten(-2)
    x = torch.stack((top, bottom), dim=-2).flatten(-3, -2)
    return x[..., :shape[0], :shape[1]]


class WaveletSkip(nn.Module):
    """Three trainable nonnegative thresholds per skip, shared over channels.

    Thresholds are in feature units, not photon-noise estimates. LL is untouched.
    identity=True is the DWT/IDWT numerical control, with no learned parameters.
    """
    def __init__(self, threshold=.01, identity=False):
        super().__init__()
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("wavelet threshold initialization must be finite and > 0")
        self.identity = identity
        if not identity:
            self.raw_threshold = nn.Parameter(torch.full((3,), threshold + math.log(-math.expm1(-threshold))))
        self.capture_stats = False
        self.last_stats = None

    def forward(self, x):
        bands, shape = haar_dwt(x)
        if self.identity:
            return haar_idwt(bands, shape)
        high = bands[:, :, 1:]
        threshold = F.softplus(self.raw_threshold).view(1, 1, 3, 1, 1)
        filtered = high.sign() * F.relu(high.abs() - threshold)
        if self.capture_stats:
            with torch.no_grad():
                self.last_stats = {
                    "thresholds": threshold.flatten().detach().cpu().tolist(),
                    "zero_fraction": (filtered == 0).float().mean((0, 1, 3, 4)).cpu().tolist(),
                    "input_zero_fraction": (high == 0).float().mean((0, 1, 3, 4)).cpu().tolist(),
                }
        return haar_idwt(torch.cat((bands[:, :, :1], filtered), dim=2), shape)
