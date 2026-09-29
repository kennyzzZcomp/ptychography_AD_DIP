"""Optional full-resolution residual path for the shared amplitude U-Net."""
import torch
from torch import nn
from functions.addip.model import ProPtyUNet


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                    nn.LeakyReLU(.2, inplace=False),
                                    nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.layers(x)


class FullResolutionHead(nn.Module):
    """Two residual blocks, no pooling/upsampling/BN, signed zero-init output."""
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.Sequential(ResidualBlock(channels), ResidualBlock(channels))
        self.out = nn.Conv2d(channels, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.blocks(x))


class DetailUNet(ProPtyUNet):
    """Reuse the original encoder/decoder, plus a direct path from e1 to z.

    e1 still has the original BatchNorm. Only the added branch is BN-free.
    Shallow features are learned from measurements, not true object edges.
    """
    def __init__(self, in_ch, base, **kwargs):
        super().__init__(in_ch, base, **kwargs)
        self.detail_head = FullResolutionHead(base)

    def forward_amplitude_raw(self, x):
        # Evaluate each encoder and its BN exactly once, as in ProPtyUNet.
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        xb = self.bot(self.pool(x3))
        y = self.d3(torch.cat([self.u3(xb), self.skip_filters[2](x3)], 1))
        y = self.d2(torch.cat([self.u2(y), self.skip_filters[1](x2)], 1))
        y = self.d1(torch.cat([self.u1(y), self.skip_filters[0](x1)], 1))
        return self.head_amp(y) + self.detail_head(x1)
