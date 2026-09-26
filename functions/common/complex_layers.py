"""Complex layers represented by (real, imag) float tensors.

Complex Glorot initialization and covariance BN follow the building blocks of
Trabelsi et al., Deep Complex Networks (ICLR 2018). No holomorphic/unitary claim.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super().__init__()
        self.stride, self.padding, self.dilation = stride, padding, dilation
        k = _pair(kernel_size)
        self.weight_real = nn.Parameter(torch.empty(out_channels, in_channels, *k))
        self.weight_imag = nn.Parameter(torch.empty_like(self.weight_real))
        std = 1 / math.sqrt((in_channels + out_channels) * math.prod(k))
        nn.init.normal_(self.weight_real, std=std)
        nn.init.normal_(self.weight_imag, std=std)
        self.bias_real = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.bias_imag = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def _conv(self, x, w):
        return F.conv2d(x, w, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def forward(self, z):
        r, i = z
        yr = self._conv(r, self.weight_real) - self._conv(i, self.weight_imag)
        yi = self._conv(r, self.weight_imag) + self._conv(i, self.weight_real)
        if self.bias_real is not None:
            yr = yr + self.bias_real[None, :, None, None]
            yi = yi + self.bias_imag[None, :, None, None]
        return yr, yi


class ComplexConvTranspose2d(ComplexConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,
                 padding=1, output_padding=1, bias=True):
        # Transposed-convolution weight layout is (in_channels, out_channels, K, K).
        super().__init__(out_channels, in_channels, kernel_size, stride, padding, bias=False)
        self.output_padding = output_padding
        self.bias_real = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.bias_imag = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def _conv(self, x, w):
        return F.conv_transpose2d(x, w, stride=self.stride, padding=self.padding,
                                  output_padding=self.output_padding)


class ComplexModReLU(nn.Module):
    def __init__(self, channels, bias_init=-.01, eps=1e-6):
        super().__init__()
        self.bias = nn.Parameter(torch.full((1, channels, 1, 1), bias_init))
        self.eps = eps

    def forward(self, z):
        r, i = z
        # Smooth norm avoids sqrt'(0); a common scale preserves phase for nonzero z.
        radius = torch.sqrt(r.square() + i.square() + self.eps**2)
        gain = F.relu(radius + self.bias) / radius
        return gain*r, gain*i


class ComplexReLU(nn.Module):
    def forward(self, z):
        return F.relu(z[0]), F.relu(z[1])


class ComplexBatchNorm2d(nn.Module):
    """Per-channel 2x2 whitening over B,H,W, with population-moment EMA.

    Symmetric learned affine matrix (not constrained positive-definite).
    Regularization is added to both covariance diagonals, not to cross covariance.
    """
    def __init__(self, channels, eps=1e-5, momentum=.1, affine=True):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.register_buffer('running_mean', torch.zeros(2, channels))
        self.register_buffer('running_covar', torch.stack((torch.ones(channels),
                                                         torch.ones(channels), torch.zeros(channels))))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.stack((torch.full((channels,), 2**-.5),
                                                    torch.full((channels,), 2**-.5), torch.zeros(channels))))
            self.bias = nn.Parameter(torch.zeros(2, channels))

    def forward(self, z):
        r, i = z
        if self.training:
            mean = torch.stack((r.mean((0, 2, 3)), i.mean((0, 2, 3))))
        else:
            mean = self.running_mean
        r = r - mean[0][None, :, None, None]
        i = i - mean[1][None, :, None, None]
        if self.training:
            cov = torch.stack((r.square().mean((0, 2, 3)), i.square().mean((0, 2, 3)),
                               (r*i).mean((0, 2, 3))))
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_covar.lerp_(cov.detach(), self.momentum)
                self.num_batches_tracked.add_(1)
        else:
            cov = self.running_covar
        a, b, c = cov[0] + self.eps, cov[1] + self.eps, cov[2]
        # Analytic SPD inverse square root. Clamp roundoff in the determinant.
        s = (a*b-c.square()).clamp_min(self.eps**2).sqrt()
        denom = s * (a+b+2*s).sqrt()
        rr, ii, ri = (b+s)/denom, (a+s)/denom, -c/denom
        view = lambda v: v[None, :, None, None]
        yr, yi = view(rr)*r + view(ri)*i, view(ri)*r + view(ii)*i
        if self.affine:
            wr, wi, wc = self.weight
            yr, yi = (view(wr)*yr + view(wc)*yi + view(self.bias[0]),
                      view(wc)*yr + view(wi)*yi + view(self.bias[1]))
        return yr, yi


class ComplexMaxPool2d(nn.Module):
    """Select by modulus, gather real and imaginary values at the SAME index."""
    def forward(self, z):
        r, i = z
        _, indices = F.max_pool2d(r.square()+i.square(), 2, 2, return_indices=True)
        gather = lambda x: x.flatten(2).gather(2, indices.flatten(2)).reshape_as(indices)
        return gather(r), gather(i)


def complex_cat(a, b):
    return torch.cat((a[0], b[0]), 1), torch.cat((a[1], b[1]), 1)


class ComplexDoubleConv(nn.Module):
    def __init__(self, cin, cout, activation='modrelu'):
        super().__init__()
        if activation not in ('modrelu', 'crelu'):
            raise ValueError('complex activation must be modrelu or crelu')
        act = lambda: ComplexModReLU(cout) if activation == 'modrelu' else ComplexReLU()
        self.layers = nn.Sequential(ComplexConv2d(cin, cout, 3, padding=1),
                                    ComplexBatchNorm2d(cout), act(),
                                    ComplexConv2d(cout, cout, 3, padding=1),
                                    ComplexBatchNorm2d(cout), act())

    def forward(self, z):
        return self.layers(z)
