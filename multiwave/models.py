"""Identical field initialization for pixel and existing U-Net parameterizations."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from functions.addip.model import ProPtyUNet


class ObjectModel(nn.Module):
    def __init__(self, cfg, method, input_stack):
        super().__init__()
        representation, self.coupling = method.split("_", 1)
        if representation not in ("pixel", "unet") or self.coupling not in ("coupled", "independent", "common"):
            raise ValueError(f"unsupported method: {method}")
        self.scale = cfg.opd_scale_um
        self.register_buffer("wavelength_um", torch.tensor(cfg.wavelengths_nm,
                             dtype=torch.float32, device=input_stack.device)*1e-3)
        self.register_buffer("input_stack", input_stack)
        count = len(cfg.weights)
        amp_count = 1 if self.coupling == "common" else count
        phase_count = count if self.coupling == "independent" else 1
        bias = math.log(math.expm1(.1))  # all methods start at A=exp(-0.1), phase=0
        self.net = None
        if representation == "unet":
            # Reuse the actual INNM U-Net backbone. Only field heads are changed.
            self.net = ProPtyUNet(input_stack.shape[1], cfg.base_channels,
                                 n_fields=amp_count, ph_ch=1)
            self.net.head_phs = nn.Conv2d(cfg.base_channels, phase_count, 3, padding=1)
            with torch.no_grad():
                self.net.head_amp.weight.zero_()
                self.net.head_amp.bias.fill_(bias)
                self.net.head_phs.weight.zero_()
                self.net.head_phs.bias.zero_()
        else:
            shape = (cfg.object_size, cfg.object_size)
            self.raw_tau = nn.Parameter(torch.full((amp_count, *shape), bias))
            self.raw_opd = nn.Parameter(torch.zeros(phase_count, *shape))

    def forward(self):
        tau_raw, opd_raw = self.net(self.input_stack) if self.net is not None else (self.raw_tau, self.raw_opd)
        tau = F.softplus(tau_raw)
        opd = self.scale*opd_raw  # unwrapped length, not a tanh-limited phase
        count = len(self.wavelength_um)
        if self.coupling == "common":
            tau = tau.expand(count, -1, -1)
            # Deliberately wrong-model control: the same complex object at every lambda.
            opd = opd.expand(count, -1, -1)*(self.wavelength_um/self.wavelength_um[0])[:, None, None]
        elif self.coupling == "coupled":
            opd = opd.expand(count, -1, -1)
        phase = 2*math.pi*opd/self.wavelength_um[:, None, None]
        return torch.polar(torch.exp(-tau), phase), tau, opd


def masked_tv(tau, opd, mask, opd_scale_um):
    """Same optional prior/domain for all methods, normalized per physical channel."""
    field = torch.cat((tau, opd/opd_scale_um), 0)
    vertical = mask[1:] & mask[:-1]
    horizontal = mask[:, 1:] & mask[:, :-1]
    terms = []
    for diff, selected in ((field[:, 1:]-field[:, :-1], vertical),
                           (field[:, :, 1:]-field[:, :, :-1], horizontal)):
        # Zero at a constant image, finite derivative at zero.
        values = (diff.square()+1e-8).sqrt()-1e-4
        terms.append(values[:, selected].mean())
    return sum(terms)
