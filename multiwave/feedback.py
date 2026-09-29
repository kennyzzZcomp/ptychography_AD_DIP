"""Online one-step, stop-gradient physical feedback (not pretrained unrolling)."""
import math
import torch
from torch import nn
from functions.addip.model import ProPtyUNet


class FeedbackAmplitudeModel(nn.Module):
    def __init__(self, cfg, method, input_stack):
        super().__init__()
        self.cfg = cfg
        self.register_buffer('amplitude', torch.full(
            (cfg.object_size, cfg.object_size), math.exp(-.1), device=input_stack.device))
        self.net = ProPtyUNet(3, cfg.base_channels, n_fields=1, ph_ch=1)
        del self.net.head_phs
        nn.init.zeros_(self.net.head_amp.weight)
        nn.init.zeros_(self.net.head_amp.bias)
        self.context = None

    def fields(self, amplitude):
        a = amplitude.expand(len(self.cfg.wavelengths_nm), -1, -1)
        zero = torch.zeros_like(a)
        return torch.complex(a, zero), -a.clamp_min(1e-8).log(), zero

    def prepare(self, scene, probes):
        from .reconstruct import backward_shared_data
        a = self.amplitude.detach().clone().requires_grad_(True)
        # Exact summed shared-object gradient, all TRAIN scans, fixed current probes.
        backward_shared_data(self.fields(a)[0], probes.detach(), scene, self.cfg)
        g = a.grad.detach()
        coverage = scene.operator.illumination(scene.train, probes.detach())
        support = coverage > 0
        scale = g[support].square().mean().sqrt().clamp_min(1e-12)
        direction = g / scale  # global positive rescaling, same for all pixels
        cov = coverage / coverage.max().clamp_min(1e-12)
        net_gradient = direction if self.cfg.feedback_mode != 'no_gradient' else torch.zeros_like(direction)
        inputs = torch.stack((self.amplitude.detach(), net_gradient, cov))[None]
        if not torch.isfinite(inputs).all():
            raise FloatingPointError('non-finite feedback input')
        self.context = (inputs.detach(), direction, support)
        self.gradient_rms = float(scale)

    def candidate(self):
        inputs, direction, support = self.context
        # BN uses batch statistics for both candidate forwards, without updating
        # running buffers twice. Original baseline BN behavior is untouched.
        for layer in self.net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.track_running_stats = False
        raw = self.net.head_amp(self.net.forward_features(inputs))[0, 0]
        weights = 1 + .5 * torch.tanh(raw)  # [0.5, 1.5], initial value exactly one
        if self.cfg.feedback_mode == 'identity':
            weights = 1 + raw * 0  # preserve a zero-gradient graph for shared loop
        self.weight_mean = float(weights.detach()[support].mean())
        return (self.amplitude - self.cfg.feedback_step * weights * direction).clamp_min(0)

    @torch.no_grad()
    def commit(self):
        # Recompute with updated network on the SAME detached state/gradient.
        a = self.candidate()
        if not torch.isfinite(a).all():
            raise FloatingPointError('non-finite feedback amplitude')
        self.amplitude.copy_(a)
        self.context = None

    def forward(self):
        return self.fields(self.amplitude if self.context is None else self.candidate())
