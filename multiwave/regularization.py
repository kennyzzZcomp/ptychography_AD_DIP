"""Shared-amplitude smoothed TGV2, using training geometry only."""
import torch
from functions.paperrepro.tgv import tgv2_terms


class AmplitudeTGV:
    """Warm-started auxiliary minimization, not an exact TGV proximal solve.

    Differences are in pixel units. No amplitude mean normalization: probe
    powers already fix the scale convention in this experiment.
    """
    def __init__(self, cfg, scene):
        self.cfg = cfg
        device = scene.objects.device
        self.mask = torch.zeros((cfg.object_size, cfg.object_size), dtype=torch.bool, device=device)
        for y, x in scene.operator.positions[scene.train].detach().cpu().tolist():
            self.mask[y:y+cfg.patch_size, x:x+cfg.patch_size] = True
        valid = self.mask[:-1, :-1] & self.mask[1:, :-1] & self.mask[:-1, 1:]
        if not bool(valid.any()):
            raise ValueError("TGV training domain has no valid stencil")
        self.v = torch.nn.Parameter(torch.zeros(2, cfg.object_size, cfg.object_size, device=device))
        self.optimizer = torch.optim.Adam([self.v], lr=cfg.tgv_lr)

    def terms(self, amplitude, vector=None):
        return tgv2_terms(amplitude, self.v.detach() if vector is None else vector,
                          self.mask, self.cfg.tgv_alpha0, self.cfg.tgv_alpha1, self.cfg.tgv_eps)

    def penalty(self, amplitude):
        for _ in range(self.cfg.tgv_inner_steps):
            self.optimizer.zero_grad(set_to_none=True)
            self.terms(amplitude.detach(), self.v)[0].backward()
            self.optimizer.step()
        return self.terms(amplitude)[0]

    def state_dict(self):
        return {"vector": self.v.detach().cpu(), "mask": self.mask.cpu(),
                "optimizer": self.optimizer.state_dict()}


class AmplitudeATV:
    """Exact |Dx A| + |Dy A|, each direction averaged over valid training edges.

    Pixel-unit forward differences; no boundary wrapping or external zero pad.
    This is anisotropic (axis-aligned), not adaptive or direction-learned TV.
    """
    def __init__(self, cfg, scene):
        self.mask = torch.zeros((cfg.object_size, cfg.object_size), dtype=torch.bool,
                                device=scene.objects.device)
        for y, x in scene.operator.positions[scene.train].detach().cpu().tolist():
            self.mask[y:y+cfg.patch_size, x:x+cfg.patch_size] = True
        self.valid_x = self.mask[:, 1:] & self.mask[:, :-1]
        self.valid_y = self.mask[1:, :] & self.mask[:-1, :]
        if not bool(self.valid_x.any()) or not bool(self.valid_y.any()):
            raise ValueError("ATV training domain has no valid edges")

    def terms(self, amplitude):
        horizontal = (amplitude[:, 1:]-amplitude[:, :-1])[self.valid_x].abs().mean()
        vertical = (amplitude[1:, :]-amplitude[:-1, :])[self.valid_y].abs().mean()
        return horizontal + vertical, horizontal, vertical

    def penalty(self, amplitude):
        return self.terms(amplitude)[0]
