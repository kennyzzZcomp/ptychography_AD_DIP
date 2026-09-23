"""Net/probe learning-rate stages; boundaries count completed updates."""
import math


def parse_lr_schedule(spec, lr_net, lr_probe, iters, cosine=False):
    if any(not math.isfinite(v) or v < 0 for v in (lr_net, lr_probe)):
        raise ValueError("net/probe learning rates must be finite and nonnegative")
    if not spec:
        return []
    if cosine:
        raise ValueError("--lr-schedule cannot be combined with --lr-cosine")
    stages = []
    for item in spec.split(","):
        try:
            step_text, net_text, probe_text = item.strip().split(":")
            step, net, probe = int(step_text), float(net_text), float(probe_text)
        except ValueError as exc:
            raise ValueError('Use --lr-schedule "1000:8e-4:2e-2" (completed updates:net:probe)') from exc
        if step < 1 or step >= iters or (stages and step <= stages[-1][0]):
            raise ValueError("LR boundaries must increase strictly and satisfy 1 <= boundary < iters")
        if any(not math.isfinite(v) or v < 0 for v in (net, probe)):
            raise ValueError("scheduled learning rates must be finite and nonnegative")
        stages.append((step, net, probe))
    return stages


def learning_rates(lr_net, lr_probe, stages, completed_updates):
    for boundary, net, probe in stages:
        if completed_updates < boundary:
            break
        lr_net, lr_probe = net, probe
    return lr_net, lr_probe
