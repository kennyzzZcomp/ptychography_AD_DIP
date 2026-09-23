"""Piecewise-constant amplitude TGV weights; boundaries count completed updates."""
import math


def parse_tgv_schedule(spec, initial, iters):
    if not spec:
        return []
    if not math.isfinite(initial) or initial <= 0:
        raise ValueError('TGV schedule requires --tgv-amp > 0')
    stages = []
    for item in spec.split(','):
        try:
            step_text, weight_text = item.strip().split(':')
            step, weight = int(step_text), float(weight_text)
        except ValueError as exc:
            raise ValueError('Use --tgv-amp-schedule "1000:0.01" or "1000:0.01,1500:0.001"') from exc
        if step < 1 or step >= iters or (stages and step <= stages[-1][0]):
            raise ValueError('TGV boundaries must increase strictly and satisfy 1 <= boundary < iters')
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('TGV schedule weights must be finite and nonnegative')
        stages.append((step, weight))
    return stages


def tgv_weight(initial, stages, completed_updates):
    weight = initial
    for boundary, value in stages:
        if completed_updates < boundary:
            break
        weight = value
    return weight
