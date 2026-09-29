"""Poisson count likelihood with a fixed, training-only normalization."""
import torch


def observed_counts(target, photons):
    # Measurements are saved as counts / photons; undo float32 roundoff.
    return (target * photons).round()


def poisson_normalizer(target, photons):
    return observed_counts(target, photons).sum().clamp_min(1)


def poisson_nll_sum(prediction, target, photons):
    """NLL minus saturated-model NLL (half deviance), summed over pixels.

    mu = photons * prediction + 1e-8 counts. Additive epsilon avoids log(0)
    without a clamp's zero-gradient region. xlogy handles zero observations.
    The subtracted term depends only on observations, so gradients equal NLL.
    """
    mu = photons * prediction + 1e-8
    y = observed_counts(target, photons)
    return (mu-y + torch.xlogy(y, y.clamp_min(1e-8))-torch.xlogy(y, mu)).sum()


def poisson_loss(prediction, target, photons):
    return poisson_nll_sum(prediction, target, photons) / poisson_normalizer(target, photons)
