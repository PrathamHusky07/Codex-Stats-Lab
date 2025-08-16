
"""Utility functions for simulating common probability distributions.

Each function returns a tuple consisting of the generated samples, the
sample mean and the sample variance.  Numpy is used as the underlying
random number generator to ensure vectorised and reproducible results.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def simulate_poisson(lam: float, size: int) -> Tuple[np.ndarray, float, float]:
    """Generate Poisson distributed samples.

    Parameters
    ----------
    lam:
        Rate :math:`\lambda` of the Poisson distribution. ``lam`` must be a
        positive real number representing the expected number of events in a
        fixed interval.
    size:
        Number of random samples to draw.

    Returns
    -------
    tuple
        ``(samples, mean, variance)`` where ``samples`` is a ``numpy`` array of
        the generated values, ``mean`` is the sample mean and ``variance`` is the
        sample variance.

    Examples
    --------
    >>> samples, mean, var = simulate_poisson(3.5, 1_000)
    >>> round(mean, 2)
    3.5
    """

    samples = np.random.poisson(lam=lam, size=size)
    return samples, float(samples.mean()), float(samples.var())


def simulate_beta(
    alpha: float, beta: float, size: int
) -> Tuple[np.ndarray, float, float]:
    """Generate Beta distributed samples.

    The Beta distribution is handy for modelling probabilities or rates that
    are bounded between ``0`` and ``1``, such as click-through rates or
    conversion probabilities in online experiments.

    Parameters
    ----------
    alpha:
        First shape parameter of the distribution. Must be positive.
    beta:
        Second shape parameter of the distribution. Must be positive.
    size:
        Number of random samples to generate.

    Returns
    -------
    tuple
        ``(samples, mean, variance)`` where ``samples`` is a ``numpy`` array of
        the generated values, ``mean`` is the sample mean and ``variance`` is the
        sample variance.

    Examples
    --------
    >>> samples, mean, var = simulate_beta(2.0, 5.0, 1_000)
    >>> 0 <= mean <= 1
    True
    """

    samples = np.random.beta(a=alpha, b=beta, size=size)
    return samples, float(samples.mean()), float(samples.var())


def simulate_lognormal(
    mu: float, sigma: float, size: int
) -> Tuple[np.ndarray, float, float]:
    """Generate Log-Normal distributed samples.

    The Log-Normal distribution is frequently used for modelling
    non-negative, right-skewed quantities such as revenue per user or
    time-on-page.

    Parameters
    ----------
    mu:
        Mean of the underlying normal distribution.
    sigma:
        Standard deviation of the underlying normal distribution. Must be
        non-negative.
    size:
        Number of random samples to generate.

    Returns
    -------
    tuple
        ``(samples, mean, variance)`` where ``samples`` is a ``numpy`` array of
        the generated values, ``mean`` is the sample mean and ``variance`` is the
        sample variance.

    Examples
    --------
    >>> samples, mean, var = simulate_lognormal(0.0, 1.0, 1_000)
    >>> mean > 0
    True
    """

    samples = np.random.lognormal(mean=mu, sigma=sigma, size=size)
    return samples, float(samples.mean()), float(samples.var())
