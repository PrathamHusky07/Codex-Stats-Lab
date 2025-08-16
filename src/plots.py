"""Visualisation utilities for probability distributions.

The functions in this module generate figures illustrating sampled
probability distributions alongside their theoretical counterparts.

All functions return the created :class:`matplotlib.figure.Figure` instance
without displaying or saving it, enabling further customisation by the
caller.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import beta as beta_dist
from scipy.stats import lognorm, poisson


def plot_poisson_distribution(samples: np.ndarray, lam: float) -> Figure:
    """Visualise a Poisson distribution using empirical samples.

    A bar plot of the observed counts is produced and the theoretical
    probability mass function (PMF) for the supplied ``lam`` parameter is
    overlaid for comparison.

    Parameters
    ----------
    samples:
        Sampled data drawn from a Poisson distribution.
    lam:
        Rate :math:`\lambda` of the Poisson distribution.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib figure.
    """

    fig, ax = plt.subplots()

    values, counts = np.unique(samples, return_counts=True)
    probabilities = counts / samples.size
    ax.bar(values, probabilities, width=0.8, alpha=0.6, label="Empirical")

    k = np.arange(0, values.max() + 1)
    ax.plot(k, poisson.pmf(k, lam), "o-", color="tab:red", label="Poisson PMF")

    ax.set_xlabel("k")
    ax.set_ylabel("Probability")
    ax.set_title(f"Poisson Distribution (\u03bb={lam})")
    ax.legend()

    return fig


def plot_beta_distribution(
    samples: np.ndarray, alpha: float, beta: float
) -> Figure:
    """Visualise a Beta distribution using empirical samples.

    Parameters
    ----------
    samples:
        Sampled data drawn from a Beta distribution.
    alpha:
        First shape parameter :math:`\alpha` of the Beta distribution.
    beta:
        Second shape parameter :math:`\beta` of the Beta distribution.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib figure.
    """

    fig, ax = plt.subplots()

    ax.hist(samples, bins=30, density=True, alpha=0.6, label="Empirical")
    x = np.linspace(0, 1, 200)
    ax.plot(x, beta_dist.pdf(x, alpha, beta), label="Beta PDF", color="tab:red")

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title(f"Beta Distribution (\u03b1={alpha}, \u03b2={beta})")
    ax.legend()

    return fig


def plot_lognormal_distribution(samples: np.ndarray, mu: float, sigma: float) -> Figure:
    """Visualise a Log-Normal distribution using empirical samples.

    Parameters
    ----------
    samples:
        Sampled data drawn from a Log-Normal distribution.
    mu:
        Mean ``\mu`` of the underlying normal distribution.
    sigma:
        Standard deviation ``\sigma`` of the underlying normal distribution.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib figure.
    """

    fig, ax = plt.subplots()

    ax.hist(samples, bins=30, density=True, alpha=0.6, label="Empirical")
    x = np.linspace(0, max(samples), 200)
    ax.plot(
        x,
        lognorm.pdf(x, s=sigma, scale=np.exp(mu)),
        label="Log-Normal PDF",
        color="tab:red",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title(f"Log-Normal Distribution (\u03bc={mu}, \u03c3={sigma})")
    ax.legend()

    return fig

