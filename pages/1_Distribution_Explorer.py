"""Streamlit application for exploring statistical distributions.

The dashboard allows users to simulate samples from several probability
models, inspect summary statistics and visualise the resulting
histograms alongside their theoretical counterparts.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import kurtosis, skew

from src.plots import (
    plot_beta_distribution,
    plot_lognormal_distribution,
    plot_poisson_distribution,
)
from src.simulations import (
    simulate_beta,
    simulate_lognormal,
    simulate_poisson,
)

# ---------------------------------------------------------------------------
# Configuration and static content
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Distribution Explorer", layout="wide")
sns.set_theme(style="whitegrid")

OVERVIEW_TEXT: Dict[str, str] = {
    "Poisson": (
        "The **Poisson distribution** models the number of events occurring in "
        "a fixed interval of time or space when these events happen with a "
        "known constant rate and independently of the time since the last "
        "event."
    ),
    "Beta": (
        "The **Beta distribution** lives on the interval [0, 1] and is "
        "parameterised by two positive shape parameters α and β. It is "
        "commonly used to describe random variables that represent "
        "probabilities or proportions."
    ),
    "Log-Normal": (
        "A **Log-Normal distribution** arises when the logarithm of a random "
        "variable follows a normal distribution. It is useful for modelling "
        "non-negative, right-skewed data such as incomes or survival times."
    ),
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def run_simulation(dist: str, size: int, params: Dict[str, float]) -> Tuple[np.ndarray, float, float]:
    """Simulate samples for the selected distribution."""
    if dist == "Poisson":
        return simulate_poisson(params["lam"], size)
    if dist == "Beta":
        return simulate_beta(params["alpha"], params["beta"], size)
    if dist == "Log-Normal":
        return simulate_lognormal(params["mu"], params["sigma"], size)
    raise ValueError("Unknown distribution")


def get_simulator(dist: str, params: Dict[str, float]) -> Callable[[int], np.ndarray]:
    """Return a function that generates samples of a given size."""
    if dist == "Poisson":
        return lambda s: simulate_poisson(params["lam"], s)[0]
    if dist == "Beta":
        return lambda s: simulate_beta(params["alpha"], params["beta"], s)[0]
    if dist == "Log-Normal":
        return lambda s: simulate_lognormal(params["mu"], params["sigma"], s)[0]
    raise ValueError("Unknown distribution")


def plot_distribution(dist: str, samples: np.ndarray, params: Dict[str, float]) -> Figure:
    """Dispatch to the appropriate plotting function."""
    if dist == "Poisson":
        return plot_poisson_distribution(samples, params["lam"])
    if dist == "Beta":
        return plot_beta_distribution(samples, params["alpha"], params["beta"])
    if dist == "Log-Normal":
        return plot_lognormal_distribution(samples, params["mu"], params["sigma"])
    raise ValueError("Unknown distribution")


def clt_sample_means(simulator: Callable[[int], np.ndarray], n: int, k: int) -> np.ndarray:
    """Generate ``k`` sample means of size ``n`` using ``simulator``."""
    return np.array([simulator(n).mean() for _ in range(k)])


# ---------------------------------------------------------------------------
# Sidebar - user inputs
# ---------------------------------------------------------------------------

st.sidebar.header("Simulation Settings")
dist = st.sidebar.selectbox("Distribution", ["Poisson", "Beta", "Log-Normal"])
size = int(
    st.sidebar.number_input(
        "Sample size",
        min_value=1,
        value=10_000,
        step=1,
        help="Number of random samples to generate",
    )
)

params: Dict[str, float] = {}
if dist == "Poisson":
    params["lam"] = st.sidebar.number_input(
        "λ (rate)",
        min_value=0.0,
        value=3.0,
        help="λ controls the rate in the Poisson distribution",
    )
elif dist == "Beta":
    params["alpha"] = st.sidebar.number_input(
        "α (shape)",
        min_value=0.0001,
        value=2.0,
        help="α controls the concentration of the distribution",
    )
    params["beta"] = st.sidebar.number_input(
        "β (shape)",
        min_value=0.0001,
        value=5.0,
        help="β controls the concentration of the distribution",
    )
elif dist == "Log-Normal":
    params["mu"] = st.sidebar.number_input(
        "μ (mean)", value=0.0, help="μ is the mean of the underlying normal distribution"
    )
    params["sigma"] = st.sidebar.number_input(
        "σ (std)",
        min_value=0.0001,
        value=1.0,
        help="σ (>0) controls the spread of the distribution",
    )
    if params["sigma"] <= 0:
        st.error("σ must be greater than 0")
        st.stop()

# ---------------------------------------------------------------------------
# Simulation and metrics
# ---------------------------------------------------------------------------

samples, mean, variance = run_simulation(dist, size, params)
skewness = float(skew(samples))
kurt = float(kurtosis(samples))

st.markdown("### 📈 Summary Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean", f"{mean:.3f}")
col2.metric("Variance", f"{variance:.3f}")
col3.metric("Skewness", f"{skewness:.3f}", help="Measure of asymmetry")
col4.metric("Kurtosis", f"{kurt:.3f}", help="Measure of tail heaviness")

STAT_INSIGHTS = {
    "Poisson": (
        "**Did you know?**\n\n"
        "The **Poisson distribution** is often used to model the number of times an event occurs in a fixed interval of time or space. "
        "A key property is that **its mean and variance are both equal to λ (lambda)**, which makes it unique among discrete distributions. \n\n"
        "It's widely used in real-world scenarios such as **modeling call arrivals at a call center**, **the number of emails received per hour**, or **accident counts at an intersection**. "
        "Another fascinating aspect is that for large λ, the Poisson distribution starts to resemble a Normal distribution — a phenomenon known as the **law of rare events**."
    ),
    "Beta": (
        "**Did you know?**\n\n"
        "The **Beta distribution** is a continuous probability distribution defined on the interval [0, 1]. It's incredibly versatile — "
        "**by tuning the shape parameters α and β**, it can resemble a **uniform, bell-shaped, J-shaped, or U-shaped** distribution. \n\n"
        "This makes it especially useful in **Bayesian statistics**, where it's commonly used to represent prior distributions of probabilities. "
        "For example, in A/B testing, the Beta distribution is often used to **model uncertainty in conversion rates**."
    ),
    "Log-Normal": (
        "**Did you know?**\n\n"
        "In a **Log-Normal distribution**, the logarithm of the variable is normally distributed. That means if X is Log-Normal, then log(X) follows a Normal distribution. \n\n"
        "This distribution is positively skewed, with a long right tail, and is often used to model **real-world quantities that can’t be negative** and tend to grow multiplicatively, such as **income**, **stock prices**, **time to failure**, or **biological measurements**. \n\n"
        "It’s particularly useful in **financial modeling** where returns often exhibit multiplicative effects rather than additive ones."
    ),
}

with st.expander("📘 Statistical Insight"):
    st.markdown(STAT_INSIGHTS[dist])

# ---------------------------------------------------------------------------
# Tabs for content
# ---------------------------------------------------------------------------

overview_tab, plot_tab, clt_tab, data_tab, download_tab = st.tabs(
    ["Overview", "Plot", "CLT Demo", "Raw Data", "Download"]
)

with overview_tab:
    st.markdown(OVERVIEW_TEXT[dist])

def get_plot_description(dist: str, log_scale: bool) -> str:
    if dist == "Poisson":
        scale_note = "logarithmic scale" if log_scale else "linear scale"
        return (
            f"This plot displays a histogram of Poisson-distributed event counts "
            f"overlaid with the theoretical Poisson PMF. The x-axis shows the number "
            f"of occurrences (k), while the y-axis shows the probability of each count. "
            f"The red curve is the expected PMF (λ = {params['lam']:.2f}) and the blue bars represent "
            f"empirical data. Using a {scale_note} helps interpret rare vs frequent events."
        )
    elif dist == "Beta":
        scale_note = "logarithmic scale" if log_scale else "linear scale"
        return (
            f"This plot shows the Beta distribution over [0,1] with parameters "
            f"α = {params['alpha']:.2f}, β = {params['beta']:.2f}. The red curve shows the theoretical "
            f"Beta PDF while the histogram shows the simulated data. The x-axis shows probability-like values "
            f"and the y-axis shows density. {scale_note.capitalize()} helps visualize distribution skew and mode."
        )
    elif dist == "Log-Normal":
        scale_note = "logarithmic scale" if log_scale else "linear scale"
        return (
            f"This histogram shows a Log-Normal distribution (μ = {params['mu']:.2f}, σ = {params['sigma']:.2f}) "
            f"with the theoretical PDF overlay. The x-axis represents non-negative values, and the y-axis shows "
            f"density. A log scale is especially useful to reveal long-tail behavior in skewed distributions."
        )
    return "No description available."

with plot_tab:
    log_scale = st.checkbox("Log scale", value=False)
    fig = plot_distribution(dist, samples, params)
    if log_scale:
        fig.axes[0].set_yscale("log")
    st.pyplot(fig)

    st.markdown(
        f"### You generated {size:,} samples with mean = {mean:.3f} and skewness = {skewness:.3f}"
    )
    st.markdown(get_plot_description(dist, log_scale))

with clt_tab:
    st.subheader("Central Limit Theorem Demo")
    with st.expander("What is CLT?"):
        st.markdown(
            "The Central Limit Theorem (CLT) states that the distribution of "
            "sample means approaches a normal distribution as the sample size "
            "n grows, regardless of the original distribution."
        )
    n = int(
        st.number_input(
            "Sample size per run (n)", min_value=1, value=30, step=1
        )
    )
    k = int(
        st.number_input(
            "Number of repetitions (k)", min_value=1, value=1_000, step=1
        )
    )
    simulator = get_simulator(dist, params)
    means = clt_sample_means(simulator, n, k)
    fig_clt, ax_clt = plt.subplots()
    ax_clt.hist(means, bins=30, density=True, alpha=0.7)
    ax_clt.set_xlabel("Sample mean")
    ax_clt.set_ylabel("Density")
    ax_clt.set_title("Distribution of Sample Means")
    st.pyplot(fig_clt)

with data_tab:
    st.dataframe(pd.DataFrame(samples, columns=["value"]).head(10))

with download_tab:
    csv = pd.DataFrame(samples, columns=["value"]).to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name="samples.csv", mime="text/csv")
