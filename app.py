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
seed_input = st.sidebar.text_input(
    "Seed (optional)", help="Set an integer seed for reproducibility"
)
if seed_input:
    try:
        np.random.seed(int(seed_input))
    except ValueError:
        st.sidebar.warning("Seed must be an integer")

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

st.markdown(
    f"### You generated {size:,} samples with mean = {mean:.3f} and skewness = {skewness:.3f}"
)

st.markdown("### 📈 Summary Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean", f"{mean:.3f}")
col2.metric("Variance", f"{variance:.3f}")
col3.metric("Skewness", f"{skewness:.3f}", help="Measure of asymmetry")
col4.metric("Kurtosis", f"{kurt:.3f}", help="Measure of tail heaviness")

with st.expander("What does Skewness mean?"):
    st.markdown(
        "Skewness quantifies the asymmetry of a distribution. A positive value "
        "indicates a longer right tail, while a negative value indicates a "
        "longer left tail."
    )

# ---------------------------------------------------------------------------
# Tabs for content
# ---------------------------------------------------------------------------

overview_tab, plot_tab, clt_tab, data_tab, download_tab = st.tabs(
    ["Overview", "Plot", "CLT Demo", "Raw Data", "Download"]
)

with overview_tab:
    st.markdown(OVERVIEW_TEXT[dist])

with plot_tab:
    log_scale = st.checkbox("Log scale", value=False)
    fig = plot_distribution(dist, samples, params)
    if log_scale:
        fig.axes[0].set_yscale("log")
    st.pyplot(fig)
    st.caption("Your histogram vs. ideal curve")

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
