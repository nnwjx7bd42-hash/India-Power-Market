"""
Scenario diagnostics and visualisation.

- Fan chart: all scenarios + median + confidence bands + actual prices
- Spread distribution: daily max-min spread histogram vs historical
- Correlation heatmap: empirical scenario correlation vs target
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------
# Fan chart
# ------------------------------------------------------------------

def plot_fan_chart(
    scenarios: np.ndarray,
    quantile_forecasts: Optional[np.ndarray] = None,
    quantiles: Optional[np.ndarray] = None,
    actuals: Optional[np.ndarray] = None,
    timestamps: Optional[Sequence] = None,
    *,
    title: str = "Price Scenario Fan Chart",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plot all scenarios (faded lines), median + 80%/90% bands, and actuals.

    Parameters
    ----------
    scenarios          : (n_scenarios, n_hours)
    quantile_forecasts : (n_hours, n_quantiles) — optional, for band overlays
    quantiles          : (n_quantiles,) — needed if quantile_forecasts given
    actuals            : (n_hours,) — optional, overlaid as thick line
    timestamps         : sequence of length n_hours — x-axis labels
    """
    n_scenarios, n_hours = scenarios.shape
    x = np.arange(n_hours) if timestamps is None else np.arange(n_hours)

    fig, ax = plt.subplots(figsize=(14, 6))

    # All scenarios (faded)
    for s in range(n_scenarios):
        ax.plot(x, scenarios[s, :], color="steelblue", alpha=0.08, linewidth=0.5)

    # Scenario-based percentile bands
    p5 = np.percentile(scenarios, 5, axis=0)
    p10 = np.percentile(scenarios, 10, axis=0)
    p50 = np.percentile(scenarios, 50, axis=0)
    p90 = np.percentile(scenarios, 90, axis=0)
    p95 = np.percentile(scenarios, 95, axis=0)

    ax.fill_between(x, p5, p95, alpha=0.15, color="steelblue", label="90% band (scenarios)")
    ax.fill_between(x, p10, p90, alpha=0.25, color="steelblue", label="80% band (scenarios)")
    ax.plot(x, p50, color="navy", linewidth=1.5, label="Median (scenarios)")

    # Quantile forecast bands (if provided)
    if quantile_forecasts is not None and quantiles is not None:
        quantiles = np.asarray(quantiles)
        lo_90 = int(np.argmin(np.abs(quantiles - 0.05)))
        hi_90 = int(np.argmin(np.abs(quantiles - 0.95)))
        med_idx = int(np.argmin(np.abs(quantiles - 0.50)))
        ax.plot(x, quantile_forecasts[:, med_idx], "--", color="darkorange",
                linewidth=1.2, label="Median (quantile model)")

    # Actual prices
    if actuals is not None:
        ax.plot(x, actuals, color="black", linewidth=2, label="Actual", zorder=5)

    # X-axis labels
    if timestamps is not None:
        tick_step = max(1, n_hours // 14)
        tick_idx = list(range(0, n_hours, tick_step))
        ax.set_xticks(tick_idx)
        labels = [str(timestamps[i])[:13] for i in tick_idx]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    ax.set_xlabel("Hour")
    ax.set_ylabel("Price (INR/MWh)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ------------------------------------------------------------------
# Spread distribution
# ------------------------------------------------------------------

def plot_spread_distribution(
    scenarios: np.ndarray,
    historical_prices: Optional[np.ndarray] = None,
    hours_per_day: int = 24,
    *,
    title: str = "Daily Spread Distribution",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Compare the distribution of daily max-min spreads across scenarios
    vs historical.
    """
    n_scenarios, n_hours = scenarios.shape
    n_days = n_hours // hours_per_day

    # Scenario daily spreads
    sc_spreads = []
    for s in range(n_scenarios):
        for d in range(n_days):
            day_slice = scenarios[s, d * hours_per_day : (d + 1) * hours_per_day]
            sc_spreads.append(day_slice.max() - day_slice.min())
    sc_spreads = np.array(sc_spreads)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sc_spreads, bins=40, alpha=0.6, density=True, label="Scenarios", color="steelblue")

    if historical_prices is not None:
        n_hist_days = len(historical_prices) // hours_per_day
        hist_spreads = []
        for d in range(n_hist_days):
            day_slice = historical_prices[d * hours_per_day : (d + 1) * hours_per_day]
            hist_spreads.append(day_slice.max() - day_slice.min())
        hist_spreads = np.array(hist_spreads)
        ax.hist(hist_spreads, bins=40, alpha=0.5, density=True,
                label="Historical", color="coral", edgecolor="black")

    ax.set_xlabel("Daily Max-Min Spread (INR/MWh)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ------------------------------------------------------------------
# Correlation heatmap
# ------------------------------------------------------------------

def plot_correlation_check(
    scenarios: np.ndarray,
    target_corr: np.ndarray,
    *,
    title: str = "Scenario vs Target Correlation",
    save_path: Optional[str | Path] = None,
) -> float:
    """
    Compare empirical correlation of generated scenarios with the target
    rank correlation matrix.

    Returns mean absolute error between the two correlation matrices.
    """
    from scipy.stats import spearmanr

    emp_corr, _ = spearmanr(scenarios, axis=0)
    if emp_corr.ndim == 0:
        emp_corr = np.array([[emp_corr]])

    diff = np.abs(emp_corr - target_corr)
    mae = float(np.mean(diff))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(target_corr, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title("Target Rank Correlation")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(emp_corr, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("Empirical (Scenarios)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff, aspect="auto", cmap="Reds", vmin=0, vmax=0.3)
    axes[2].set_title(f"|Difference| (MAE={mae:.3f})")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("Hour")
        ax.set_ylabel("Hour")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return mae
