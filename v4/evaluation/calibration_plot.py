"""
Calibration diagnostics for probabilistic quantile forecasts.

- Reliability diagram (nominal vs empirical coverage)
- PIT histogram (probability integral transform)
- Calibration summary table
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------
# Reliability diagram
# ------------------------------------------------------------------

def reliability_diagram(
    y_true: np.ndarray,
    q_preds: np.ndarray,
    quantiles: np.ndarray,
    *,
    title: str = "Reliability Diagram",
    save_path: Optional[str | Path] = None,
) -> Dict[float, float]:
    """
    Plot nominal coverage vs empirical coverage for each symmetric interval.

    A well-calibrated model lies on the diagonal.

    Returns dict mapping nominal_coverage -> empirical_coverage.
    """
    y_true = np.asarray(y_true)
    quantiles = np.asarray(quantiles)
    n_q = len(quantiles)
    coverages: Dict[float, float] = {}

    # Build symmetric pairs
    pairs = []
    for i in range(n_q // 2):
        lo, hi = quantiles[i], quantiles[-(i + 1)]
        if lo < 0.5 < hi:
            pairs.append((i, -(i + 1), hi - lo))

    for lo_idx, hi_idx, nominal in pairs:
        lower = q_preds[:, lo_idx]
        upper = q_preds[:, hi_idx if hi_idx >= 0 else n_q + hi_idx]
        inside = (y_true >= lower) & (y_true <= upper)
        coverages[float(nominal)] = float(np.mean(inside))

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    nominals = sorted(coverages.keys())
    empiricals = [coverages[n] for n in nominals]

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.scatter(nominals, empiricals, s=80, zorder=5)
    ax.plot(nominals, empiricals, "o-", markersize=8)

    # Tolerance band (±3%)
    ax.fill_between([0, 1], [0 - 0.03, 1 - 0.03], [0 + 0.03, 1 + 0.03],
                    alpha=0.15, color="green", label="+-3% tolerance")

    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return coverages


# ------------------------------------------------------------------
# PIT histogram
# ------------------------------------------------------------------

def pit_histogram(
    y_true: np.ndarray,
    q_preds: np.ndarray,
    quantiles: np.ndarray,
    *,
    n_bins: int = 10,
    title: str = "PIT Histogram",
    save_path: Optional[str | Path] = None,
) -> np.ndarray:
    """
    Probability Integral Transform histogram.

    For a well-calibrated model the PIT values should be ~ Uniform(0, 1),
    so the histogram should be approximately flat.

    Returns the PIT values array.
    """
    y_true = np.asarray(y_true)
    q_preds = np.asarray(q_preds)
    quantiles = np.asarray(quantiles)
    n = len(y_true)

    # For each observation, find the approximate CDF value F(y)
    # by interpolating through the quantile grid
    pit = np.zeros(n)
    for i in range(n):
        # Linearly interpolate: where does y_true[i] fall in q_preds[i, :]?
        qvals = q_preds[i, :]
        if y_true[i] <= qvals[0]:
            pit[i] = quantiles[0] * (y_true[i] / max(qvals[0], 1e-8))
            pit[i] = max(0.0, min(pit[i], quantiles[0]))
        elif y_true[i] >= qvals[-1]:
            pit[i] = quantiles[-1] + (1 - quantiles[-1]) * 0.5  # clamp in upper tail
            pit[i] = min(1.0, pit[i])
        else:
            pit[i] = float(np.interp(y_true[i], qvals, quantiles))

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pit, bins=n_bins, density=True, edgecolor="black", alpha=0.7)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Uniform (ideal)")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return pit


# ------------------------------------------------------------------
# Calibration table (print)
# ------------------------------------------------------------------

def print_calibration_table(
    y_true: np.ndarray,
    q_preds: np.ndarray,
    quantiles: np.ndarray,
    tolerance: float = 0.03,
) -> None:
    """Print a table of nominal vs empirical coverage + pass/fail."""
    coverages = reliability_diagram.__wrapped__(y_true, q_preds, quantiles) \
        if hasattr(reliability_diagram, "__wrapped__") \
        else _compute_coverages(y_true, q_preds, quantiles)

    print(f"\n{'Nominal':>10} {'Empirical':>10} {'Gap':>8} {'Status':>8}")
    print("-" * 40)
    for nom in sorted(coverages.keys()):
        emp = coverages[nom]
        gap = abs(emp - nom)
        status = "OK" if gap <= tolerance else "MISS"
        print(f"  {nom:>7.0%}   {emp:>8.1%}   {gap:>6.1%}   {status:>6}")


def _compute_coverages(y_true, q_preds, quantiles):
    y_true = np.asarray(y_true)
    quantiles = np.asarray(quantiles)
    n_q = len(quantiles)
    coverages = {}
    for i in range(n_q // 2):
        lo_idx, hi_idx = i, n_q - 1 - i
        nominal = quantiles[hi_idx] - quantiles[lo_idx]
        inside = (y_true >= q_preds[:, lo_idx]) & (y_true <= q_preds[:, hi_idx])
        coverages[float(nominal)] = float(np.mean(inside))
    return coverages
