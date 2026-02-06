"""
Probabilistic scoring functions for quantile forecasts.

Metrics
-------
pinball_loss          — per-quantile asymmetric loss (a.k.a. quantile loss)
avg_pinball_loss      — mean pinball across all quantiles
crps_from_quantiles   — CRPS approximated from a discrete quantile grid
empirical_coverage    — fraction of actuals inside a prediction interval
winkler_score         — interval sharpness + calibration penalty
median_mape           — MAPE using the median (q=0.50) as the point forecast
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Pinball / quantile loss
# ------------------------------------------------------------------

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> np.ndarray:
    """
    Element-wise pinball loss for a single quantile level *tau*.

    L_tau(y, q) = tau * max(y - q, 0) + (1 - tau) * max(q - y, 0)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return np.where(diff >= 0, tau * diff, (tau - 1.0) * diff)


def avg_pinball_loss(
    y_true: np.ndarray,
    q_preds: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    """Mean pinball loss averaged over all quantiles and all samples."""
    y_true = np.asarray(y_true)
    q_preds = np.asarray(q_preds)
    quantiles = np.asarray(quantiles)
    total = 0.0
    for j, tau in enumerate(quantiles):
        total += np.mean(pinball_loss(y_true, q_preds[:, j], tau))
    return total / len(quantiles)


# ------------------------------------------------------------------
# CRPS (discrete approximation via quantile grid)
# ------------------------------------------------------------------

def crps_from_quantiles(
    y_true: np.ndarray,
    q_preds: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    """
    Approximate CRPS from a discrete set of quantiles using the
    trapezoidal rule on the pinball loss integral:

        CRPS ≈ 2 * ∫₀¹ pinball_tau(y, q(tau)) d(tau)

    For a dense quantile grid this converges to the exact CRPS.
    """
    y_true = np.asarray(y_true)
    q_preds = np.asarray(q_preds)
    quantiles = np.asarray(quantiles, dtype=float)
    n_q = len(quantiles)
    # Extend to boundaries 0 and 1 if not present
    taus = quantiles
    losses = np.array([
        np.mean(pinball_loss(y_true, q_preds[:, j], taus[j]))
        for j in range(n_q)
    ])
    # Trapezoidal integration of 2 * pinball(tau) over [tau_min, tau_max]
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2.0 compat
    crps_val = 2.0 * _trapz(losses, taus)
    return float(crps_val)


# ------------------------------------------------------------------
# Empirical coverage
# ------------------------------------------------------------------

def empirical_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of actuals falling within [lower, upper]."""
    y_true = np.asarray(y_true)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


# ------------------------------------------------------------------
# Winkler score
# ------------------------------------------------------------------

def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """
    Average Winkler (interval) score.

    For a (1 - alpha) prediction interval [L, U]:
        W = (U - L) + (2/alpha) * (L - y)  if y < L
        W = (U - L)                         if L <= y <= U
        W = (U - L) + (2/alpha) * (y - U)  if y > U

    Lower is better.  Rewards narrow intervals, penalises misses.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    penalty = np.where(
        y_true < lower,
        (2.0 / alpha) * (lower - y_true),
        np.where(y_true > upper, (2.0 / alpha) * (y_true - upper), 0.0),
    )
    return float(np.mean(width + penalty))


# ------------------------------------------------------------------
# Median MAPE
# ------------------------------------------------------------------

def median_mape(
    y_true: np.ndarray,
    q_preds: np.ndarray,
    quantiles: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """MAPE using the median quantile (q=0.50) as the point forecast."""
    quantiles = np.asarray(quantiles)
    median_idx = int(np.argmin(np.abs(quantiles - 0.50)))
    y_med = q_preds[:, median_idx]
    y_true = np.asarray(y_true)
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_med[mask]) / y_true[mask])) * 100)
