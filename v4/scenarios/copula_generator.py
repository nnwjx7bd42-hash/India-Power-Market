"""
Gaussian copula scenario generator.

Given marginal quantile forecasts (168 × n_quantiles) and a historical
rank-correlation matrix (168 × 168), generate N correlated 168-hour
price paths that respect both marginals and temporal dependence.

Pipeline:
    1. Estimate Spearman rank correlation from historical residuals
    2. Cholesky-decompose the correlation matrix
    3. Sample correlated standard normals → CDF → uniform [0,1]
    4. Map each uniform through the per-hour quantile forecast (interp1d)
    5. Clip to non-negative prices (IEX floor)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import cholesky
from scipy.stats import norm, spearmanr


# ------------------------------------------------------------------
# Correlation estimation
# ------------------------------------------------------------------

def estimate_rank_correlation(
    residuals: np.ndarray,
    regularise_eps: float = 1e-4,
) -> np.ndarray:
    """
    Compute Spearman rank correlation matrix from a (n_weeks, 168) residual matrix.

    Parameters
    ----------
    residuals : (n_weeks, 168)  — one row per historical week
    regularise_eps : float      — small diagonal nudge to ensure PSD

    Returns
    -------
    (168, 168) correlation matrix, guaranteed PSD.
    """
    corr, _ = spearmanr(residuals, axis=0)
    if corr.ndim == 0:
        # Edge case: single-column
        return np.array([[1.0]])
    # Ensure symmetric
    corr = (corr + corr.T) / 2.0
    # Ensure PSD via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, regularise_eps)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Renormalise to correlation (unit diagonal)
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def build_weekly_residuals(
    y_actual: np.ndarray,
    y_median: np.ndarray,
    hours_per_week: int = 168,
) -> np.ndarray:
    """
    Reshape actual - median_forecast into (n_weeks, 168).

    Drops incomplete trailing weeks.
    """
    residuals = np.asarray(y_actual) - np.asarray(y_median)
    n_full = (len(residuals) // hours_per_week) * hours_per_week
    return residuals[:n_full].reshape(-1, hours_per_week)


# ------------------------------------------------------------------
# Copula sampling
# ------------------------------------------------------------------

def generate_correlated_uniforms(
    n_scenarios: int,
    corr_matrix: np.ndarray,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate (n_scenarios, n_hours) of correlated Uniform(0,1) samples
    via a Gaussian copula.

    Steps:
        Z ~ N(0, I)  →  Z_corr = Z @ L^T  →  U = Phi(Z_corr)
    """
    rng = np.random.default_rng(seed)
    n_hours = corr_matrix.shape[0]
    L = cholesky(corr_matrix, lower=True)
    Z = rng.standard_normal((n_scenarios, n_hours))
    correlated = Z @ L.T
    U = norm.cdf(correlated)
    return U  # (n_scenarios, n_hours)


def map_uniforms_to_prices(
    U: np.ndarray,
    q_forecasts: np.ndarray,
    quantiles: np.ndarray,
    price_floor: float = 0.0,
) -> np.ndarray:
    """
    Map correlated uniform samples to price values by interpolating
    through the quantile forecast at each hour.

    Parameters
    ----------
    U            : (n_scenarios, n_hours) — from ``generate_correlated_uniforms``
    q_forecasts  : (n_hours, n_quantiles) — quantile predictions per hour
    quantiles    : (n_quantiles,) — e.g. [0.05, 0.10, ..., 0.95]
    price_floor  : float — minimum price (IEX floor)

    Returns
    -------
    (n_scenarios, n_hours) price paths.
    """
    n_scenarios, n_hours = U.shape
    quantiles = np.asarray(quantiles)
    paths = np.zeros((n_scenarios, n_hours))

    for h in range(n_hours):
        # Monotone interpolator for this hour's quantile function
        qf = interp1d(
            quantiles,
            q_forecasts[h, :],
            kind="linear",
            bounds_error=False,
            fill_value=(q_forecasts[h, 0], q_forecasts[h, -1]),
        )
        paths[:, h] = qf(U[:, h])

    paths = np.maximum(paths, price_floor)
    return paths


# ------------------------------------------------------------------
# High-level convenience
# ------------------------------------------------------------------

def generate_scenarios(
    q_forecasts: np.ndarray,
    quantiles: np.ndarray,
    corr_matrix: np.ndarray,
    n_scenarios: int = 200,
    seed: Optional[int] = None,
    price_floor: float = 0.0,
) -> np.ndarray:
    """
    End-to-end: generate *n_scenarios* correlated price paths.

    Parameters
    ----------
    q_forecasts  : (n_hours, n_quantiles)
    quantiles    : (n_quantiles,)
    corr_matrix  : (n_hours, n_hours)
    n_scenarios  : int
    seed         : int or None
    price_floor  : float

    Returns
    -------
    (n_scenarios, n_hours) array of price paths.
    """
    U = generate_correlated_uniforms(n_scenarios, corr_matrix, seed=seed)
    return map_uniforms_to_prices(U, q_forecasts, quantiles, price_floor=price_floor)
