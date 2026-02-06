"""
Backtest metrics for BESS arbitrage strategies.

Per-week metrics:
    - Realised revenue (INR)
    - Revenue capture ratio (vs perfect foresight)
    - Number of cycles consumed
    - Peak SoC, min SoC
    - Daily revenue breakdown

Aggregate metrics (across all backtest weeks):
    - Annual revenue (extrapolated): mean, median, std
    - CVaR realised (worst 5% of weeks)
    - Max drawdown of cumulative P&L
    - Negative profit weeks count
    - Cycles per year
    - Revenue per MW per year (INR Lakh/MW/year)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Per-week metrics
# ---------------------------------------------------------------------------

@dataclass
class WeeklyMetrics:
    """Metrics for a single week of BESS operation."""

    revenue: float
    capture_ratio: float          # revenue / perfect_foresight_revenue
    cycles: float                 # number of full cycles (total discharge / E_cap)
    peak_soc: float
    min_soc: float
    daily_revenues: np.ndarray    # (7,) or (n_days,) daily breakdown


def compute_weekly_metrics(
    p_ch: np.ndarray,
    p_dis: np.ndarray,
    soc: np.ndarray,
    actual_prices: np.ndarray,
    perfect_foresight_revenue: float,
    bess_dict: dict,
) -> WeeklyMetrics:
    """Compute per-week metrics from a schedule and actual prices."""
    C_deg = bess_dict["C_deg"]
    E_cap = bess_dict["E_cap"]

    # Revenue
    revenue = float(np.sum(actual_prices * (p_dis - p_ch) - C_deg * p_dis))

    # Capture ratio
    capture = revenue / perfect_foresight_revenue if perfect_foresight_revenue > 0 else 0.0

    # Cycles: total discharge energy / capacity
    total_discharge_mwh = float(np.sum(p_dis))
    cycles = total_discharge_mwh / E_cap if E_cap > 0 else 0.0

    # SoC stats
    peak_soc = float(np.max(soc))
    min_soc = float(np.min(soc))

    # Daily revenue breakdown
    T = len(actual_prices)
    n_days = T // 24
    hourly_rev = actual_prices * (p_dis - p_ch) - C_deg * p_dis
    daily_revs = np.array([hourly_rev[d * 24: (d + 1) * 24].sum() for d in range(n_days)])

    return WeeklyMetrics(
        revenue=revenue,
        capture_ratio=capture,
        cycles=cycles,
        peak_soc=peak_soc,
        min_soc=min_soc,
        daily_revenues=daily_revs,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    """Aggregate backtest metrics across all weeks."""

    mean_weekly_revenue: float
    median_weekly_revenue: float
    std_weekly_revenue: float
    annual_revenue_mean: float       # extrapolated: mean * 52
    annual_revenue_median: float
    annual_revenue_std: float
    cvar_realized: float             # mean of worst alpha-fraction of weeks
    max_drawdown: float
    negative_weeks: int
    total_weeks: int
    mean_cycles_per_week: float
    cycles_per_year: float
    revenue_per_mw_per_year_lakh: float  # INR lakh per MW per year
    capture_ratio_mean: float
    capture_ratio_std: float


def compute_aggregate_metrics(
    weekly_results: List[WeeklyMetrics],
    power_mw: float = 10.0,
    alpha: float = 0.05,
) -> AggregateMetrics:
    """
    Aggregate per-week metrics into summary statistics.

    Parameters
    ----------
    weekly_results : list of WeeklyMetrics from each backtest week
    power_mw       : battery power rating (for per-MW normalisation)
    alpha          : fraction of worst weeks for CVaR (default 5%)
    """
    n = len(weekly_results)
    revenues = np.array([w.revenue for w in weekly_results])
    captures = np.array([w.capture_ratio for w in weekly_results])
    cycles = np.array([w.cycles for w in weekly_results])

    # Annual extrapolation
    mean_rev = float(np.mean(revenues))
    median_rev = float(np.median(revenues))
    std_rev = float(np.std(revenues, ddof=1)) if n > 1 else 0.0

    # CVaR realised: mean of worst alpha fraction of weeks
    k = max(1, int(np.ceil(alpha * n)))
    sorted_revs = np.sort(revenues)  # ascending
    cvar_realized = float(np.mean(sorted_revs[:k]))

    # Max drawdown of cumulative P&L
    cum_pnl = np.cumsum(revenues)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Negative weeks
    neg_weeks = int(np.sum(revenues < 0))

    # Revenue per MW per year (lakh INR)
    annual_mean = mean_rev * 52
    rev_per_mw_year = (annual_mean / power_mw) / 1e5  # convert to lakh

    return AggregateMetrics(
        mean_weekly_revenue=mean_rev,
        median_weekly_revenue=median_rev,
        std_weekly_revenue=std_rev,
        annual_revenue_mean=annual_mean,
        annual_revenue_median=median_rev * 52,
        annual_revenue_std=std_rev * np.sqrt(52),
        cvar_realized=cvar_realized,
        max_drawdown=max_dd,
        negative_weeks=neg_weeks,
        total_weeks=n,
        mean_cycles_per_week=float(np.mean(cycles)),
        cycles_per_year=float(np.mean(cycles)) * 52,
        revenue_per_mw_per_year_lakh=rev_per_mw_year,
        capture_ratio_mean=float(np.mean(captures)),
        capture_ratio_std=float(np.std(captures, ddof=1)) if n > 1 else 0.0,
    )
