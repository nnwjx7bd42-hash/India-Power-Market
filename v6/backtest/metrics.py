"""
Backtest metrics for BESS arbitrage strategies (V6: with cost breakdown).

Per-week metrics:
    - Gross revenue, all cost components, net revenue
    - Revenue capture ratio (vs perfect foresight)
    - Number of cycles consumed
    - Peak SoC, min SoC, SoC statistics
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

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

import sys
from pathlib import Path

# Ensure v6 is importable
_V6_ROOT = Path(__file__).resolve().parent.parent
if str(_V6_ROOT) not in sys.path:
    sys.path.insert(0, str(_V6_ROOT))

from optimizer.transaction_costs import compute_total_transaction_costs
from optimizer.dsm_costs import compute_dsm_cost_schedule, simulate_bess_deviation


# ---------------------------------------------------------------------------
# Per-week metrics
# ---------------------------------------------------------------------------

@dataclass
class WeeklyMetrics:
    """Metrics for a single week of BESS operation (V6: extended with cost breakdown)."""

    # Revenue components (all *simulated* under actual price path)
    gross_revenue_inr: float
    iex_fees_inr: float
    sldc_rldc_fees_inr: float
    tx_loss_inr: float
    dsm_cost_inr: float
    degradation_cost_inr: float
    net_revenue_inr: float
    
    # Capture ratios (set externally after all strategies are evaluated)
    capture_vs_pf_pct: float = float("nan")       # 100 * net_revenue / pf_net_revenue
    capture_vs_naive_pct: float = float("nan")     # 100 * net_revenue / naive_net_revenue
    
    # Cycles
    cycles: float = 0.0                 # number of full cycles (total discharge / E_usable)
    cumulative_cycles_ytd: float = 0.0   # running total for annual cap check
    
    # SoC statistics
    peak_soc: float = 0.0
    min_soc: float = 0.0
    avg_soc_pct: float = 0.0            # average SoC as percentage of E_cap
    hours_at_soc_min: int = 0           # hours sitting at floor
    hours_at_soc_max: int = 0           # hours sitting at ceiling
    
    # Daily breakdown
    daily_revenues: np.ndarray = None    # (7,) or (n_days,) daily net revenue breakdown


def compute_weekly_metrics(
    p_ch: np.ndarray,
    p_dis: np.ndarray,
    soc: np.ndarray,
    actual_prices: np.ndarray,
    bess_dict: dict,
    transaction_costs_dict: Dict[str, float],
    compute_dsm: bool = True,
    rng: np.random.Generator | None = None,
) -> WeeklyMetrics:
    """
    Compute per-week metrics from a schedule and actual prices (V6: with full cost breakdown).
    
    All revenue numbers are *simulated* under the actual price path.
    Capture ratios are NOT set here — they are set externally by the
    rolling_backtest after all strategies have been evaluated for the same week.
    
    Parameters
    ----------
    p_ch : (T,) charge power MW per hour
    p_dis : (T,) discharge power MW per hour
    soc : (T,) state of charge MWh at end of each hour
    actual_prices : (T,) actual clearing prices INR/MWh
    bess_dict : BESS parameters dict
    transaction_costs_dict : transaction costs dict
    compute_dsm : if True, compute full DSM costs using deviation simulation
    rng : numpy random generator for DSM deviation simulation
    """
    T = len(actual_prices)
    dt = 1.0
    E_usable = bess_dict.get("E_usable", bess_dict.get("E_cap", 40.0) * 0.8)
    E_cap = bess_dict.get("E_cap", 40.0)
    C_deg = bess_dict.get("C_deg", 1471.0)
    eta_discharge = bess_dict.get("eta_discharge", bess_dict.get("eta", 0.9220))
    
    # Gross revenue (before any costs)
    gross_revenue = float(np.sum(actual_prices * (p_dis * eta_discharge - p_ch)))
    
    # Transaction costs
    tx_costs = compute_total_transaction_costs(
        p_dis, p_ch, actual_prices, dt, transaction_costs_dict, bess_dict
    )
    iex_fees = tx_costs["iex_fees_inr"]
    sldc_rldc_fees = tx_costs["sldc_rldc_fees_inr"]
    tx_loss_value = tx_costs["tx_loss_inr"]
    
    # Degradation cost
    total_throughput = float(np.sum(p_dis * dt + p_ch * dt))
    degradation_cost = total_throughput * C_deg / 1000.0
    
    # DSM cost (full calculation with deviation simulation)
    dsm_cost = 0.0
    if compute_dsm:
        # Convert hourly schedule to 96 × 15-min blocks
        scheduled_15min = []
        actual_15min = []
        normal_rates_15min = []
        
        for h in range(T):
            # Each hour = 4 × 15-min blocks
            for q in range(4):
                scheduled_15min.append(p_dis[h] if q == 0 else 0.0)  # simplified: discharge at hour start
                # Simulate deviation
                actual_mw = simulate_bess_deviation(p_dis[h], rng=rng) if p_dis[h] > 0 else 0.0
                actual_15min.append(actual_mw)
                normal_rates_15min.append(actual_prices[h])  # use hourly price for all 4 blocks
        
        scheduled_15min = np.array(scheduled_15min[:96])  # clip to 96 blocks
        actual_15min = np.array(actual_15min[:96])
        normal_rates_15min = np.array(normal_rates_15min[:96])
        
        dsm_cost = compute_dsm_cost_schedule(
            scheduled_15min, actual_15min, normal_rates_15min
        )
    
    # Net revenue (simulated, under actual price path)
    net_revenue = gross_revenue - iex_fees - sldc_rldc_fees - tx_loss_value - degradation_cost - dsm_cost
    
    # Cycles: total discharge energy / usable capacity
    total_discharge_mwh = float(np.sum(p_dis * dt))
    cycles = total_discharge_mwh / E_usable if E_usable > 0 else 0.0
    
    # SoC statistics
    peak_soc = float(np.max(soc))
    min_soc = float(np.min(soc))
    avg_soc_pct = float(np.mean(soc) / E_cap * 100.0)
    
    soc_min_threshold = bess_dict.get("E_min", E_cap * 0.1)
    soc_max_threshold = bess_dict.get("E_max", E_cap * 0.9)
    hours_at_soc_min = int(np.sum(soc <= soc_min_threshold * 1.01))  # within 1% of floor
    hours_at_soc_max = int(np.sum(soc >= soc_max_threshold * 0.99))  # within 1% of ceiling
    
    # Daily revenue breakdown (net)
    n_days = T // 24
    daily_revs = []
    for d in range(n_days):
        day_start = d * 24
        day_end = min((d + 1) * 24, T)
        day_p_ch = p_ch[day_start:day_end]
        day_p_dis = p_dis[day_start:day_end]
        day_prices = actual_prices[day_start:day_end]
        
        day_gross = float(np.sum(day_prices * (day_p_dis * eta_discharge - day_p_ch)))
        day_tx = compute_total_transaction_costs(
            day_p_dis, day_p_ch, day_prices, dt, transaction_costs_dict, bess_dict
        )
        day_deg = float(np.sum(day_p_dis * dt + day_p_ch * dt)) * C_deg / 1000.0
        day_rev = day_gross - day_tx["total_transaction_costs_inr"] - day_deg
        daily_revs.append(day_rev)
    
    daily_revs = np.array(daily_revs)
    
    return WeeklyMetrics(
        gross_revenue_inr=gross_revenue,
        iex_fees_inr=iex_fees,
        sldc_rldc_fees_inr=sldc_rldc_fees,
        tx_loss_inr=tx_loss_value,
        dsm_cost_inr=dsm_cost,
        degradation_cost_inr=degradation_cost,
        net_revenue_inr=net_revenue,
        # capture ratios left as NaN — set by rolling_backtest
        cycles=cycles,
        cumulative_cycles_ytd=0.0,  # will be set by rolling_backtest
        peak_soc=peak_soc,
        min_soc=min_soc,
        avg_soc_pct=avg_soc_pct,
        hours_at_soc_min=hours_at_soc_min,
        hours_at_soc_max=hours_at_soc_max,
        daily_revenues=daily_revs,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    """Aggregate backtest metrics across all weeks (V6: extended)."""

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
    
    # V6 additions
    mean_gross_revenue: float
    mean_total_costs: float
    capture_vs_pf_pct_mean: float        # mean of 100*net/pf_net
    capture_vs_pf_pct_std: float


def compute_aggregate_metrics(
    weekly_results: List[WeeklyMetrics],
    power_mw: float = 20.0,
    alpha: float = 0.05,
) -> AggregateMetrics:
    """
    Aggregate per-week metrics into summary statistics (V6: extended).
    
    Parameters
    ----------
    weekly_results : list of WeeklyMetrics from each backtest week
    power_mw       : battery power rating (for per-MW normalisation)
    alpha          : fraction of worst weeks for CVaR (default 5%)
    """
    n = len(weekly_results)
    net_revenues = np.array([w.net_revenue_inr for w in weekly_results])
    gross_revenues = np.array([w.gross_revenue_inr for w in weekly_results])
    capture_pf = np.array([w.capture_vs_pf_pct for w in weekly_results])
    cycles = np.array([w.cycles for w in weekly_results])
    
    # Total costs per week
    total_costs = np.array([
        w.iex_fees_inr + w.sldc_rldc_fees_inr + w.tx_loss_inr + 
        w.dsm_cost_inr + w.degradation_cost_inr
        for w in weekly_results
    ])
    
    # Annual extrapolation (using net revenue)
    mean_rev = float(np.mean(net_revenues))
    median_rev = float(np.median(net_revenues))
    std_rev = float(np.std(net_revenues, ddof=1)) if n > 1 else 0.0
    
    # CVaR realised: mean of worst alpha fraction of weeks
    k = max(1, int(np.ceil(alpha * n)))
    sorted_revs = np.sort(net_revenues)  # ascending
    cvar_realized = float(np.mean(sorted_revs[:k]))
    
    # Max drawdown of cumulative P&L
    cum_pnl = np.cumsum(net_revenues)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    # Negative weeks
    neg_weeks = int(np.sum(net_revenues < 0))
    
    # Revenue per MW per year (lakh INR)
    annual_mean = mean_rev * 52
    rev_per_mw_year = (annual_mean / power_mw) / 1e5  # convert to lakh
    
    # Capture ratio stats (filter NaNs from PF itself where ratio=100)
    valid_cap = capture_pf[~np.isnan(capture_pf)]
    cap_mean = float(np.mean(valid_cap)) if len(valid_cap) > 0 else float("nan")
    cap_std = float(np.std(valid_cap, ddof=1)) if len(valid_cap) > 1 else 0.0
    
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
        mean_gross_revenue=float(np.mean(gross_revenues)),
        mean_total_costs=float(np.mean(total_costs)),
        capture_vs_pf_pct_mean=cap_mean,
        capture_vs_pf_pct_std=cap_std,
    )
