"""
CERC Deviation Settlement Mechanism (DSM) cost calculation.

Full slab structure for backtest evaluation.  Optimizer uses a simplified
conservative buffer term (flat INR/MWh adder) to avoid non-linearity in the LP.

CERC DSM Slab Structure (2024-25 Regulations):
    Under-injection (actual < scheduled):
        ≤ 2%    → 100% of Normal Rate (NR)
        2–5%    → 105–115% of NR (frequency-dependent)
        5–10%   → 115–150% of NR
        10–20%  → 150–200% of NR
        > 20%   → 200% of NR (flat)
    Over-injection (actual > scheduled):
        ≤ 2%    → paid at NR
        2–10%   → 90–50% of NR (decreasing)
        10–20%  → 50–0% of NR
        > 20%   → 0% (no payment for excess)

    NR ≈ DAM Area Clearing Price for the relevant time block.
    Grid frequency affects the exact multiplier for under-injection;
    the implementation uses a conservative (worst-case) interpolation.
"""
from __future__ import annotations

import numpy as np


def compute_dsm_cost(
    scheduled_mw: float,
    actual_mw: float,
    normal_rate_inr: float,
    freq_hz: float = 50.0,
) -> float:
    """
    Compute DSM cost for a single 15-minute block using CERC DSM slab structure.
    
    Parameters
    ----------
    scheduled_mw : scheduled injection/withdrawal MW
    actual_mw    : actual injection/withdrawal MW
    normal_rate_inr : Normal Rate (NR) in INR/MWh (typically ≈ DAM ACP)
    freq_hz      : grid frequency Hz (affects under-injection rates)
    
    Returns
    -------
    DSM cost in INR (positive = penalty, negative = payment for over-injection)
    """
    if scheduled_mw == 0:
        return 0.0
    
    # Deviation in MWh (15-min block = 0.25 hours)
    deviation_mwh = (actual_mw - scheduled_mw) * 0.25
    deviation_pct = abs(deviation_mwh / (scheduled_mw * 0.25)) * 100.0
    
    if deviation_mwh < 0:  # Under-injection (discharged less than scheduled)
        # Frequency-dependent rate multiplier (simplified: worst case)
        if deviation_pct <= 2.0:
            rate_multiplier = 1.0  # 100% of NR
        elif deviation_pct <= 5.0:
            # 105-115% depending on frequency (use worst case 115%)
            rate_multiplier = 1.10 + (deviation_pct - 2.0) / 3.0 * 0.05
        elif deviation_pct <= 10.0:
            # 115-150% depending on frequency
            rate_multiplier = 1.15 + (deviation_pct - 5.0) / 5.0 * 0.35
        elif deviation_pct <= 20.0:
            # 150-200% depending on frequency
            rate_multiplier = 1.50 + (deviation_pct - 10.0) / 10.0 * 0.50
        else:
            rate_multiplier = 2.0  # 200% flat
        
        return abs(deviation_mwh) * normal_rate_inr * rate_multiplier
    
    else:  # Over-injection (discharged more than scheduled)
        if deviation_pct <= 2.0:
            rate_multiplier = 1.0  # paid at NR
        elif deviation_pct <= 10.0:
            # Paid at decreasing rate: 90-115% → 50%
            rate_multiplier = 0.90 - (deviation_pct - 2.0) / 8.0 * 0.40
        elif deviation_pct <= 20.0:
            # Paid at 50-0%
            rate_multiplier = 0.50 - (deviation_pct - 10.0) / 10.0 * 0.50
        else:
            rate_multiplier = 0.0  # paid ZERO for excess >20%
        
        # Negative = payment (revenue), but we return as cost (positive = penalty)
        # So over-injection reduces cost (negative return)
        return -abs(deviation_mwh) * normal_rate_inr * rate_multiplier


def simulate_bess_deviation(
    scheduled_mw: float,
    systematic_bias_pct: float = 0.5,
    random_std_pct: float = 1.0,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Simulate realistic BESS deviation from scheduled injection.
    
    BESS with SCADA typically achieves <2% deviation:
    - Systematic bias: slight under-delivery (~0.5%)
    - Random noise: ~1% standard deviation
    
    Parameters
    ----------
    scheduled_mw : scheduled injection MW
    systematic_bias_pct : systematic under-delivery percentage
    random_std_pct : random deviation standard deviation percentage
    rng : numpy random generator (for reproducibility)
    
    Returns
    -------
    actual_mw : actual injection MW with noise
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    noise_pct = -systematic_bias_pct + rng.normal(0, random_std_pct)
    actual_mw = scheduled_mw * (1 + noise_pct / 100.0)
    
    # Clamp to physical limits (can't discharge negative or exceed P_max)
    actual_mw = max(0.0, actual_mw)
    
    return actual_mw


def compute_dsm_cost_schedule(
    scheduled_mw: np.ndarray,
    actual_mw: np.ndarray,
    normal_rates_inr: np.ndarray,
    freq_hz: float = 50.0,
) -> float:
    """
    Compute total DSM cost over a full schedule (96 × 15-min blocks).
    
    Parameters
    ----------
    scheduled_mw : (96,) scheduled injection MW per block
    actual_mw     : (96,) actual injection MW per block
    normal_rates_inr : (96,) Normal Rate INR/MWh per block
    freq_hz      : grid frequency (assumed constant)
    
    Returns
    -------
    Total DSM cost in INR
    """
    total_cost = 0.0
    for i in range(len(scheduled_mw)):
        cost = compute_dsm_cost(
            scheduled_mw[i], actual_mw[i], normal_rates_inr[i], freq_hz
        )
        total_cost += cost
    return total_cost
