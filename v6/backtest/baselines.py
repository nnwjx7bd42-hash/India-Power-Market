"""
Naive baseline strategies for BESS arbitrage backtesting (V6: with cycle limits).

Each function returns a dict with keys: p_ch, p_dis, soc (numpy arrays).
"""
from __future__ import annotations

import numpy as np


def naive_threshold_schedule(
    prices: np.ndarray,
    bess_dict: dict,
    charge_below: float = 3000.0,
    discharge_above: float = 6000.0,
) -> dict:
    """
    Simple rule: charge when price < charge_below, discharge when price > discharge_above.
    V6: Respects cycle limits (max 2 cycles/day, 8.08 cycles/week).

    Parameters
    ----------
    prices : (T,) hourly prices
    bess_dict : dict with P_max, E_min, E_max, E_init, eta_charge, eta_discharge, E_usable
    charge_below : INR/MWh threshold to start charging
    discharge_above : INR/MWh threshold to start discharging

    Returns
    -------
    dict with p_ch, p_dis, soc arrays (all shape (T,))
    """
    T = len(prices)
    eta_charge = bess_dict.get("eta_charge", bess_dict.get("eta", 0.9220))
    eta_discharge = bess_dict.get("eta_discharge", bess_dict.get("eta", 0.9220))
    P_max = bess_dict["P_max"]
    E_min = bess_dict["E_min"]
    E_max = bess_dict["E_max"]
    E_usable = bess_dict.get("E_usable", bess_dict.get("E_cap", 40.0) * 0.8)
    max_cycles_per_day = bess_dict.get("max_cycles_per_day", 2.0)
    max_cycles_per_week = bess_dict.get("max_cycles_per_week", 8.08)
    
    soc = bess_dict["E_init"]
    dt = 1.0  # hourly

    p_ch = np.zeros(T)
    p_dis = np.zeros(T)
    soc_arr = np.zeros(T)
    
    # Track cumulative discharge per day and per week
    daily_discharge = {}  # day_index -> cumulative MWh discharged
    weekly_discharge = 0.0

    for t in range(T):
        day_idx = t // 24
        if day_idx not in daily_discharge:
            daily_discharge[day_idx] = 0.0
        
        # Check cycle limits
        daily_cycles = daily_discharge[day_idx] / E_usable if E_usable > 0 else 0.0
        weekly_cycles = weekly_discharge / E_usable if E_usable > 0 else 0.0
        
        can_discharge = (daily_cycles < max_cycles_per_day) and (weekly_cycles < max_cycles_per_week)
        can_charge = True  # no explicit charge limit, but constrained by SoC
        
        if prices[t] <= charge_below and can_charge:
            max_ch = min(P_max, (E_max - soc) / eta_charge)
            p_ch[t] = max(0.0, max_ch)
        elif prices[t] >= discharge_above and can_discharge:
            max_dis = min(P_max, (soc - E_min) * eta_discharge)
            p_dis[t] = max(0.0, max_dis)
            daily_discharge[day_idx] += p_dis[t] * dt
            weekly_discharge += p_dis[t] * dt
        # else: hold

        soc = soc + eta_charge * p_ch[t] - p_dis[t] / eta_discharge
        soc_arr[t] = soc

    return {"p_ch": p_ch, "p_dis": p_dis, "soc": soc_arr}


def do_nothing_schedule(T: int, E_init: float) -> dict:
    """Baseline that does absolutely nothing (holds SoC constant)."""
    return {
        "p_ch": np.zeros(T),
        "p_dis": np.zeros(T),
        "soc": np.full(T, E_init),
    }
