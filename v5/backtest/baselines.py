"""
Naive baseline strategies for BESS arbitrage backtesting.

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

    Parameters
    ----------
    prices : (T,) hourly prices
    bess_dict : dict with P_max, E_min, E_max, E_init, eta
    charge_below : INR/MWh threshold to start charging
    discharge_above : INR/MWh threshold to start discharging

    Returns
    -------
    dict with p_ch, p_dis, soc arrays (all shape (T,))
    """
    T = len(prices)
    eta = bess_dict["eta"]
    P_max = bess_dict["P_max"]
    E_min = bess_dict["E_min"]
    E_max = bess_dict["E_max"]
    soc = bess_dict["E_init"]

    p_ch = np.zeros(T)
    p_dis = np.zeros(T)
    soc_arr = np.zeros(T)

    for t in range(T):
        if prices[t] <= charge_below:
            max_ch = min(P_max, (E_max - soc) / eta)
            p_ch[t] = max(0.0, max_ch)
        elif prices[t] >= discharge_above:
            max_dis = min(P_max, (soc - E_min) * eta)
            p_dis[t] = max(0.0, max_dis)
        # else: hold

        soc = soc + eta * p_ch[t] - p_dis[t] / eta
        soc_arr[t] = soc

    return {"p_ch": p_ch, "p_dis": p_dis, "soc": soc_arr}


def do_nothing_schedule(T: int, E_init: float) -> dict:
    """Baseline that does absolutely nothing (holds SoC constant)."""
    return {
        "p_ch": np.zeros(T),
        "p_dis": np.zeros(T),
        "soc": np.full(T, E_init),
    }
