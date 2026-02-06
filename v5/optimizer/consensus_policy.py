"""
Columbia consensus heuristic for BESS arbitrage.

Algorithm (per hour, rolling):
  1. Sample N price paths uniformly between conformal q_lower and q_upper.
  2. Solve deterministic LP for each sample.
  3. Vote: if ALL (or >= threshold fraction) samples agree on charge at hour 0
     -> charge (conservative amount = min across samples).
     If ALL agree on discharge -> discharge. Otherwise -> hold.
  4. Execute only hour 0's action. Advance. Repeat.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .bess_params import BESSParams
from .deterministic_lp import solve_deterministic_lp


# ---------------------------------------------------------------------------
# Single-hour consensus action
# ---------------------------------------------------------------------------

def consensus_action(
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    current_soc: float,
    bess: BESSParams | dict,
    n_samples: int = 100,
    threshold: float = 1.0,
    rng: np.random.Generator | None = None,
    solver_name: str = "highs",
) -> Tuple[float, float]:
    """
    Determine this hour's (charge, discharge) via consensus.

    Parameters
    ----------
    q_lower  : (H,) lower conformal bound for the remaining horizon
    q_upper  : (H,) upper conformal bound for the remaining horizon
    current_soc : current state of charge in MWh
    bess     : BESSParams or dict
    n_samples: number of Monte Carlo price samples
    threshold: fraction of samples that must agree (1.0 = unanimous)
    rng      : numpy random generator
    solver_name : Pyomo solver name

    Returns
    -------
    (p_ch, p_dis) for the current hour
    """
    if isinstance(bess, dict):
        bp_dict = bess
    else:
        bp_dict = bess.as_dict()

    if rng is None:
        rng = np.random.default_rng(42)

    H = len(q_lower)

    # Create a temporary bess dict with current SoC as initial
    bp_mod = bp_dict.copy()
    bp_mod["E_init"] = current_soc

    # Sample N price paths: uniform between q_lower and q_upper
    # shape (N, H)
    uniforms = rng.uniform(size=(n_samples, H))
    sampled_prices = q_lower[np.newaxis, :] + uniforms * (q_upper - q_lower)[np.newaxis, :]

    # Solve deterministic LP for each sample -> collect hour-0 actions
    ch_actions = np.zeros(n_samples)
    dis_actions = np.zeros(n_samples)

    for i in range(n_samples):
        sched = solve_deterministic_lp(sampled_prices[i], bp_mod, solver_name)
        ch_actions[i] = sched.p_ch[0]
        dis_actions[i] = sched.p_dis[0]

    # Classify each sample's action for hour 0
    eps = 1e-6
    is_charging = ch_actions > eps
    is_discharging = dis_actions > eps

    frac_charge = is_charging.mean()
    frac_discharge = is_discharging.mean()

    # Apply threshold
    if frac_charge >= threshold:
        # Charge: conservative amount = minimum charge across agreeing samples
        p_ch = float(np.min(ch_actions[is_charging]))
        return (p_ch, 0.0)
    elif frac_discharge >= threshold:
        # Discharge: conservative amount = minimum discharge across agreeing samples
        p_dis = float(np.min(dis_actions[is_discharging]))
        return (0.0, p_dis)
    else:
        # No consensus -> hold
        return (0.0, 0.0)


# ---------------------------------------------------------------------------
# Full rolling consensus schedule
# ---------------------------------------------------------------------------

def run_consensus_rolling(
    q_lower_full: np.ndarray,
    q_upper_full: np.ndarray,
    bess: BESSParams | dict,
    n_samples: int = 100,
    threshold: float = 1.0,
    solver_name: str = "highs",
    seed: int = 42,
) -> dict:
    """
    Run the full consensus policy over the entire horizon.

    Parameters
    ----------
    q_lower_full : (T,) lower conformal bound for all T hours
    q_upper_full : (T,) upper conformal bound for all T hours
    bess         : BESSParams or dict
    n_samples    : MC samples per step
    threshold    : consensus threshold
    solver_name  : Pyomo solver
    seed         : random seed

    Returns
    -------
    dict with keys: p_ch, p_dis, soc, revenue (arrays + float)
    """
    if isinstance(bess, dict):
        bp_dict = bess
    else:
        bp_dict = bess.as_dict()

    T = len(q_lower_full)
    rng = np.random.default_rng(seed)

    p_ch_arr = np.zeros(T)
    p_dis_arr = np.zeros(T)
    soc_arr = np.zeros(T)

    eta = bp_dict["eta"]
    current_soc = bp_dict["E_init"]

    for t in range(T):
        # Remaining horizon
        remaining = T - t
        q_lo = q_lower_full[t:]
        q_hi = q_upper_full[t:]

        p_ch, p_dis = consensus_action(
            q_lower=q_lo,
            q_upper=q_hi,
            current_soc=current_soc,
            bess=bp_dict,
            n_samples=n_samples,
            threshold=threshold,
            rng=rng,
            solver_name=solver_name,
        )

        # Clamp to SoC limits
        max_charge = (bp_dict["E_max"] - current_soc) / eta
        max_discharge = (current_soc - bp_dict["E_min"]) * eta
        p_ch = min(p_ch, max_charge, bp_dict["P_max"])
        p_dis = min(p_dis, max_discharge, bp_dict["P_max"])

        p_ch_arr[t] = p_ch
        p_dis_arr[t] = p_dis

        # Update SoC
        current_soc = current_soc + eta * p_ch - p_dis / eta
        soc_arr[t] = current_soc

    return {
        "p_ch": p_ch_arr,
        "p_dis": p_dis_arr,
        "soc": soc_arr,
    }
