"""
Perfect-foresight deterministic LP for BESS arbitrage.

Pure LP — no binaries needed since IEX prices are non-negative
(Columbia relaxation: simultaneous charge/discharge is never optimal).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pyomo.environ as pyo

from .bess_params import BESSParams


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BESSSchedule:
    """Result of a single deterministic BESS optimisation."""

    p_ch: np.ndarray       # (T,) charge power MW per hour
    p_dis: np.ndarray      # (T,) discharge power MW per hour
    soc: np.ndarray        # (T,) state of charge MWh at end of each hour
    revenue: float         # total revenue INR over horizon
    status: str = "optimal"


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_deterministic_lp(
    prices: np.ndarray,
    bess: BESSParams | dict,
    solver_name: str = "highs",
) -> BESSSchedule:
    """
    Solve the perfect-foresight BESS arbitrage LP.

    Parameters
    ----------
    prices : (T,) array of hourly prices in INR/MWh
    bess   : BESSParams instance or dict from bess.as_dict()
    solver_name : Pyomo solver name (default 'highs')

    Returns
    -------
    BESSSchedule with optimal charge/discharge/SoC and revenue.
    """
    if isinstance(bess, dict):
        bp = bess
    else:
        bp = bess.as_dict()

    T = len(prices)
    time_set = range(T)

    P_max = bp["P_max"]
    E_min = bp["E_min"]
    E_max = bp["E_max"]
    E_init = bp["E_init"]
    eta = bp["eta"]
    C_deg = bp["C_deg"]
    tol = bp["terminal_soc_tolerance"]

    # --- Pyomo model ---
    m = pyo.ConcreteModel("DeterministicBESS")

    m.T = pyo.Set(initialize=time_set)

    m.p_dis = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, P_max))
    m.p_ch = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, P_max))
    m.soc = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(E_min, E_max))

    # Objective: maximise revenue minus degradation cost
    def _obj_rule(model):
        return sum(
            prices[t] * (model.p_dis[t] - model.p_ch[t]) - C_deg * model.p_dis[t]
            for t in model.T
        )
    m.obj = pyo.Objective(rule=_obj_rule, sense=pyo.maximize)

    # SoC dynamics
    def _soc_rule(model, t):
        soc_prev = E_init if t == 0 else model.soc[t - 1]
        return model.soc[t] == soc_prev + eta * model.p_ch[t] - model.p_dis[t] / eta
    m.soc_dynamics = pyo.Constraint(m.T, rule=_soc_rule)

    # Terminal SoC: soc[T-1] >= E_init * (1 - tolerance)
    def _terminal_rule(model):
        return model.soc[T - 1] >= E_init * (1 - tol)
    m.terminal_soc = pyo.Constraint(rule=_terminal_rule)

    # --- Solve ---
    solver = pyo.SolverFactory(solver_name)
    result = solver.solve(m, tee=False)
    status = str(result.solver.termination_condition)

    if status not in ("optimal", "feasible"):
        return BESSSchedule(
            p_ch=np.zeros(T),
            p_dis=np.zeros(T),
            soc=np.full(T, E_init),
            revenue=0.0,
            status=status,
        )

    # Extract solution
    p_ch_arr = np.array([pyo.value(m.p_ch[t]) for t in time_set])
    p_dis_arr = np.array([pyo.value(m.p_dis[t]) for t in time_set])
    soc_arr = np.array([pyo.value(m.soc[t]) for t in time_set])
    revenue = float(pyo.value(m.obj))

    return BESSSchedule(
        p_ch=p_ch_arr,
        p_dis=p_dis_arr,
        soc=soc_arr,
        revenue=revenue,
        status=status,
    )


def simulate_schedule(
    p_ch: np.ndarray,
    p_dis: np.ndarray,
    actual_prices: np.ndarray,
    bess: BESSParams | dict,
) -> float:
    """
    Simulate a given schedule against actual (realised) prices.

    Returns the total realised revenue (INR).
    """
    if isinstance(bess, dict):
        bp = bess
    else:
        bp = bess.as_dict()
    C_deg = bp["C_deg"]
    revenue = float(np.sum(actual_prices * (p_dis - p_ch) - C_deg * p_dis))
    return revenue
