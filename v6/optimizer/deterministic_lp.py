"""
Perfect-foresight deterministic LP for BESS arbitrage (V6: with transaction costs and cycle limits).

Pure LP — no binaries needed since IEX prices are non-negative
(Columbia relaxation: simultaneous charge/discharge is never optimal).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pyomo.environ as pyo

from .bess_params import BESSParams
from .transaction_costs import net_revenue_per_timestep


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BESSSchedule:
    """Result of a single deterministic BESS optimisation."""

    p_ch: np.ndarray       # (T,) charge power MW per hour
    p_dis: np.ndarray      # (T,) discharge power MW per hour
    soc: np.ndarray        # (T,) state of charge MWh at end of each hour
    revenue: float         # total net revenue INR over horizon
    status: str = "optimal"


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_deterministic_lp(
    prices: np.ndarray,
    bess: BESSParams | dict,
    transaction_costs_dict: Dict[str, float] | None = None,
    solver_name: str = "highs",
) -> BESSSchedule:
    """
    Solve the perfect-foresight BESS arbitrage LP with transaction costs and cycle limits.

    Parameters
    ----------
    prices : (T,) array of hourly prices in INR/MWh
    bess   : BESSParams instance or dict from bess.as_dict()
    transaction_costs_dict : dict with IEX/SLDC/RLDC fees, tx_loss, dsm_buffer
    solver_name : Pyomo solver name (default 'highs')

    Returns
    -------
    BESSSchedule with optimal charge/discharge/SoC and net revenue.
    """
    if isinstance(bess, dict):
        bp = bess
    else:
        bp = bess.as_dict()

    if transaction_costs_dict is None:
        transaction_costs_dict = {}

    T = len(prices)
    time_set = range(T)
    dt = 1.0  # hourly timestep

    P_max = bp["P_max"]
    E_min = bp["E_min"]
    E_max = bp["E_max"]
    E_init = bp["E_init"]
    eta_charge = bp.get("eta_charge", bp.get("eta", 0.9220))
    eta_discharge = bp.get("eta_discharge", bp.get("eta", 0.9220))
    C_deg = bp["C_deg"]
    tol = bp["terminal_soc_tolerance"]
    E_usable = bp.get("E_usable", bp["E_cap"] * 0.8)  # fallback if not in dict
    max_cycles_per_day = bp.get("max_cycles_per_day", 2.0)
    max_cycles_per_week = bp.get("max_cycles_per_week", 8.08)

    # --- Pyomo model ---
    m = pyo.ConcreteModel("DeterministicBESS_V6")

    m.T = pyo.Set(initialize=time_set)

    m.p_dis = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, P_max))
    m.p_ch = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, P_max))
    m.soc = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(E_min, E_max))

    # Objective: maximise net revenue (including all transaction costs)
    def _obj_rule(model):
        total_revenue = 0.0
        for t in model.T:
            # Use net revenue calculation with all costs
            rev = net_revenue_per_timestep(
                float(pyo.value(model.p_dis[t])),
                float(pyo.value(model.p_ch[t])),
                prices[t],
                dt,
                transaction_costs_dict,
                bp,
            )
            total_revenue += rev
        return total_revenue
    
    # For Pyomo, we need to build the expression directly
    def _obj_expr(model):
        total = 0.0
        for t in model.T:
            # Net revenue per timestep: sell_revenue - buy_cost - degradation - dsm_buffer
            eta_dis = eta_discharge
            tx_loss = transaction_costs_dict.get("transmission_loss_pct", 0.03)
            fee_per_mwh = (
                transaction_costs_dict.get("iex_transaction_fee_inr_mwh", 20.0)
                + transaction_costs_dict.get("sldc_charge_inr_mwh", 5.0)
                + transaction_costs_dict.get("rldc_charge_inr_mwh", 2.0)
            )
            dsm_buffer = transaction_costs_dict.get("dsm_cost_buffer_inr_mwh", 25.0)
            
            # Sell revenue (after efficiency and tx losses)
            effective_injection = model.p_dis[t] * eta_dis * (1 - tx_loss) * dt
            sell_revenue = effective_injection * prices[t]
            sell_fees = effective_injection * fee_per_mwh
            
            # Buy cost
            energy_drawn = model.p_ch[t] * dt
            buy_cost = energy_drawn * (prices[t] + fee_per_mwh)
            
            # Degradation
            degradation = (model.p_dis[t] + model.p_ch[t]) * dt * C_deg / 1000.0
            
            # DSM buffer
            dsm = (model.p_dis[t] + model.p_ch[t]) * dt * dsm_buffer / 1000.0
            
            total += sell_revenue - sell_fees - buy_cost - degradation - dsm
        return total
    
    m.obj = pyo.Objective(rule=_obj_expr, sense=pyo.maximize)

    # SoC dynamics (separate charge/discharge efficiencies)
    def _soc_rule(model, t):
        soc_prev = E_init if t == 0 else model.soc[t - 1]
        return model.soc[t] == soc_prev + eta_charge * model.p_ch[t] - model.p_dis[t] / eta_discharge
    m.soc_dynamics = pyo.Constraint(m.T, rule=_soc_rule)

    # Terminal SoC: soc[T-1] >= E_init * (1 - tolerance)
    def _terminal_rule(model):
        return model.soc[T - 1] >= E_init * (1 - tol)
    m.terminal_soc = pyo.Constraint(rule=_terminal_rule)

    # Cycle limit constraints
    # Daily: sum(P_dis[t] * dt for t in day_d) <= max_cycles_per_day * E_usable
    n_days = T // 24
    for d in range(n_days):
        day_start = d * 24
        day_end = min((d + 1) * 24, T)
        day_hours = list(range(day_start, day_end))
        
        def _make_daily_rule(hours):
            def _daily_cycle_rule(model):
                return sum(model.p_dis[t] * dt for t in hours) <= max_cycles_per_day * E_usable
            return _daily_cycle_rule
        setattr(m, f"daily_cycle_day{d}", pyo.Constraint(rule=_make_daily_rule(day_hours)))

    # Weekly: sum(P_dis[t] * dt for t in T) <= max_cycles_per_week * E_usable
    def _weekly_cycle_rule(model):
        return sum(model.p_dis[t] * dt for t in model.T) <= max_cycles_per_week * E_usable
    m.weekly_cycle = pyo.Constraint(rule=_weekly_cycle_rule)

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
    transaction_costs_dict: Dict[str, float] | None = None,
) -> float:
    """
    Simulate a given schedule against actual (realised) prices with transaction costs.

    Returns the total realised net revenue (INR).
    """
    if isinstance(bess, dict):
        bp = bess
    else:
        bp = bess.as_dict()
    
    if transaction_costs_dict is None:
        transaction_costs_dict = {}
    
    dt = 1.0
    total_revenue = 0.0
    
    for t in range(len(actual_prices)):
        rev = net_revenue_per_timestep(
            p_dis[t], p_ch[t], actual_prices[t], dt, transaction_costs_dict, bp
        )
        total_revenue += rev
    
    return total_revenue
