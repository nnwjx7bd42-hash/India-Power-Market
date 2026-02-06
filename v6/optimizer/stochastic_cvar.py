"""
Two-stage stochastic LP with CVaR risk measure for BESS arbitrage (V6: with transaction costs and cycle limits).

Rockafellar-Uryasev linearisation:
    max  (1-beta)*E[R] + beta*CVaR
    s.t. CVaR = zeta - 1/(1-alpha) * sum(w_s * u_s)
         u_s >= zeta - R_s   for all s
         u_s >= 0            for all s
         + physical BESS constraints (same schedule for all scenarios)

WHY PURE LP (no binary mutual-exclusion constraints):
    Standard BESS formulations require binary variables to prevent simultaneous
    charging and discharging in the same timestep.  However, IEX DAM prices are
    non-negative (floor = 0 INR/MWh), so simultaneous charge + discharge would
    always *increase* cost without increasing revenue.  Any LP-optimal solution
    therefore naturally satisfies p_ch[t] * p_dis[t] == 0 for all t.
    This eliminates the need for big-M or SOS1 constraints, keeping the problem
    a pure LP solvable in polynomial time by HiGHS's simplex/IPM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pyomo.environ as pyo

from .bess_params import BESSParams


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StochasticResult:
    """Result of the stochastic CVaR optimisation."""

    p_ch: np.ndarray           # (T,) charge power
    p_dis: np.ndarray          # (T,) discharge power
    soc: np.ndarray            # (T,) state of charge
    expected_revenue: float    # weighted mean net revenue across scenarios
    cvar: float                # conditional value-at-risk
    var: float                 # value-at-risk (zeta)
    per_scenario_revenue: np.ndarray   # (S,) net revenue per scenario
    objective: float           # optimal objective value
    status: str = "optimal"


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_stochastic_cvar(
    scenarios: np.ndarray,
    weights: np.ndarray,
    bess: BESSParams | dict,
    transaction_costs_dict: Dict[str, float] | None = None,
    beta: float = 0.3,
    alpha: float = 0.95,
    solver_name: str = "highs",
) -> StochasticResult:
    """
    Solve the scenario-based stochastic BESS LP with CVaR, transaction costs, and cycle limits.

    Parameters
    ----------
    scenarios : (S, T) price paths
    weights   : (S,) probability weights summing to 1
    bess      : BESSParams or dict from bess.as_dict()
    transaction_costs_dict : dict with IEX/SLDC/RLDC fees, tx_loss, dsm_buffer
    beta      : risk-aversion weight (0=risk-neutral, 1=fully risk-averse)
    alpha     : CVaR confidence level (e.g. 0.95 for 95th percentile tail)
    solver_name : Pyomo solver

    Returns
    -------
    StochasticResult
    """
    if isinstance(bess, dict):
        bp = bess
    else:
        bp = bess.as_dict()

    if transaction_costs_dict is None:
        transaction_costs_dict = {}

    S, T = scenarios.shape
    time_set = range(T)
    scen_set = range(S)
    dt = 1.0  # hourly timestep

    P_max = bp["P_max"]
    E_min = bp["E_min"]
    E_max = bp["E_max"]
    E_init = bp["E_init"]
    eta_charge = bp.get("eta_charge", bp.get("eta", 0.9220))
    eta_discharge = bp.get("eta_discharge", bp.get("eta", 0.9220))
    C_deg = bp["C_deg"]
    tol = bp["terminal_soc_tolerance"]
    E_usable = bp.get("E_usable", bp["E_cap"] * 0.8)
    max_cycles_per_day = bp.get("max_cycles_per_day", 2.0)
    max_cycles_per_week = bp.get("max_cycles_per_week", 8.08)

    # Extract cost parameters
    tx_loss = transaction_costs_dict.get("transmission_loss_pct", 0.03)
    fee_per_mwh = (
        transaction_costs_dict.get("iex_transaction_fee_inr_mwh", 20.0)
        + transaction_costs_dict.get("sldc_charge_inr_mwh", 5.0)
        + transaction_costs_dict.get("rldc_charge_inr_mwh", 2.0)
    )
    dsm_buffer = transaction_costs_dict.get("dsm_cost_buffer_inr_mwh", 25.0)

    # normalise weights just in case
    w = weights / weights.sum()

    # --- Pyomo model ---
    m = pyo.ConcreteModel("StochasticCVaR_BESS_V6")

    m.T = pyo.Set(initialize=time_set)
    m.S = pyo.Set(initialize=scen_set)

    # Decision variables (scenario-independent — "here-and-now" decisions).
    # The dispatch schedule is the SAME across all scenarios — we commit to a
    # single schedule at bid time, before the uncertain price is realised.
    m.p_dis = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, P_max))
    m.p_ch = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, P_max))
    m.soc = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(E_min, E_max))

    # CVaR auxiliary variables (Rockafellar-Uryasev linearisation):
    #   zeta ≈ VaR_alpha (the alpha-quantile of revenue distribution)
    #   u[s] ≥ max(0, zeta - R_s)  captures the "shortfall" below VaR
    m.zeta = pyo.Var(domain=pyo.Reals)
    m.u = pyo.Var(m.S, domain=pyo.NonNegativeReals)

    # Per-scenario net revenue expression (with all transaction costs)
    def _revenue_expr(model, s):
        total = 0.0
        for t in model.T:
            # Sell revenue (after efficiency and tx losses)
            effective_injection = model.p_dis[t] * eta_discharge * (1 - tx_loss) * dt
            sell_revenue = effective_injection * scenarios[s, t]
            sell_fees = effective_injection * fee_per_mwh
            
            # Buy cost
            energy_drawn = model.p_ch[t] * dt
            buy_cost = energy_drawn * (scenarios[s, t] + fee_per_mwh)
            
            # Degradation
            degradation = (model.p_dis[t] + model.p_ch[t]) * dt * C_deg / 1000.0
            
            # DSM buffer
            dsm = (model.p_dis[t] + model.p_ch[t]) * dt * dsm_buffer / 1000.0
            
            total += sell_revenue - sell_fees - buy_cost - degradation - dsm
        return total
    m.R = pyo.Expression(m.S, rule=_revenue_expr)

    # Expected revenue
    def _expected_revenue(model):
        return sum(w[s] * model.R[s] for s in model.S)
    m.E_R = pyo.Expression(rule=_expected_revenue)

    # CVaR expression
    def _cvar_expr(model):
        return model.zeta - (1.0 / (1.0 - alpha)) * sum(w[s] * model.u[s] for s in model.S)
    m.CVaR = pyo.Expression(rule=_cvar_expr)

    # Objective: max (1-beta)*E[R] + beta*CVaR
    def _obj_rule(model):
        return (1.0 - beta) * model.E_R + beta * model.CVaR
    m.obj = pyo.Objective(rule=_obj_rule, sense=pyo.maximize)

    # CVaR linearisation constraints: u_s >= zeta - R_s
    def _cvar_shortfall(model, s):
        return model.u[s] >= model.zeta - model.R[s]
    m.cvar_shortfall = pyo.Constraint(m.S, rule=_cvar_shortfall)

    # SoC dynamics (separate charge/discharge efficiencies)
    def _soc_rule(model, t):
        soc_prev = E_init if t == 0 else model.soc[t - 1]
        return model.soc[t] == soc_prev + eta_charge * model.p_ch[t] - model.p_dis[t] / eta_discharge
    m.soc_dynamics = pyo.Constraint(m.T, rule=_soc_rule)

    # Terminal SoC
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
        return StochasticResult(
            p_ch=np.zeros(T),
            p_dis=np.zeros(T),
            soc=np.full(T, E_init),
            expected_revenue=0.0,
            cvar=0.0,
            var=0.0,
            per_scenario_revenue=np.zeros(S),
            objective=0.0,
            status=status,
        )

    # Extract solution
    p_ch_arr = np.array([pyo.value(m.p_ch[t]) for t in time_set])
    p_dis_arr = np.array([pyo.value(m.p_dis[t]) for t in time_set])
    soc_arr = np.array([pyo.value(m.soc[t]) for t in time_set])
    per_scen_rev = np.array([pyo.value(m.R[s]) for s in scen_set])
    expected_rev = float(pyo.value(m.E_R))
    cvar_val = float(pyo.value(m.CVaR))
    var_val = float(pyo.value(m.zeta))
    obj_val = float(pyo.value(m.obj))

    return StochasticResult(
        p_ch=p_ch_arr,
        p_dis=p_dis_arr,
        soc=soc_arr,
        expected_revenue=expected_rev,
        cvar=cvar_val,
        var=var_val,
        per_scenario_revenue=per_scen_rev,
        objective=obj_val,
        status=status,
    )
