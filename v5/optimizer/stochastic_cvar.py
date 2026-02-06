"""
Two-stage stochastic LP with CVaR risk measure for BESS arbitrage.

Rockafellar-Uryasev linearisation:
    max  (1-beta)*E[R] + beta*CVaR
    s.t. CVaR = zeta - 1/(1-alpha) * sum(w_s * u_s)
         u_s >= zeta - R_s   for all s
         u_s >= 0            for all s
         + physical BESS constraints (same schedule for all scenarios)

Pure LP (no binaries) — valid for non-negative IEX prices.
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
    expected_revenue: float    # weighted mean revenue across scenarios
    cvar: float                # conditional value-at-risk
    var: float                 # value-at-risk (zeta)
    per_scenario_revenue: np.ndarray   # (S,) revenue per scenario
    objective: float           # optimal objective value
    status: str = "optimal"


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_stochastic_cvar(
    scenarios: np.ndarray,
    weights: np.ndarray,
    bess: BESSParams | dict,
    beta: float = 0.3,
    alpha: float = 0.95,
    solver_name: str = "highs",
) -> StochasticResult:
    """
    Solve the scenario-based stochastic BESS LP with CVaR.

    Parameters
    ----------
    scenarios : (S, T) price paths
    weights   : (S,) probability weights summing to 1
    bess      : BESSParams or dict from bess.as_dict()
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

    S, T = scenarios.shape
    time_set = range(T)
    scen_set = range(S)

    P_max = bp["P_max"]
    E_min = bp["E_min"]
    E_max = bp["E_max"]
    E_init = bp["E_init"]
    eta = bp["eta"]
    C_deg = bp["C_deg"]
    tol = bp["terminal_soc_tolerance"]

    # normalise weights just in case
    w = weights / weights.sum()

    # --- Pyomo model ---
    m = pyo.ConcreteModel("StochasticCVaR_BESS")

    m.T = pyo.Set(initialize=time_set)
    m.S = pyo.Set(initialize=scen_set)

    # Decision variables (scenario-independent — here-and-now)
    m.p_dis = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, P_max))
    m.p_ch = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, P_max))
    m.soc = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(E_min, E_max))

    # CVaR auxiliary variables
    m.zeta = pyo.Var(domain=pyo.Reals)       # VaR threshold
    m.u = pyo.Var(m.S, domain=pyo.NonNegativeReals)  # shortfall per scenario

    # Per-scenario revenue expression
    def _revenue_expr(model, s):
        return sum(
            scenarios[s, t] * (model.p_dis[t] - model.p_ch[t]) - C_deg * model.p_dis[t]
            for t in model.T
        )
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

    # SoC dynamics
    def _soc_rule(model, t):
        soc_prev = E_init if t == 0 else model.soc[t - 1]
        return model.soc[t] == soc_prev + eta * model.p_ch[t] - model.p_dis[t] / eta
    m.soc_dynamics = pyo.Constraint(m.T, rule=_soc_rule)

    # Terminal SoC
    def _terminal_rule(model):
        return model.soc[T - 1] >= E_init * (1 - tol)
    m.terminal_soc = pyo.Constraint(rule=_terminal_rule)

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
