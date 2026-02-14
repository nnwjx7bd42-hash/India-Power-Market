import pulp
import numpy as np
from typing import Dict, List
from src.optimizer.bess_params import BESSParams


class MultiDayBESS:
    """7-Day Extensive Form Stochastic BESS Optimizer.
    
    Solves a single LP spanning n_days × 24 hours with SoC flowing
    continuously across day boundaries. Only Day 1's DAM schedule
    is actionable; remaining days are lookahead.
    
    CRITICAL: This class mirrors TwoStageBESS exactly in:
    - In-LP cost structure (IEX fee + degradation + ₹135 DSM friction)
    - CVaR linearization (Rockafellar-Uryasev)
    - Deviation penalty (lambda_dev, dev_max)
    - SoC dynamics formula
    - Revenue formula (IEX deviation-based settlement)
    """
    
    def __init__(self, params: BESSParams, config: Dict):
        self.params = params
        self.config = config
        self.solver_name = config.get('solver', 'CBC')
        self.lambda_risk = config.get('lambda_risk', 0.0)
        self.lambda_dev = config.get('lambda_dev', 10.0)
        self.dev_max = config.get('dev_max_mw', 50.0)
        self.risk_alpha = config.get('risk_alpha', 0.1)
        self.time_limit = config.get('solver_time_limit', 600)
    
    def solve(self, daily_scenarios: List[Dict], n_days: int = 7) -> Dict:
        """
        daily_scenarios: list of n_days dicts, each from ScenarioLoader.get_day_scenarios()
            Keys: 'dam' (S,24), 'rtm' (S,24)
        Returns: {'dam_schedule': list(24), 'status': str, ...}
        """
        S = daily_scenarios[0]['dam'].shape[0]
        T = n_days * 24
        prob = pulp.LpProblem("MultiDay_BESS", pulp.LpMaximize)
        
        # --- DAM variables (non-anticipative PER DAY) ---
        x_c, x_d = {}, {}
        for d in range(n_days):
            x_c[d] = pulp.LpVariable.dicts(f"dam_c_{d}", range(24), 0, self.params.p_max_mw)
            x_d[d] = pulp.LpVariable.dicts(f"dam_d_{d}", range(24), 0, self.params.p_max_mw)
        
        # --- RTM + deviation variables (scenario-specific, per day) ---
        y_c, y_d = {}, {}
        dev_pos, dev_neg = {}, {}
        for s in range(S):
            y_c[s], y_d[s] = {}, {}
            dev_pos[s], dev_neg[s] = {}, {}
            for d in range(n_days):
                y_c[s][d] = pulp.LpVariable.dicts(f"rtm_c_{s}_{d}", range(24), 0, self.params.p_max_mw)
                y_d[s][d] = pulp.LpVariable.dicts(f"rtm_d_{s}_{d}", range(24), 0, self.params.p_max_mw)
                dev_pos[s][d] = pulp.LpVariable.dicts(f"dp_{s}_{d}", range(24), lowBound=0)
                dev_neg[s][d] = pulp.LpVariable.dicts(f"dn_{s}_{d}", range(24), lowBound=0)
        
        # --- SoC: continuous across ALL days ---
        soc = {}
        for s in range(S):
            soc[s] = pulp.LpVariable.dicts(f"soc_{s}", range(T + 1),
                                            self.params.e_min_mwh, self.params.e_max_mwh)
        
        # --- CVaR variables (IDENTICAL to TwoStageBESS) ---
        zeta = pulp.LpVariable("zeta")
        u = pulp.LpVariable.dicts("u", range(S), lowBound=0)
        
        # --- Build scenario revenues ---
        scenario_revenues = []
        for s in range(S):
            prob += soc[s][0] == self.params.soc_initial_mwh
            rev = 0
            
            for d in range(n_days):
                dam_prices = daily_scenarios[d]['dam']
                rtm_prices = daily_scenarios[d]['rtm']
                
                for h in range(24):
                    t = d * 24 + h
                    
                    # SoC dynamics (IDENTICAL formula)
                    prob += soc[s][t+1] == (soc[s][t]
                        + self.params.eta_charge * y_c[s][d][h]
                        - (1.0 / self.params.eta_discharge) * y_d[s][d][h])
                    
                    # Revenue (IDENTICAL IEX settlement)
                    rev += dam_prices[s, h] * (x_d[d][h] - x_c[d][h])
                    rev += rtm_prices[s, h] * (
                        (y_d[s][d][h] - y_c[s][d][h]) - (x_d[d][h] - x_c[d][h])
                    )
                    
                    # In-LP costs (ALL THREE — matching TwoStageBESS)
                    rev -= self.params.iex_fee_rs_mwh * (y_c[s][d][h] + y_d[s][d][h])
                    rev -= self.params.degradation_cost_rs_mwh * y_d[s][d][h]
                    rev -= 135.0 * (y_c[s][d][h] + y_d[s][d][h])  # DSM friction
                    
                    # Deviation constraints (IDENTICAL to TwoStageBESS)
                    prob += ((y_d[s][d][h] - y_c[s][d][h]) - (x_d[d][h] - x_c[d][h])
                             == dev_pos[s][d][h] - dev_neg[s][d][h])
                    prob += dev_pos[s][d][h] + dev_neg[s][d][h] <= self.dev_max
            
            # Terminal SoC: ONLY at end of horizon (not between days)
            prob += soc[s][T] >= self.params.soc_terminal_min_mwh
            
            # Optional per-day cycling constraint
            if self.params.max_cycles_per_day is not None:
                usable_energy = self.params.e_max_mwh - self.params.e_min_mwh
                for d in range(n_days):
                    d_discharge = pulp.lpSum([y_d[s][d][h] for h in range(24)])
                    prob += d_discharge <= self.params.max_cycles_per_day * usable_energy
            
            # CVaR shortfall
            prob += u[s] >= zeta - rev
            scenario_revenues.append(rev)
        
        # --- Objective (IDENTICAL structure to TwoStageBESS) ---
        avg_revenue = pulp.lpSum(scenario_revenues) / S
        cvar_expr = zeta - (1.0 / (S * self.risk_alpha)) * pulp.lpSum(
            [u[s] for s in range(S)])
        stability_penalty = (self.lambda_dev / S) * pulp.lpSum([
            dev_pos[s][d][h] + dev_neg[s][d][h]
            for s in range(S) for d in range(n_days) for h in range(24)
        ])
        
        prob.setObjective(avg_revenue + self.lambda_risk * cvar_expr - stability_penalty)
        
        # --- Solve ---
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit)
        
        if self.solver_name == 'HiGHS':
            try:
                solver = pulp.HiGHS_CMD(msg=0, timeLimit=self.time_limit)
            except Exception:
                solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit)
        
        status = prob.solve(solver)
        
        if pulp.LpStatus[status] != 'Optimal':
            return {"status": pulp.LpStatus[status]}
        
        # --- Extract Day 1 DAM schedule ONLY ---
        return {
            "status": "Optimal",
            "dam_schedule": [pulp.value(x_d[0][h] - x_c[0][h]) for h in range(24)],
            "expected_revenue": pulp.value(avg_revenue),
            "cvar_zeta_rs": pulp.value(zeta),
            "cvar_value_rs": pulp.value(cvar_expr),
            "total_horizon_revenue": pulp.value(prob.objective),
            "terminal_socs": {s: pulp.value(soc[s][T]) for s in range(S)},
            "overnight_socs": {
                d: {s: pulp.value(soc[s][d * 24]) for s in range(S)}
                for d in range(1, n_days)
            },
            "scenarios": [
                {
                    "id": s,
                    "rtm_dispatch": [pulp.value(y_d[s][0][h] - y_c[s][0][h]) for h in range(24)],
                    "soc": [pulp.value(soc[s][t]) for t in range(25)],
                    "revenue": pulp.value(scenario_revenues[s])
                }
                for s in range(S)
            ]
        }
