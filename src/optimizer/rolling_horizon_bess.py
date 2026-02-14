import pulp
import numpy as np
from typing import Dict, List, Optional
from src.optimizer.bess_params import BESSParams


class RollingHorizonBESS:
    """48-hour Two-Stage Stochastic BESS Optimizer.
    
    Solves a 2-day problem but only COMMITS to Day 1's DAM schedule.
    Day 2 is lookahead only — provides the value function for overnight SoC.
    
    SoC flows continuously across midnight (no terminal constraint at hour 24).
    Terminal constraint applies only at hour 48.
    
    Cost structure, CVaR, and deviation penalty mirror TwoStageBESS exactly.
    """
    
    def __init__(self, params: BESSParams, config: Dict):
        self.params = params
        self.config = config
        self.solver_name = config.get('solver', 'CBC')
        self.lambda_risk = config.get('lambda_risk', 0.0)
        self.lambda_dev = config.get('lambda_dev', 10.0)
        self.dev_max = config.get('dev_max_mw', 50.0)
        self.risk_alpha = config.get('risk_alpha', 0.1)
    
    def solve(self, dam_scenarios_d1: np.ndarray, rtm_scenarios_d1: np.ndarray,
              dam_scenarios_d2: np.ndarray, rtm_scenarios_d2: np.ndarray) -> Dict:
        """
        Build 48-hour LP.
        Day 1: x_c[t], x_d[t] for t in [0,23] — committed DAM schedule
        Day 2: x_c_look[t], x_d_look[t] for t in [0,23] — lookahead only
        RTM: y_c[s][t], y_d[s][t] for t in [0,47] — scenario-specific
        SoC: soc[s][t] for t in [0,48] — continuous across midnight
        
        Returns: Only Day 1's DAM schedule and metrics.
        """
        n_scenarios = dam_scenarios_d1.shape[0]
        T = 48  # 2 days × 24 hours
        
        prob = pulp.LpProblem("RollingHorizon_48h_BESS", pulp.LpMaximize)
        
        # --- DAM variables (non-anticipative PER DAY) ---
        # Day 1: committed
        x_c_d1 = pulp.LpVariable.dicts("dam_c_d1", range(24), 0, self.params.p_max_mw)
        x_d_d1 = pulp.LpVariable.dicts("dam_d_d1", range(24), 0, self.params.p_max_mw)
        # Day 2: lookahead (still non-anticipative across scenarios)
        x_c_d2 = pulp.LpVariable.dicts("dam_c_d2", range(24), 0, self.params.p_max_mw)
        x_d_d2 = pulp.LpVariable.dicts("dam_d_d2", range(24), 0, self.params.p_max_mw)
        
        # --- RTM / SoC / Deviation variables (scenario-specific, 48h) ---
        y_c = pulp.LpVariable.dicts("rtm_c", (range(n_scenarios), range(T)), 
                                    0, self.params.p_max_mw)
        y_d = pulp.LpVariable.dicts("rtm_d", (range(n_scenarios), range(T)),
                                    0, self.params.p_max_mw)
        
        soc = pulp.LpVariable.dicts("soc", (range(n_scenarios), range(T + 1)),
                                    self.params.e_min_mwh, self.params.e_max_mwh)
        
        dev_pos = pulp.LpVariable.dicts("dev_pos", (range(n_scenarios), range(T)), lowBound=0)
        dev_neg = pulp.LpVariable.dicts("dev_neg", (range(n_scenarios), range(T)), lowBound=0)
        
        # --- CVaR variables (IDENTICAL to TwoStageBESS) ---
        zeta = pulp.LpVariable("zeta")
        u = pulp.LpVariable.dicts("u", range(n_scenarios), lowBound=0)
        
        # --- Build scenario revenues ---
        scenario_revenues_expr = []
        
        for s in range(n_scenarios):
            # SoC Initial
            prob += soc[s][0] == self.params.soc_initial_mwh
            
            rev = 0
            total_discharge = 0
            
            for t in range(T):
                h = t % 24  # Hour within the day
                
                # Select the right DAM variables and prices per day
                if t < 24:
                    # Day 1
                    x_c_t, x_d_t = x_c_d1[h], x_d_d1[h]
                    dam_price = dam_scenarios_d1[s, h]
                    rtm_price = rtm_scenarios_d1[s, h]
                else:
                    # Day 2 (lookahead)
                    x_c_t, x_d_t = x_c_d2[h], x_d_d2[h]
                    dam_price = dam_scenarios_d2[s, h]
                    rtm_price = rtm_scenarios_d2[s, h]
                
                # SoC dynamics (IDENTICAL formula)
                prob += soc[s][t+1] == (soc[s][t]
                    + self.params.eta_charge * y_c[s][t]
                    - (1.0 / self.params.eta_discharge) * y_d[s][t])
                
                # Revenue (IDENTICAL IEX settlement)
                rev += dam_price * (x_d_t - x_c_t)
                rev += rtm_price * ((y_d[s][t] - y_c[s][t]) - (x_d_t - x_c_t))
                
                # In-LP costs (ALL THREE — matching TwoStageBESS)
                rev -= self.params.iex_fee_rs_mwh * (y_c[s][t] + y_d[s][t])
                rev -= self.params.degradation_cost_rs_mwh * y_d[s][t]
                rev -= 135.0 * (y_c[s][t] + y_d[s][t])  # DSM friction proxy
                
                # Deviation constraints (IDENTICAL to TwoStageBESS)
                prob += ((y_d[s][t] - y_c[s][t]) - (x_d_t - x_c_t)
                         == dev_pos[s][t] - dev_neg[s][t])
                prob += dev_pos[s][t] + dev_neg[s][t] <= self.dev_max
                
                total_discharge += y_d[s][t]
            
            # Terminal SoC: ONLY at end of 48h horizon (not at midnight)
            prob += soc[s][T] >= self.params.soc_terminal_min_mwh
            
            # Optional Cycling Constraint (per-day, applied to both days)
            if self.params.max_cycles_per_day is not None:
                usable_energy = self.params.e_max_mwh - self.params.e_min_mwh
                # Day 1 cycles
                d1_discharge = pulp.lpSum([y_d[s][t] for t in range(24)])
                prob += d1_discharge <= self.params.max_cycles_per_day * usable_energy
                # Day 2 cycles
                d2_discharge = pulp.lpSum([y_d[s][t] for t in range(24, T)])
                prob += d2_discharge <= self.params.max_cycles_per_day * usable_energy
            
            # CVaR shortfall
            prob += u[s] >= zeta - rev
            scenario_revenues_expr.append(rev)
        
        # --- Objective (IDENTICAL structure to TwoStageBESS) ---
        avg_revenue = pulp.lpSum(scenario_revenues_expr) / n_scenarios
        
        cvar_expr = zeta - (1.0 / (n_scenarios * self.risk_alpha)) * pulp.lpSum(
            [u[s] for s in range(n_scenarios)])
        
        stability_penalty = (self.lambda_dev / n_scenarios) * pulp.lpSum([
            dev_pos[s][t] + dev_neg[s][t]
            for s in range(n_scenarios) for t in range(T)
        ])
        
        objective = avg_revenue + self.lambda_risk * cvar_expr - stability_penalty
        prob.setObjective(objective)
        
        # --- Solve ---
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=300)
        
        if self.solver_name == 'HiGHS':
            try:
                solver = pulp.HiGHS_CMD(msg=0, timeLimit=300)
                print("Using HiGHS solver (48h)...")
            except Exception:
                solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=300)
        
        status = prob.solve(solver)
        
        if pulp.LpStatus[status] != 'Optimal':
            return {"status": pulp.LpStatus[status]}
        
        # --- Extract Day 1 DAM schedule ONLY ---
        scen_rev_vals = [pulp.value(expr) for expr in scenario_revenues_expr]
        
        results = {
            "status": "Optimal",
            "expected_revenue": pulp.value(avg_revenue),
            "cvar_zeta_rs": pulp.value(zeta),
            "cvar_value_rs": pulp.value(cvar_expr),
            "dam_schedule": [pulp.value(x_d_d1[t] - x_c_d1[t]) for t in range(24)],
            "overnight_soc": {
                s: pulp.value(soc[s][24]) for s in range(n_scenarios)
            },
            "scenarios": []
        }
        
        for s in range(n_scenarios):
            scen_res = {
                "id": s,
                "rtm_dispatch": [pulp.value(y_d[s][t] - y_c[s][t]) for t in range(24)],
                "soc": [pulp.value(soc[s][t]) for t in range(25)],
                "revenue": scen_rev_vals[s]
            }
            results["scenarios"].append(scen_res)
        
        return results
