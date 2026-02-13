import pulp
import numpy as np
from typing import Dict, List, Optional
from src.optimizer.bess_params import BESSParams

class TwoStageBESS:
    """
    Two-Stage Stochastic BESS Optimizer.
    Stage 1: DAM commitment (x_t) - non-anticipative.
    Stage 2: RTM recourse (y_{s,t}) - scenario dependent.
    """
    def __init__(self, params: BESSParams, config: Dict):
        self.params = params
        self.config = config
        self.solver_name = config.get('solver', 'HiGHS')
        self.lambda_risk = config.get('lambda_risk', 0.0)
        self.lambda_dev = config.get('lambda_dev', 0.0)
        self.dev_max = config.get('dev_max_mw', 50.0)
        self.risk_alpha = config.get('risk_alpha', 0.1)

    def solve(self, dam_scenarios: np.ndarray, rtm_scenarios: np.ndarray) -> Dict:
        """
        Build and solve the two-stage stochastic program with realistic costs.
        """
        n_scenarios = dam_scenarios.shape[0]
        prob = pulp.LpProblem("CostAware_BESS_Optimization", pulp.LpMaximize)

        # --- Decision Variables ---
        
        # Stage 1: DAM schedule (split to apply fees)
        x_c = pulp.LpVariable.dicts("dam_charge", range(24), 0, self.params.p_max_mw)
        x_d = pulp.LpVariable.dicts("dam_discharge", range(24), 0, self.params.p_max_mw)
        
        # Helper: x_net = x_d - x_c
        
        # Stage 2: RTM physical dispatch 
        y_c = pulp.LpVariable.dicts("rtm_charge", (range(n_scenarios), range(24)), 
                                    0, self.params.p_max_mw)
        y_d = pulp.LpVariable.dicts("rtm_discharge", (range(n_scenarios), range(24)), 
                                    0, self.params.p_max_mw)
        
        # SOC variables
        soc = pulp.LpVariable.dicts("soc", (range(n_scenarios), range(25)), 
                                    self.params.e_min_mwh, self.params.e_max_mwh)

        # Deviation auxiliary variables: y_net - x_net = dev_pos - dev_neg
        dev_pos = pulp.LpVariable.dicts("dev_pos", (range(n_scenarios), range(24)), lowBound=0)
        dev_neg = pulp.LpVariable.dicts("dev_neg", (range(n_scenarios), range(24)), lowBound=0)

        # CVaR Support: Rockafellar-Uryasev Linearization
        # zeta = VaR threshold (unbounded)
        # u[s] = shortfall for scenario s (non-negative)
        zeta = pulp.LpVariable("zeta") # Default is unbounded unless lowBound/upBound set
        u = pulp.LpVariable.dicts("u", range(n_scenarios), lowBound=0)

        # --- Objective Function ---
        
        # Realized Revenue R_s = sum_t [ p_dam * (x_d - x_c) + p_rtm * ((y_d - y_c) - (x_d - x_c)) ]
        # Realized Fees F_s = f_iex * sum_t [ (x_d + x_c) + (dev_pos + dev_neg) ]
        # Degradation D_s = f_deg * sum_t [ y_d ]
        
        scenario_revenues_expr = []
        for s in range(n_scenarios):
            # Gross Arbitrage (Deviation-based)
            arbitrage = pulp.lpSum([
                dam_scenarios[s, t] * (x_d[t] - x_c[t]) + 
                rtm_scenarios[s, t] * ((y_d[s][t] - y_c[s][t]) - (x_d[t] - x_c[t]))
                for t in range(24)
            ])
            
            # Transaction Fees (Stage 1 + Stage 2 deviations) - Market Churn Model
            # This matches the Phase 3D optimizer objective exactly.
            fees = self.params.iex_fee_rs_mwh * pulp.lpSum([
                (x_d[t] + x_c[t]) + (dev_pos[s][t] + dev_neg[s][t])
                for t in range(24)
            ])
            
            # Physical Degradation
            degradation = self.params.degradation_cost_rs_mwh * pulp.lpSum([y_d[s][t] for t in range(24)])
            
            scenario_revenues_expr.append(arbitrage - fees - degradation)

        avg_revenue = pulp.lpSum(scenario_revenues_expr) / n_scenarios
        
        # CVaR = zeta - (1/(S * alpha)) * sum(u_s)
        cvar_expr = zeta - (1.0 / (n_scenarios * self.risk_alpha)) * pulp.lpSum([u[s] for s in range(n_scenarios)])
        
        # Secondary Penalty (for numerical stability only)
        # Small penalty on deviations to prefer DAM over RTM for the same spread
        stability_penalty = (self.lambda_dev / n_scenarios) * pulp.lpSum([
            dev_pos[s][t] + dev_neg[s][t] for s in range(n_scenarios) for t in range(24)
        ])

        # Composite Objective: Maximize E[R] + lambda * CVaR
        objective = avg_revenue + self.lambda_risk * cvar_expr - stability_penalty
        prob.setObjective(objective)

        # --- Constraints ---
        
        for s in range(n_scenarios):
            # 1. CVaR Shortfall Constraint: u[s] >= zeta - R_s
            prob += u[s] >= zeta - scenario_revenues_expr[s]
            
            # 2. SOC Initial
            prob += soc[s][0] == self.params.soc_initial_mwh
            
            total_discharge = 0
            for t in range(24):
                # Dynamics
                prob += soc[s][t+1] == soc[s][t] + self.params.eta_charge * y_c[s][t] - (1.0 / self.params.eta_discharge) * y_d[s][t]
                
                # Deviation: (y_d - y_c) - (x_d - x_c) = dev_pos - dev_neg
                prob += (y_d[s][t] - y_c[s][t]) - (x_d[t] - x_c[t]) == dev_pos[s][t] - dev_neg[s][t]
                
                # Bounds
                prob += dev_pos[s][t] + dev_neg[s][t] <= self.dev_max
                
                total_discharge += y_d[s][t]
                
            # Terminal
            prob += soc[s][24] >= self.params.soc_terminal_min_mwh
            
            # Optional Cycling Constraint
            if self.params.max_cycles_per_day is not None:
                usable_energy = self.params.e_max_mwh - self.params.e_min_mwh
                prob += total_discharge <= self.params.max_cycles_per_day * usable_energy

        # --- Solve ---
        # Default to CBC for maximum compatibility in this environment
        solver = pulp.PULP_CBC_CMD(msg=0)
        
        if self.solver_name == 'HiGHS':
            try:
                solver = pulp.HiGHS_CMD(msg=0)
                print("Using HiGHS solver...")
            except Exception as e:
                print(f"HiGHS solver initialization failed: {e}. Falling back to CBC.")
                solver = pulp.PULP_CBC_CMD(msg=0)
        else:
            print("Using CBC solver...")
            
        status = prob.solve(solver)
        
        if pulp.LpStatus[status] != 'Optimal':
            return {"status": pulp.LpStatus[status]}

        # --- Extract Results ---
        scen_rev_vals = [pulp.value(expr) for expr in scenario_revenues_expr]
        
        results = {
            "status": "Optimal",
            "expected_revenue": pulp.value(avg_revenue),
            "cvar_zeta_rs": pulp.value(zeta),
            "cvar_value_rs": pulp.value(cvar_expr),
            "dam_schedule": [pulp.value(x_d[t] - x_c[t]) for t in range(24)],
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
