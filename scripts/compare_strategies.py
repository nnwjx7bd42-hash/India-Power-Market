import pandas as pd
import numpy as np
import yaml
import json
import sys
from pathlib import Path
import argparse
import pulp

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.optimizer.bess_params import BESSParams
from src.optimizer.scenario_loader import ScenarioLoader
from src.optimizer.costs import CostModel
from scripts.run_phase3b_backtest import evaluate_actuals

def solve_deterministic(bess_params, dam_scenarios, rtm_scenarios, lambda_dev=0.0):
    """
    Solve for q50 deterministic forecast.
    """
    # Simply take the mean of scenarios
    dam_mean = np.mean(dam_scenarios, axis=0) # [24]
    rtm_mean = np.mean(rtm_scenarios, axis=0) # [h, scenario] -> [24]
    
    # Simple LP for single trajectory
    prob = pulp.LpProblem("Deterministic_BESS", pulp.LpMaximize)
    
    x = pulp.LpVariable.dicts("dam_dispatch", range(24), -bess_params.p_max_mw, bess_params.p_max_mw)
    y = pulp.LpVariable.dicts("rtm_dispatch", range(24), -bess_params.p_max_mw, bess_params.p_max_mw)
    y_c = pulp.LpVariable.dicts("y_c", range(24), 0, bess_params.p_max_mw)
    y_d = pulp.LpVariable.dicts("y_d", range(24), 0, bess_params.p_max_mw)
    soc = pulp.LpVariable.dicts("soc", range(25), bess_params.e_min_mwh, bess_params.e_max_mwh)
    dev = pulp.LpVariable.dicts("dev", range(24), lowBound=0)

    prob += soc[0] == bess_params.soc_initial_mwh
    for t in range(24):
        prob += y[t] == y_d[t] - y_c[t]
        prob += soc[t+1] == soc[t] + (y_c[t] * bess_params.eta_charge) - (y_d[t] / bess_params.eta_discharge)
        prob += dev[t] >= y[t] - x[t]
        prob += dev[t] >= x[t] - y[t]

    prob += soc[24] >= bess_params.soc_terminal_min_mwh
    
    revenue = pulp.lpSum([dam_mean[t] * x[t] + rtm_mean[t] * (y[t] - x[t]) for t in range(24)])
    fees = bess_params.iex_fee_rs_mwh * pulp.lpSum([y_c[t] + y_d[t] for t in range(24)])
    degradation = bess_params.degradation_cost_rs_mwh * pulp.lpSum([y_d[t] for t in range(24)])
    dsm_friction = 135.0 * pulp.lpSum([y_c[t] + y_d[t] for t in range(24)])
    prob.objective = revenue - fees - degradation - dsm_friction - lambda_dev * pulp.lpSum([dev[t] for t in range(24)])
    
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)
    
    return [pulp.value(x[t]) for t in range(24)]

def solve_perfect_foresight(bess_params, dam_actual, rtm_actual, lambda_dev=0.0):
    """
    The Oracle: Knows exact price for the day.
    """
    prob = pulp.LpProblem("Perfect_Foresight", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("dam_dispatch", range(24), -bess_params.p_max_mw, bess_params.p_max_mw)
    y = pulp.LpVariable.dicts("rtm_dispatch", range(24), -bess_params.p_max_mw, bess_params.p_max_mw)
    y_c = pulp.LpVariable.dicts("y_c", range(24), 0, bess_params.p_max_mw)
    y_d = pulp.LpVariable.dicts("y_d", range(24), 0, bess_params.p_max_mw)
    soc = pulp.LpVariable.dicts("soc", range(25), bess_params.e_min_mwh, bess_params.e_max_mwh)
    dev = pulp.LpVariable.dicts("dev", range(24), lowBound=0)

    prob += soc[0] == bess_params.soc_initial_mwh
    for t in range(24):
        prob += y[t] == y_d[t] - y_c[t]
        prob += soc[t+1] == soc[t] + (y_c[t] * bess_params.eta_charge) - (y_d[t] / bess_params.eta_discharge)
        prob += dev[t] >= y[t] - x[t]
        prob += dev[t] >= x[t] - y[t]

    prob += soc[24] >= bess_params.soc_terminal_min_mwh
    
    revenue = pulp.lpSum([dam_actual[t] * x[t] + rtm_actual[t] * (y[t] - x[t]) for t in range(24)])
    fees = bess_params.iex_fee_rs_mwh * pulp.lpSum([y_c[t] + y_d[t] for t in range(24)])
    degradation = bess_params.degradation_cost_rs_mwh * pulp.lpSum([y_d[t] for t in range(24)])
    dsm_friction = 135.0 * pulp.lpSum([y_c[t] + y_d[t] for t in range(24)])
    prob.objective = revenue - fees - degradation - dsm_friction - lambda_dev * pulp.lpSum([dev[t] for t in range(24)])
    
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)
    
    return [pulp.value(x[t]) for t in range(24)]

def run_benchmarks():
    print("============================================================")
    print("PHASE 3B: STRATEGY COMPARISON (BENCHMARKS)")
    print("============================================================")
    
    bess_params = BESSParams.from_yaml("config/bess.yaml")
    with open("config/phase3b.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    loader = ScenarioLoader(
        dam_path=config['paths']['scenarios_dam'],
        rtm_path=config['paths']['scenarios_rtm'],
        actuals_dam_path=config['paths']['actuals_dam'],
        actuals_rtm_path=config['paths']['actuals_rtm']
    )
    
    # Load Cost Model
    cost_model = None
    if Path("config/costs_config.yaml").exists():
        cost_model = CostModel.from_yaml("config/costs_config.yaml")
        print("Loaded CostModel from config/costs_config.yaml")
    
    dates = loader.common_dates
    print(f"Comparing benchmarks for {len(dates)} days...")
    
    bench_results = []
    
    for i, date in enumerate(dates):
        print(f"[{i+1}/{len(dates)}] {date}...", end=" ", flush=True)
        
        day_data = loader.get_day_scenarios(date, n_scenarios=config['n_scenarios'])
        
        # 1. Deterministic Commit
        x_det = solve_deterministic(bess_params, day_data['dam'], day_data['rtm'], lambda_dev=config['lambda_dev'])
        eval_det = evaluate_actuals(bess_params, x_det, day_data['dam_actual'], day_data['rtm_actual'], cost_model=cost_model, lambda_dev=config['lambda_dev'])
        
        # 2. Perfect Foresight (Upper Bound)
        x_oracle = solve_perfect_foresight(bess_params, day_data['dam_actual'], day_data['rtm_actual'], lambda_dev=config['lambda_dev'])
        eval_oracle = evaluate_actuals(bess_params, x_oracle, day_data['dam_actual'], day_data['rtm_actual'], cost_model=cost_model, lambda_dev=config['lambda_dev'])
        
        # 3. Naive (Zero DAM Commitment, purely RTM)
        x_naive = [0.0] * 24
        eval_naive = evaluate_actuals(bess_params, x_naive, day_data['dam_actual'], day_data['rtm_actual'], cost_model=cost_model, lambda_dev=config['lambda_dev'])
        
        net_rev_det = eval_det['net_revenue']
        net_rev_oracle = eval_oracle['net_revenue']
        net_rev_naive = eval_naive['net_revenue']
        
        print(f"Naive: ₹{net_rev_naive:,.0f} | Det: ₹{net_rev_det:,.0f} | Oracle: ₹{net_rev_oracle:,.0f}")
        
        bench_results.append({
            "target_date": date,
            "realized_revenue_naive": eval_naive['revenue'],
            "realized_revenue_det": eval_det['revenue'],
            "realized_revenue_oracle": eval_oracle['revenue'],
            "net_revenue_naive": net_rev_naive,
            "net_revenue_det": net_rev_det,
            "net_revenue_oracle": net_rev_oracle
        })
        
    df = pd.DataFrame(bench_results)
    df.to_csv(Path(config['paths']['results_dir']) / "benchmark_results.csv", index=False)
    
    summary = df.mean(numeric_only=True).to_dict()
    print("\nBenchmark Avg Daily Realized:")
    print(df.mean(numeric_only=True))

if __name__ == "__main__":
    run_benchmarks()
