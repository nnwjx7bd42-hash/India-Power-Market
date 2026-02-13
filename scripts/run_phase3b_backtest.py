import pandas as pd
import numpy as np
import yaml
import json
import sys
from pathlib import Path
from datetime import datetime
import argparse
import pulp

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.optimizer.bess_params import BESSParams
from src.optimizer.scenario_loader import ScenarioLoader
from src.optimizer.two_stage_bess import TwoStageBESS

def evaluate_actuals(bess_params, dam_schedule, dam_actual, rtm_actual, lambda_dev=0.0):
    """
    Rigorous evaluation: Given Stage 1 DAM schedule, solve for optimal RTM recourse 
    using ACTUAL realized prices.
    """
    prob = pulp.LpProblem("RTM_Recourse_Actuals", pulp.LpMaximize)
    
    # Variables
    y = pulp.LpVariable.dicts("rtm_dispatch", range(24), 
                              lowBound=-bess_params.p_max_mw, 
                              upBound=bess_params.p_max_mw)
    y_c = pulp.LpVariable.dicts("y_c", range(24), lowBound=0, upBound=bess_params.p_max_mw)
    y_d = pulp.LpVariable.dicts("y_d", range(24), lowBound=0, upBound=bess_params.p_max_mw)
    
    soc = pulp.LpVariable.dicts("soc", range(25), 
                                lowBound=bess_params.e_min_mwh, 
                                upBound=bess_params.e_max_mwh)
    
    # SoC Dynamics and Terminal Constraint
    prob += soc[0] == bess_params.soc_initial_mwh
    for t in range(24):
        prob += y[t] == y_d[t] - y_c[t]
        prob += soc[t+1] == soc[t] + (y_c[t] * bess_params.eta_charge) - (y_d[t] / bess_params.eta_discharge)
        
    prob += soc[24] >= bess_params.soc_terminal_min_mwh
    
    # Revenue Calculation: R = sum( p_dam * x + p_rtm * (y - x) )
    revenue = pulp.lpSum([(dam_actual[t] - rtm_actual[t]) * dam_schedule[t] + rtm_actual[t] * y[t] for t in range(24)])
    
    # Costs
    # Fees apply to total throughput in RTM (including DAM adjustment)
    # y = physical dispatch. dev = |y - x|
    dev = pulp.LpVariable.dicts("dev", range(24), lowBound=0)
    for t in range(24):
        prob += dev[t] >= y[t] - dam_schedule[t]
        prob += dev[t] >= dam_schedule[t] - y[t]
    
    # Stage 1 Fees (sunk) + Stage 2 Fees
    # Reverting to physical-flow fee to match Phase 3D baseline
    fees = bess_params.iex_fee_rs_mwh * pulp.lpSum([y_c[t] + y_d[t] for t in range(24)])
    degradation = bess_params.degradation_cost_rs_mwh * pulp.lpSum([y_d[t] for t in range(24)])
    
    prob.objective = revenue - fees - degradation - lambda_dev * pulp.lpSum([dev[t] for t in range(24)])
    
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        return None
        
    return {
        "revenue": pulp.value(revenue),
        "net_revenue": pulp.value(revenue - fees - degradation),
        "rtm_schedule": [pulp.value(y[t]) for t in range(24)],
        "soc": [pulp.value(soc[t]) for t in range(25)]
    }

def run_backtest(args):
    print("============================================================")
    print("PHASE 3B: TWO-STAGE STOCHASTIC BESS BACKTEST")
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
    
    optimizer = TwoStageBESS(bess_params, config)
    
    results_dir = Path(config['paths']['results_dir'])
    daily_results_dir = results_dir / "daily"
    daily_results_dir.mkdir(parents=True, exist_ok=True)
    
    dates = loader.common_dates
    if args.day:
        dates = [args.day]
    elif args.limit:
        dates = dates[:args.limit]
        
    print(f"Running backtest for {len(dates)} days...")
    
    backtest_results = []
    
    for i, date in enumerate(dates):
        print(f"[{i+1}/{len(dates)}] {date}...", end=" ", flush=True)
        
        day_data = loader.get_day_scenarios(date, n_scenarios=config['n_scenarios'])
        
        # Solve Two-Stage Stochastic
        res = optimizer.solve(day_data['dam'], day_data['rtm'])
        
        if res['status'] != 'Optimal':
            print(f"FAILED (Status: {res['status']})")
            continue
            
        # Evaluate against actuals
        eval_res = evaluate_actuals(
            bess_params, 
            res['dam_schedule'], 
            day_data['dam_actual'], 
            day_data['rtm_actual'],
            lambda_dev=config['lambda_dev']
        )
        
        if eval_res is None:
            print("EVAL FAILED")
            continue
            
        realized_rev = eval_res['revenue']
        expected_rev = res['expected_revenue']
        
        print(f"Exp: ₹{expected_rev:,.0f} | Realized: ₹{realized_rev:,.0f}")
        
        # Save daily detailed results
        daily_output = {
            "date": date,
            "status": res['status'],
            "expected_revenue": expected_rev,
            "realized_revenue": realized_rev,
            "dam_schedule": res['dam_schedule'],
            "rtm_realized_schedule": eval_res['rtm_schedule'],
            "soc_realized": eval_res['soc'],
            "scenarios": res['scenarios']
        }
        
        with open(daily_results_dir / f"result_{date}.json", 'w') as f:
            json.dump(daily_output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.generic) else x)
            
        backtest_results.append({
            "target_date": date,
            "expected_revenue": expected_rev,
            "realized_revenue": realized_rev
        })
        
    # Save Summary
    results_df = pd.DataFrame(backtest_results)
    results_df.to_csv(results_dir / "backtest_results.csv", index=False)
    
    summary = {
        "n_days": len(results_df),
        "total_expected_revenue": results_df['expected_revenue'].sum(),
        "total_realized_revenue": results_df['realized_revenue'].sum(),
        "avg_daily_realized": results_df['realized_revenue'].mean(),
        "std_daily_realized": results_df['realized_revenue'].std(),
        "min_daily_realized": results_df['realized_revenue'].min()
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("\n============================================================")
    print("BACKTEST COMPLETE")
    print(f"Total Realized Revenue: ₹{summary['total_realized_revenue']:,.2f}")
    print(f"Avg Daily Realized: ₹{summary['avg_daily_realized']:,.2f}")
    print(f"Summary saved to {results_dir / 'summary.json'}")
    print("============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=str, help='Run for a specific date (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, help='Limit number of days to run')
    args = parser.parse_args()
    run_backtest(args)
