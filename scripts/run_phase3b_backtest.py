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
from src.optimizer.costs import CostModel

def evaluate_actuals(bess_params, dam_schedule, dam_actual, rtm_actual, cost_model: CostModel = None, lambda_dev=0.0, force_terminal_mode=None):
    """
    Rigorous evaluation: Given Stage 1 DAM schedule, solve for optimal RTM recourse 
    using ACTUAL realized prices.
    
    Args:
        force_terminal_mode: Override terminal constraint. Options:
            None: use bess_params.soc_terminal_mode as-is
            'hard': soc[24] >= soc_terminal_min_mwh (100 MWh)
            'physical': soc[24] >= e_min_mwh only (20 MWh)
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
    # Terminal constraint logic
    effective_mode = force_terminal_mode if force_terminal_mode else bess_params.soc_terminal_mode
    if effective_mode == "hard":
        prob += soc[24] >= bess_params.soc_terminal_min_mwh
    elif effective_mode == "physical":
        pass  # physical floor already enforced via soc variable lb = e_min_mwh
    # else: "soft" — physical bounds still enforced via variable bounds
    
    # Revenue Calculation: R = sum( p_dam * x + p_rtm * (y - x) )
    revenue = pulp.lpSum([(dam_actual[t] - rtm_actual[t]) * dam_schedule[t] + rtm_actual[t] * y[t] for t in range(24)])
    
    # Costs — must mirror the planning LP in TwoStageBESS.solve()
    fees = bess_params.iex_fee_rs_mwh * pulp.lpSum([y_c[t] + y_d[t] for t in range(24)])
    degradation = bess_params.degradation_cost_rs_mwh * pulp.lpSum([y_d[t] for t in range(24)])
    
    # Fix #1: DSM Friction Proxy (CERC 2024) — matches planning stage
    # 3% physical error * ₹4500 Normal Rate = ₹135/MWh throughput friction
    dsm_friction = 135.0 * pulp.lpSum([y_c[t] + y_d[t] for t in range(24)])
    
    # Deviation auxiliary variables for λ penalty
    dev = pulp.LpVariable.dicts("dev", range(24), lowBound=0)
    for t in range(24):
        prob += dev[t] >= y[t] - dam_schedule[t]
        prob += dev[t] >= dam_schedule[t] - y[t]
        
    prob.objective = revenue - fees - degradation - dsm_friction - lambda_dev * pulp.lpSum([dev[t] for t in range(24)])
    
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        return None
        
    rtm_schedule = [pulp.value(y[t]) for t in range(24)]
    charge_array = np.maximum([-v for v in rtm_schedule], 0)
    discharge_array = np.maximum(rtm_schedule, 0)
    
    realized_rev = pulp.value(revenue)
    
    cost_breakdown = None
    net_revenue = realized_rev - pulp.value(fees + degradation + dsm_friction)
    
    if cost_model:
        # Fix #2: Forward actual prices for block-wise NR (not flat fallback)
        cost_breakdown = cost_model.compute_costs(
            charge=charge_array, 
            discharge=discharge_array,
            dam_actual=dam_actual,
            rtm_actual=rtm_actual
        )
        net_revenue = realized_rev - cost_breakdown['total_costs']

    return {
        "revenue": realized_rev,
        "net_revenue": net_revenue,
        "fees_breakdown": cost_breakdown,
        "rtm_schedule": rtm_schedule,
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
    
    # Load Cost Model
    cost_model = None
    if Path("config/costs_config.yaml").exists():
        cost_model = CostModel.from_yaml("config/costs_config.yaml")
        print("Loaded CostModel from config/costs_config.yaml")
    
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
    prev_soc = bess_params.soc_initial_mwh  # SoC chaining for multi-day
    
    for i, date in enumerate(dates):
        # --- Multi-day SoC: update initial SoC from previous day ---
        bess_params.soc_initial_mwh = prev_soc
        
        # --- Multi-day SoC: estimate continuation value for soft terminal ---
        if bess_params.soc_terminal_mode == "soft" and i < len(dates) - 1:
            next_data = loader.get_day_scenarios(dates[i + 1], n_scenarios=20)
            spread_per_scenario = np.max(next_data['dam'], axis=1) - np.min(next_data['dam'], axis=1)
            expected_spread = np.mean(spread_per_scenario)
            bess_params.soc_terminal_value_rs_mwh = max(0, (
                expected_spread * bess_params.eta_charge * bess_params.eta_discharge
                - bess_params.iex_fee_rs_mwh * 2   # round-trip IEX cost
                - bess_params.degradation_cost_rs_mwh  # degradation
                - 135.0 * 2  # round-trip DSM friction
            ))
        else:
            bess_params.soc_terminal_value_rs_mwh = 0.0  # last day or hard mode
        
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
            cost_model=cost_model,
            lambda_dev=config['lambda_dev']
        )
        
        if eval_res is None:
            print("EVAL FAILED")
            continue
            
        realized_rev = eval_res['revenue']
        net_rev = eval_res['net_revenue']
        expected_rev = res['expected_revenue']
        
        print(f"Exp: ₹{expected_rev:,.0f} | Realized: ₹{realized_rev:,.0f} | Net: ₹{net_rev:,.0f}")
        
        # Save daily detailed results
        daily_output = {
            "date": date,
            "status": res['status'],
            "expected_revenue": expected_rev,
            "realized_revenue": realized_rev,
            "soc_initial": prev_soc,
            "soc_terminal": eval_res['soc'][-1],
            "continuation_value": bess_params.soc_terminal_value_rs_mwh,
            "dam_schedule": res['dam_schedule'],
            "rtm_realized_schedule": eval_res['rtm_schedule'],
            "soc_realized": eval_res['soc'],
            "actual_dam_prices": day_data['dam_actual'].tolist(),
            "actual_rtm_prices": day_data['rtm_actual'].tolist(),
            "scenarios": res['scenarios']
        }
        
        # --- Multi-day SoC: carry terminal SoC forward ---
        if eval_res is not None:
            prev_soc = eval_res['soc'][-1]
        
        with open(daily_results_dir / f"result_{date}.json", 'w') as f:
            json.dump(daily_output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.generic) else x)
            
        backtest_results.append({
            "target_date": date,
            "expected_revenue": expected_rev,
            "realized_revenue": realized_rev,
            "net_revenue": net_rev
        })
        
    # Save Summary
    results_df = pd.DataFrame(backtest_results)
    results_df.to_csv(results_dir / "backtest_results.csv", index=False)
    
    summary = {
        "n_days": len(results_df),
        "total_expected_revenue": results_df['expected_revenue'].sum(),
        "total_realized_revenue": results_df['realized_revenue'].sum(),
        "total_net_revenue": results_df['net_revenue'].sum(),
        "avg_daily_realized": results_df['realized_revenue'].mean(),
        "avg_daily_net": results_df['net_revenue'].mean(),
        "std_daily_realized": results_df['realized_revenue'].std(),
        "min_daily_realized": results_df['realized_revenue'].min()
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("\n============================================================")
    print("BACKTEST COMPLETE")
    print(f"Total Realized Revenue: ₹{summary['total_realized_revenue']:,.2f}")
    print(f"Total Net Revenue:      ₹{summary['total_net_revenue']:,.2f}")
    print(f"Avg Daily Net:          ₹{summary['avg_daily_net']:,.2f}")
    print(f"Summary saved to {results_dir / 'summary.json'}")
    print("============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=str, help='Run for a specific date (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, help='Limit number of days to run')
    args = parser.parse_args()
    run_backtest(args)
