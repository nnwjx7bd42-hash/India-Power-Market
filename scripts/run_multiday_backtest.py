"""
Phase 5.3: 7-Day Extensive Form BESS Backtest

Solves a 7-day (n_lookahead_days) stochastic program per day, commits Day 1's
DAM schedule, evaluates against actuals, and chains SoC across day boundaries.

Expected solve time: 1-5 min/day (CBC), ~12-48 hours for full 143-day backtest.
"""
import pandas as pd
import numpy as np
import yaml
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.optimizer.bess_params import BESSParams
from src.optimizer.scenario_loader import ScenarioLoader
from src.optimizer.multiday_bess import MultiDayBESS
from src.optimizer.costs import CostModel
from scripts.run_phase3b_backtest import evaluate_actuals


def run_multiday_backtest(args):
    print("============================================================")
    print("PHASE 5.3: 7-DAY EXTENSIVE FORM BESS BACKTEST")
    print("============================================================")
    
    bess_params = BESSParams.from_yaml("config/bess.yaml")
    with open("config/phase4_multiday.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    n_lookahead = config.get('n_lookahead_days', 7)
    
    loader = ScenarioLoader(
        dam_path=config['paths']['scenarios_dam'],
        rtm_path=config['paths']['scenarios_rtm'],
        actuals_dam_path=config['paths']['actuals_dam'],
        actuals_rtm_path=config['paths']['actuals_rtm']
    )
    
    optimizer = MultiDayBESS(bess_params, config)
    
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
    
    print(f"Running {n_lookahead}-day extensive form backtest for {len(dates)} days...")
    print(f"Scenarios per day: {config['n_scenarios']}, Solver time limit: {config.get('solver_time_limit', 600)}s")
    
    backtest_results = []
    prev_soc = bess_params.soc_initial_mwh
    solve_times = []
    
    for i, date in enumerate(dates):
        # --- Multi-day SoC: carry forward from previous day ---
        bess_params.soc_initial_mwh = prev_soc
        
        # Determine lookahead window (shrinks near end of backtest)
        effective_lookahead = min(n_lookahead, len(dates) - i)
        
        print(f"[{i+1}/{len(dates)}] {date} (SoC₀={prev_soc:.1f}, look={effective_lookahead}d)...", 
              end=" ", flush=True)
        
        # Load scenarios for all lookahead days
        daily_scenarios = []
        for d in range(effective_lookahead):
            day_idx = i + d
            day_data = loader.get_day_scenarios(dates[day_idx], n_scenarios=config['n_scenarios'])
            daily_scenarios.append(day_data)
        
        # Solve multi-day problem
        t0 = time.time()
        res = optimizer.solve(daily_scenarios, n_days=effective_lookahead)
        solve_time = time.time() - t0
        solve_times.append(solve_time)
        
        if res['status'] != 'Optimal':
            print(f"FAILED (Status: {res['status']}, {solve_time:.1f}s)")
            continue
        
        # Evaluate Day 1 against actuals
        # Force hard terminal for evaluation — the multi-day optimizer handles
        # overnight value internally; evaluation should enforce SoC floor.
        saved_mode = bess_params.soc_terminal_mode
        bess_params.soc_terminal_mode = "hard"
        eval_res = evaluate_actuals(
            bess_params,
            res['dam_schedule'],
            daily_scenarios[0]['dam_actual'],
            daily_scenarios[0]['rtm_actual'],
            cost_model=cost_model,
            lambda_dev=config['lambda_dev']
        )
        bess_params.soc_terminal_mode = saved_mode
        
        if eval_res is None:
            print("EVAL FAILED")
            continue
        
        realized_rev = eval_res['revenue']
        net_rev = eval_res['net_revenue']
        expected_rev = res['expected_revenue']
        
        print(f"Net: ₹{net_rev:,.0f} | SoC₂₄: {eval_res['soc'][-1]:.1f} | {solve_time:.1f}s")
        
        # Save daily detailed results
        daily_output = {
            "date": date,
            "status": res['status'],
            "expected_revenue": expected_rev,
            "realized_revenue": realized_rev,
            "net_revenue": net_rev,
            "soc_initial": prev_soc,
            "soc_terminal": eval_res['soc'][-1],
            "lookahead_days": effective_lookahead,
            "solve_time_s": solve_time,
            "dam_schedule": res['dam_schedule'],
            "rtm_realized_schedule": eval_res['rtm_schedule'],
            "soc_realized": eval_res['soc'],
            "actual_dam_prices": daily_scenarios[0]['dam_actual'].tolist(),
            "actual_rtm_prices": daily_scenarios[0]['rtm_actual'].tolist(),
            "scenarios": res['scenarios']
        }
        
        # Carry terminal SoC forward
        prev_soc = eval_res['soc'][-1]
        
        with open(daily_results_dir / f"result_{date}.json", 'w') as f:
            json.dump(daily_output, f, indent=2,
                      default=lambda x: float(x) if isinstance(x, np.generic) else x)
        
        backtest_results.append({
            "target_date": date,
            "expected_revenue": expected_rev,
            "realized_revenue": realized_rev,
            "net_revenue": net_rev,
            "soc_initial": daily_output['soc_initial'],
            "soc_terminal": daily_output['soc_terminal'],
            "solve_time_s": solve_time
        })
    
    # Save Summary
    results_df = pd.DataFrame(backtest_results)
    results_df.to_csv(results_dir / "backtest_results.csv", index=False)
    
    summary = {
        "n_days": len(results_df),
        "n_lookahead_days": n_lookahead,
        "n_scenarios": config['n_scenarios'],
        "total_expected_revenue": results_df['expected_revenue'].sum(),
        "total_realized_revenue": results_df['realized_revenue'].sum(),
        "total_net_revenue": results_df['net_revenue'].sum(),
        "avg_daily_net": results_df['net_revenue'].mean(),
        "std_daily_net": results_df['net_revenue'].std(),
        "min_daily_net": results_df['net_revenue'].min(),
        "max_daily_net": results_df['net_revenue'].max(),
        "avg_terminal_soc": results_df['soc_terminal'].mean(),
        "min_terminal_soc": results_df['soc_terminal'].min(),
        "max_terminal_soc": results_df['soc_terminal'].max(),
        "avg_solve_time_s": np.mean(solve_times),
        "max_solve_time_s": np.max(solve_times),
        "total_compute_time_h": np.sum(solve_times) / 3600
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n============================================================")
    print(f"{n_lookahead}-DAY EXTENSIVE FORM BACKTEST COMPLETE")
    print(f"Total Realized Revenue: ₹{summary['total_realized_revenue']:,.2f}")
    print(f"Total Net Revenue:      ₹{summary['total_net_revenue']:,.2f}")
    print(f"Avg Daily Net:          ₹{summary['avg_daily_net']:,.2f}")
    print(f"Worst Day Net:          ₹{summary['min_daily_net']:,.2f}")
    print(f"SoC Range: [{summary['min_terminal_soc']:.1f}, {summary['max_terminal_soc']:.1f}] MWh")
    print(f"Avg Solve Time:         {summary['avg_solve_time_s']:.1f}s")
    print(f"Total Compute Time:     {summary['total_compute_time_h']:.1f}h")
    print(f"Summary saved to {results_dir / 'summary.json'}")
    print("============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=str, help='Run for a specific date (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, help='Limit number of days to run')
    args = parser.parse_args()
    run_multiday_backtest(args)
