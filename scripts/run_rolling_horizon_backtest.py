"""
Phase 5.2: 48-Hour Rolling Horizon BESS Backtest

Solves a 48-hour stochastic program per day, commits Day 1's DAM schedule,
evaluates against actuals, and chains SoC across day boundaries.
"""
import pandas as pd
import numpy as np
import yaml
import json
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.optimizer.bess_params import BESSParams
from src.optimizer.scenario_loader import ScenarioLoader
from src.optimizer.rolling_horizon_bess import RollingHorizonBESS
from src.optimizer.costs import CostModel
from scripts.run_phase3b_backtest import evaluate_actuals


def run_rolling_backtest(args):
    print("============================================================")
    print("PHASE 5.2: 48-HOUR ROLLING HORIZON BESS BACKTEST")
    print("============================================================")
    
    bess_params = BESSParams.from_yaml("config/bess.yaml")
    config_path = args.config if hasattr(args, 'config') and args.config else "config/phase4_rolling.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Config-driven eval terminal mode: option_a → hard, option_b → physical
    eval_mode_map = {'option_a': 'hard', 'option_b': 'physical'}
    eval_terminal_mode = eval_mode_map.get(config.get('eval_terminal_mode', 'option_a'), 'hard')
    print(f"Eval terminal mode: {config.get('eval_terminal_mode', 'option_a')} → force_terminal_mode={eval_terminal_mode}")
    
    
    loader = ScenarioLoader(
        dam_path=config['paths']['scenarios_dam'],
        rtm_path=config['paths']['scenarios_rtm'],
        actuals_dam_path=config['paths']['actuals_dam'],
        actuals_rtm_path=config['paths']['actuals_rtm']
    )
    
    optimizer = RollingHorizonBESS(bess_params, config)
    
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
    
    print(f"Running 48h rolling backtest for {len(dates)} days...")
    
    backtest_results = []
    prev_soc = bess_params.soc_initial_mwh
    
    for i, date in enumerate(dates):
        # --- Multi-day SoC: carry forward from previous day ---
        bess_params.soc_initial_mwh = prev_soc
        
        print(f"[{i+1}/{len(dates)}] {date} (SoC₀={prev_soc:.1f})...", end=" ", flush=True)
        
        # Load Day 1 scenarios
        day1_data = loader.get_day_scenarios(date, n_scenarios=config['n_scenarios'])
        
        # Load Day 2 scenarios (lookahead)
        if i < len(dates) - 1:
            day2_data = loader.get_day_scenarios(dates[i + 1], n_scenarios=config['n_scenarios'])
        else:
            # Last day: use same-day scenarios as proxy
            day2_data = day1_data
        
        # Solve 48-hour problem
        res = optimizer.solve(
            day1_data['dam'], day1_data['rtm'],
            day2_data['dam'], day2_data['rtm']
        )
        
        if res['status'] != 'Optimal':
            print(f"FAILED (Status: {res['status']})")
            continue
        
        # Evaluate Day 1 against actuals
        eval_res = evaluate_actuals(
            bess_params,
            res['dam_schedule'],
            day1_data['dam_actual'],
            day1_data['rtm_actual'],
            cost_model=cost_model,
            lambda_dev=config['lambda_dev'],
            force_terminal_mode=eval_terminal_mode
        )
        
        if eval_res is None:
            print("EVAL FAILED")
            continue
        
        realized_rev = eval_res['revenue']
        net_rev = eval_res['net_revenue']
        expected_rev = res['expected_revenue']
        
        # Overnight SoC statistics
        overnight_socs = list(res.get('overnight_soc', {}).values())
        avg_overnight = np.mean(overnight_socs) if overnight_socs else prev_soc
        
        print(f"Exp: ₹{expected_rev:,.0f} | Net: ₹{net_rev:,.0f} | SoC₂₄: {eval_res['soc'][-1]:.1f}")
        
        # Save daily detailed results
        daily_output = {
            "date": date,
            "status": res['status'],
            "expected_revenue": expected_rev,
            "realized_revenue": realized_rev,
            "net_revenue": net_rev,
            "soc_initial": prev_soc,
            "soc_terminal": eval_res['soc'][-1],
            "avg_overnight_soc_planned": avg_overnight,
            "dam_schedule": res['dam_schedule'],
            "rtm_realized_schedule": eval_res['rtm_schedule'],
            "soc_realized": eval_res['soc'],
            "actual_dam_prices": day1_data['dam_actual'].tolist(),
            "actual_rtm_prices": day1_data['rtm_actual'].tolist(),
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
            "soc_terminal": daily_output['soc_terminal']
        })
        
        # SoC drift monitor: warn if < 50 MWh for 3+ consecutive days
        recent_socs = [r['soc_terminal'] for r in backtest_results[-3:]]
        if len(recent_socs) >= 3 and all(s < 50.0 for s in recent_socs):
            print(f"  ⚠️ SoC DRIFT WARNING: SoC < 50 MWh for {len(recent_socs)} consecutive days")
    
    # Save Summary
    results_df = pd.DataFrame(backtest_results)
    results_df.to_csv(results_dir / "backtest_results.csv", index=False)
    
    summary = {
        "n_days": len(results_df),
        "total_expected_revenue": results_df['expected_revenue'].sum(),
        "total_realized_revenue": results_df['realized_revenue'].sum(),
        "total_net_revenue": results_df['net_revenue'].sum(),
        "avg_daily_net": results_df['net_revenue'].mean(),
        "std_daily_net": results_df['net_revenue'].std(),
        "min_daily_net": results_df['net_revenue'].min(),
        "max_daily_net": results_df['net_revenue'].max(),
        "avg_terminal_soc": results_df['soc_terminal'].mean(),
        "min_terminal_soc": results_df['soc_terminal'].min(),
        "max_terminal_soc": results_df['soc_terminal'].max()
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n============================================================")
    print("48H ROLLING HORIZON BACKTEST COMPLETE")
    print(f"Total Realized Revenue: ₹{summary['total_realized_revenue']:,.2f}")
    print(f"Total Net Revenue:      ₹{summary['total_net_revenue']:,.2f}")
    print(f"Avg Daily Net:          ₹{summary['avg_daily_net']:,.2f}")
    print(f"Worst Day Net:          ₹{summary['min_daily_net']:,.2f}")
    print(f"SoC Range: [{summary['min_terminal_soc']:.1f}, {summary['max_terminal_soc']:.1f}] MWh")
    print(f"Summary saved to {results_dir / 'summary.json'}")
    print("============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=str, help='Run for a specific date (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, help='Limit number of days to run')
    parser.add_argument('--config', type=str, default='config/phase4_rolling.yaml',
                        help='Config file path')
    args = parser.parse_args()
    run_rolling_backtest(args)
