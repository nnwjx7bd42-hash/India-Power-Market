import pandas as pd
import numpy as np
import yaml
import json
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.optimizer.bess_params import BESSParams
from src.optimizer.scenario_loader import ScenarioLoader
from src.optimizer.two_stage_bess import TwoStageBESS
from scripts.run_phase3b_backtest import evaluate_actuals

def run_cvar_sweep(args):
    print("============================================================")
    print(f"PHASE 4: CVaR RISK-RETURN SWEEP ({args.scenarios.upper()})")
    print("============================================================")
    
    # Load configs
    bess_params = BESSParams.from_yaml("config/bess.yaml")
    with open("config/phase3b.yaml", 'r') as f:
        p3b_config = yaml.safe_load(f)
    with open("config/cvar_config.yaml", 'r') as f:
        cvar_config = yaml.safe_load(f)
        
    dam_path = p3b_config['paths']['scenarios_dam']
    rtm_path = p3b_config['paths']['scenarios_rtm']
    suffix = ""
    
    if args.scenarios == "recalibrated":
        dam_path = "Data/Predictions/joint_dam_scenarios_backtest_recalibrated.parquet"
        rtm_path = "Data/Predictions/joint_rtm_scenarios_backtest_recalibrated.parquet"
        suffix = "_recalibrated"
        
    loader = ScenarioLoader(
        dam_path=dam_path,
        rtm_path=rtm_path,
        actuals_dam_path=p3b_config['paths']['actuals_dam'],
        actuals_rtm_path=p3b_config['paths']['actuals_rtm']
    )
    
    dates = loader.common_dates
    lambdas = cvar_config['cvar']['lambda_values']
    alpha = cvar_config['cvar']['alpha']
    
    results_dir = Path("Data/Backtest/cvar_sweep")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    sweep_summary = []
    
    for l_val in lambdas:
        print(f"\nRunning sweep for Lambda = {l_val}...")
        
        # Override config for this sweep step
        step_config = p3b_config.copy()
        step_config['lambda_risk'] = l_val
        step_config['risk_alpha'] = alpha
        
        optimizer = TwoStageBESS(bess_params, step_config)
        
        day_results = []
        for i, date in enumerate(dates):
            if i % 20 == 0:
                print(f"  [{i}/{len(dates)}] {date}...")
                
            day_data = loader.get_day_scenarios(date, n_scenarios=step_config['n_scenarios'])
            
            res = optimizer.solve(day_data['dam'], day_data['rtm'])
            if res['status'] != 'Optimal':
                continue
                
            eval_res = evaluate_actuals(
                bess_params, 
                res['dam_schedule'], 
                day_data['dam_actual'], 
                day_data['rtm_actual'],
                lambda_dev=step_config['lambda_dev']
            )
            
            if eval_res is None:
                continue
                
            sched = np.array(eval_res['rtm_schedule'])
            cycles = np.sum(np.where(sched > 0, sched, 0)) / (bess_params.e_max_mwh - bess_params.e_min_mwh)
            
            day_results.append({
                'date': date,
                'expected_revenue': res['expected_revenue'],
                'realized_revenue': eval_res['net_revenue'],
                'cvar_zeta': res['cvar_zeta_rs'],
                'cvar_value': res['cvar_value_rs'],
                'cycles': cycles
            })
            
        df = pd.DataFrame(day_results)
        df.to_csv(results_dir / f"results_lambda_{l_val}{suffix}.csv", index=False)
        
        total_net = df['realized_revenue'].sum()
        worst_day = df['realized_revenue'].min()
        avg_cvar = df['cvar_value'].mean()
        avg_cycles = df['cycles'].mean()
        sharpe = (df['realized_revenue'].mean() / df['realized_revenue'].std()) * np.sqrt(365) if len(df) > 1 else 0
        
        sweep_summary.append({
            'lambda': l_val,
            'net_revenue_m': total_net / 1e6,
            'worst_day_k': worst_day / 1e3,
            'avg_cvar_k': avg_cvar / 1e3,
            'sharpe': sharpe,
            'avg_cycles': avg_cycles
        })
        print(f"  Results: Net=₹{total_net/1e6:.2f}M | Worst=₹{worst_day/1e3:.1f}K | Sharpe={sharpe:.2f}")

    summary_df = pd.DataFrame(sweep_summary)
    summary_df.to_csv(results_dir / f"efficient_frontier{suffix}.csv", index=False)
    summary_df.to_parquet(results_dir / f"efficient_frontier{suffix}.parquet", index=False)
    
    with open(f"results/cvar_sweep_summary{suffix}.json", 'w') as f:
        json.dump(sweep_summary, f, indent=2)
        
    print("\n" + "="*60)
    print(f"EFFICIENT FRONTIER ({args.scenarios.upper()})")
    print("="*60)
    print(summary_df)
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", choices=["original", "recalibrated"],
                    default="original", help="Which scenario set to use")
    args = parser.parse_args()
    
    Path("results").mkdir(exist_ok=True)
    run_cvar_sweep(args)
