import pandas as pd
import numpy as np
import yaml
import json
import sys
from pathlib import Path
import argparse

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.optimizer.costs import CostModel

def load_data(results_dir):
    backtest_file = results_dir / "backtest_results.csv"
    bench_file = results_dir / "benchmark_results.csv"
    
    backtest_df = pd.read_csv(backtest_file)
    bench_df = pd.read_csv(bench_file) if bench_file.exists() else None
    
    return backtest_df, bench_df

def process_strategy_costs(df, cost_model, results_dir, col_prefix=""):
    daily_results_dir = results_dir / "daily"
    final_rows = []
    
    for i, row in df.iterrows():
        date = row['target_date']
        # For benchmarks, we oracles don't have daily JSONs in the same way, 
        # but wait, compare_strategies.py doesn't save daily JSONs.
        # This is a limitation. I'll focus on the Stochastic for now or 
        # approximate benchmarks using Stochastic throughput if needed.
        # Actually, let's keep it simple: Stochastic is what we care about most.
        pass

def run_backtest_with_costs():
    print("============================================================")
    print("PHASE 3C: APPLYING TRADING COSTS TO BACKTEST")
    print("============================================================")
    
    with open("config/costs_config.yaml", 'r') as f:
        costs_config = yaml.safe_load(f)
    
    cost_model = CostModel(costs_config)
    
    results_dir = Path("Data/Results/phase3b")
    daily_results_dir = results_dir / "daily"
    
    backtest_df, bench_df = load_data(results_dir)
    
    final_output = []
    
    print(f"Processing {len(backtest_df)} days...")
    
    for i, row in backtest_df.iterrows():
        date = row['target_date']
        res_path = daily_results_dir / f"result_{date}.json"
        with open(res_path, 'r') as f:
            daily_res = json.load(f)
            
        y_stoch = daily_res['rtm_realized_schedule']
        costs_stoch = cost_model.compute_costs(
            np.where(np.array(y_stoch) < 0, -np.array(y_stoch), 0),
            np.where(np.array(y_stoch) > 0, np.array(y_stoch), 0),
            scheduled=np.array(y_stoch)
        )
        
        row_data = {
            'target_date': date,
            'gross_revenue': row['realized_revenue'],
            'total_costs': costs_stoch['total_costs'],
            'net_revenue': row['realized_revenue'] - costs_stoch['total_costs'],
            'iex_fee': costs_stoch['iex_transaction_fee'],
            'scheduling': costs_stoch['scheduling_charges'],
            'degradation': costs_stoch['degradation_cost'],
            'dsm': costs_stoch['dsm_penalty_estimate']
        }
        final_output.append(row_data)
        
    df_with_costs = pd.DataFrame(final_output)
    
    # Sensitivity: Degradation Sweep
    deg_levels = [0, 300, 500, 650, 800, 1000]
    sweep_results = []
    for level in deg_levels:
        total_net = 0
        for i, row in df_with_costs.iterrows():
            # Adjust degradation only
            # col 'degradation' was calculated with 650
            original_deg = row['degradation']
            new_deg = (original_deg / 650.0) * level if original_deg > 0 else 0
            total_net += (row['gross_revenue'] - (row['total_costs'] - original_deg + new_deg))
        sweep_results.append({'Level': level, 'Net Revenue (₹M)': total_net / 1e6})
    
    # Save results
    output_dir = Path("Data/Results/phase3c")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_with_costs.to_csv(output_dir / "backtest_results_with_costs.csv", index=False)
    
    summary = {
        "n_days": len(df_with_costs),
        "total_gross_revenue": df_with_costs['gross_revenue'].sum(),
        "total_costs": df_with_costs['total_costs'].sum(),
        "total_net_revenue": df_with_costs['net_revenue'].sum(),
        "haircut_pct": (df_with_costs['total_costs'].sum() / df_with_costs['gross_revenue'].sum()) * 100,
        "avg_daily_net": df_with_costs['net_revenue'].mean(),
        "cost_breakdown": {
            "iex": df_with_costs['iex_fee'].sum(),
            "scheduling": df_with_costs['scheduling'].sum(),
            "degradation": df_with_costs['degradation'].sum(),
            "dsm": df_with_costs['dsm'].sum()
        },
        "degradation_sweep": sweep_results
    }
    
    with open(output_dir / "summary_with_costs.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Total Net Revenue: ₹{summary['total_net_revenue']:,.2f}")
    print(f"Revenue Haircut: {summary['haircut_pct']:.2f}%")

if __name__ == "__main__":
    run_backtest_with_costs()
