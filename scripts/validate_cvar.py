import pandas as pd
import numpy as np
import json
from pathlib import Path
import yaml

def validate_cvar():
    print("============================================================")
    print("PHASE 4 VALIDATION REPORT")
    print("============================================================")
    
    results_dir = Path("Data/Backtest/cvar_sweep")
    with open("config/cvar_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    frontier_path = results_dir / "efficient_frontier.csv"
    if not frontier_path.exists():
        print("ERROR: efficient_frontier.csv not found. Run sweep first.")
        return
        
    df = pd.read_csv(frontier_path)
    baseline_rev_m = config['phase3d_baseline']['net_revenue_m']
    tolerance = config['phase3d_baseline']['regression_tolerance_pct'] / 100.0
    
    print(f"Basline Revenue (Phase 3D): ₹{baseline_rev_m:.2f}M")
    
    # 1. Regression Check (lambda=0)
    row_0 = df[df['lambda'] == 0].iloc[0]
    rev_0 = row_0['net_revenue_m']
    print(f"Sweep Revenue (lambda=0): ₹{rev_0:.2f}M")
    
    diff_pct = abs(rev_0 - baseline_rev_m) / baseline_rev_m
    assert diff_pct <= tolerance, f"Regression failed! Lambda=0 diff {diff_pct:.4%} > {tolerance:.4%}"
    print("✓ Regression gate passed (matches Phase 3D).")
    
    # 2. Monotonicity: Revenue non-increasing
    # We allow small numerical noise (0.1M)
    revs = df['net_revenue_m'].values
    for i in range(len(revs)-1):
        assert revs[i+1] <= revs[i] + 0.1, f"Monotonicity failed! Revenue increased at lambda={df.iloc[i+1]['lambda']}"
    print("✓ Revenue monotonicity passed.")
    
    # 3. Monotonicity: Worst-day non-decreasing
    # Note: Worst-day might stay flat or have noise, but should generally trend up.
    worst_days = df['worst_day_k'].values
    for i in range(len(worst_days)-1):
        # Allow small noise (10k)
        if worst_days[i+1] < worst_days[i] - 10.0:
             print(f"WARNING: Worst-day decreased at lambda={df.iloc[i+1]['lambda']}: {worst_days[i]} -> {worst_days[i+1]}")
    print("✓ Worst-day trend checked.")
    
    # 4. Efficiency Ratio
    # worst_day_improved_k / rev_sacrificed_m
    if len(df) > 1:
        print("\nRisk-Return Efficiency:")
        for i in range(1, len(df)):
            l = df.iloc[i]['lambda']
            dr = rev_0 - df.iloc[i]['net_revenue_m']
            dw = df.iloc[i]['worst_day_k'] - row_0['worst_day_k']
            eff = dw / dr if dr > 0 else 0
            print(f"  Lambda {l:4}: Sacrificed ₹{dr:4.2f}M to improve worst day by ₹{dw:5.1f}K (Eff: ₹{eff:5.1f}K/₹M)")
            
    # 5. Feasibility
    # Scipy linprog failures would have raised error in sweep, but we check if all lambdas present
    assert len(df) == len(config['cvar']['lambda_values']), "Some lambda steps missing!"
    print("\n✓ All feasibility checks passed.")
    
    print("\nValidated Successfully.")

if __name__ == "__main__":
    validate_cvar()
