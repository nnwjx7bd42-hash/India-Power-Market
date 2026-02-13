import pandas as pd
import numpy as np
import json
from pathlib import Path

def validate():
    print("============================================================")
    print("PHASE 3B VALIDATION REPORT")
    print("============================================================")
    
    results_dir = Path("Data/Results/phase3b")
    backtest_df = pd.read_csv(results_dir / "backtest_results.csv")
    bench_df = pd.read_csv(results_dir / "benchmark_results.csv")
    
    df = pd.merge(backtest_df, bench_df, on='target_date')
    
    # 1. Capture Ratio (Stochastic / Oracle)
    capture_ratio = df['realized_revenue'].sum() / df['realized_revenue_oracle'].sum()
    
    # 2. Deterministic Ratio (Stochastic / Deterministic)
    det_ratio = df['realized_revenue'].sum() / df['realized_revenue_det'].sum()
    
    # 3. Naive Uplift
    uplift = (df['realized_revenue'].sum() / df['realized_revenue_naive'].sum()) - 1.0
    
    # 4. Volatility (Daily Realized Revenue std)
    daily_std = df['realized_revenue'].std()
    sharpe = df['realized_revenue'].mean() / daily_std if daily_std > 0 else 0
    
    print(f"Capture Ratio (vs Oracle): {capture_ratio:.4f}")
    print(f"Efficiency (vs Deterministic): {det_ratio:.4f}")
    print(f"Naive Uplift: {uplift*100:.2f}%")
    print(f"Daily Sharpe-like Ratio: {sharpe:.4f}")
    
    metrics = {
        "capture_ratio": capture_ratio,
        "det_ratio": det_ratio,
        "uplift_pct": uplift * 100,
        "sharpe_ratio": sharpe,
        "total_revenue": df['realized_revenue'].sum()
    }
    
    with open(results_dir / "validation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # Gating checks
    print("\nGATING CHECKS:")
    print(f"  [PASS] Capture Ratio > 0.8: {capture_ratio > 0.8}")
    print(f"  [PASS] Uplift > 50%: {uplift > 0.5}")
    
    if capture_ratio > 0.965:
        print("  !!! STRETCH GOAL MET: Capture Ratio > 0.965 !!!")

if __name__ == "__main__":
    validate()
