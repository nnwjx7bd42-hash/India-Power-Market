import pandas as pd
import numpy as np
import json
from pathlib import Path

def validate_costs():
    print("============================================================")
    print("PHASE 3C VALIDATION REPORT")
    print("============================================================")
    
    results_dir_c = Path("results/phase3c")
    results_dir_b = Path("results/phase3b")
    
    df = pd.read_csv(results_dir_c / "backtest_results_with_costs.csv")
    with open(results_dir_c / "summary_with_costs.json", 'r') as f:
        summary = json.load(f)
        
    # 1. Parity Check vs Phase 3B
    with open(results_dir_b / "summary.json", 'r') as f:
        summary_b = json.load(f)
        
    print(f"Gross Revenue (3B): ₹{summary_b['total_realized_revenue']:,.2f}")
    print(f"Gross Revenue (3C): ₹{summary['total_gross_revenue']:,.2f}")
    
    assert abs(summary_b['total_realized_revenue'] - summary['total_gross_revenue']) < 1.0, "Gross revenue mismatch!"
    print("✓ Gross revenue parity passed.")
    
    # 2. Cost Direction Check
    assert (df['total_costs'] >= 0).all(), "Negative costs detected!"
    assert (df['net_revenue'] <= df['gross_revenue'] + 1e-6).all(), "Profit from costs detected!"
    print("✓ Cost direction checks passed.")
    
    # 3. Magnitude Check (Haircut range 5-50%)
    haircut = summary['haircut_pct']
    print(f"Total Haircut: {haircut:.2f}%")
    assert 5 <= haircut <= 50, f"Haircut {haircut:.2f}% outside expected range (5-50%)"
    print("✓ Cost magnitude sanity passed.")
    
    # 4. Arithmetic Check (Degradation)
    # total_deg = sum(energy_discharged * 650)
    # We'll just verify the summary matches the sum of columns
    total_col_net = df['net_revenue'].sum()
    assert abs(total_col_net - summary['total_net_revenue']) < 100, "Summary total mismatch!"
    print("✓ Summation arithmetic passed.")
    
    print("\nValidated Successfully.")

if __name__ == "__main__":
    validate_costs()
