import pandas as pd
import numpy as np
import yaml
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import time

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.optimizer.bess_params import BESSParams
from src.optimizer.scenario_loader import ScenarioLoader
from src.optimizer.two_stage_bess import TwoStageBESS

def run_convergence_test(n_range=[50, 100, 200, 500, 1000], n_days=3):
    print("============================================================")
    print("SCENARIO CONVERGENCE TEST")
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
    
    # Select sample dates (common dates)
    dates = loader.common_dates[:n_days]
    print(f"Testing convergence over {n_days} days: {dates}")
    print(f"Scenario counts: {n_range}")
    
    results = []
    
    for date in dates:
        print(f"\nDate: {date}")
        for n in n_range:
            print(f"  Running N={n}...", end=" ", flush=True)
            
            day_data = loader.get_day_scenarios(date, n_scenarios=n)
            
            start_time = time.time()
            res = optimizer.solve(day_data['dam'], day_data['rtm'])
            solve_time = time.time() - start_time
            
            if res['status'] == 'Optimal':
                print(f"Done ({solve_time:.2f}s) | Exp Rev: ₹{res['expected_revenue']:,.0f}")
                results.append({
                    "date": date,
                    "n_scenarios": n,
                    "expected_revenue": res['expected_revenue'],
                    "solve_time": solve_time
                })
            else:
                print(f"FAILED ({res['status']})")
                
    df = pd.DataFrame(results)
    
    # Summary Statistics
    summary = df.groupby('n_scenarios').agg({
        'expected_revenue': ['mean', 'std'],
        'solve_time': ['mean', 'max']
    })
    print("\nSummary Results:")
    print(summary)
    
    # Save Results
    output_dir = Path("results/convergence")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "convergence_results.csv", index=False)
    
    # Plotting
    try:
        plt.figure(figsize=(12, 6))
        
        # subplot 1: Revenue Convergence
        plt.subplot(1, 2, 1)
        for date in dates:
            day_df = df[df['date'] == date]
            plt.plot(day_df['n_scenarios'], day_df['expected_revenue'], marker='o', label=date)
        plt.title("Expected Revenue vs Scenario Count")
        plt.xlabel("N Scenarios")
        plt.ylabel("Expected Revenue (₹)")
        plt.legend()
        plt.grid(True)
        
        # subplot 2: Solve Time
        plt.subplot(1, 2, 2)
        plt.plot(summary.index, summary[('solve_time', 'mean')], marker='s', color='red')
        plt.title("Mean Solve Time vs Scenario Count")
        plt.xlabel("N Scenarios")
        plt.ylabel("Solve Time (s)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "convergence_plot.png")
        print(f"\nConvergence plot saved to {output_dir / 'convergence_plot.png'}")
    except Exception as e:
        print(f"\nCould not generate plots: {e}")

if __name__ == "__main__":
    run_convergence_test()
