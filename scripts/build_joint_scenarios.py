import pandas as pd
import numpy as np
import json
import scipy.stats
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.scenarios.joint_copula import generate_correlated_uniforms, inverse_cdf_vectorized

def run_build_scenarios():
    print("============================================================")
    print("BUILDING JOINT DAM<->RTM SCENARIOS")
    print("============================================================")
    
    # Paths
    results_dir = Path("results")
    pred_dir = Path("Data/Predictions")
    params_path = results_dir / "joint_copula_params.json"
    
    if not params_path.exists():
        print(f"Error: Joint copula parameters not found at {params_path}")
        return
        
    with open(params_path, 'r') as f:
        params = json.load(f)
        
    rho_by_hour = params['rho_by_hour']
    dam_corr = np.array(params['dam_copula_correlation'])
    
    # Load backtest predictions
    print("Loading backtest quantile predictions...")
    dam_preds = pd.read_parquet(pred_dir / "dam_quantiles_backtest.parquet")
    rtm_preds = pd.read_parquet(pred_dir / "rtm_quantiles_backtest.parquet")
    
    # Normalize RTM
    if 'target_hour' not in rtm_preds.columns:
        if 'delivery_start_ist' in rtm_preds.columns:
            rtm_preds['target_hour'] = pd.to_datetime(rtm_preds['delivery_start_ist']).dt.hour
            rtm_preds['target_date'] = pd.to_datetime(rtm_preds['delivery_start_ist']).dt.date.astype(str)
            
    # Find common dates
    dam_dates = set(dam_preds.groupby('target_date').filter(lambda x: len(x) == 24)['target_date'].unique())
    rtm_dates = set(rtm_preds.groupby('target_date').filter(lambda x: len(x) == 24)['target_date'].unique())
    common_dates = sorted(list(dam_dates.intersection(rtm_dates)))
    n_days = len(common_dates)
    
    print(f"Generating scenarios for {n_days} common backtest days...")
    
    # Prepare DAM Cholesky
    L = np.linalg.cholesky(dam_corr)
    
    n_scenarios = 200
    dam_rows = []
    rtm_rows = []
    
    for date_idx, date in enumerate(common_dates):
        rng = np.random.default_rng(seed=42 + date_idx)
        
        # Extract day's quantiles
        dam_day = dam_preds[dam_preds['target_date'] == date].sort_values('target_hour')
        rtm_day = rtm_preds[rtm_preds['target_date'] == date].sort_values('target_hour')
        
        # Latent Gaussians
        z_indep = rng.standard_normal((n_scenarios, 24))
        z_dam = z_indep @ L.T
        
        # Correlated Uniforms
        u_dam = scipy.stats.norm.cdf(z_dam)
        u_rtm = generate_correlated_uniforms(z_dam, rho_by_hour, rng)
        
        # Map to prices
        dam_prices = np.zeros((n_scenarios, 24))
        rtm_prices = np.zeros((n_scenarios, 24))
        
        for h in range(24):
            # DAM
            row_d = dam_day.iloc[h]
            q_d = {k: row_d[k] for k in ['q10', 'q25', 'q50', 'q75', 'q90']}
            dam_prices[:, h] = inverse_cdf_vectorized(u_dam[:, h], q_d)
            
            # RTM
            row_r = rtm_day.iloc[h]
            q_r = {k: row_r[k] for k in ['q10', 'q25', 'q50', 'q75', 'q90']}
            rtm_prices[:, h] = inverse_cdf_vectorized(u_rtm[:, h], q_r)
            
        # Clamp non-negative
        dam_prices = np.maximum(dam_prices, 0)
        rtm_prices = np.maximum(rtm_prices, 0)
        
        # Collect rows
        for s in range(n_scenarios):
            dam_rows.append({
                'target_date': date,
                'scenario_id': s,
                **{f'h{h:02d}': float(dam_prices[s, h]) for h in range(24)}
            })
            rtm_rows.append({
                'target_date': date,
                'scenario_id': s,
                **{f'h{h:02d}': float(rtm_prices[s, h]) for h in range(24)}
            })
            
        if (date_idx + 1) % 20 == 0:
            print(f"  Generated {date_idx + 1}/{n_days} days")
            
    # Save
    dam_df = pd.DataFrame(dam_rows)
    rtm_df = pd.DataFrame(rtm_rows)
    
    dam_df.to_parquet(pred_dir / "joint_dam_scenarios_backtest.parquet", index=False)
    dam_df.to_csv(pred_dir / "joint_dam_scenarios_backtest.csv", index=False)
    rtm_df.to_parquet(pred_dir / "joint_rtm_scenarios_backtest.parquet", index=False)
    rtm_df.to_csv(pred_dir / "joint_rtm_scenarios_backtest.csv", index=False)
    
    # Also export existing quantiles to CSV
    for f in ["dam_quantiles_backtest.parquet", "rtm_quantiles_backtest.parquet"]:
        path = pred_dir / f
        if path.exists():
            df = pd.read_parquet(path)
            df.to_csv(pred_dir / f.replace('.parquet', '.csv'), index=False)
            
    print("\nJOINT SCENARIO GENERATION COMPLETE")
    print(f"Backtest dates: {n_days} days ({common_dates[0]} to {common_dates[-1]})")
    print(f"Scenarios per day: {n_scenarios}")
    
    print("\nDAM scenario statistics (all paths):")
    all_dam = dam_df.filter(regex='^h').values.flatten()
    print(f"  Mean: ₹{np.mean(all_dam):.2f}   Std: ₹{np.std(all_dam):.2f}   Min: ₹{np.min(all_dam):.2f}   Max: ₹{np.max(all_dam):.2f}")
    
    print("\nRTM scenario statistics (all paths):")
    all_rtm = rtm_df.filter(regex='^h').values.flatten()
    print(f"  Mean: ₹{np.mean(all_rtm):.2f}   Std: ₹{np.std(all_rtm):.2f}   Min: ₹{np.min(all_rtm):.2f}   Max: ₹{np.max(all_rtm):.2f}")
    
    print(f"\nFiles saved in {pred_dir}")
    print("============================================================")

if __name__ == "__main__":
    run_build_scenarios()
