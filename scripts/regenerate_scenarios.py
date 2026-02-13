import pandas as pd
import numpy as np
import json
import scipy.stats
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.scenarios.joint_copula import inverse_cdf_vectorized

def regenerate_scenarios():
    print("=" * 64)
    print("SCENARIO REGENERATION (RECALIBRATED + STATIC COPULA)")
    print("=" * 64)
    
    # Paths
    pred_dir = Path("Data/Predictions")
    results_dir = Path("results")
    params_path = results_dir / "joint_copula_params.json"
    
    with open(params_path, 'r') as f:
        params = json.load(f)
        
    rho_by_hour = params['rho_by_hour']
    dam_corr = np.array(params['dam_copula_correlation'])
    L = np.linalg.cholesky(dam_corr)
    
    # Load Recalibrated Quantiles
    print("Loading recalibrated quantiles...")
    dam_preds = pd.read_parquet(pred_dir / "dam_quantiles_backtest_recalibrated.parquet")
    rtm_preds = pd.read_parquet(pred_dir / "rtm_quantiles_backtest_recalibrated.parquet")
    
    # Common Dates
    dam_dates = set(dam_preds['target_date'].unique())
    rtm_dates = set(rtm_preds['target_date'].unique())
    common_dates = sorted(list(dam_dates & rtm_dates))
    
    n_scenarios = 100
    dam_rows = []
    rtm_rows = []
    
    print(f"Generating scenarios for {len(common_dates)} days...")
    for date_idx, date in enumerate(tqdm(common_dates)):
        rng = np.random.default_rng(seed=42 + date_idx)
        
        # Day's quantiles
        dam_day = dam_preds[dam_preds['target_date'] == date].sort_values('target_hour')
        rtm_day = rtm_preds[rtm_preds['target_date'] == date].sort_values('target_hour')
        
        if len(dam_day) != 24 or len(rtm_day) != 24:
            continue
            
        # Latent Gaussians
        z_indep = rng.standard_normal((n_scenarios, 24))
        z_dam = z_indep @ L.T
        u_dam = scipy.stats.norm.cdf(z_dam)
        
        # Correlated RTM Latents
        z_rtm = np.zeros_like(z_dam)
        eps_rtm = rng.standard_normal((n_scenarios, 24))
        for h in range(24):
            rho = rho_by_hour[h]
            z_rtm[:, h] = rho * z_dam[:, h] + np.sqrt(1 - rho**2) * eps_rtm[:, h]
        u_rtm = scipy.stats.norm.cdf(z_rtm)
        
        # Prices
        dam_prices = np.zeros((n_scenarios, 24))
        rtm_prices = np.zeros((n_scenarios, 24))
        
        for h in range(24):
            q_d = {k: dam_day.iloc[h][k] for k in ['q10', 'q25', 'q50', 'q75', 'q90']}
            dam_prices[:, h] = inverse_cdf_vectorized(u_dam[:, h], q_d)
            
            q_r = {k: rtm_day.iloc[h][k] for k in ['q10', 'q25', 'q50', 'q75', 'q90']}
            rtm_prices[:, h] = inverse_cdf_vectorized(u_rtm[:, h], q_r)
            
        # Clamp non-negative
        dam_prices = np.maximum(dam_prices, 0)
        rtm_prices = np.maximum(rtm_prices, 0)
        
        for s in range(n_scenarios):
            dam_row = {"target_date": date, "scenario_id": s}
            rtm_row = {"target_date": date, "scenario_id": s}
            for t in range(24):
                dam_row[f"h{t:02d}"] = float(dam_prices[s, t])
                rtm_row[f"h{t:02d}"] = float(rtm_prices[s, t])
            dam_rows.append(dam_row)
            rtm_rows.append(rtm_row)
            
    # Save
    df_dam_scen = pd.DataFrame(dam_rows)
    df_rtm_scen = pd.DataFrame(rtm_rows)
    
    print("\nSaving joint recalibrated scenarios...")
    df_dam_scen.to_parquet(pred_dir / "joint_dam_scenarios_backtest_recalibrated.parquet", index=False)
    df_dam_scen.to_csv(pred_dir / "joint_dam_scenarios_backtest_recalibrated.csv", index=False)
    df_rtm_scen.to_parquet(pred_dir / "joint_rtm_scenarios_backtest_recalibrated.parquet", index=False)
    df_rtm_scen.to_csv(pred_dir / "joint_rtm_scenarios_backtest_recalibrated.csv", index=False)
    
    print("\nREGENERATION COMPLETE")

if __name__ == "__main__":
    regenerate_scenarios()
