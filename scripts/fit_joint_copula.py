import pandas as pd
import numpy as np
import json
import scipy.stats
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.scenarios.joint_copula import fit_cross_market_rho, estimate_dam_copula_correlation, generate_correlated_uniforms, inverse_cdf_vectorized

def run_fitting():
    print("============================================================")
    print("FITTING JOINT DAM<->RTM COPULA")
    print("============================================================")
    
    # Paths
    pred_dir = Path("Data/Predictions")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    dam_val_path = pred_dir / "dam_quantiles_val.parquet"
    rtm_val_path = pred_dir / "rtm_quantiles_val.parquet"
    
    if not dam_val_path.exists() or not rtm_val_path.exists():
        print(f"Error: Validation predictions not found in {pred_dir}")
        return
        
    print(f"Loading validation predictions...")
    dam_val = pd.read_parquet(dam_val_path)
    rtm_val = pd.read_parquet(rtm_val_path)
    
    # Fit Cross-Market rho
    print("Fitting cross-market correlation (rho)...")
    rho_result = fit_cross_market_rho(dam_val, rtm_val)
    
    # Estimate DAM Copula
    print("Estimating DAM 24x24 copula matrix...")
    dam_corr = estimate_dam_copula_correlation(dam_val)
    
    # Prepare result
    val_dates = dam_val['target_date'].unique()
    
    params = {
        "rho_by_hour": rho_result['rho_by_hour'],
        "rho_by_hour_raw": rho_result['rho_by_hour_raw'],
        "rho_global": rho_result['rho_global'],
        "n_observations_by_hour": rho_result['n_observations_by_hour'],
        "n_common_dates": rho_result['n_common_dates'],
        "shrinkage_factor": rho_result['shrinkage_factor'],
        "dam_copula_correlation": dam_corr.tolist(),
        "validation_period": f"{min(val_dates)} to {max(val_dates)}",
    }
    
    with open(results_dir / "joint_copula_params.json", 'w') as f:
        json.dump(params, f, indent=2)
        
    # Output Table
    print("\nJOINT DAM<->RTM COPULA PARAMETERS")
    print(f"Validation period: {params['validation_period']} ({params['n_common_dates']} common dates)")
    print(f"Total observation pairs: {rho_result['n_total_observations']}")
    
    print("\nCross-Market Correlation (rho) by Hour:")
    print("Hour  rho_raw    rho_shrunk   n_obs")
    for h in range(24):
        print(f"{h:<6d}{rho_result['rho_by_hour_raw'][h]:<11.3f}{rho_result['rho_by_hour'][h]:<13.3f}{rho_result['n_observations_by_hour'][h]}")
        
    print("\nSummary:")
    print(f"Global rho:     {rho_result['rho_global']:.3f}")
    print(f"Mean rho_h:     {np.mean(rho_result['rho_by_hour']):.3f}")
    print(f"Min rho_h:      {np.min(rho_result['rho_by_hour']):.3f}")
    print(f"Max rho_h:      {np.max(rho_result['rho_by_hour']):.3f}")
    
    print(f"\nDAM Copula Correlation Matrix:")
    print(f"Shape: {dam_corr.shape}")
    off_diag = dam_corr[~np.eye(24, dtype=bool)]
    print(f"Mean off-diagonal: {np.mean(off_diag):.3f}")
    print(f"Min off-diagonal:  {np.min(off_diag):.3f}")
    print(f"Max off-diagonal:  {np.max(off_diag):.3f}")
    
    print(f"\nSaved to: results/joint_copula_params.json")
    print("============================================================")

if __name__ == "__main__":
    run_fitting()
