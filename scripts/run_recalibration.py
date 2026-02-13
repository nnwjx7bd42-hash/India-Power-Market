import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from scipy.stats import spearmanr

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.forecasting.recalibrate import compute_cqr_corrections, apply_cqr_corrections

def run_recalibration():
    print("=" * 64)
    print("QUANTILE RECALIBRATION RESULTS (CQR)")
    print("=" * 64)
    
    markets = ["dam", "rtm"]
    corrections_output = {}
    
    for market in markets:
        print(f"\nProcessing {market.upper()} market...")
        
        # 1. Load Validation Data
        val_pred_path = f"Data/Predictions/{market}_quantiles_val.parquet"
        val_feat_path = f"Data/Features/{market}_features_val.parquet"
        
        if not os.path.exists(val_pred_path):
             print(f"ERROR: {val_pred_path} not found.")
             continue
             
        df_val_pred = pd.read_parquet(val_pred_path)
        df_val_feat = pd.read_parquet(val_feat_path)
        
        # Align actuals
        # target_mcp_rs_mwh is the target column
        val_actuals = df_val_feat["target_mcp_rs_mwh"]
        # Ensure indices match if not already
        common_idx = df_val_pred.index.intersection(val_actuals.index)
        df_val_pred = df_val_pred.loc[common_idx]
        val_actuals = val_actuals.loc[common_idx]
        
        # 2. Compute Corrections
        market_corrections = compute_cqr_corrections(df_val_pred, val_actuals)
        corrections_output[market] = market_corrections
        
        # 3. Apply to Backtest Data
        bt_pred_path = f"Data/Predictions/{market}_quantiles_backtest.parquet"
        bt_feat_path = f"Data/Features/{market}_features_backtest.parquet"
        
        if not os.path.exists(bt_pred_path):
            print(f"ERROR: {bt_pred_path} not found.")
            continue
            
        df_bt_pred = pd.read_parquet(bt_pred_path)
        df_bt_feat = pd.read_parquet(bt_feat_path)
        
        # Align backtest actuals for reporting
        bt_actuals = df_bt_feat["target_mcp_rs_mwh"]
        common_idx_bt = df_bt_pred.index.intersection(bt_actuals.index)
        df_bt_pred_aligned = df_bt_pred.loc[common_idx_bt]
        bt_actuals_aligned = bt_actuals.loc[common_idx_bt]
        
        # Apply
        print(f"\nApplying to Backtest...")
        df_bt_recal = apply_cqr_corrections(df_bt_pred, market_corrections)
        
        # Save results
        recal_parquet = f"Data/Predictions/{market}_quantiles_backtest_recalibrated.parquet"
        recal_csv = f"Data/Predictions/{market}_quantiles_backtest_recalibrated.csv"
        df_bt_recal.to_parquet(recal_parquet)
        df_bt_recal.to_csv(recal_csv, index=True)
        
        # 4. Report Backtest Coverage
        print(f"\nBACKTEST COVERAGE (out-of-sample) - {market.upper()}:")
        print(f"{'Quantile':<10} {'Target':<10} {'Original':<10} {'Recal':<10} {'Delta':<10}")
        df_bt_recal_aligned = df_bt_recal.loc[common_idx_bt]
        
        for q in ["q10", "q25", "q50", "q75", "q90"]:
            tau = float(q[1:]) / 100
            cov_orig = (bt_actuals_aligned <= df_bt_pred_aligned[q]).mean()
            cov_recal = (bt_actuals_aligned <= df_bt_recal_aligned[q]).mean()
            print(f"{q:<10} {tau:<10.3f} {cov_orig:<10.3f} {cov_recal:<10.3f} {cov_recal - cov_orig:<10.3f}")
            
        # Spearman Correlation on q50
        rho, _ = spearmanr(df_bt_pred["q50"], df_bt_recal["q50"])
        print(f"\nRank preservation (Spearman raw vs recalibrated q50): {rho:.4f}")

    # Save corrections JSON
    corrections_output["method"] = "conformal_quantile_regression"
    corrections_output["fitted_on"] = "validation_2024-10-01_to_2025-01-31"
    
    with open("Data/Predictions/cqr_corrections.json", 'w') as f:
        json.dump(corrections_output, f, indent=2)
        
    print("\n" + "=" * 64)
    print("RECALIBRATION COMPLETE")
    print("=" * 64)

if __name__ == "__main__":
    run_recalibration()
