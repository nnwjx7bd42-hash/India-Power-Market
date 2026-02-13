import pandas as pd
import numpy as np

def compute_cqr_corrections(val_predictions_df, val_actuals, quantile_levels=[0.10, 0.25, 0.50, 0.75, 0.90]):
    """
    Compute Conformal Quantile Regression (CQR) corrections (delta) for each quantile level.
    delta_tau = np.quantile(q_pred - actual, 1 - tau)
    """
    corrections = {}
    
    print(f"{'Quantile':<10} {'Target':<10} {'Before':<10} {'Delta (Rs)':<12} {'After':<10}")
    print("-" * 55)
    
    for tau in quantile_levels:
        q_col = f"q{int(tau * 100)}"
        q_pred = val_predictions_df[q_col]
        
        # Conformity scores: positive means predicted > actual (over-coverage)
        scores = q_pred - val_actuals
        
        # Correction delta
        delta = np.quantile(scores, 1 - tau)
        
        # Check coverage before and after
        cov_before = (val_actuals <= q_pred).mean()
        cov_after = (val_actuals <= (q_pred - delta)).mean()
        
        corrections[q_col] = float(delta)
        
        print(f"{q_col:<10} {tau:<10.3f} {cov_before:<10.3f} {delta:<12.2f} {cov_after:<10.3f}")
        
        # Sanity Checks
        if tau == 0.10:
            if delta > 1000:
                print(f"WARN: q10 delta ({delta:.1f}) too large - check val predictions alignment")
            elif delta < 50 and cov_before > 0.15:
                print(f"WARN: q10 over-coverage is distributional, not level-shift - consider isotonic v2")

    return corrections

def apply_cqr_corrections(predictions_df, corrections_dict):
    """
    Apply CQR corrections, enforce non-negativity, and repair monotonicity.
    """
    recal_df = predictions_df.copy()
    
    # 1. Shift
    for q_col, delta in corrections_dict.items():
        recal_df[q_col] = recal_df[q_col] - delta
        
    # 2. Non-negativity
    print("\nNon-negativity clips:")
    for q_col in corrections_dict.keys():
        n_clipped = (recal_df[q_col] < 0).sum()
        recal_df[q_col] = recal_df[q_col].clip(lower=0)
        print(f"  {q_col}: {n_clipped}")
        
    # 3. Monotonicity Repair
    # q10 <= q25 <= q50 <= q75 <= q90
    q_cols = ["q10", "q25", "q50", "q75", "q90"]
    repaired_count = 0
    for i in range(1, len(q_cols)):
        violation_mask = recal_df[q_cols[i]] < recal_df[q_cols[i-1]]
        if violation_mask.any():
            repaired_count += violation_mask.sum()
            recal_df.loc[violation_mask, q_cols[i]] = recal_df.loc[violation_mask, q_cols[i-1]]
            
    print(f"Monotonicity repairs: {repaired_count} ({repaired_count / len(recal_df) * 100:.2f}%)")
    
    return recal_df

def validate_recalibration(recal_df, actuals, quantile_levels=[0.10, 0.25, 0.50, 0.75, 0.90]):
    """
    Compute coverage for recalibrated data.
    """
    stats = {}
    for tau in quantile_levels:
        q_col = f"q{int(tau * 100)}"
        stats[q_col] = (actuals <= recal_df[q_col]).mean()
    return stats
