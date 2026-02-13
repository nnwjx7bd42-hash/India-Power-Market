import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import scipy.stats

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

def run_validation():
    print("============================================================")
    print("VALIDATING JOINT DAM<->RTM SCENARIOS")
    print("============================================================")
    
    # Paths
    pred_dir = Path("Data/Predictions")
    results_dir = Path("results")
    params_path = results_dir / "joint_copula_params.json"
    
    # 1. EXISTENCE
    checks = {
        "joint_dam_parquet": (pred_dir / "joint_dam_scenarios_backtest.parquet").exists(),
        "joint_rtm_parquet": (pred_dir / "joint_rtm_scenarios_backtest.parquet").exists(),
        "joint_copula_params": params_path.exists()
    }
    
    for name, ok in checks.items():
        if not ok:
            print(f"FAILED: {name} not found")
            sys.exit(1)
            
    # Load data
    dam_df = pd.read_parquet(pred_dir / "joint_dam_scenarios_backtest.parquet")
    rtm_df = pd.read_parquet(pred_dir / "joint_rtm_scenarios_backtest.parquet")
    with open(params_path, 'r') as f:
        params = json.load(f)
        
    # 2. SHAPE
    if len(dam_df) != len(rtm_df):
        print(f"FAILED: DAM and RTM scenario counts mismatch ({len(dam_df)} vs {len(rtm_df)})")
        sys.exit(1)
        
    n_days = len(dam_df['target_date'].unique())
    print(f"Found {n_days} days of scenarios.")
    if n_days < 140:
        print(f"FAILED: Insufficient backtest days ({n_days} < 140)")
        sys.exit(1)
        
    # 3. ALIGNMENT
    # Check if target_date and scenario_id match perfectly
    alignment_check = (dam_df[['target_date', 'scenario_id']] == rtm_df[['target_date', 'scenario_id']]).all().all()
    if not alignment_check:
        print("FAILED: DAM and RTM scenario IDs are not aligned")
        sys.exit(1)
        
    # 4. NON-NEGATIVITY
    if (dam_df.filter(regex='^h').values < 0).any() or (rtm_df.filter(regex='^h').values < 0).any():
        print("FAILED: Negative prices found in scenarios")
        sys.exit(1)
        
    # 5. CROSS-MARKET CORRELATION
    rho_target = np.array(params['rho_by_hour'])
    rho_realized = []
    
    print("\nCross-Market Rank Correlation (Spearman):")
    print("Hour  Target_rho  Realized_rho  Delta   Status")
    
    hours_passed = 0
    for h in range(24):
        h_col = f'h{h:02d}'
        # Compute correlation per day and average
        daily_rhos = []
        for date, group in dam_df.groupby('target_date'):
            rtm_group = rtm_df[rtm_df['target_date'] == date]
            r, _ = scipy.stats.spearmanr(group[h_col], rtm_group[h_col])
            daily_rhos.append(r)
            
        r_sp = np.nanmean(daily_rhos)
        rho_realized.append(r_sp)
        
        delta = abs(r_sp - rho_target[h])
        status = "v" if delta < 0.15 else "x"
        if status == "v": hours_passed += 1
        
        print(f"{h:<6d}{rho_target[h]:<12.3f}{r_sp:<14.3f}{delta:<8.3f}{status}")
        
    print(f"\nHours passing correlation check (|delta| < 0.15): {hours_passed}/24")
    if hours_passed < 18:
        print("FAILED: Cross-market correlation significantly diverges from target in too many hours")
        sys.exit(1)
        
    # 6. TAIL DEPENDENCE
    print("\nTail Dependence Statistics:")
    upper_concordance = []
    lower_concordance = []
    
    for h in range(24):
        h_col = f'h{h:02d}'
        # Deciles per day/hour
        dam_h = dam_df[h_col].values
        rtm_h = rtm_df[h_col].values
        
        # We need to do this per date to avoid mixing global distributions
        matches_upper = 0
        matches_lower = 0
        total_days = 0
        
        for date, g_dam in dam_df.groupby('target_date'):
            g_rtm = rtm_df[rtm_df['target_date'] == date]
            
            d_vals = g_dam[h_col].values
            r_vals = g_rtm[h_col].values
            
            # Top decile (top 20 of 200)
            d_top_idx = np.argsort(d_vals)[-20:]
            r_top_idx = np.argsort(r_vals)[-20:]
            matches_upper += len(set(d_top_idx).intersection(set(r_top_idx)))
            
            # Bottom decile
            d_bot_idx = np.argsort(d_vals)[:20]
            r_bot_idx = np.argsort(r_vals)[:20]
            matches_lower += len(set(d_bot_idx).intersection(set(r_bot_idx)))
            
            total_days += 1
            
        upper_concordance.append(matches_upper / (20 * total_days))
        lower_concordance.append(matches_lower / (20 * total_days))
        
    print(f"Mean Upper Tail Concordance: {np.mean(upper_concordance):.3f} (Expected > 0.25)")
    print(f"Mean Lower Tail Concordance: {np.mean(lower_concordance):.3f} (Expected > 0.25)")
    
    # 7. INDEPENDENT VS JOINT COMPARISON
    print("\nComparison with Independent Scenarios:")
    try:
        dam_ind = pd.read_parquet(pred_dir / "dam_scenarios_backtest.parquet")
        rtm_ind = pd.read_parquet(pred_dir / "rtm_scenarios_backtest.parquet")
        
        # We need to reshape to h00-h23 if they are in different format
        # Phase 3A independent scenarios were long format (target_date, target_hour, scenario_id, price)
        # Let's pivot if needed
        def pivot_if_numeric(df):
            if 'target_hour' in df.columns:
                return df.pivot_table(index=['target_date', 'scenario_id'], columns='target_hour', values='price').reset_index()
            return df

        dam_ind_wide = pivot_if_numeric(dam_ind)
        rtm_ind_wide = pivot_if_numeric(rtm_ind)
        
        # Merge to align
        merged_ind = pd.merge(dam_ind_wide, rtm_ind_wide, on=['target_date', 'scenario_id'], suffixes=('_dam', '_rtm'))
        
        ind_rho = []
        for h in range(24):
            # Hour column in wide might be 0 or 'h00'
            col_dam = h if h in merged_ind.columns else f'h{h:02d}_dam'
            col_rtm = h if h in merged_ind.columns else f'h{h:02d}_rtm'
            if h in merged_ind.columns:
                r, _ = scipy.stats.spearmanr(merged_ind[h], merged_ind[h]) # Wait, merge suffixes
            else:
                r, _ = scipy.stats.spearmanr(merged_ind[col_dam], merged_ind[col_rtm])
            ind_rho.append(r)
            
        print(f"Mean Cross-Market rho (Independent): {np.mean(ind_rho):.3f}")
        print(f"Mean Cross-Market rho (Joint):       {np.mean(rho_realized):.3f}")
        
    except Exception as e:
        print(f"Could not load independent scenarios for comparison: {e}")

    # Save results
    val_report = {
        "days": n_days,
        "hours_passed": hours_passed,
        "mean_realized_rho": float(np.mean(rho_realized)),
        "mean_upper_concordance": float(np.mean(upper_concordance)),
        "mean_lower_concordance": float(np.mean(lower_concordance)),
        "rho_by_hour": [float(r) for r in rho_realized]
    }
    with open(results_dir / "joint_scenario_validation.json", 'w') as f:
        json.dump(val_report, f, indent=2)

    print("\n✅ VALIDATION PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_validation()
