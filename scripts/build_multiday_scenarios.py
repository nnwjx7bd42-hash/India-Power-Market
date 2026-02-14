"""
Build multi-day joint scenarios for Rolling Horizon and Extensive Form optimizers.

For each backtest date D, generates:
  - Day D:   (DAM, RTM) scenarios using existing Day D quantile predictions
  - Day D+1: (DAM, RTM) scenarios using Day D+1 quantile predictions
  - Days D+2..D+6: (DAM, RTM) scenarios using climatological fallback

Cross-day correlation: latent Gaussian z-vectors are correlated across days via AR(1).
"""
import pandas as pd
import numpy as np
import json
import scipy.stats
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.scenarios.joint_copula import (
    generate_correlated_uniforms, inverse_cdf_vectorized,
    generate_multiday_latent
)


def get_climatological_quantiles(date: str, hour: int, dam_preds_df: pd.DataFrame,
                                  lookback_days: int = 60) -> dict:
    """
    Expanding window: median of quantile predictions for same hour
    over the past `lookback_days` days.
    """
    cutoff = pd.Timestamp(date) - pd.Timedelta(days=1)
    start = cutoff - pd.Timedelta(days=lookback_days)
    
    mask = (
        (dam_preds_df['target_date'] >= str(start.date()))
        & (dam_preds_df['target_date'] <= str(cutoff.date()))
        & (dam_preds_df['target_hour'] == hour)
    )
    
    subset = dam_preds_df[mask]
    if len(subset) < 5:
        # Not enough history — widen to all available
        mask_all = (
            (dam_preds_df['target_date'] <= str(cutoff.date()))
            & (dam_preds_df['target_hour'] == hour)
        )
        subset = dam_preds_df[mask_all]
    
    return {k: float(subset[k].median()) for k in ['q10', 'q25', 'q50', 'q75', 'q90']}


def run_build_multiday_scenarios():
    parser = argparse.ArgumentParser(description='Build multi-day scenarios')
    parser.add_argument('--n-scenarios', type=int, default=200)
    parser.add_argument('--n-days', type=int, default=7)
    args = parser.parse_args()
    
    print("============================================================")
    print(f"BUILDING MULTI-DAY JOINT SCENARIOS ({args.n_days}-DAY, {args.n_scenarios} scenarios)")
    print("============================================================")
    
    pred_dir = Path("Data/Predictions")
    results_dir = Path("results")
    params_path = results_dir / "joint_copula_params.json"
    
    if not params_path.exists():
        print(f"Error: Joint copula parameters not found at {params_path}")
        print("Run: python scripts/fit_joint_copula.py first")
        return
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    rho_by_hour = params['rho_by_hour']
    dam_corr = np.array(params['dam_copula_correlation'])
    cross_day_rho = params.get('cross_day_rho', 0.3)  # fallback
    
    print(f"Using cross-day rho: {cross_day_rho:.3f}")
    
    # Load predictions
    print("Loading predictions...")
    dam_preds = pd.read_parquet(pred_dir / "dam_quantiles_backtest.parquet")
    rtm_preds = pd.read_parquet(pred_dir / "rtm_quantiles_backtest.parquet")
    
    # Load D+1 predictions if available
    d1_path = pred_dir / "dam_d1_quantiles_backtest.parquet"
    dam_d1_preds = None
    if d1_path.exists():
        dam_d1_preds = pd.read_parquet(d1_path)
        print(f"Loaded D+1 DAM predictions: {len(dam_d1_preds)} rows")
    else:
        print("WARNING: D+1 DAM predictions not found. Using climatological for Day D+1.")
    
    # Normalize RTM dates
    if 'target_hour' not in rtm_preds.columns:
        if 'delivery_start_ist' in rtm_preds.columns:
            rtm_preds['target_hour'] = pd.to_datetime(rtm_preds['delivery_start_ist']).dt.hour
            rtm_preds['target_date'] = pd.to_datetime(rtm_preds['delivery_start_ist']).dt.date.astype(str)
    
    # Find common dates with 24 hours
    dam_dates = set(dam_preds.groupby('target_date').filter(lambda x: len(x) == 24)['target_date'].unique())
    rtm_dates = set(rtm_preds.groupby('target_date').filter(lambda x: len(x) == 24)['target_date'].unique())
    common_dates = sorted(list(dam_dates.intersection(rtm_dates)))
    n_backtest_days = len(common_dates)
    
    print(f"Backtest dates: {n_backtest_days} days ({common_dates[0]} to {common_dates[-1]})")
    
    n_scenarios = args.n_scenarios
    n_days = args.n_days
    
    dam_rows = []
    rtm_rows = []
    
    for date_idx, date in enumerate(common_dates):
        rng = np.random.default_rng(seed=42 + date_idx)
        
        # Generate multi-day correlated latent vectors
        z_all = generate_multiday_latent(n_scenarios, n_days, dam_corr, cross_day_rho, rng)
        
        for day_offset in range(n_days):
            # Determine quantile source for this day offset
            future_date_str = (pd.Timestamp(date) + pd.Timedelta(days=day_offset)).strftime('%Y-%m-%d')
            
            # Get quantiles for each hour
            if day_offset == 0:
                # Day D: use existing predictions
                dam_day = dam_preds[dam_preds['target_date'] == date].sort_values('target_hour')
                rtm_day = rtm_preds[rtm_preds['target_date'] == date].sort_values('target_hour')
                use_rtm_direct = True
            elif day_offset == 1 and dam_d1_preds is not None:
                # Day D+1: use D+1 predictions
                dam_day = dam_d1_preds[dam_d1_preds['target_date'] == future_date_str].sort_values('target_hour')
                if len(dam_day) < 24:
                    # D+1 prediction not available for this date — fall back to climatological
                    dam_day = None
                use_rtm_direct = False
            else:
                dam_day = None
                use_rtm_direct = False
            
            # Extract day's z-scores
            z_dam_day = z_all[:, day_offset*24:(day_offset+1)*24]
            u_dam = scipy.stats.norm.cdf(z_dam_day)
            
            # Generate correlated RTM uniforms
            u_rtm = generate_correlated_uniforms(z_dam_day, rho_by_hour, rng)
            
            # Map to prices
            dam_prices = np.zeros((n_scenarios, 24))
            rtm_prices = np.zeros((n_scenarios, 24))
            
            for h in range(24):
                # DAM quantiles
                if dam_day is not None and len(dam_day) >= 24:
                    row_d = dam_day.iloc[h]
                    q_d = {k: row_d[k] for k in ['q10', 'q25', 'q50', 'q75', 'q90']}
                else:
                    q_d = get_climatological_quantiles(date, h, dam_preds)
                
                dam_prices[:, h] = inverse_cdf_vectorized(u_dam[:, h], q_d)
                
                # RTM quantiles
                if use_rtm_direct and len(rtm_day) >= 24:
                    row_r = rtm_day.iloc[h]
                    q_r = {k: row_r[k] for k in ['q10', 'q25', 'q50', 'q75', 'q90']}
                else:
                    # Cross-market correlation from DAM
                    q_r = get_climatological_quantiles(date, h, rtm_preds)
                
                rtm_prices[:, h] = inverse_cdf_vectorized(u_rtm[:, h], q_r)
            
            # Clamp non-negative and handle NaN (boundary dates may lack quantiles)
            dam_median = np.nanmedian(dam_prices) if np.any(~np.isnan(dam_prices)) else 3500.0
            rtm_median = np.nanmedian(rtm_prices) if np.any(~np.isnan(rtm_prices)) else 3500.0
            dam_prices = np.nan_to_num(dam_prices, nan=dam_median)
            rtm_prices = np.nan_to_num(rtm_prices, nan=rtm_median)
            dam_prices = np.maximum(dam_prices, 0)
            rtm_prices = np.maximum(rtm_prices, 0)
            
            # Collect rows
            for s in range(n_scenarios):
                dam_rows.append({
                    'target_date': date,
                    'scenario_id': s,
                    'day_offset': day_offset,
                    **{f'h{h:02d}': float(dam_prices[s, h]) for h in range(24)}
                })
                rtm_rows.append({
                    'target_date': date,
                    'scenario_id': s,
                    'day_offset': day_offset,
                    **{f'h{h:02d}': float(rtm_prices[s, h]) for h in range(24)}
                })
        
        if (date_idx + 1) % 20 == 0:
            print(f"  Generated {date_idx + 1}/{n_backtest_days} dates")
    
    # Save
    print("Saving scenario files...")
    dam_df = pd.DataFrame(dam_rows)
    rtm_df = pd.DataFrame(rtm_rows)
    
    dam_df.to_parquet(pred_dir / "multiday_dam_scenarios_backtest.parquet", index=False)
    rtm_df.to_parquet(pred_dir / "multiday_rtm_scenarios_backtest.parquet", index=False)
    
    # Sanity checks
    print("\n─── SANITY CHECKS ───")
    
    # Non-negativity
    all_dam = dam_df.filter(regex='^h').values.flatten()
    all_rtm = rtm_df.filter(regex='^h').values.flatten()
    assert all_dam.min() >= 0, f"FAIL: negative DAM price {all_dam.min()}"
    assert all_rtm.min() >= 0, f"FAIL: negative RTM price {all_rtm.min()}"
    print(f"✓ Non-negativity: DAM min ₹{all_dam.min():.2f}, RTM min ₹{all_rtm.min():.2f}")
    
    # Cross-day correlation check (Day D avg vs Day D+1 avg)
    day0 = dam_df[dam_df['day_offset'] == 0].groupby('target_date').apply(
        lambda x: x.filter(regex='^h').values.mean()
    )
    day1 = dam_df[dam_df['day_offset'] == 1].groupby('target_date').apply(
        lambda x: x.filter(regex='^h').values.mean()
    )
    common = day0.index.intersection(day1.index)
    if len(common) > 5:
        realized_corr = np.corrcoef(
            [day0[d] for d in common],
            [day1[d] for d in common]
        )[0, 1]
        print(f"✓ Cross-day correlation (Day 0 vs Day 1): {realized_corr:.3f} (target rho: {cross_day_rho:.3f})")
    
    print(f"\nDAM scenarios: {len(dam_df)} rows ({n_backtest_days} dates × {n_days} days × {n_scenarios} scenarios)")
    print(f"RTM scenarios: {len(rtm_df)} rows")
    
    print(f"\nDAM stats: Mean ₹{all_dam.mean():.2f}  Std ₹{all_dam.std():.2f}  Min ₹{all_dam.min():.2f}  Max ₹{all_dam.max():.2f}")
    print(f"RTM stats: Mean ₹{all_rtm.mean():.2f}  Std ₹{all_rtm.std():.2f}  Min ₹{all_rtm.min():.2f}  Max ₹{all_rtm.max():.2f}")
    
    print(f"\nSaved to {pred_dir}")
    print("============================================================")


if __name__ == "__main__":
    run_build_multiday_scenarios()
