import numpy as np
import scipy.stats
import pandas as pd
from typing import Dict, List, Union
from sklearn.covariance import LedoitWolf
from src.scenarios.utils import inverse_cdf

def compute_pit(actual: float, quantiles_dict: Dict[str, float]) -> float:
    """
    Compute Probability Integral Transform (PIT) residual.
    Maps actual price to (0, 1) space based on quantile predictions.
    """
    # Keys might be 'q10' or 0.1
    q_levels = [0.10, 0.25, 0.50, 0.75, 0.90]
    q_keys = ['q10', 'q25', 'q50', 'q75', 'q90']
    
    q_values = []
    for k in q_keys:
        val = quantiles_dict.get(k)
        if val is None:
            # Try float key
            val = quantiles_dict.get(float(k[1:])/100)
        q_values.append(val)
        
    # Enforce monotonicity
    q_values = np.maximum.accumulate(q_values)
    
    # Extend with pseudo-boundaries for better tail behavior
    q_levels_ext = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    q10, q25, q50, q75, q90 = q_values[0], q_values[1], q_values[2], q_values[3], q_values[4]
    
    low_ext = q10 - 1.5 * (q25 - q10)
    high_ext = q90 + 1.5 * (q90 - q75)
    
    q_values_ext = [low_ext, q10, q25, q50, q75, q90, high_ext]
    
    # Interpolate actual to get u
    u = np.interp(actual, q_values_ext, q_levels_ext)
    
    # Clip to avoid infinity in ppf
    return np.clip(u, 0.001, 0.999)

def fit_cross_market_rho(dam_val_df: pd.DataFrame, rtm_val_df: pd.DataFrame) -> dict:
    """
    Estimate per-hour cross-market correlation from validation PIT residuals.
    """
    # Ensure RTM has target_date and target_hour
    if 'target_hour' not in rtm_val_df.columns:
        if 'delivery_start_ist' in rtm_val_df.columns:
            rtm_val_df['target_hour'] = pd.to_datetime(rtm_val_df['delivery_start_ist']).dt.hour
            rtm_val_df['target_date'] = pd.to_datetime(rtm_val_df['delivery_start_ist']).dt.date.astype(str)
            
    # Normalize DAM date format
    dam_val_df['target_date'] = dam_val_df['target_date'].astype(str)
    
    # Merge on common date/hour
    merged = pd.merge(
        dam_val_df, rtm_val_df, 
        on=['target_date', 'target_hour'], 
        suffixes=('_dam', '_rtm')
    )
    
    z_dam_list = []
    z_rtm_list = []
    
    # Compute PIT for each row
    for _, row in merged.iterrows():
        q_dam = {k: row[f'{k}_dam'] for k in ['q10', 'q25', 'q50', 'q75', 'q90']}
        q_rtm = {k: row[f'{k}_rtm'] for k in ['q10', 'q25', 'q50', 'q75', 'q90']}
        
        # Get actuals - column name might vary
        act_dam = row.get('actual_mcp_dam', row.get('target_mcp_rs_mwh_dam'))
        act_rtm = row.get('actual_mcp_rtm', row.get('target_mcp_rs_mwh_rtm'))
        
        u_dam = compute_pit(act_dam, q_dam)
        u_rtm = compute_pit(act_rtm, q_rtm)
        
        z_dam_list.append(scipy.stats.norm.ppf(u_dam))
        z_rtm_list.append(scipy.stats.norm.ppf(u_rtm))
        
    merged['z_dam'] = z_dam_list
    merged['z_rtm'] = z_rtm_list
    
    # Hour-wise correlation
    rho_by_hour_raw = []
    n_obs_by_hour = []
    
    for h in range(24):
        hourly_data = merged[merged['target_hour'] == h]
        if len(hourly_data) > 1:
            r = np.corrcoef(hourly_data['z_dam'], hourly_data['z_rtm'])[0, 1]
            rho_by_hour_raw.append(float(r))
            n_obs_by_hour.append(len(hourly_data))
        else:
            rho_by_hour_raw.append(0.0)
            n_obs_by_hour.append(len(hourly_data))
            
    # Global rho
    global_rho = float(np.corrcoef(merged['z_dam'], merged['z_rtm'])[0, 1])
    
    # Shrinkage
    rho_by_hour = []
    shrink_factor = 0.3
    for h in range(24):
        if n_obs_by_hour[h] < 15:
            rho_h = global_rho
        else:
            rho_h = (1 - shrink_factor) * rho_by_hour_raw[h] + shrink_factor * global_rho
        rho_by_hour.append(np.clip(rho_h, -0.99, 0.99))
        
    return {
        'rho_by_hour': rho_by_hour,
        'rho_by_hour_raw': rho_by_hour_raw,
        'rho_global': global_rho,
        'n_observations_by_hour': n_obs_by_hour,
        'shrinkage_factor': shrink_factor,
        'n_total_observations': len(merged),
        'n_common_dates': len(merged['target_date'].unique())
    }

def estimate_dam_copula_correlation(dam_val_df: pd.DataFrame) -> np.ndarray:
    """
    Estimate the 24x24 DAM hour-to-hour copula correlation matrix.
    """
    dam_val_df['target_date'] = dam_val_df['target_date'].astype(str)
    
    z_data = []
    for _, row in dam_val_df.iterrows():
        q_dict = {k: row[k] for k in ['q10', 'q25', 'q50', 'q75', 'q90']}
        act = row.get('actual_mcp', row.get('target_mcp_rs_mwh'))
        u = compute_pit(act, q_dict)
        z = scipy.stats.norm.ppf(u)
        z_data.append({'date': row['target_date'], 'hour': row['target_hour'], 'z': z})
        
    z_df = pd.DataFrame(z_data)
    z_pivoted = z_df.pivot(index='date', columns='hour', values='z').dropna()
    
    # Ledoit-Wolf shrinkage
    lw = LedoitWolf().fit(z_pivoted.values)
    cov = lw.covariance_
    
    # Cov to Corr
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    
    # Ensure PSD
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    corr_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Renormalize diagonal to 1.0
    d = np.sqrt(np.diag(corr_psd))
    corr_final = corr_psd / np.outer(d, d)
    
    return corr_final

def generate_correlated_uniforms(z_dam: np.ndarray, rho_by_hour: List[float], rng: np.random.Generator) -> np.ndarray:
    """
    Generate RTM uniforms correlated with DAM latent Gaussians.
    z_dam shape: (n_scenarios, 24)
    """
    n_scenarios = z_dam.shape[0]
    z_rtm = np.zeros_like(z_dam)
    epsilon = rng.standard_normal((n_scenarios, 24))
    
    for h in range(24):
        rho = rho_by_hour[h]
        z_rtm[:, h] = rho * z_dam[:, h] + np.sqrt(1 - rho**2) * epsilon[:, h]
        
    return scipy.stats.norm.cdf(z_rtm)

def inverse_cdf_vectorized(u: np.ndarray, quantiles_dict: Dict[str, float]) -> np.ndarray:
    """
    Map an array of uniform values to prices using quantile predictions.
    Reuses existing inverse_cdf but prepared for simpler interface if needed.
    """
    # Map str keys to float for existing inverse_cdf if needed
    q_dict_float = {
        0.10: quantiles_dict['q10'],
        0.25: quantiles_dict['q25'],
        0.50: quantiles_dict['q50'],
        0.75: quantiles_dict['q75'],
        0.90: quantiles_dict['q90']
    }
    
    # Monotonicity check for values
    q_values = sorted(q_dict_float.values())
    q_levels = sorted(q_dict_float.keys())
    q_dict_sorted = dict(zip(q_levels, q_values))
    
    # We can use the existing inverse_cdf which already handles extrapolation
    return inverse_cdf(u, q_dict_sorted)
