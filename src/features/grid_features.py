import pandas as pd
import numpy as np

def build_grid_features(grid_df):
    """
    Build grid features.
    Input: hourly grid data.
    Output: DataFrame indexed on delivery_start_ist.
    """
    df = grid_df.set_index('delivery_start_ist').sort_index()
    feats = pd.DataFrame(index=df.index)
    
    # Passthrough Features
    # Renaming as per spec
    feats['grid_demand_mw'] = df['all_india_demand_mw']
    feats['grid_net_demand_mw'] = df['net_demand_mw']
    feats['grid_solar_mw'] = df['all_india_solar_mw']
    feats['grid_wind_mw'] = df['all_india_wind_mw']
    feats['grid_total_gen_mw'] = df['total_generation_mw']
    
    if 'fuel_mix_imputed' in df.columns:
        feats['grid_fuel_mix_imputed'] = df['fuel_mix_imputed'].astype(int) # bool to int for safety
    else:
        feats['grid_fuel_mix_imputed'] = 0
            
    # Derived Features
    # Delta 1h
    feats['grid_net_demand_delta_1h'] = feats['grid_net_demand_mw'] - feats['grid_net_demand_mw'].shift(1)
    
    # Lag 24h
    feats['grid_net_demand_lag_24h'] = feats['grid_net_demand_mw'].shift(24)
    
    # Solar Ramp 1h
    feats['grid_solar_ramp_1h'] = feats['grid_solar_mw'] - feats['grid_solar_mw'].shift(1)
    
    # Demand-Gen Gap
    feats['grid_demand_gen_gap'] = feats['grid_demand_mw'] - feats['grid_total_gen_mw']
    
    # Thermal Utilization
    # Using 'total_thermal_mw' from grid columns
    thermal_col = 'total_thermal_mw'
    if thermal_col in df.columns:
        thermal = df[thermal_col]
        # Check if missing/zero and impute?
        # If thermal is 0, it might be missing data or actually 0 (unlikely for India).
        # But we'll trust the data.
    else:
        # Fallback if somehow missing
        thermal = df['grid_total_gen_mw'] - (df['grid_solar_mw'] + df['grid_wind_mw'])
    
    feats['grid_thermal_util'] = thermal / 180000.0
    
    # Renewable share
    feats['grid_renewable_share'] = (feats['grid_solar_mw'] + feats['grid_wind_mw']) / feats['grid_demand_mw']
    feats['grid_renewable_share'] = feats['grid_renewable_share'].replace([np.inf, -np.inf], 0).fillna(0)
    
    return feats
