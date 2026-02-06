"""
Supply-side breakdown estimation from demand forecast
Estimates Thermal, Hydro, Wind, Solar, Gas, Nuclear from demand and weather forecasts
"""

import pandas as pd
import numpy as np
from datetime import datetime


def estimate_renewable_generation(demand_df, weather_df, historical_df=None):
    """
    Estimate renewable generation (Wind, Solar) from weather forecasts
    
    Parameters:
    -----------
    demand_df : pd.DataFrame
        Hourly demand forecast (index: timestamp, column: load_mw)
    weather_df : pd.DataFrame
        Weather forecast with columns: direct_radiation_national, wind_speed_10m_national
    historical_df : pd.DataFrame, optional
        Historical data to learn relationships
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Wind and Solar generation estimates
    """
    # Merge demand and weather on timestamp
    # Use merge_asof to handle timestamp misalignment (weather API returns forecasts from "now")
    # Sort both dataframes for merge_asof
    demand_sorted = demand_df.sort_index()
    weather_sorted = weather_df[['direct_radiation_national', 'wind_speed_10m_national']].sort_index()
    
    # Use merge_asof to find nearest weather data for each demand timestamp
    merged = pd.merge_asof(
        demand_sorted,
        weather_sorted,
        left_index=True,
        right_index=True,
        direction='nearest',
        tolerance=pd.Timedelta('1h')  # Allow 1 hour tolerance
    )
    
    # If merge_asof fails (no overlap), try reindexing weather to match demand timestamps
    if merged[['direct_radiation_national', 'wind_speed_10m_national']].isnull().all().all():
        # Reindex weather to match demand timestamps (for testing with past dates)
        weather_reindexed = weather_sorted.reindex(demand_sorted.index, method='nearest')
        merged = demand_sorted.copy()
        merged['direct_radiation_national'] = weather_reindexed['direct_radiation_national'].values
        merged['wind_speed_10m_national'] = weather_reindexed['wind_speed_10m_national'].values
    
    # Extract hour and month for patterns
    merged['Hour'] = merged.index.hour
    merged['Month'] = merged.index.month
    
    # Estimate Solar generation from direct radiation
    # Simple model: Solar ≈ f(direct_radiation, hour, season)
    # Use historical relationship if available
    if historical_df is not None and 'Solar' in historical_df.columns and 'direct_radiation_national' in historical_df.columns:
        # Learn relationship from historical data
        hist_merged = historical_df[['Solar', 'direct_radiation_national', 'Hour', 'Month']].copy()
        hist_merged = hist_merged.dropna()
        
        if len(hist_merged) > 0:
            # Calculate ratio: Solar / direct_radiation (avoid division by zero)
            hist_merged['ratio'] = hist_merged['Solar'] / (hist_merged['direct_radiation_national'] + 1e-6)
            # Cap ratio at reasonable maximum (e.g., 0.5 MW per W/m²)
            hist_merged['ratio'] = hist_merged['ratio'].clip(0, 0.5)
            
            # Group by hour and month, calculate median ratio (more robust than mean)
            solar_factors = hist_merged.groupby(['Hour', 'Month'])['ratio'].median()
            
            # Apply factors
            merged['Solar'] = merged.apply(
                lambda row: row['direct_radiation_national'] * solar_factors.get(
                    (row['Hour'], row['Month']), 
                    solar_factors.median() if len(solar_factors) > 0 else 0.05
                ) if pd.notna(row['direct_radiation_national']) else 0,
                axis=1
            )
        else:
            # Fallback: simple linear relationship
            merged['Solar'] = np.where(
                (merged['Hour'] >= 6) & (merged['Hour'] <= 18) & pd.notna(merged['direct_radiation_national']),
                merged['direct_radiation_national'] * 0.05,  # Conservative factor
                0
            )
    else:
        # Fallback: simple model
        # Solar generation is proportional to direct radiation, but only during daylight hours
        merged['Solar'] = np.where(
            (merged['Hour'] >= 6) & (merged['Hour'] <= 18) & pd.notna(merged['direct_radiation_national']),
            merged['direct_radiation_national'] * 0.05,  # Conservative conversion factor
            0
        )
    
    # Estimate Wind generation from wind speed
    # Simple model: Wind ≈ f(wind_speed^3, hour)
    if historical_df is not None and 'Wind' in historical_df.columns and 'wind_speed_10m_national' in historical_df.columns:
        # Learn relationship from historical data
        hist_merged = historical_df[['Wind', 'wind_speed_10m_national', 'Hour']].copy()
        hist_merged = hist_merged.dropna()
        
        if len(hist_merged) > 0:
            # Calculate wind_speed^3
            hist_merged['wind_speed_cubed'] = hist_merged['wind_speed_10m_national'] ** 3
            # Calculate ratio: Wind / wind_speed^3
            hist_merged['ratio'] = hist_merged['Wind'] / (hist_merged['wind_speed_cubed'] + 1e-6)
            # Cap ratio at reasonable maximum
            hist_merged['ratio'] = hist_merged['ratio'].clip(0, 100)  # Max 100 MW per (m/s)^3
            
            # Group by hour, calculate median ratio
            wind_factors = hist_merged.groupby('Hour')['ratio'].median()
            
            # Apply factors
            merged['Wind'] = merged.apply(
                lambda row: (row['wind_speed_10m_national'] ** 3) * wind_factors.get(
                    row['Hour'],
                    wind_factors.median() if len(wind_factors) > 0 else 0.5
                ) if pd.notna(row['wind_speed_10m_national']) else 0,
                axis=1
            )
        else:
            # Fallback: simple cubic relationship
            merged['Wind'] = np.where(
                pd.notna(merged['wind_speed_10m_national']),
                (merged['wind_speed_10m_national'] ** 3) * 0.5,  # Conservative factor
                0
            )
    else:
        # Fallback: simple cubic relationship
        merged['Wind'] = np.where(
            pd.notna(merged['wind_speed_10m_national']),
            (merged['wind_speed_10m_national'] ** 3) * 0.5,  # Conservative factor
            0
        )
    
    # Ensure non-negative and fill any remaining NaN with 0
    merged['Solar'] = np.maximum(0, merged['Solar'].fillna(0))
    merged['Wind'] = np.maximum(0, merged['Wind'].fillna(0))
    
    # Cap at reasonable maximums (e.g., 50 GW for solar, 20 GW for wind in India)
    merged['Solar'] = merged['Solar'].clip(0, 50000)
    merged['Wind'] = merged['Wind'].clip(0, 20000)
    
    return merged[['Solar', 'Wind']]


def estimate_conventional_generation(demand_df, re_generation_df, historical_df=None):
    """
    Estimate conventional generation (Hydro, Gas, Nuclear) and Thermal
    
    Parameters:
    -----------
    demand_df : pd.DataFrame
        Hourly demand forecast
    re_generation_df : pd.DataFrame
        Renewable generation (Wind, Solar)
    historical_df : pd.DataFrame, optional
        Historical data for averages
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Thermal, Hydro, Gas, Nuclear estimates
    """
    # Start with demand
    result = demand_df.copy()
    result = result.rename(columns={'load_mw': 'Demand'})
    
    # Add renewable generation
    result['RE_Generation'] = re_generation_df['Wind'] + re_generation_df['Solar']
    result['Wind'] = re_generation_df['Wind']
    result['Solar'] = re_generation_df['Solar']
    
    # Estimate Hydro, Gas, Nuclear from historical averages
    if historical_df is not None:
        hist_cols = ['Hydro', 'Gas', 'Nuclear']
        available_cols = [col for col in hist_cols if col in historical_df.columns]
        
        if available_cols:
            # Calculate average by hour of day (these are less variable)
            result['Hour'] = result.index.hour
            
            for col in available_cols:
                hist_hourly_avg = historical_df.groupby(historical_df.index.hour)[col].mean()
                result[col] = result['Hour'].map(hist_hourly_avg)
                result[col] = result[col].fillna(historical_df[col].mean())
        else:
            # Use overall averages
            for col in hist_cols:
                if col in historical_df.columns:
                    result[col] = historical_df[col].mean()
                else:
                    result[col] = 0
    else:
        # Default estimates (rough averages for India)
        result['Hydro'] = result['Demand'] * 0.10  # ~10% of demand
        result['Gas'] = result['Demand'] * 0.05     # ~5% of demand
        result['Nuclear'] = result['Demand'] * 0.03  # ~3% of demand
    
    # Thermal is residual: Demand - RE - Hydro - Gas - Nuclear
    result['Thermal'] = (
        result['Demand'] 
        - result['RE_Generation'] 
        - result['Hydro'] 
        - result['Gas'] 
        - result['Nuclear']
    )
    
    # Ensure non-negative
    result['Thermal'] = np.maximum(0, result['Thermal'])
    
    # Drop helper column
    if 'Hour' in result.columns:
        result = result.drop(columns=['Hour'])
    
    return result


def estimate_supply_breakdown(demand_df, weather_df, historical_df=None):
    """
    Estimate complete supply-side breakdown from demand forecast
    
    Parameters:
    -----------
    demand_df : pd.DataFrame
        Hourly demand forecast (index: timestamp, column: load_mw)
    weather_df : pd.DataFrame
        Weather forecast with national-level variables
    historical_df : pd.DataFrame, optional
        Historical data to learn relationships
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all supply-side variables:
        Demand, Thermal, Hydro, Gas, Nuclear, Wind, Solar, RE_Generation
    """
    print("\n" + "="*80)
    print("ESTIMATING SUPPLY-SIDE BREAKDOWN")
    print("="*80)
    
    # Step 1: Estimate renewable generation
    print("\n1. Estimating renewable generation (Wind, Solar)...")
    re_generation = estimate_renewable_generation(demand_df, weather_df, historical_df)
    
    print(f"   Solar: {re_generation['Solar'].mean():,.0f} MW (avg), {re_generation['Solar'].max():,.0f} MW (max)")
    print(f"   Wind:  {re_generation['Wind'].mean():,.0f} MW (avg), {re_generation['Wind'].max():,.0f} MW (max)")
    
    # Step 2: Estimate conventional generation
    print("\n2. Estimating conventional generation (Hydro, Gas, Nuclear, Thermal)...")
    supply_df = estimate_conventional_generation(demand_df, re_generation, historical_df)
    
    print(f"   Hydro:   {supply_df['Hydro'].mean():,.0f} MW (avg)")
    print(f"   Gas:     {supply_df['Gas'].mean():,.0f} MW (avg)")
    print(f"   Nuclear: {supply_df['Nuclear'].mean():,.0f} MW (avg)")
    print(f"   Thermal: {supply_df['Thermal'].mean():,.0f} MW (avg), {supply_df['Thermal'].max():,.0f} MW (max)")
    
    # Calculate RE Penetration
    supply_df['RE_Penetration'] = supply_df['RE_Generation'] / supply_df['Demand'].replace(0, np.nan)
    supply_df['RE_Penetration'] = supply_df['RE_Penetration'].fillna(0)
    
    # Net Load
    supply_df['Net_Load'] = supply_df['Demand'] - supply_df['RE_Generation']
    
    # Total Generation
    gen_cols = ['Thermal', 'Hydro', 'Gas', 'Nuclear', 'Wind', 'Solar']
    supply_df['Total_Generation'] = supply_df[gen_cols].sum(axis=1)
    
    # Thermal Share
    supply_df['Thermal_Share'] = supply_df['Thermal'] / supply_df['Total_Generation'].replace(0, np.nan)
    supply_df['Thermal_Share'] = supply_df['Thermal_Share'].fillna(0)
    
    print(f"\n✓ Supply breakdown estimated")
    print(f"   RE Penetration: {supply_df['RE_Penetration'].mean():.2%} (avg)")
    print(f"   Net Load: {supply_df['Net_Load'].mean():,.0f} MW (avg)")
    
    return supply_df
