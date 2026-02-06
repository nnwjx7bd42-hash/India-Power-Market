"""
Load and preprocess POSOCO (GRID-INDIA) data
Converts daily data to hourly resolution using historical profiles
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_posoco_data(filepath='POSOCO_data.csv', start_date='2023-08-30', end_date='2026-01-28'):
    """
    Load POSOCO CSV data and extract India-level columns
    
    Parameters:
    -----------
    filepath : str
        Path to POSOCO_data.csv
    start_date : str
        Start date filter (YYYY-MM-DD)
    end_date : str
        End date filter (YYYY-MM-DD)
    
    Returns:
    --------
    pd.DataFrame
        Daily POSOCO data with India-level columns
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"POSOCO file not found: {filepath}")
        return None
    
    print(f"Loading POSOCO data from {filepath}...")
    
    # Read CSV
    df = pd.read_csv(filepath)
    
    # Parse date (avoid fragmentation warning by creating new dataframe)
    df = df.copy()
    df['date'] = pd.to_datetime(df['yyyymmdd'].astype(str), format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Filter by date range
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    df = df[(df['date'] >= start_ts) & (df['date'] <= end_ts)]
    df = df.sort_values('date')
    
    # Set date as index
    df = df.set_index('date')
    
    # Set timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')
    
    print(f"  Loaded {len(df):,} daily records")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def extract_india_level_columns(posoco_df):
    """
    Extract India-level columns from POSOCO data
    
    Parameters:
    -----------
    posoco_df : pd.DataFrame
        POSOCO dataframe
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with India-level columns only
    """
    # India-level columns
    india_cols = [col for col in posoco_df.columns if col.startswith('India:')]
    
    if not india_cols:
        print("  WARNING: No India-level columns found")
        return None
    
    # Extract India columns
    india_df = posoco_df[india_cols].copy()
    
    # Rename columns (remove 'India:' prefix)
    rename_dict = {col: col.replace('India: ', '') for col in india_cols}
    india_df = india_df.rename(columns=rename_dict)
    
    print(f"  Extracted {len(india_cols)} India-level columns")
    
    return india_df


def extract_hourly_profiles(nerdc_df):
    """
    Extract hourly load and generation profiles from NERLDC data
    
    Parameters:
    -----------
    nerdc_df : pd.DataFrame
        Historical NERLDC hourly data
    
    Returns:
    --------
    dict
        Dictionary with profiles for each variable
    """
    if nerdc_df is None or len(nerdc_df) == 0:
        return None
    
    profiles = {}
    
    # Extract day-of-week patterns for demand
    if 'Demand' in nerdc_df.columns:
        nerdc_df['DayOfWeek'] = nerdc_df.index.dayofweek
        nerdc_df['Hour'] = nerdc_df.index.hour
        
        # Calculate average hourly pattern by day of week
        demand_profile = nerdc_df.groupby(['DayOfWeek', 'Hour'])['Demand'].mean().unstack(level=0)
        profiles['Demand'] = demand_profile
    
    # Extract hourly patterns for generation types
    gen_types = ['Thermal', 'Hydro', 'Nuclear', 'Gas', 'Wind', 'Solar']
    
    for gen_type in gen_types:
        if gen_type in nerdc_df.columns:
            nerdc_df['Hour'] = nerdc_df.index.hour
            nerdc_df['DayOfWeek'] = nerdc_df.index.dayofweek
            
            # For Solar: use hour pattern (daylight hours)
            if gen_type == 'Solar':
                solar_profile = nerdc_df.groupby('Hour')[gen_type].mean()
                profiles['Solar'] = solar_profile
            # For Wind: use hour pattern
            elif gen_type == 'Wind':
                wind_profile = nerdc_df.groupby('Hour')[gen_type].mean()
                profiles['Wind'] = wind_profile
            # For others: use day-of-week and hour pattern
            else:
                gen_profile = nerdc_df.groupby(['DayOfWeek', 'Hour'])[gen_type].mean().unstack(level=0)
                profiles[gen_type] = gen_profile
    
    print(f"  Extracted profiles for: {list(profiles.keys())}")
    
    return profiles


def convert_daily_to_hourly(daily_df, profiles=None):
    """
    Convert daily POSOCO data to hourly resolution using profiles
    
    Parameters:
    -----------
    daily_df : pd.DataFrame
        Daily POSOCO data
    profiles : dict, optional
        Hourly profiles from historical NERLDC data
    
    Returns:
    --------
    pd.DataFrame
        Hourly data
    """
    print("Converting daily data to hourly resolution...")
    
    # Create hourly timestamp range
    start_date = daily_df.index.min().date()
    end_date = daily_df.index.max().date()
    hourly_range = pd.date_range(
        start=pd.Timestamp(start_date, tz='Asia/Kolkata'),
        end=pd.Timestamp(end_date, tz='Asia/Kolkata') + pd.Timedelta(days=1) - pd.Timedelta(hours=1),
        freq='h',  # Hourly frequency
        tz='Asia/Kolkata'
    )
    
    # Create hourly dataframe with date and hour info
    hourly_df = pd.DataFrame(index=hourly_range)
    hourly_df['date'] = hourly_df.index.date
    hourly_df['hour'] = hourly_df.index.hour
    hourly_df['day_of_week'] = hourly_df.index.dayofweek
    
    # Map daily values to hourly using vectorized operations
    for col in daily_df.columns:
        if col == 'yyyymmdd':
            continue
        
        # Determine profile column name
        profile_col = col
        if 'WindGen' in col:
            profile_col = 'Wind'
        elif 'SolarGen' in col:
            profile_col = 'Solar'
        elif 'DemandMet' in col:
            profile_col = 'Demand'
        
        # Get daily values for each date using merge (more efficient)
        # Create a mapping from date to daily value
        date_to_value = {}
        for date_idx in daily_df.index:
            date_only = date_idx.date()
            date_to_value[date_only] = daily_df.loc[date_idx, col]
        
        # Map daily values to hourly rows
        hourly_values = hourly_df['date'].map(date_to_value)
        hourly_df[col] = hourly_values.values
        
        # Apply profiles if available
        if profiles and profile_col in profiles:
            profile = profiles[profile_col]
            
            if isinstance(profile, pd.Series):
                # Simple hourly profile (Solar, Wind)
                profile_mean = profile.mean()
                if profile_mean > 0:
                    # Map hour to profile factor
                    hour_factors = hourly_df['hour'].map(lambda h: profile.get(h, profile_mean) / profile_mean)
                    hourly_df[col] = hourly_df[col] * hour_factors
            elif isinstance(profile, pd.DataFrame):
                # Day-of-week and hour profile
                profile_mean = profile.mean().mean()
                if profile_mean > 0:
                    # Vectorized lookup using merge
                    profile_flat = profile.stack().reset_index()
                    profile_flat.columns = ['hour', 'day_of_week', 'value']
                    profile_flat['factor'] = profile_flat['value'] / profile_mean
                    
                    # Merge with hourly_df
                    merge_df = hourly_df[['hour', 'day_of_week']].merge(
                        profile_flat[['hour', 'day_of_week', 'factor']],
                        on=['hour', 'day_of_week'],
                        how='left'
                    )
                    merge_df['factor'] = merge_df['factor'].fillna(1.0)
                    hourly_df[col] = hourly_df[col] * merge_df['factor'].values
        
        # Handle special cases
        if 'EnergyMet' in col:
            # Daily energy -> hourly average (divide by 24)
            hourly_df[col] = hourly_df[col] / 24
        elif 'DemandMet' not in col and 'MaximumDemand' not in col and 'EnergyMet' not in col:
            # For generation columns without profiles, distribute evenly
            if profiles is None or profile_col not in profiles:
                hourly_df[col] = hourly_df[col] / 24
    
    # Remove helper columns
    hourly_df = hourly_df.drop(columns=['date', 'hour', 'day_of_week'], errors='ignore')
    
    print(f"  Created {len(hourly_df):,} hourly records")
    print(f"  Date range: {hourly_df.index.min()} to {hourly_df.index.max()}")
    
    return hourly_df


def map_posoco_to_nerldc_format(posoco_hourly_df):
    """
    Map POSOCO columns to NERLDC format
    
    Parameters:
    -----------
    posoco_hourly_df : pd.DataFrame
        Hourly POSOCO data
    
    Returns:
    --------
    pd.DataFrame
        Data in NERLDC format
    """
    print("Mapping POSOCO columns to NERLDC format...")
    
    # Column mapping (POSOCO column -> NERLDC column)
    column_mapping = {
        'DemandMet': 'Demand',  # Use DemandMet as Demand
        'Thermal': 'Thermal',   # India: Thermal -> Thermal
        'Hydro': 'Hydro',       # India: Hydro -> Hydro (not HydroGen)
        'Nuclear': 'Nuclear',
        'Gas': 'Gas',
        'WindGen': 'Wind',      # India: WindGen -> Wind
        'SolarGen': 'Solar',    # India: SolarGen -> Solar
    }
    
    # Note: POSOCO has both 'Hydro' (generation total) and 'HydroGen' (renewable hydro)
    # We use 'Hydro' which matches NERLDC format better
    
    # Alternative: use EnergyMet/24 for Demand if DemandMet not available
    if 'DemandMet' not in posoco_hourly_df.columns and 'EnergyMet' in posoco_hourly_df.columns:
        posoco_hourly_df['DemandMet'] = posoco_hourly_df['EnergyMet'] / 24
    
    # Create output dataframe
    nerdc_format_df = pd.DataFrame(index=posoco_hourly_df.index)
    
    # Map columns
    for posoco_col, nerdc_col in column_mapping.items():
        if posoco_col in posoco_hourly_df.columns:
            nerdc_format_df[nerdc_col] = posoco_hourly_df[posoco_col]
        else:
            print(f"  WARNING: {posoco_col} not found in POSOCO data")
    
    # Ensure all required columns exist
    required_cols = ['Demand', 'Thermal', 'Hydro', 'Gas', 'Nuclear', 'Wind', 'Solar']
    for col in required_cols:
        if col not in nerdc_format_df.columns:
            print(f"  WARNING: {col} not available after mapping")
    
    print(f"  Mapped to NERLDC format: {list(nerdc_format_df.columns)}")
    
    return nerdc_format_df


def load_posoco_hourly_data(posoco_file='POSOCO_data.csv', 
                            start_date='2023-08-30', 
                            end_date='2026-01-28',
                            nerdc_df=None):
    """
    Main function to load POSOCO data and convert to hourly NERLDC format
    
    Parameters:
    -----------
    posoco_file : str
        Path to POSOCO_data.csv
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    nerdc_df : pd.DataFrame, optional
        Historical NERLDC data for profile extraction
    
    Returns:
    --------
    pd.DataFrame
        Hourly data in NERLDC format
    """
    print("="*80)
    print("LOADING POSOCO DATA")
    print("="*80)
    
    # Load POSOCO data
    posoco_daily = load_posoco_data(posoco_file, start_date, end_date)
    
    if posoco_daily is None or len(posoco_daily) == 0:
        return None
    
    # Extract India-level columns
    india_daily = extract_india_level_columns(posoco_daily)
    
    if india_daily is None:
        return None
    
    # Extract profiles from NERLDC data
    profiles = None
    if nerdc_df is not None:
        print("\nExtracting hourly profiles from NERLDC data...")
        profiles = extract_hourly_profiles(nerdc_df)
    
    # Convert daily to hourly
    india_hourly = convert_daily_to_hourly(india_daily, profiles)
    
    # Map to NERLDC format
    nerdc_format = map_posoco_to_nerldc_format(india_hourly)
    
    print(f"\n✓ POSOCO data loaded and converted")
    print(f"  Shape: {nerdc_format.shape}")
    print(f"  Date range: {nerdc_format.index.min()} to {nerdc_format.index.max()}")
    
    return nerdc_format


if __name__ == "__main__":
    # Test loading
    posoco_df = load_posoco_hourly_data()
    
    if posoco_df is not None:
        print("\nSample data:")
        print(posoco_df.head())
        print("\nData summary:")
        print(posoco_df.describe())
