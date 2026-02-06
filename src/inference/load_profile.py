"""
Load profile conversion from daily peak to hourly profile
Uses historical patterns to convert daily peak values to hourly load shapes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def analyze_load_patterns(df, load_col='L(T-1)'):
    """
    Extract hourly load patterns by day of week from historical data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data with datetime index and load column
    load_col : str
        Name of load column (default: 'L(T-1)')
    
    Returns:
    --------
    dict
        Dictionary mapping day of week (0=Monday) to hourly load factors
        Format: {0: [24 factors], 1: [24 factors], ..., 6: [24 factors]}
    """
    if load_col not in df.columns:
        raise ValueError(f"Load column '{load_col}' not found in dataframe")
    
    print("Analyzing historical load patterns by day of week...")
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    # Add day of week (0=Monday, 6=Sunday)
    df_with_dow = df.copy()
    df_with_dow['DayOfWeek'] = df_with_dow.index.dayofweek
    df_with_dow['Hour'] = df_with_dow.index.hour
    
    # Group by day of week and hour
    patterns = {}
    
    for dow in range(7):  # 0-6 (Monday-Sunday)
        dow_data = df_with_dow[df_with_dow['DayOfWeek'] == dow]
        
        if len(dow_data) == 0:
            print(f"  Warning: No data for day of week {dow}, using average pattern")
            # Use average across all days
            hourly_means = df_with_dow.groupby('Hour')[load_col].mean()
        else:
            # Calculate average load by hour for this day of week
            hourly_means = dow_data.groupby('Hour')[load_col].mean()
        
        # Normalize to peak = 1.0
        peak_value = hourly_means.max()
        if peak_value > 0:
            hourly_factors = hourly_means / peak_value
        else:
            # Fallback: use uniform pattern if no data
            hourly_factors = pd.Series([1.0] * 24, index=range(24))
        
        patterns[dow] = hourly_factors.values.tolist()
        
        # Print pattern summary
        peak_hour = hourly_means.idxmax()
        print(f"  Day {dow} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]}): Peak at hour {peak_hour}, factors range: {hourly_factors.min():.3f} - {hourly_factors.max():.3f}")
    
    return patterns


def daily_peak_to_hourly(daily_peaks, start_date, load_patterns=None, historical_df=None):
    """
    Convert daily peak load values to hourly load profile
    
    Parameters:
    -----------
    daily_peaks : list or array
        List of daily peak load values (MW) for each day
    start_date : str or datetime
        Start date/time for first day (will start at 00:00 IST)
    load_patterns : dict, optional
        Pre-computed load patterns by day of week
        If None, will compute from historical_df
    historical_df : pd.DataFrame, optional
        Historical data to compute patterns if load_patterns not provided
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with hourly load profile
        Columns: timestamp, load_mw
    """
    # Parse start date
    if isinstance(start_date, str):
        start_dt = pd.to_datetime(start_date)
    else:
        start_dt = start_date
    
    # Ensure timezone
    if start_dt.tz is None:
        start_dt = start_dt.tz_localize('Asia/Kolkata')
    else:
        start_dt = start_dt.tz_convert('Asia/Kolkata')
    
    # Start at midnight
    start_dt = start_dt.normalize()  # Set to 00:00:00
    
    # Get load patterns if not provided
    if load_patterns is None:
        if historical_df is None:
            raise ValueError("Either load_patterns or historical_df must be provided")
        load_patterns = analyze_load_patterns(historical_df)
    
    # Generate hourly timestamps for all days
    num_days = len(daily_peaks)
    num_hours = num_days * 24
    
    timestamps = pd.date_range(
        start=start_dt,
        periods=num_hours,
        freq='h',
        tz='Asia/Kolkata'
    )
    
    # Generate hourly load values
    hourly_loads = []
    
    for day_idx, peak_load in enumerate(daily_peaks):
        day_start_hour = day_idx * 24
        
        # Get day of week (0=Monday, 6=Sunday)
        day_timestamp = timestamps[day_start_hour]
        day_of_week = day_timestamp.dayofweek
        
        # Get load pattern for this day of week
        pattern = load_patterns[day_of_week]
        
        # Apply pattern: hourly_load = pattern_factor * daily_peak
        for hour_idx in range(24):
            factor = pattern[hour_idx]
            hourly_load = factor * peak_load
            hourly_loads.append(hourly_load)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'load_mw': hourly_loads
    })
    
    df = df.set_index('timestamp')
    
    print(f"\n✓ Converted {num_days} daily peaks to {num_hours} hourly values")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Load range: {df['load_mw'].min():,.0f} - {df['load_mw'].max():,.0f} MW")
    
    return df
