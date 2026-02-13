import pandas as pd
import numpy as np

def build_calendar_features(timestamps, holidays_df):
    """
    Build calendar features.
    Input: Series of delivery_start_ist (timestamps), holidays DataFrame.
    Output: DataFrame indexed on delivery_start_ist.
    """
    # Ensure timestamps is a Series
    # specific fix: convert to DatetimeIndex to avoid index alignment issues with Series
    ts = pd.to_datetime(timestamps).dropna().sort_values()
    ts = pd.DatetimeIndex(ts)
    
    feats = pd.DataFrame(index=ts)
    
    # Time components
    # Ensure properties are accessed safely - DatetimeIndex uses attributes directly
    feats['cal_hour'] = ts.hour
    feats['cal_day_of_week'] = ts.dayofweek
    feats['cal_month'] = ts.month
    feats['cal_quarter'] = ts.quarter
    
    # Weekend (Sat=5, Sun=6)
    feats['cal_is_weekend'] = feats['cal_day_of_week'].isin([5, 6]).astype(int)
    
    # Holidays
    holiday_dates = set(pd.to_datetime(holidays_df['date']).dt.date)
    dates = ts.date # numpy array of objects
    feats['cal_is_holiday'] = pd.Series(dates).isin(holiday_dates).astype(int).values
    
    # Monsoon (Jun 15 - Sep 30)
    # Cast to int to be safe
    m = feats['cal_month'].astype(int)
    d = ts.day.astype(int)
    is_monsoon = (
        ((m == 6) & (d >= 15)) |
        (m.isin([7, 8])) |
        ((m == 9) & (d <= 30))
    )
    feats['cal_is_monsoon'] = is_monsoon.astype(int)
    
    # Days to nearest holiday
    # If holidays empty, 999
    if not holiday_dates:
        feats['cal_days_to_nearest_holiday'] = 999
    else:
        # Compute min distance
        # Vectorized is hard, do simple approximation or nearest merge
        # Create a DF of holidays
        hol_series = pd.to_datetime(list(holiday_dates))
        
        # We need distance for each date in feats
        # Since unique dates are ~1000 and holidays ~100, we can use merge_asof or reindex
        
        # Reset index to allow merge
        temp_df = pd.DataFrame({'date': pd.to_datetime(dates.unique())}).sort_values('date')
        
        # Expand holidays to a series of dates
        hol_df = pd.DataFrame({'hol_date': hol_series}).sort_values('hol_date')
        
        # Use merge_asof to find backward nearest
        temp_df = pd.merge_asof(temp_df, hol_df, left_on='date', right_on='hol_date', direction='backward')
        temp_df = temp_df.rename(columns={'hol_date': 'prev_hol'})
        
        # Use merge_asof to find forward nearest
        temp_df_fwd = pd.DataFrame({'date': pd.to_datetime(dates.unique())}).sort_values('date')
        temp_df_fwd = pd.merge_asof(temp_df_fwd, hol_df, left_on='date', right_on='hol_date', direction='forward')
        
        temp_df['next_hol'] = temp_df_fwd['hol_date']
        
        # Calc diffs
        temp_df['diff_prev'] = (temp_df['date'] - temp_df['prev_hol']).dt.days.abs()
        temp_df['diff_next'] = (temp_df['date'] - temp_df['next_hol']).dt.days.abs()
        
        temp_df['min_dist'] = temp_df[['diff_prev', 'diff_next']].min(axis=1).fillna(999)
        
        # Map back to result
        dist_map = temp_df.set_index('date')['min_dist']
        feats['cal_days_to_nearest_holiday'] = dates.map(dist_map).fillna(999).astype(int)

    # Cyclical
    feats['cal_hour_sin'] = np.sin(2 * np.pi * feats['cal_hour'] / 24)
    feats['cal_hour_cos'] = np.cos(2 * np.pi * feats['cal_hour'] / 24)
    
    feats['cal_month_sin'] = np.sin(2 * np.pi * (feats['cal_month'] - 1) / 12)
    feats['cal_month_cos'] = np.cos(2 * np.pi * (feats['cal_month'] - 1) / 12)
    
    return feats
