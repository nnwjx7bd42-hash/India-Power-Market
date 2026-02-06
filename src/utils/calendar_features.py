"""
Calendar feature engineering
Adds time-based features like DayOfWeek, Month, Hour, etc.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def add_calendar_features(df):
    """
    Add calendar-based features to dataframe with datetime index
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added calendar features
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index")
    
    # Ensure timezone is IST
    if df.index.tz is None:
        df.index = df.index.tz_localize('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')
    
    # Basic time features
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek  # 0=Monday, 6=Sunday
    df['DayOfMonth'] = df.index.day
    df['Month'] = df.index.month
    df['WeekOfYear'] = df.index.isocalendar().week
    df['DayOfYear'] = df.index.dayofyear
    df['Quarter'] = df.index.quarter
    
    # Derived features
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)  # Saturday=5, Sunday=6
    df['IsMonday'] = (df['DayOfWeek'] == 0).astype(int)
    df['IsFriday'] = (df['DayOfWeek'] == 4).astype(int)
    
    # Cyclical encoding for hour (sine/cosine)
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    # Cyclical encoding for month
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Cyclical encoding for day of year
    df['DayOfYear_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['DayOfYear_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
    
    # Cyclical encoding for day of week
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    # Season mapping (based on month)
    # Winter: Dec-Feb (12, 1, 2), Spring: Mar-May (3,4,5), 
    # Monsoon: Jun-Sep (6,7,8,9), Post-Monsoon: Oct-Nov (10,11)
    def map_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Summer/Spring
        elif month in [6, 7, 8, 9]:
            return 2  # Monsoon
        else:  # 10, 11
            return 3  # Post-Monsoon
    
    df['Season_Month'] = df['Month'].apply(map_season)
    
    # Time of day categories
    def time_of_day(hour):
        if 5 <= hour < 12:
            return 0  # Morning
        elif 12 <= hour < 17:
            return 1  # Afternoon
        elif 17 <= hour < 21:
            return 2  # Evening
        else:
            return 3  # Night
    
    df['TimeOfDay'] = df['Hour'].apply(time_of_day)
    
    return df


def add_holiday_features(df, holidays_file=None):
    """
    Add holiday indicators (placeholder - requires holiday calendar)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    holidays_file : str, optional
        Path to CSV file with holiday dates
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with IsHoliday column
    """
    df['IsHoliday'] = 0  # Default to no holiday
    
    if holidays_file:
        try:
            holidays_df = pd.read_csv(holidays_file, parse_dates=['date'])
            holiday_dates = pd.to_datetime(holidays_df['date']).dt.date
            df_dates = df.index.date
            df['IsHoliday'] = df_dates.isin(holiday_dates).astype(int)
        except Exception as e:
            print(f"Warning: Could not load holidays file: {e}")
    
    return df
