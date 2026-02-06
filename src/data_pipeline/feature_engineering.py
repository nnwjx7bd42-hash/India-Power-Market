"""
Feature engineering for IEX price forecasting
Creates derived features from raw data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from calendar_features import add_calendar_features


def calculate_heat_index(temperature_c, humidity_pct):
    """
    Calculate heat index (feels-like temperature)
    Formula from NOAA
    
    Parameters:
    -----------
    temperature_c : float or Series
        Temperature in Celsius
    humidity_pct : float or Series
        Relative humidity as percentage (0-100)
    
    Returns:
    --------
    float or Series
        Heat index in Celsius
    """
    # Convert to Fahrenheit for calculation
    T_f = (temperature_c * 9/5) + 32
    RH = humidity_pct
    
    # Heat index formula (simplified, valid for T > 80°F and RH > 40%)
    # For lower temps, heat index = temperature
    heat_index_f = np.where(
        (T_f >= 80) & (RH >= 40),
        -42.379 + 2.04901523*T_f + 10.14333127*RH - 0.22475541*T_f*RH
        - 6.83783e-3*T_f**2 - 5.481717e-2*RH**2 + 1.22874e-3*T_f**2*RH
        + 8.5282e-4*T_f*RH**2 - 1.99e-6*T_f**2*RH**2,
        T_f  # Below threshold, heat index = temperature
    )
    
    # Convert back to Celsius
    heat_index_c = (heat_index_f - 32) * 5/9
    
    return heat_index_c


def add_weather_features(df):
    """
    Add weather-derived features
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with weather columns (temperature_2m_national, etc.)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added weather features
    """
    if 'temperature_2m_national' not in df.columns:
        print("Warning: No temperature data found, skipping weather features")
        return df
    
    temp_col = 'temperature_2m_national'
    humidity_col = 'relative_humidity_2m_national'
    
    # Cooling Degree Hours (CDH) - AC demand threshold
    df['CDH'] = np.maximum(0, df[temp_col] - 24)
    
    # Heating Degree Hours (HDH) - heating demand threshold
    df['HDH'] = np.maximum(0, 18 - df[temp_col])
    
    # Heat Index
    if humidity_col in df.columns:
        df['Heat_Index'] = calculate_heat_index(df[temp_col], df[humidity_col])
    else:
        df['Heat_Index'] = df[temp_col]  # Fallback to temperature
    
    # Temperature deviation from rolling mean
    df['Temp_Deviation'] = df[temp_col] - df[temp_col].rolling(window=720, min_periods=1).mean()  # 30-day rolling
    
    # Humidity deviation
    if humidity_col in df.columns:
        df['Humidity_Deviation'] = df[humidity_col] - df[humidity_col].rolling(window=720, min_periods=1).mean()
    
    # Solar generation proxy from irradiance
    if 'direct_radiation_national' in df.columns and 'cloud_cover_national' in df.columns:
        # Effective irradiance accounting for clouds
        df['Solar_Irradiance_Effective'] = (
            df['direct_radiation_national'] * (1 - df['cloud_cover_national'] / 100)
        )
    
    # Wind generation proxy (simple relationship)
    if 'wind_speed_10m_national' in df.columns:
        # Simple cubic relationship (wind power ~ wind_speed^3)
        df['Wind_Power_Proxy'] = df['wind_speed_10m_national'] ** 3
    
    return df


def add_supply_features(df):
    """
    Add supply-side derived features from NERLDC data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with NERLDC columns (Thermal, Hydro, Wind, Solar, Demand, etc.)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added supply features
    """
    # Renewable generation
    if 'Wind' in df.columns and 'Solar' in df.columns:
        df['RE_Generation'] = df['Wind'] + df['Solar']
    
    # Renewable penetration
    if 'RE_Generation' in df.columns and 'Demand' in df.columns:
        df['RE_Penetration'] = df['RE_Generation'] / df['Demand'].replace(0, np.nan)
        df['RE_Penetration'] = df['RE_Penetration'].fillna(0)
    
    # Net load (residual load that thermal must serve)
    if 'Demand' in df.columns and 'RE_Generation' in df.columns:
        df['Net_Load'] = df['Demand'] - df['RE_Generation']
    
    # Total generation
    gen_cols = ['Thermal', 'Hydro', 'Gas', 'Nuclear', 'Wind', 'Solar']
    available_gen = [col for col in gen_cols if col in df.columns]
    if available_gen:
        df['Total_Generation'] = df[available_gen].sum(axis=1)
    
    # Thermal share
    if 'Thermal' in df.columns and 'Total_Generation' in df.columns:
        df['Thermal_Share'] = df['Thermal'] / df['Total_Generation'].replace(0, np.nan)
        df['Thermal_Share'] = df['Thermal_Share'].fillna(0)
    
    # Solar ramp (change from previous hour)
    if 'Solar' in df.columns:
        df['Solar_Ramp'] = df['Solar'] - df['Solar'].shift(1)
        df['Solar_Ramp'] = df['Solar_Ramp'].fillna(0)
    
    # Hydro availability
    if 'Hydro' in df.columns and 'Demand' in df.columns:
        df['Hydro_Availability'] = df['Hydro'] / df['Demand'].replace(0, np.nan)
        df['Hydro_Availability'] = df['Hydro_Availability'].fillna(0)
    
    # Renewable volatility (rolling std)
    if 'RE_Generation' in df.columns:
        df['RE_Volatility'] = df['RE_Generation'].rolling(window=24, min_periods=1).std()
        df['RE_Volatility'] = df['RE_Volatility'].fillna(0)
    
    return df


def add_lag_features(df):
    """
    Add extended lag features
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price and load columns
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added lag features
    """
    # Weekly lags (T-168 = same hour last week)
    if 'P(T)' in df.columns:
        df['P_T-168'] = df['P(T)'].shift(168)
    
    if 'L(T-1)' in df.columns:
        df['L_T-168'] = df['L(T-1)'].shift(167)  # Adjust for existing lag
    
    # Extended short-term price lags
    if 'P(T)' in df.columns:
        df['P_T-3'] = df['P(T)'].shift(3)
        df['P_T-4'] = df['P(T)'].shift(4)
    
    # Rolling statistics
    if 'P(T)' in df.columns:
        df['Price_MA_24h'] = df['P(T)'].rolling(window=24, min_periods=1).mean()
        df['Price_Std_24h'] = df['P(T)'].rolling(window=24, min_periods=1).std()
        df['Price_Std_24h'] = df['Price_Std_24h'].fillna(0)
    
    if 'L(T-1)' in df.columns:
        df['Load_MA_24h'] = df['L(T-1)'].rolling(window=24, min_periods=1).mean()
        df['Load_Std_24h'] = df['L(T-1)'].rolling(window=24, min_periods=1).std()
        df['Load_Std_24h'] = df['Load_Std_24h'].fillna(0)
    
    return df


def add_interaction_features(df):
    """
    Add interaction features
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with base features
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added interaction features
    """
    # Hour × NetLoad (evening peak dynamics)
    if 'Hour' in df.columns and 'Net_Load' in df.columns:
        df['Hour_x_NetLoad'] = df['Hour'] * df['Net_Load']
    
    # Hour × CDH (cooling load varies by time)
    if 'Hour' in df.columns and 'CDH' in df.columns:
        df['Hour_x_CDH'] = df['Hour'] * df['CDH']
    
    # Month × Hour (seasonal intraday patterns)
    if 'Month' in df.columns and 'Hour' in df.columns:
        df['Month_x_Hour'] = df['Month'] * df['Hour']
    
    # RE_Pen × Hour (renewable impact varies by time)
    if 'RE_Penetration' in df.columns and 'Hour' in df.columns:
        df['RE_Pen_x_Hour'] = df['RE_Penetration'] * df['Hour']
    
    # Temperature × Hour
    if 'temperature_2m_national' in df.columns and 'Hour' in df.columns:
        df['Temp_x_Hour'] = df['temperature_2m_national'] * df['Hour']
    
    # Season × Temperature
    if 'Season' in df.columns and 'temperature_2m_national' in df.columns:
        df['Season_x_Temp'] = df['Season'] * df['temperature_2m_national']
    
    return df


def create_features(df):
    """
    Main function to create all features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw unified dataset with datetime index
    
    Returns:
    --------
    pd.DataFrame
        Dataset with all engineered features
    """
    print("="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    print(f"\nStarting with {len(df.columns)} columns")
    
    # 1. Calendar features (including holidays)
    print("\n1. Adding calendar features...")
    df = add_calendar_features(df)
    
    # Add holiday features
    holidays_file = Path('data/raw/indian_holidays.csv')
    if holidays_file.exists():
        print("   Adding holiday features...")
        from calendar_features import add_holiday_features
        df = add_holiday_features(df, str(holidays_file))
    else:
        print("   Warning: Holiday calendar not found, skipping holiday features")
    
    # 2. Weather features
    print("2. Adding weather-derived features...")
    df = add_weather_features(df)
    
    # 3. Supply-side features
    print("3. Adding supply-side features...")
    df = add_supply_features(df)
    
    # 4. Extended lag features
    print("4. Adding extended lag features...")
    df = add_lag_features(df)
    
    # 5. Interaction features
    print("5. Adding interaction features...")
    df = add_interaction_features(df)
    
    print(f"\n✓ Feature engineering complete")
    print(f"  Final columns: {len(df.columns)}")
    print(f"  Rows: {len(df):,}")
    
    return df
