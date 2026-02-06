"""
Feature engineering for forecast period
Builds all 64 features needed for enhanced model prediction
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Import feature engineering functions
sys.path.insert(0, str(Path(__file__).parent.parent / 'data_pipeline'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from feature_engineering import (
    add_weather_features,
    add_supply_features,
    add_interaction_features,
    calculate_heat_index
)
from calendar_features import add_calendar_features


def build_features_for_forecast(
    historical_df,
    forecast_timestamps,
    weather_forecast_df,
    load_forecast_df,
    supply_forecast_df,
    price_predictions=None
):
    """
    Build complete feature matrix for forecast period
    
    Parameters:
    -----------
    historical_df : pd.DataFrame
        Last 168+ hours of historical data (for lag features)
    forecast_timestamps : pd.DatetimeIndex
        Future timestamps for forecast period
    weather_forecast_df : pd.DataFrame
        Weather forecasts (national level)
    load_forecast_df : pd.DataFrame
        Hourly load forecast (column: load_mw)
    supply_forecast_df : pd.DataFrame
        Supply breakdown forecast
    price_predictions : array-like, optional
        Previous price predictions (for autoregressive lags)
    
    Returns:
    --------
    pd.DataFrame
        Feature matrix with all 64 features
    """
    print("\n" + "="*80)
    print("BUILDING FEATURES FOR FORECAST PERIOD")
    print("="*80)
    
    # Create base DataFrame with timestamps
    forecast_df = pd.DataFrame(index=forecast_timestamps)
    forecast_df.index.name = 'Timestamp'
    
    # Ensure timezone
    if forecast_df.index.tz is None:
        forecast_df.index = forecast_df.index.tz_localize('Asia/Kolkata')
    
    print(f"\nForecast period: {forecast_df.index.min()} to {forecast_df.index.max()}")
    print(f"Total hours: {len(forecast_df)}")
    
    # Step 1: Add calendar features
    print("\n1. Adding calendar features...")
    forecast_df = add_calendar_features(forecast_df)
    
    # Step 2: Add weather features
    print("2. Adding weather features...")
    # Merge weather forecast
    forecast_df = forecast_df.merge(
        weather_forecast_df,
        left_index=True,
        right_index=True,
        how='left'
    )
    # Ensure float64 for weather columns
    weather_cols = [c for c in weather_forecast_df.columns if c in forecast_df.columns]
    for col in weather_cols:
        if forecast_df[col].dtype in ['float32', 'int32', 'int64']:
            forecast_df[col] = forecast_df[col].astype('float64')
    # Add weather-derived features
    forecast_df = add_weather_features(forecast_df)
    
    # Step 3: Add load and supply features
    print("3. Adding load and supply features...")
    # Add load (rename to match training data format)
    forecast_df['L(T-1)'] = load_forecast_df['load_mw'].values.astype('float64')
    
    # Merge supply features
    supply_cols = ['Thermal', 'Hydro', 'Gas', 'Nuclear', 'Wind', 'Solar', 'Demand']
    for col in supply_cols:
        if col in supply_forecast_df.columns:
            forecast_df[col] = supply_forecast_df[col].values.astype('float64')
    
    # Add supply-derived features
    forecast_df = add_supply_features(forecast_df)
    
    # Step 4: Add lag features (autoregressive)
    print("4. Adding lag features (autoregressive)...")
    forecast_df = add_autoregressive_lags(
        forecast_df,
        historical_df,
        price_predictions
    )
    
    # Step 5: Add interaction features
    print("5. Adding interaction features...")
    forecast_df = add_interaction_features(forecast_df)
    
    # Step 6: Add Day and Season (if missing)
    if 'Day' not in forecast_df.columns:
        # Day: 0 = weekday, 1 = weekend
        forecast_df['Day'] = (forecast_df['DayOfWeek'] >= 5).astype(int)
    
    if 'Season' not in forecast_df.columns:
        # Season: 0=Winter (Dec-Feb), 1=Summer (Mar-May), 2=Monsoon (Jun-Sep), 3=Post-Monsoon (Oct-Nov)
        forecast_df['Season'] = forecast_df['Month'].apply(
            lambda m: 0 if m in [12, 1, 2] else (1 if m in [3, 4, 5] else (2 if m in [6, 7, 8, 9] else 3))
        )
    
    # Ensure all required features exist
    print("\n6. Validating features...")
    required_features = get_required_features()
    missing_features = set(required_features) - set(forecast_df.columns)
    
    if missing_features:
        print(f"  Warning: Missing features: {missing_features}")
        # Fill with defaults
        for feat in missing_features:
            forecast_df[feat] = 0
    
    print(f"\n✓ Feature engineering complete")
    print(f"  Total features: {len(forecast_df.columns)}")
    print(f"  Required features: {len(required_features)}")
    
    return forecast_df


def add_autoregressive_lags(forecast_df, historical_df, price_predictions=None):
    """
    Add lag features using historical data and previous predictions
    
    Parameters:
    -----------
    forecast_df : pd.DataFrame
        Forecast DataFrame to add lags to
    historical_df : pd.DataFrame
        Historical data (last 168+ hours)
    price_predictions : array-like, optional
        Previous price predictions
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with lag features added
    """
    # Ensure historical data is sorted
    historical_df = historical_df.sort_index()
    
    # Convert all numeric columns to float64 to avoid dtype issues
    numeric_cols = forecast_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if forecast_df[col].dtype != 'float64':
            forecast_df[col] = forecast_df[col].astype('float64')
    
    # Get last values from historical data
    last_price = historical_df['P(T)'].iloc[-1] if 'P(T)' in historical_df.columns else None
    last_load = historical_df['L(T-1)'].iloc[-1] if 'L(T-1)' in historical_df.columns else None
    
    # Price lags - ensure all are float64
    if 'P(T)' in historical_df.columns:
        # Initialize all price lag columns as float64
        price_lag_cols = ['P(T-1)', 'P(T-2)', 'P(T-24)', 'P(T-48)', 'P_T-168']
        for col in price_lag_cols:
            forecast_df[col] = pd.Series(np.nan, index=forecast_df.index, dtype='float64')
        
        # P(T-1): Use previous prediction or last historical
        if price_predictions is not None and len(price_predictions) > 0:
            forecast_df.loc[forecast_df.index[0], 'P(T-1)'] = float(last_price) if last_price is not None else np.nan
            for i in range(1, len(forecast_df)):
                if i <= len(price_predictions):
                    forecast_df.loc[forecast_df.index[i], 'P(T-1)'] = float(price_predictions[i-1])
        else:
            forecast_df['P(T-1)'] = float(last_price) if last_price is not None else np.nan
        
        # P(T-2): Two steps back
        if price_predictions is not None and len(price_predictions) >= 2:
            forecast_df.loc[forecast_df.index[0], 'P(T-2)'] = float(historical_df['P(T)'].iloc[-2]) if len(historical_df) >= 2 else float(last_price) if last_price else np.nan
            forecast_df.loc[forecast_df.index[1], 'P(T-2)'] = float(price_predictions[0]) if len(price_predictions) > 0 else float(last_price) if last_price else np.nan
            for i in range(2, len(forecast_df)):
                if i <= len(price_predictions):
                    forecast_df.loc[forecast_df.index[i], 'P(T-2)'] = float(price_predictions[i-2])
        else:
            forecast_df['P(T-2)'] = float(historical_df['P(T)'].iloc[-2]) if len(historical_df) >= 2 else float(last_price) if last_price else np.nan
        
        # P(T-24): Same hour yesterday
        for i, ts in enumerate(forecast_df.index):
            yesterday_ts = ts - pd.Timedelta(days=1)
            if yesterday_ts in historical_df.index:
                forecast_df.loc[ts, 'P(T-24)'] = float(historical_df.loc[yesterday_ts, 'P(T)'])
            elif i >= 24 and price_predictions is not None:
                forecast_df.loc[ts, 'P(T-24)'] = float(price_predictions[i-24])
            else:
                forecast_df.loc[ts, 'P(T-24)'] = float(last_price) if last_price else np.nan
        
        # P(T-48): Same hour 2 days ago
        for i, ts in enumerate(forecast_df.index):
            two_days_ago_ts = ts - pd.Timedelta(days=2)
            if two_days_ago_ts in historical_df.index:
                forecast_df.loc[ts, 'P(T-48)'] = float(historical_df.loc[two_days_ago_ts, 'P(T)'])
            elif i >= 48 and price_predictions is not None:
                forecast_df.loc[ts, 'P(T-48)'] = float(price_predictions[i-48])
            else:
                forecast_df.loc[ts, 'P(T-48)'] = float(last_price) if last_price else np.nan
        
        # P_T-168: Same hour last week
        for i, ts in enumerate(forecast_df.index):
            last_week_ts = ts - pd.Timedelta(days=7)
            if last_week_ts in historical_df.index:
                forecast_df.loc[ts, 'P_T-168'] = float(historical_df.loc[last_week_ts, 'P(T)'])
            elif i >= 168 and price_predictions is not None:
                forecast_df.loc[ts, 'P_T-168'] = float(price_predictions[i-168])
            else:
                forecast_df.loc[ts, 'P_T-168'] = float(last_price) if last_price else np.nan
        
        # P_T-3, P_T-4: Short-term lags (initialize as float64)
        if 'P_T-3' not in forecast_df.columns:
            forecast_df['P_T-3'] = pd.Series(np.nan, index=forecast_df.index, dtype='float64')
        if 'P_T-4' not in forecast_df.columns:
            forecast_df['P_T-4'] = pd.Series(np.nan, index=forecast_df.index, dtype='float64')
        
        if price_predictions is not None and len(price_predictions) >= 3:
            forecast_df.loc[forecast_df.index[0], 'P_T-3'] = float(historical_df['P(T)'].iloc[-3]) if len(historical_df) >= 3 else float(last_price) if last_price else np.nan
            for i in range(1, len(forecast_df)):
                if i <= len(price_predictions):
                    forecast_df.loc[forecast_df.index[i], 'P_T-3'] = float(price_predictions[i-1]) if i >= 1 else float(last_price) if last_price else np.nan
        
        if price_predictions is not None and len(price_predictions) >= 4:
            forecast_df.loc[forecast_df.index[0], 'P_T-4'] = float(historical_df['P(T)'].iloc[-4]) if len(historical_df) >= 4 else float(last_price) if last_price else np.nan
            for i in range(1, len(forecast_df)):
                if i <= len(price_predictions):
                    forecast_df.loc[forecast_df.index[i], 'P_T-4'] = float(price_predictions[i-2]) if i >= 2 else float(last_price) if last_price else np.nan
    
    # Load lags
    if 'L(T-1)' in forecast_df.columns:
        # L(T-2): Two steps back
        forecast_df['L(T-2)'] = forecast_df['L(T-1)'].shift(1)
        forecast_df['L(T-2)'] = forecast_df['L(T-2)'].fillna(last_load if last_load is not None else forecast_df['L(T-1)'].iloc[0])
        
        # L(T-24): Same hour yesterday
        forecast_df['L(T-24)'] = forecast_df['L(T-1)'].shift(24)
        for i, ts in enumerate(forecast_df.index):
            if pd.isna(forecast_df.loc[ts, 'L(T-24)']):
                yesterday_ts = ts - pd.Timedelta(days=1)
                if yesterday_ts in historical_df.index and 'L(T-1)' in historical_df.columns:
                    forecast_df.loc[ts, 'L(T-24)'] = historical_df.loc[yesterday_ts, 'L(T-1)']
                else:
                    forecast_df.loc[ts, 'L(T-24)'] = forecast_df['L(T-1)'].iloc[i]
        
        # L(T-48): Same hour 2 days ago
        forecast_df['L(T-48)'] = forecast_df['L(T-1)'].shift(48)
        for i, ts in enumerate(forecast_df.index):
            if pd.isna(forecast_df.loc[ts, 'L(T-48)']):
                two_days_ago_ts = ts - pd.Timedelta(days=2)
                if two_days_ago_ts in historical_df.index and 'L(T-1)' in historical_df.columns:
                    forecast_df.loc[ts, 'L(T-48)'] = historical_df.loc[two_days_ago_ts, 'L(T-1)']
                else:
                    forecast_df.loc[ts, 'L(T-48)'] = forecast_df['L(T-1)'].iloc[i]
        
        # L_T-168: Same hour last week
        forecast_df['L_T-168'] = forecast_df['L(T-1)'].shift(167)
        for i, ts in enumerate(forecast_df.index):
            if pd.isna(forecast_df.loc[ts, 'L_T-168']):
                last_week_ts = ts - pd.Timedelta(days=7)
                if last_week_ts in historical_df.index and 'L(T-1)' in historical_df.columns:
                    forecast_df.loc[ts, 'L_T-168'] = historical_df.loc[last_week_ts, 'L(T-1)']
                else:
                    forecast_df.loc[ts, 'L_T-168'] = forecast_df['L(T-1)'].iloc[i]
    
    # Rolling statistics (need to combine historical + predictions)
    if 'P(T-1)' in forecast_df.columns:
        # Initialize rolling stat columns as float64
        if 'Price_MA_24h' not in forecast_df.columns:
            forecast_df['Price_MA_24h'] = pd.Series(np.nan, index=forecast_df.index, dtype='float64')
        if 'Price_Std_24h' not in forecast_df.columns:
            forecast_df['Price_Std_24h'] = pd.Series(np.nan, index=forecast_df.index, dtype='float64')
        
        # Combine historical prices with predictions for rolling stats
        all_prices = []
        if 'P(T)' in historical_df.columns:
            all_prices.extend(historical_df['P(T)'].iloc[-24:].values.astype('float64'))
        if price_predictions is not None:
            all_prices.extend(np.array(price_predictions).astype('float64'))
        
        # Price rolling stats
        for i in range(len(forecast_df)):
            if len(all_prices) >= 24:
                window_prices = all_prices[max(0, len(all_prices)-24-i):len(all_prices)-i] if i < len(all_prices) else all_prices[-24:]
                forecast_df.loc[forecast_df.index[i], 'Price_MA_24h'] = float(np.mean(window_prices))
                forecast_df.loc[forecast_df.index[i], 'Price_Std_24h'] = float(np.std(window_prices)) if len(window_prices) > 1 else 0.0
            else:
                # Use historical average
                if 'P(T)' in historical_df.columns:
                    forecast_df.loc[forecast_df.index[i], 'Price_MA_24h'] = float(historical_df['P(T)'].iloc[-24:].mean())
                    forecast_df.loc[forecast_df.index[i], 'Price_Std_24h'] = float(historical_df['P(T)'].iloc[-24:].std())
                else:
                    forecast_df.loc[forecast_df.index[i], 'Price_MA_24h'] = float(last_price) if last_price else 0.0
                    forecast_df.loc[forecast_df.index[i], 'Price_Std_24h'] = 0.0
    
    if 'L(T-1)' in forecast_df.columns:
        # Initialize load rolling stat columns as float64
        if 'Load_MA_24h' not in forecast_df.columns:
            forecast_df['Load_MA_24h'] = pd.Series(np.nan, index=forecast_df.index, dtype='float64')
        if 'Load_Std_24h' not in forecast_df.columns:
            forecast_df['Load_Std_24h'] = pd.Series(np.nan, index=forecast_df.index, dtype='float64')
        
        # Load rolling stats
        all_loads = []
        if 'L(T-1)' in historical_df.columns:
            all_loads.extend(historical_df['L(T-1)'].iloc[-24:].values.astype('float64'))
        all_loads.extend(forecast_df['L(T-1)'].values.astype('float64'))
        
        for i in range(len(forecast_df)):
            if len(all_loads) >= 24:
                window_loads = all_loads[max(0, len(all_loads)-24-i):len(all_loads)-i] if i < len(all_loads) else all_loads[-24:]
                forecast_df.loc[forecast_df.index[i], 'Load_MA_24h'] = float(np.mean(window_loads))
                forecast_df.loc[forecast_df.index[i], 'Load_Std_24h'] = float(np.std(window_loads)) if len(window_loads) > 1 else 0.0
            else:
                forecast_df.loc[forecast_df.index[i], 'Load_MA_24h'] = float(forecast_df['L(T-1)'].iloc[i])
                forecast_df.loc[forecast_df.index[i], 'Load_Std_24h'] = 0.0
    
    # Fill any remaining NaN values
    forecast_df = forecast_df.ffill().bfill()
    
    # Ensure all numeric columns are float64 (XGBoost requirement)
    for col in forecast_df.select_dtypes(include=['int64', 'int32', 'float32']).columns:
        forecast_df[col] = forecast_df[col].astype('float64')
    
    return forecast_df


def get_required_features():
    """Get list of all required features for enhanced model"""
    # This should match the features used in training
    # Based on dataset_cleaned.parquet columns (excluding P(T))
    return [
        # Price/Load lags
        'P(T-1)', 'P(T-2)', 'P(T-24)', 'P(T-48)', 'P_T-168', 'P_T-3', 'P_T-4',
        'L(T-1)', 'L(T-2)', 'L(T-24)', 'L(T-48)', 'L_T-168',
        'Price_MA_24h', 'Price_Std_24h', 'Load_MA_24h', 'Load_Std_24h',
        # Weather
        'temperature_2m_national', 'relative_humidity_2m_national',
        'direct_radiation_national', 'diffuse_radiation_national',
        'wind_speed_10m_national', 'cloud_cover_national',
        'CDH', 'HDH', 'Temp_Deviation', 'Humidity_Deviation',
        'Solar_Irradiance_Effective', 'Wind_Power_Proxy',
        # Calendar
        'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'IsWeekend', 'IsMonday',
        'Season_Month', 'TimeOfDay', 'Hour_Sin', 'Hour_Cos',
        'Month_Sin', 'Month_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos',
        'DayOfYear_Sin', 'DayOfYear_Cos',
        # Supply-side
        'Thermal', 'Hydro', 'Gas', 'Nuclear', 'Wind', 'Solar', 'Demand',
        'RE_Generation', 'RE_Penetration', 'Net_Load', 'Thermal_Share',
        'Solar_Ramp', 'RE_Volatility',
        # Interactions
        'Hour_x_CDH', 'RE_Pen_x_Hour', 'Month_x_Hour', 'Season_x_Temp',
        # Original
        'Day', 'Season'
    ]
