"""
Weather forecast fetching from Open-Meteo Forecast API
Fetches forecasts for 5 cities and aggregates to national level
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Import CITY_CONFIG from load_weather_data
sys.path.insert(0, str(Path(__file__).parent.parent / 'data_pipeline'))
from load_weather_data import CITY_CONFIG


def fetch_city_forecast(city_name, forecast_days=7):
    """
    Fetch weather forecast for a single city from Open-Meteo Forecast API
    
    Parameters:
    -----------
    city_name : str
        Name of city (must be in CITY_CONFIG)
    forecast_days : int
        Number of days to forecast (default: 7, max: 16)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with hourly weather forecasts
    """
    if city_name not in CITY_CONFIG:
        raise ValueError(f"City {city_name} not in CITY_CONFIG")
    
    config = CITY_CONFIG[city_name]
    lat = config['lat']
    lon = config['lon']
    
    print(f"  Fetching forecast for {city_name} ({config['grid']})...")
    
    # Open-Meteo Forecast API endpoint
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': [
            'temperature_2m',
            'relative_humidity_2m',
            'direct_radiation',
            'diffuse_radiation',
            'shortwave_radiation',
            'wind_speed_10m',
            'cloud_cover'
        ],
        'forecast_days': min(forecast_days, 16),  # Max 16 days
        'timezone': 'Asia/Kolkata'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Extract hourly data
        hourly = data.get('hourly', {})
        if not hourly:
            print(f"    WARNING: No hourly data returned for {city_name}")
            return None
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': pd.to_datetime(hourly['time']),
            'temperature_2m': hourly.get('temperature_2m'),
            'relative_humidity_2m': hourly.get('relative_humidity_2m'),
            'direct_radiation': hourly.get('direct_radiation'),
            'diffuse_radiation': hourly.get('diffuse_radiation'),
            'shortwave_radiation': hourly.get('shortwave_radiation'),
            'wind_speed_10m': hourly.get('wind_speed_10m'),
            'cloud_cover': hourly.get('cloud_cover')
        })
        
        # Set time as index and ensure IST timezone
        df = df.set_index('time')
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            df.index = df.index.tz_convert('Asia/Kolkata')
        
        # Add city metadata
        df['city'] = city_name
        df['grid'] = config['grid']
        df['weight'] = config['weight']
        
        print(f"    ✓ Fetched {len(df):,} hourly forecast records")
        print(f"      Date range: {df.index.min()} to {df.index.max()}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"    ERROR fetching forecast for {city_name}: {e}")
        return None
    except Exception as e:
        print(f"    ERROR processing forecast for {city_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def fetch_weather_forecast(forecast_days=7, cities=None):
    """
    Fetch weather forecasts for all cities
    
    Parameters:
    -----------
    forecast_days : int
        Number of days to forecast (default: 7)
    cities : list, optional
        List of city names (default: all cities in CITY_CONFIG)
    
    Returns:
    --------
    dict
        Dictionary of DataFrames keyed by city name
    """
    if cities is None:
        cities = list(CITY_CONFIG.keys())
    
    print("="*80)
    print("FETCHING WEATHER FORECASTS FROM OPEN-METEO")
    print("="*80)
    print(f"Forecast horizon: {forecast_days} days")
    print(f"Cities: {', '.join(cities)}")
    
    weather_forecasts = {}
    
    for city_name in cities:
        df = fetch_city_forecast(city_name, forecast_days)
        if df is not None:
            weather_forecasts[city_name] = df
        
        # Rate limiting
        time.sleep(0.5)
    
    print(f"\n✓ Fetched forecasts for {len(weather_forecasts)} cities")
    return weather_forecasts


def aggregate_forecast_to_national(weather_forecasts_dict):
    """
    Aggregate city-level weather forecasts to national level using demand-weighted average
    
    Parameters:
    -----------
    weather_forecasts_dict : dict
        Dictionary of DataFrames keyed by city name
    
    Returns:
    --------
    pd.DataFrame
        National-level aggregated weather forecast
    """
    print("\n" + "="*80)
    print("AGGREGATING WEATHER FORECASTS TO NATIONAL LEVEL")
    print("="*80)
    
    if not weather_forecasts_dict:
        raise ValueError("No weather forecasts to aggregate")
    
    # Combine all city dataframes
    all_cities = []
    for city_name, df in weather_forecasts_dict.items():
        all_cities.append(df)
    
    # Combine and align on timestamp
    combined = pd.concat(all_cities, axis=0)
    
    # Weather columns to aggregate
    weather_cols = [
        'temperature_2m',
        'relative_humidity_2m',
        'direct_radiation',
        'diffuse_radiation',
        'shortwave_radiation',
        'wind_speed_10m',
        'cloud_cover'
    ]
    
    # Create weighted averages
    aggregated = []
    
    for timestamp in combined.index.unique():
        timestamp_data = combined.loc[timestamp]
        
        # Handle both single row and multiple rows
        if isinstance(timestamp_data, pd.Series):
            timestamp_data = timestamp_data.to_frame().T
        
        weights = timestamp_data['weight'].values
        weights = weights / weights.sum()  # Normalize weights
        
        row = {'time': timestamp}
        
        for col in weather_cols:
            if col in timestamp_data.columns:
                values = timestamp_data[col].values
                # Weighted average (handle NaN)
                valid_mask = ~pd.isna(values)
                if valid_mask.sum() > 0:
                    weighted_avg = np.average(
                        values[valid_mask],
                        weights=weights[valid_mask]
                    )
                    row[col] = weighted_avg
                else:
                    row[col] = np.nan
        
        aggregated.append(row)
    
    df_national = pd.DataFrame(aggregated)
    df_national = df_national.set_index('time')
    df_national = df_national.sort_index()
    
    # Add prefix to indicate national aggregation
    df_national.columns = [f"{col}_national" for col in df_national.columns]
    
    # Ensure all numeric columns are float64
    for col in df_national.select_dtypes(include=[np.number]).columns:
        if df_national[col].dtype != 'float64':
            df_national[col] = df_national[col].astype('float64')
    
    print(f"✓ Aggregated to national level")
    print(f"  Total rows: {len(df_national):,}")
    print(f"  Date range: {df_national.index.min()} to {df_national.index.max()}")
    print(f"  Columns: {list(df_national.columns)}")
    
    return df_national
