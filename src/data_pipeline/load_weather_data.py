"""
Load historical weather data from Open-Meteo API
Fetches data for 5 proxy cities representing India's regional grids
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
from pathlib import Path


# City coordinates and metadata for India's 5 regional grids
CITY_CONFIG = {
    'Delhi': {
        'lat': 28.6139,
        'lon': 77.2090,
        'grid': 'NR',
        'weight': 0.30
    },
    'Mumbai': {
        'lat': 19.0760,
        'lon': 72.8777,
        'grid': 'WR',
        'weight': 0.28
    },
    'Chennai': {
        'lat': 13.0827,
        'lon': 80.2707,
        'grid': 'SR',
        'weight': 0.25
    },
    'Kolkata': {
        'lat': 22.5726,
        'lon': 88.3639,
        'grid': 'ER',
        'weight': 0.12
    },
    'Guwahati': {
        'lat': 26.1445,
        'lon': 91.7362,
        'grid': 'NER',
        'weight': 0.05
    }
}


def fetch_city_weather(city_name, start_date, end_date):
    """
    Fetch historical weather data for a single city from Open-Meteo
    
    Parameters:
    -----------
    city_name : str
        Name of city (must be in CITY_CONFIG)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with hourly weather data
    """
    if city_name not in CITY_CONFIG:
        raise ValueError(f"City {city_name} not in CITY_CONFIG")
    
    config = CITY_CONFIG[city_name]
    lat = config['lat']
    lon = config['lon']
    
    print(f"  Fetching weather for {city_name} ({config['grid']})...")
    
    # Open-Meteo Historical Weather API endpoint
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': [
            'temperature_2m',
            'relative_humidity_2m',
            'direct_radiation',
            'diffuse_radiation',
            'shortwave_radiation',
            'wind_speed_10m',
            'cloud_cover'
        ],
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
        
        # Set time as index and convert to IST
        df = df.set_index('time')
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            df.index = df.index.tz_convert('Asia/Kolkata')
        
        # Add city identifier
        df['city'] = city_name
        df['grid'] = config['grid']
        df['weight'] = config['weight']
        
        print(f"    ✓ Fetched {len(df):,} hourly records")
        print(f"      Date range: {df.index.min()} to {df.index.max()}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"    ERROR fetching {city_name}: {e}")
        return None
    except Exception as e:
        print(f"    ERROR processing {city_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def fetch_all_cities_weather(start_date, end_date, save_dir='data/raw/weather'):
    """
    Fetch weather data for all 5 cities
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    save_dir : str
        Directory to save individual city files
    
    Returns:
    --------
    dict
        Dictionary of DataFrames, keyed by city name
    """
    print("="*80)
    print("FETCHING WEATHER DATA FROM OPEN-METEO")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Cities: {', '.join(CITY_CONFIG.keys())}")
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    weather_data = {}
    
    for city_name in CITY_CONFIG.keys():
        # Check if file already exists and covers the full period
        city_file = save_path / f"{city_name}_weather.parquet"
        
        df = None
        should_fetch = True
        
        if city_file.exists():
            print(f"  Checking cached data for {city_name}...")
            try:
                df_cached = pd.read_parquet(city_file)
                # Check if cached data covers the requested period
                cache_start = df_cached.index.min().date()
                cache_end = df_cached.index.max().date()
                req_start = pd.to_datetime(start_date).date()
                req_end = pd.to_datetime(end_date).date()
                
                if cache_start <= req_start and cache_end >= req_end:
                    print(f"    ✓ Using cached data ({len(df_cached):,} records)")
                    print(f"      Cache covers: {cache_start} to {cache_end}")
                    df = df_cached
                    should_fetch = False
                else:
                    print(f"    Cache incomplete (covers {cache_start} to {cache_end}, need {req_start} to {req_end})")
                    print(f"    Fetching fresh data...")
            except Exception as e:
                print(f"    WARNING: Error loading cache, fetching fresh data: {e}")
        
        if should_fetch:
            # Fetch from API
            df = fetch_city_weather(city_name, start_date, end_date)
        
        if df is not None:
            weather_data[city_name] = df
            # Save to cache if fetched fresh
            if should_fetch:
                df.to_parquet(city_file)
                print(f"    ✓ Saved to {city_file}")
        
        # Rate limiting - be polite to API
        time.sleep(1)
    
    print(f"\n✓ Fetched weather data for {len(weather_data)} cities")
    return weather_data


def aggregate_to_national(weather_data_dict):
    """
    Aggregate 5-city weather data to national level using demand-weighted average
    
    Parameters:
    -----------
    weather_data_dict : dict
        Dictionary of DataFrames keyed by city name
    
    Returns:
    --------
    pd.DataFrame
        National-level aggregated weather data
    """
    print("\n" + "="*80)
    print("AGGREGATING WEATHER TO NATIONAL LEVEL")
    print("="*80)
    
    # Combine all city dataframes
    all_cities = []
    for city_name, df in weather_data_dict.items():
        all_cities.append(df)
    
    if not all_cities:
        raise ValueError("No weather data to aggregate")
    
    # Combine and align on timestamp
    combined = pd.concat(all_cities, axis=0)
    
    # Group by timestamp and calculate weighted average
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
    
    print(f"✓ Aggregated to national level")
    print(f"  Total rows: {len(df_national):,}")
    print(f"  Date range: {df_national.index.min()} to {df_national.index.max()}")
    print(f"  Columns: {list(df_national.columns)}")
    
    return df_national


def load_weather_data(start_date='2021-01-01', end_date='2023-08-31', 
                     use_cache=True, save_dir='data/raw/weather'):
    """
    Main function to load and aggregate weather data
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    use_cache : bool
        Whether to use cached files if available
    save_dir : str
        Directory for caching weather data
    
    Returns:
    --------
    pd.DataFrame
        National-level aggregated weather data with datetime index
    """
    # Fetch all cities
    weather_data = fetch_all_cities_weather(start_date, end_date, save_dir)
    
    if not weather_data:
        raise ValueError("No weather data fetched")
    
    # Aggregate to national level
    national_weather = aggregate_to_national(weather_data)
    
    return national_weather


if __name__ == "__main__":
    # Test fetching weather data
    print("Testing weather data fetch...")
    
    # Fetch for a shorter period first to test
    test_start = '2021-09-01'
    test_end = '2021-09-30'
    
    print(f"\nFetching test data: {test_start} to {test_end}")
    weather_df = load_weather_data(test_start, test_end)
    
    print(f"\nSample data:")
    print(weather_df.head())
    print(f"\nData summary:")
    print(weather_df.describe())
