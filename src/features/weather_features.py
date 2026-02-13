import pandas as pd
import numpy as np

def build_weather_features(weather_df):
    """
    Build weather features.
    Input: national aggregated weather.
    Output: DataFrame indexed on delivery_start_ist.
    """
    df = weather_df.set_index('delivery_start_ist').sort_index()
    feats = pd.DataFrame(index=df.index)
    
    # Passthrough
    feats['wx_national_temp'] = df['national_temperature']
    feats['wx_delhi_temp'] = df['delhi_temperature']
    feats['wx_national_shortwave'] = df['national_shortwave']
    feats['wx_chennai_wind'] = df['chennai_wind_speed']
    feats['wx_national_cloud'] = df['national_cloud_cover']
    
    # Derived
    # Cooling Degree Hours: max(0, temp - 24)
    feats['wx_cooling_degree_hours'] = (feats['wx_national_temp'] - 24).clip(lower=0)
    
    # Heat Index: temp * (humidity / 100)
    # Need humidity
    if 'national_humidity' in df.columns:
        feats['wx_heat_index'] = df['national_temperature'] * (df['national_humidity'] / 100.0)
    else:
        feats['wx_heat_index'] = np.nan
        
    # Lags
    feats['wx_temp_lag_24h'] = feats['wx_national_temp'].shift(24)
    feats['wx_shortwave_delta_1h'] = feats['wx_national_shortwave'] - feats['wx_national_shortwave'].shift(1)
    
    # Spread
    feats['wx_temp_spread'] = feats['wx_delhi_temp'] - feats['wx_national_temp']
    
    return feats
