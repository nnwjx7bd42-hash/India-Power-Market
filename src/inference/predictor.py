"""
Main inference pipeline for IEX price forecasting
Orchestrates weather fetching, load conversion, supply estimation, feature building, and prediction
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Import inference modules
from .weather_forecast import fetch_weather_forecast, aggregate_forecast_to_national
from .load_profile import analyze_load_patterns, daily_peak_to_hourly
from .supply_estimator import estimate_supply_breakdown
from .feature_builder import build_features_for_forecast, get_required_features

# Import models
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.enhanced import EnhancedModel, get_enhanced_features
from models.quantile_regression import QuantileModel


def load_historical_context(historical_data_path, hours_needed=168):
    """
    Load historical data for lag features
    
    Parameters:
    -----------
    historical_data_path : str
        Path to historical dataset
    hours_needed : int
        Minimum hours of historical data needed (default: 168 for weekly lags)
    
    Returns:
    --------
    pd.DataFrame
        Historical data (last hours_needed+ hours)
    """
    print(f"\nLoading historical data from {historical_data_path}...")
    
    df = pd.read_parquet(historical_data_path)
    
    # Ensure sorted by timestamp
    df = df.sort_index()
    
    # Get last hours_needed+ hours
    if len(df) < hours_needed:
        print(f"  Warning: Only {len(df)} hours available, need at least {hours_needed}")
    else:
        df = df.iloc[-hours_needed:]
    
    print(f"  ✓ Loaded {len(df):,} hours of historical data")
    print(f"    Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def autoregressive_predict(model, feature_df, historical_df, training_features=None):
    """
    Make autoregressive multi-step predictions
    
    Parameters:
    -----------
    model : EnhancedModel
        Trained model
    feature_df : pd.DataFrame
        Feature matrix for forecast period (will be updated iteratively)
    historical_df : pd.DataFrame
        Historical data for initial lag values
    
    Returns:
    --------
    np.array
        Array of price predictions
    """
    print("\n" + "="*80)
    print("AUTOREGRESSIVE PREDICTION")
    print("="*80)
    
    predictions = []
    feature_df = feature_df.copy()
    
    # Get last historical values
    last_price = historical_df['P(T)'].iloc[-1] if 'P(T)' in historical_df.columns else None
    last_load = historical_df['L(T-1)'].iloc[-1] if 'L(T-1)' in historical_df.columns else None
    
    # Prepare feature matrix - use exact features from training dataset
    # Priority: training_features parameter > model.feature_names > historical_df columns
    if training_features is not None:
        expected_features = training_features
    elif model.feature_names and len(model.feature_names) > 0:
        expected_features = model.feature_names
    else:
        # Use features from historical dataset (what model was trained with)
        expected_features = [c for c in historical_df.columns if c != 'P(T)']
    
    # Ensure we have all expected features
    missing_features = set(expected_features) - set(feature_df.columns)
    if missing_features:
        print(f"  Warning: Missing features: {missing_features}")
        # Fill with defaults
        for feat in missing_features:
            feature_df[feat] = 0
    
    # Select only expected features in correct order (critical for XGBoost)
    feature_names = [f for f in expected_features if f in feature_df.columns]
    
    # If we still have more features than expected, trim to match
    if len(feature_names) > len(expected_features):
        print(f"  Warning: More features than expected, using first {len(expected_features)}")
        feature_names = feature_names[:len(expected_features)]
    
    print(f"  Using {len(feature_names)} features for prediction (expected: {len(expected_features)})")
    
    X = feature_df[feature_names].copy()
    
    # Ensure float dtype for all features (XGBoost expects float)
    for col in X.columns:
        if X[col].dtype == 'int64' or X[col].dtype == 'int32':
            X[col] = X[col].astype('float64')
    
    # Fill any missing values
    X = X.fillna(X.median())
    
    print(f"\nMaking predictions for {len(X)} hours...")
    
    for hour_idx in range(len(X)):
        # Get features for this hour
        X_hour = X.iloc[hour_idx:hour_idx+1]
        
        # Update lag features with previous predictions
        if hour_idx > 0 and 'P(T-1)' in X_hour.columns:
            X_hour['P(T-1)'] = predictions[-1]
        
        if hour_idx >= 1 and 'P(T-2)' in X_hour.columns:
            X_hour['P(T-2)'] = predictions[-2] if len(predictions) >= 2 else last_price
        
        if hour_idx >= 24 and 'P(T-24)' in X_hour.columns:
            X_hour['P(T-24)'] = predictions[-24]
        
        if hour_idx >= 48 and 'P(T-48)' in X_hour.columns:
            X_hour['P(T-48)'] = predictions[-48]
        
        if hour_idx >= 168 and 'P_T-168' in X_hour.columns:
            X_hour['P_T-168'] = predictions[-168]
        
        if hour_idx >= 3 and 'P_T-3' in X_hour.columns:
            X_hour['P_T-3'] = predictions[-3]
        
        if hour_idx >= 4 and 'P_T-4' in X_hour.columns:
            X_hour['P_T-4'] = predictions[-4]
        
        # Update rolling statistics
        if hour_idx > 0:
            # Combine historical + predictions for rolling window
            all_prices = []
            if 'P(T)' in historical_df.columns:
                all_prices.extend(historical_df['P(T)'].iloc[-24:].values)
            all_prices.extend(predictions)
            
            if len(all_prices) >= 24:
                window_prices = all_prices[-24:]
                if 'Price_MA_24h' in X_hour.columns:
                    X_hour['Price_MA_24h'] = np.mean(window_prices)
                if 'Price_Std_24h' in X_hour.columns:
                    X_hour['Price_Std_24h'] = np.std(window_prices) if len(window_prices) > 1 else 0
        
        # Make prediction
        pred = model.predict(X_hour)
        predictions.append(pred[0])
        
        # Update feature_df for next iteration
        X.iloc[hour_idx] = X_hour.iloc[0]
        
        if (hour_idx + 1) % 24 == 0:
            print(f"  Completed {hour_idx + 1}/{len(X)} hours")
    
    print(f"\n✓ Predictions complete")
    print(f"  Price range: ₹{np.min(predictions):,.2f} - ₹{np.max(predictions):,.2f}")
    print(f"  Average: ₹{np.mean(predictions):,.2f}")
    
    return np.array(predictions)


def generate_forecast(
    start_date,
    daily_peak_loads,
    weather_forecasts=None,
    model_path='models/training/enhanced_model.pkl',
    quantile_model_path='models/training/quantile_models.pkl',
    historical_data_path='data/processed/dataset_cleaned.parquet',
    forecast_days=7
):
    """
    Main inference pipeline to generate price forecasts
    
    Parameters:
    -----------
    start_date : str or datetime
        Forecast start date/time (will start at 00:00 IST)
    daily_peak_loads : list
        List of daily peak load values (MW) for forecast period
    weather_forecasts : dict, optional
        Pre-fetched weather forecasts (if None, will fetch from API)
    model_path : str
        Path to enhanced model
    quantile_model_path : str
        Path to quantile models
    historical_data_path : str
        Path to historical dataset
    forecast_days : int
        Number of days to forecast (default: 7)
    
    Returns:
    --------
    dict
        Dictionary with predictions and metadata
    """
    print("="*80)
    print("IEX PRICE FORECAST GENERATION")
    print("="*80)
    
    # Parse start date
    if isinstance(start_date, str):
        start_dt = pd.to_datetime(start_date)
    else:
        start_dt = start_date
    
    if start_dt.tz is None:
        start_dt = start_dt.tz_localize('Asia/Kolkata')
    else:
        start_dt = start_dt.tz_convert('Asia/Kolkata')
    
    start_dt = start_dt.normalize()  # Set to 00:00:00
    
    print(f"\nForecast start: {start_dt}")
    print(f"Forecast days: {forecast_days}")
    print(f"Daily peaks provided: {len(daily_peak_loads)}")
    
    # Step 1: Load historical context
    historical_df = load_historical_context(historical_data_path, hours_needed=168)
    
    # Get exact feature list from historical dataset (what model was trained with)
    training_features = [c for c in historical_df.columns if c != 'P(T)']
    
    # Step 2: Fetch weather forecasts
    if weather_forecasts is None:
        print("\nFetching weather forecasts from Open-Meteo API...")
        city_forecasts = fetch_weather_forecast(forecast_days=forecast_days)
        weather_forecast_df = aggregate_forecast_to_national(city_forecasts)
    else:
        print("\nUsing provided weather forecasts...")
        weather_forecast_df = aggregate_forecast_to_national(weather_forecasts)
    
    # Step 3: Convert daily peak to hourly load profile
    print("\nConverting daily peak loads to hourly profile...")
    load_patterns = analyze_load_patterns(historical_df)
    load_forecast_df = daily_peak_to_hourly(
        daily_peak_loads,
        start_dt,
        load_patterns=load_patterns
    )
    
    # Step 4: Estimate supply breakdown
    print("\nEstimating supply-side breakdown...")
    supply_forecast_df = estimate_supply_breakdown(
        load_forecast_df,
        weather_forecast_df,
        historical_df=historical_df
    )
    
    # Step 5: Generate forecast timestamps
    forecast_timestamps = pd.date_range(
        start=start_dt,
        periods=forecast_days * 24,
        freq='h',
        tz='Asia/Kolkata'
    )
    
    # Step 6: Build features (initial, without autoregressive updates)
    print("\nBuilding initial feature matrix...")
    feature_df = build_features_for_forecast(
        historical_df,
        forecast_timestamps,
        weather_forecast_df,
        load_forecast_df,
        supply_forecast_df,
        price_predictions=None
    )
    
    # Step 7: Load models
    print("\nLoading models...")
    model_data = joblib.load(model_path)
    if isinstance(model_data, dict) and 'model' in model_data:
        enhanced_model = EnhancedModel()
        enhanced_model.model = model_data['model']
        # Use saved feature_names if available
        saved_feature_names = model_data.get('feature_names')
        if saved_feature_names is not None and len(saved_feature_names) > 0:
            enhanced_model.feature_names = saved_feature_names
        else:
            # Try to infer from XGBoost model
            if hasattr(model_data['model'], 'get_booster'):
                booster = model_data['model'].get_booster()
                if booster.feature_names and len(booster.feature_names) > 0:
                    enhanced_model.feature_names = list(booster.feature_names)
        enhanced_model.config = model_data.get('config', {})
    else:
        # Fallback: assume it's the model directly
        enhanced_model = EnhancedModel()
        enhanced_model.model = model_data
    print(f"  ✓ Loaded enhanced model from {model_path}")
    if enhanced_model.feature_names:
        print(f"    Model expects {len(enhanced_model.feature_names)} features")
    
    quantile_model = QuantileModel.load(quantile_model_path)
    print(f"  ✓ Loaded quantile models from {quantile_model_path}")
    
    # Step 8: Make autoregressive predictions
    price_predictions = autoregressive_predict(
        enhanced_model,
        feature_df,
        historical_df,
        training_features=training_features
    )
    
    # Step 9: Generate quantile predictions
    print("\nGenerating quantile predictions...")
    # Rebuild features with predictions for quantile model
    feature_df_quantile = build_features_for_forecast(
        historical_df,
        forecast_timestamps,
        weather_forecast_df,
        load_forecast_df,
        supply_forecast_df,
        price_predictions=price_predictions
    )
    
    # Use same feature names as enhanced model
    if enhanced_model.feature_names is not None:
        feature_names_quantile = [f for f in enhanced_model.feature_names if f in feature_df_quantile.columns]
    else:
        feature_names_quantile = training_features
    
    # Ensure we have all required features
    missing_features = set(feature_names_quantile) - set(feature_df_quantile.columns)
    if missing_features:
        for feat in missing_features:
            feature_df_quantile[feat] = 0
    
    X_quantile = feature_df_quantile[feature_names_quantile].copy()
    
    # Ensure float dtype for all features
    for col in X_quantile.columns:
        if X_quantile[col].dtype == 'int64' or X_quantile[col].dtype == 'int32' or X_quantile[col].dtype == 'float32':
            X_quantile[col] = X_quantile[col].astype('float64')
    
    X_quantile = X_quantile.fillna(X_quantile.median())
    
    quantile_predictions = quantile_model.predict_intervals(X_quantile)
    
    # Step 10: Prepare output
    print("\nPreparing output...")
    results_df = pd.DataFrame({
        'timestamp': forecast_timestamps,
        'price_p50': price_predictions,  # Median from enhanced model
        'price_p10': quantile_predictions[0.1],
        'price_p90': quantile_predictions[0.9],
        'load_forecast': load_forecast_df['load_mw'].values,
        'temperature': weather_forecast_df['temperature_2m_national'].values,
        'demand': supply_forecast_df['Demand'].values,
        're_generation': supply_forecast_df['RE_Generation'].values,
        'thermal': supply_forecast_df['Thermal'].values,
    })
    
    results_df = results_df.set_index('timestamp')
    
    # Get feature count for metadata
    if enhanced_model.feature_names:
        feature_count = len(enhanced_model.feature_names)
    else:
        feature_count = len(training_features)
    
    metadata = {
        'model': 'enhanced',
        'forecast_start': str(start_dt),
        'forecast_end': str(forecast_timestamps[-1]),
        'forecast_hours': len(forecast_timestamps),
        'features_used': feature_count,
        'price_range': {
            'min': float(np.min(price_predictions)),
            'max': float(np.max(price_predictions)),
            'mean': float(np.mean(price_predictions))
        }
    }
    
    print("\n" + "="*80)
    print("FORECAST GENERATION COMPLETE")
    print("="*80)
    print(f"\nForecast Summary:")
    print(f"  Period: {metadata['forecast_start']} to {metadata['forecast_end']}")
    print(f"  Hours: {metadata['forecast_hours']}")
    print(f"  Price Range: ₹{metadata['price_range']['min']:,.2f} - ₹{metadata['price_range']['max']:,.2f}")
    print(f"  Average Price: ₹{metadata['price_range']['mean']:,.2f}")
    
    return {
        'predictions': results_df,
        'metadata': metadata
    }
