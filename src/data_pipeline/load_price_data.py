"""
Load and preprocess IEX price data from combined parquet file
Replaces old Price.xlsx with combined IEX dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_price_data(filepath='data/raw/price/iex_dam_combined.parquet', start_date='2021-01-01', end_date='2025-12-31'):
    """
    Load IEX price data from combined parquet file
    
    Parameters:
    -----------
    filepath : str
        Path to combined IEX price parquet file (default: iex_dam_combined.parquet)
    start_date : str
        Start date filter in 'YYYY-MM-DD' format
    end_date : str
        End date filter in 'YYYY-MM-DD' format
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with datetime index and all price/load features
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Price data file not found: {filepath}")
    
    print(f"Loading IEX price data from {filepath}...")
    
    # Read parquet file (already has datetime index and lagged features)
    df = pd.read_parquet(filepath)
    
    print(f"  Original shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    # Filter by date range if specified
    if start_date or end_date:
        start_ts = pd.Timestamp(start_date, tz='Asia/Kolkata') if start_date else df.index.min()
        end_ts = pd.Timestamp(end_date, tz='Asia/Kolkata') if end_date else df.index.max()
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        print(f"  Filtered to: {start_ts} to {end_ts}")
        print(f"  Filtered shape: {df.shape}")
    
    # Ensure index is named
    df.index.name = 'Timestamp'
    
    # Validate column names
    expected_cols = [
        'L(T-1)', 'L(T-2)', 'L(T-24)', 'L(T-48)',
        'P(T-1)', 'P(T-2)', 'P(T-24)', 'P(T-48)',
        'Day', 'Season', 'P(T)'
    ]
    
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        print(f"  WARNING: Missing columns: {missing_cols}")
    
    # Data quality checks
    print("\nData Quality Checks:")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Price range: ₹{df['P(T)'].min():,.2f} - ₹{df['P(T)'].max():,.2f}")
    print(f"  Load range: {df['L(T-1)'].min():,.2f} - {df['L(T-1)'].max():,.2f} MW")
    print(f"  Prices at ceiling (₹20,000): {(df['P(T)'] == 20000).sum()} ({100*(df['P(T)'] == 20000).sum()/len(df):.2f}%)")
    
    # Season encoding check
    print(f"\nSeason encoding:")
    print(f"  Unique values: {sorted(df['Season'].unique())}")
    print(f"  Value counts:\n{df['Season'].value_counts().sort_index()}")
    
    # Day encoding check
    print(f"\nDay encoding:")
    print(f"  Unique values: {sorted(df['Day'].unique())}")
    print(f"  Value counts:\n{df['Day'].value_counts().sort_index()}")
    
    # Check for gaps in timestamps
    time_diffs = df.index.to_series().diff()
    gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]
    if len(gaps) > 0:
        print(f"\n  WARNING: Found {len(gaps)} gaps > 1 hour in timestamps")
        print(f"  Largest gap: {gaps.max()}")
    
    print(f"\n✓ Price data loaded successfully")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Total hours: {len(df):,}")
    
    return df


def load_extended_price_data(extended_file=None, start_date='2023-08-30', end_date='2026-01-28'):
    """
    Load extended price data from combined IEX dataset
    
    Parameters:
    -----------
    extended_file : str, optional
        Path to combined price parquet file (default: iex_dam_combined.parquet)
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    
    Returns:
    --------
    pd.DataFrame or None
        Extended price dataframe
    """
    # Use combined dataset as default
    if extended_file is None:
        extended_file = Path('data/raw/price/iex_dam_combined.parquet')
    else:
        extended_file = Path(extended_file)
    
    if extended_file.exists():
        print(f"Loading extended price data from {extended_file}...")
        try:
            df = pd.read_parquet(extended_file)
            
            # Filter by date range
            start_ts = pd.Timestamp(start_date, tz='Asia/Kolkata')
            end_ts = pd.Timestamp(end_date, tz='Asia/Kolkata')
            df = df[(df.index >= start_ts) & (df.index <= end_ts)]
            
            print(f"  Loaded {len(df):,} rows")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
            return df
        except Exception as e:
            print(f"  ERROR loading extended file: {e}")
            import traceback
            traceback.print_exc()
    
    return None


def merge_price_dataframes(original_df, extended_df):
    """
    Merge original and extended price dataframes
    
    Parameters:
    -----------
    original_df : pd.DataFrame
        Original price data
    extended_df : pd.DataFrame
        Extended price data
    
    Returns:
    --------
    pd.DataFrame
        Merged dataframe
    """
    if extended_df is None or len(extended_df) == 0:
        return original_df
    
    # Ensure same columns
    common_cols = set(original_df.columns) & set(extended_df.columns)
    if not common_cols:
        print("  WARNING: No common columns between original and extended data")
        return original_df
    
    # Align columns
    original_df = original_df[list(common_cols)]
    extended_df = extended_df[list(common_cols)]
    
    # Combine dataframes
    combined_df = pd.concat([original_df, extended_df], axis=0)
    
    # Remove duplicates (keep first occurrence where overlap exists)
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df = combined_df.sort_index()
    
    print(f"  Merged: {len(original_df):,} original + {len(extended_df):,} extended = {len(combined_df):,} total")
    print(f"  Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    
    return combined_df


def validate_price_data(df):
    """
    Validate price data for common issues
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price dataframe with datetime index
    
    Returns:
    --------
    dict
        Validation results
    """
    results = {
        'is_valid': True,
        'issues': []
    }
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        results['is_valid'] = False
        results['issues'].append('Missing values detected')
    
    # Check price range
    if 'P(T)' in df.columns:
        if df['P(T)'].min() < 0:
            results['is_valid'] = False
            results['issues'].append('Negative prices found')
        
        if df['P(T)'].max() > 20000:
            results['is_valid'] = False
            results['issues'].append('Prices exceed ceiling (₹20,000)')
    
    # Check load range (reasonable for India)
    if 'L(T-1)' in df.columns:
        if df['L(T-1)'].min() < 0:
            results['is_valid'] = False
            results['issues'].append('Negative load values found')
        
        if df['L(T-1)'].max() > 250000:  # Very high but possible during peak
            results['issues'].append('Very high load values (>250 GW) - verify')
    
    # Check timestamp continuity
    expected_freq = pd.Timedelta(hours=1)
    time_diffs = df.index.to_series().diff().dropna()
    if (time_diffs > expected_freq * 2).any():
        results['issues'].append('Large gaps in timestamps detected')
    
    return results


if __name__ == "__main__":
    # Test loading
    df = load_price_data()
    validation = validate_price_data(df)
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"Valid: {validation['is_valid']}")
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    # Save sample
    print(f"\nSample data (first 5 rows):")
    print(df.head())
