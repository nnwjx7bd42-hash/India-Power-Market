"""
Load extended IEX DAM price data from Aug 30, 2023 to Jan 28, 2026
Supports manual CSV files or scraping from IEX website
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import requests
import time
import sys

# Import parse_iex_dam from same directory
sys.path.insert(0, str(Path(__file__).parent))
from parse_iex_dam import parse_iex_dam_csv, parse_iex_dam_directory


def download_iex_dam_daily(date_str, output_dir='data/raw/price/iex_dam'):
    """
    Download IEX DAM data for a specific date
    
    Parameters:
    -----------
    date_str : str
        Date in 'YYYY-MM-DD' format
    output_dir : str
        Directory to save CSV files
    
    Returns:
    --------
    bool
        True if successful
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # IEX Market Snapshot URL
    # Note: This is a placeholder - actual URL structure may vary
    base_url = "https://www.iexindia.com/market-data/day-ahead-market/market-snapshot"
    
    # Alternative: Third-party aggregator
    alt_url = f"https://iexrtmprice.com/DSM_Data/"
    
    print(f"  Attempting to download data for {date_str}...")
    print(f"    NOTE: Manual download may be required from:")
    print(f"    - {base_url}")
    print(f"    - {alt_url}")
    
    # For now, return False - user needs to manually download
    # In production, implement actual scraping logic here
    return False


def load_extended_iex_prices(start_date='2023-08-30', end_date='2026-01-28',
                             data_dir='data/raw/price/iex_dam',
                             manual_csv_dir=None):
    """
    Load extended IEX DAM price data
    
    Parameters:
    -----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    data_dir : str
        Directory for downloaded CSV files
    manual_csv_dir : str, optional
        Directory containing manually downloaded CSV files
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with datetime index, MCP, MCV columns
    """
    print("="*80)
    print("LOADING EXTENDED IEX DAM PRICE DATA")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    
    # Check for manually downloaded files first
    if manual_csv_dir:
        manual_dir = Path(manual_csv_dir)
        if manual_dir.exists():
            print(f"\nChecking manual CSV directory: {manual_csv_dir}")
            df = parse_iex_dam_directory(manual_dir, start_date, end_date)
            if df is not None and len(df) > 0:
                return df
    
    # Check data_dir for existing files
    data_path = Path(data_dir)
    if data_path.exists():
        csv_files = list(data_path.glob('*.csv'))
        if csv_files:
            print(f"\nFound {len(csv_files)} CSV files in {data_dir}")
            df = parse_iex_dam_directory(data_dir, start_date, end_date)
            if df is not None and len(df) > 0:
                return df
    
    # If no files found, provide instructions
    print("\n" + "="*80)
    print("NO IEX DAM DATA FOUND")
    print("="*80)
    print("\nTo acquire IEX DAM price data:")
    print("1. Visit: https://www.iexindia.com/market-data/day-ahead-market/market-snapshot")
    print("2. Download daily CSV files for Aug 30, 2023 → Jan 28, 2026")
    print("3. Save files to: data/raw/price/iex_dam/")
    print("\nOR")
    print("1. Visit: https://iexrtmprice.com/DSM_Data/")
    print("2. Download aggregated data")
    print("3. Save to: data/raw/price/iex_dam/")
    print("\nThen re-run this script.")
    
    return None


def create_extended_price_dataframe(iex_df, start_date='2023-08-30', end_date='2026-01-28'):
    """
    Create price dataframe in the same format as Price.xlsx
    
    Parameters:
    -----------
    iex_df : pd.DataFrame
        IEX DAM data with MCP, MCV columns
    start_date : str
        Start date
    end_date : str
        End date
    
    Returns:
    --------
    pd.DataFrame
        DataFrame matching Price.xlsx format
    """
    if iex_df is None or len(iex_df) == 0:
        return None
    
    # Create hourly timestamps for the period
    start_ts = pd.Timestamp(start_date, tz='Asia/Kolkata')
    end_ts = pd.Timestamp(end_date, tz='Asia/Kolkata')
    hourly_range = pd.date_range(start=start_ts, end=end_ts, freq='1H', tz='Asia/Kolkata')
    
    # Create base dataframe
    result_df = pd.DataFrame(index=hourly_range)
    
    # Fill MCP (price) and MCV (load/volume)
    result_df['MCP'] = np.nan
    result_df['MCV'] = np.nan
    
    # Merge IEX data
    for ts in iex_df.index:
        if ts in result_df.index:
            result_df.loc[ts, 'MCP'] = iex_df.loc[ts, 'MCP']
            if 'MCV' in iex_df.columns:
                result_df.loc[ts, 'MCV'] = iex_df.loc[ts, 'MCV']
    
    # Create lag features (similar to Price.xlsx format)
    result_df['P(T)'] = result_df['MCP']  # Target variable
    result_df['P(T-1)'] = result_df['P(T)'].shift(1)
    result_df['P(T-2)'] = result_df['P(T)'].shift(2)
    result_df['P(T-24)'] = result_df['P(T)'].shift(24)
    result_df['P(T-48)'] = result_df['P(T)'].shift(48)
    
    # Use MCV as load, create load lags
    if 'MCV' in result_df.columns:
        result_df['L(T-1)'] = result_df['MCV'].shift(1)
        result_df['L(T-2)'] = result_df['MCV'].shift(2)
        result_df['L(T-24)'] = result_df['MCV'].shift(24)
        result_df['L(T-48)'] = result_df['MCV'].shift(48)
    else:
        # If no MCV, create placeholder (will need to be filled from demand data)
        result_df['L(T-1)'] = np.nan
        result_df['L(T-2)'] = np.nan
        result_df['L(T-24)'] = np.nan
        result_df['L(T-48)'] = np.nan
    
    # Add Day and Season features
    result_df['Day'] = (result_df.index.dayofweek >= 5).astype(int)  # Weekend=1
    result_df['Season'] = result_df.index.month.apply(
        lambda m: 0 if m in [12, 1, 2] else (1 if m in [3, 4, 5] else (2 if m in [6, 7, 8, 9] else 3))
    )
    
    # Select columns matching Price.xlsx format
    columns = [
        'L(T-1)', 'L(T-2)', 'L(T-24)', 'L(T-48)',
        'P(T-1)', 'P(T-2)', 'P(T-24)', 'P(T-48)',
        'Day', 'Season', 'P(T)'
    ]
    
    result_df = result_df[columns]
    
    return result_df


if __name__ == "__main__":
    # Load extended IEX prices
    iex_df = load_extended_iex_prices()
    
    if iex_df is not None:
        # Create formatted dataframe
        price_df = create_extended_price_dataframe(iex_df)
        
        if price_df is not None:
            # Save to parquet
            output_path = Path('data/raw/price/iex_dam_extended.parquet')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            price_df.to_parquet(output_path)
            
            print(f"\n✓ Extended price data saved to: {output_path}")
            print(f"  Shape: {price_df.shape}")
            print(f"  Date range: {price_df.index.min()} to {price_df.index.max()}")
