"""
Parser for IEX DAM (Day-Ahead Market) CSV files
Handles 15-minute block data and aggregates to hourly
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def parse_iex_dam_csv(filepath, aggregate_to_hourly=True):
    """
    Parse IEX DAM CSV file with 15-minute blocks
    
    Parameters:
    -----------
    filepath : str or Path
        Path to IEX DAM CSV file
    aggregate_to_hourly : bool
        If True, aggregate 15-minute blocks to hourly (mean)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with datetime index and MCP, MCV columns
    """
    print(f"  Parsing IEX DAM file: {filepath}")
    
    try:
        # Try reading CSV - format may vary
        df = pd.read_csv(filepath)
        
        # Common column name patterns in IEX DAM files
        # Date column might be: Date, DATE, Date/Time, Timestamp
        # Time block: Block, Block No, Time Block, Block_No
        # MCP: MCP, Market Clearing Price, Price
        # MCV: MCV, Market Clearing Volume, Volume
        
        # Find date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            # Try first column
            date_col = df.columns[0]
        
        # Find block/time column
        block_col = None
        for col in df.columns:
            if 'block' in col.lower() or 'time' in col.lower():
                block_col = col
                break
        
        # Find MCP column
        mcp_col = None
        for col in df.columns:
            if 'mcp' in col.lower() or ('price' in col.lower() and 'clearing' in col.lower()):
                mcp_col = col
                break
        
        # Find MCV column
        mcv_col = None
        for col in df.columns:
            if 'mcv' in col.lower() or ('volume' in col.lower() and 'clearing' in col.lower()):
                mcv_col = col
                break
        
        if mcp_col is None or mcv_col is None:
            print(f"    WARNING: Could not find MCP/MCV columns. Available columns: {list(df.columns)}")
            return None
        
        # Parse date
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        
        # If we have block column, create hourly timestamps
        if block_col:
            # Blocks are typically 1-96 for a day (15-minute intervals)
            # Or 1-24 for hourly
            # Convert to datetime
            if df[block_col].max() <= 24:
                # Already hourly
                df['hour'] = df[block_col] - 1  # Convert to 0-23
                df['datetime'] = df[date_col] + pd.to_timedelta(df['hour'], unit='h')
            else:
                # 15-minute blocks (1-96)
                df['block'] = df[block_col]
                df['hour'] = (df['block'] - 1) // 4  # Convert to hour (0-23)
                df['minute'] = ((df['block'] - 1) % 4) * 15
                df['datetime'] = df[date_col] + pd.to_timedelta(df['hour'], unit='h') + pd.to_timedelta(df['minute'], unit='m')
        else:
            # Assume date column has full datetime
            df['datetime'] = df[date_col]
        
        # Set timezone
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('Asia/Kolkata')
        else:
            df['datetime'] = df['datetime'].dt.tz_convert('Asia/Kolkata')
        
        # Extract MCP and MCV
        result_df = pd.DataFrame({
            'MCP': df[mcp_col].values,
            'MCV': df[mcv_col].values if mcv_col else None
        }, index=df['datetime'])
        
        result_df = result_df.sort_index()
        
        # Aggregate to hourly if needed
        if aggregate_to_hourly and len(result_df) > 0:
            # Check if already hourly
            time_diffs = result_df.index.to_series().diff().dropna()
            if (time_diffs <= pd.Timedelta('1h')).all() and (time_diffs >= pd.Timedelta('1h')).all():
                # Already hourly
                pass
            else:
                # Aggregate 15-minute blocks to hourly
                result_df = result_df.resample('1H', label='right', closed='right').agg({
                    'MCP': 'mean',  # Average price
                    'MCV': 'sum'    # Sum volume
                })
        
        print(f"    ✓ Parsed {len(result_df):,} hourly records")
        print(f"      Date range: {result_df.index.min()} to {result_df.index.max()}")
        
        return result_df
        
    except Exception as e:
        print(f"    ERROR parsing file: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_iex_dam_directory(directory, start_date='2023-08-30', end_date='2026-01-28'):
    """
    Parse all IEX DAM CSV files in a directory
    
    Parameters:
    -----------
    directory : str or Path
        Directory containing IEX DAM CSV files
    start_date : str
        Start date filter (YYYY-MM-DD)
    end_date : str
        End date filter (YYYY-MM-DD)
    
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all parsed data
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(directory.glob('*.csv'))
    
    if not csv_files:
        print(f"  No CSV files found in {directory}")
        return None
    
    print(f"  Found {len(csv_files)} CSV files")
    
    dataframes = []
    for csv_file in csv_files:
        df = parse_iex_dam_csv(csv_file)
        if df is not None:
            # Filter by date range
            start_ts = pd.Timestamp(start_date, tz='Asia/Kolkata')
            end_ts = pd.Timestamp(end_date, tz='Asia/Kolkata')
            df = df[(df.index >= start_ts) & (df.index <= end_ts)]
            if len(df) > 0:
                dataframes.append(df)
    
    if not dataframes:
        print("  No data parsed")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, axis=0)
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
    combined_df = combined_df.sort_index()
    
    print(f"\n  ✓ Combined {len(combined_df):,} hourly records")
    print(f"    Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    
    return combined_df
