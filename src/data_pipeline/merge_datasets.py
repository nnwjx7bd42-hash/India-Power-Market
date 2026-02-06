"""
Merge IEX price data and NERLDC data on timestamp
Creates unified dataset ready for feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from load_price_data import load_price_data, load_extended_price_data, merge_price_dataframes
from load_nerdc_data import load_all_nerdc_files


def merge_price_nerdc(price_df, nerdc_df):
    """
    Merge Price and NERLDC datasets on timestamp
    
    Parameters:
    -----------
    price_df : pd.DataFrame
        Price data with datetime index
    nerdc_df : pd.DataFrame
        NERLDC data with datetime index
    
    Returns:
    --------
    pd.DataFrame
        Merged dataset
    """
    print("="*80)
    print("MERGING PRICE AND NERLDC DATA")
    print("="*80)
    
    # Ensure both have datetime index
    if not isinstance(price_df.index, pd.DatetimeIndex):
        raise ValueError("Price dataframe must have datetime index")
    if not isinstance(nerdc_df.index, pd.DatetimeIndex):
        raise ValueError("NERLDC dataframe must have datetime index")
    
    # Align timezones
    if price_df.index.tz is None:
        price_df.index = price_df.index.tz_localize('Asia/Kolkata')
    if nerdc_df.index.tz is None:
        nerdc_df.index = nerdc_df.index.tz_localize('Asia/Kolkata')
    
    # Find overlap period
    overlap_start = max(price_df.index.min(), nerdc_df.index.min())
    overlap_end = min(price_df.index.max(), nerdc_df.index.max())
    
    print(f"\nPrice data range: {price_df.index.min()} to {price_df.index.max()}")
    print(f"NERLDC data range: {nerdc_df.index.min()} to {nerdc_df.index.max()}")
    print(f"Overlap period: {overlap_start} to {overlap_end}")
    
    # Filter to overlap period
    price_overlap = price_df[(price_df.index >= overlap_start) & (price_df.index <= overlap_end)]
    nerdc_overlap = nerdc_df[(nerdc_df.index >= overlap_start) & (nerdc_df.index <= overlap_end)]
    
    print(f"\nPrice rows in overlap: {len(price_overlap):,}")
    print(f"NERLDC rows in overlap: {len(nerdc_overlap):,}")
    
    # Merge on timestamp (inner join to keep only matching timestamps)
    merged_df = pd.merge(
        price_overlap,
        nerdc_overlap,
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    print(f"\nMerged dataset:")
    print(f"  Total rows: {len(merged_df):,}")
    print(f"  Date range: {merged_df.index.min()} to {merged_df.index.max()}")
    print(f"  Total columns: {len(merged_df.columns)}")
    
    # Check for missing values after merge
    print(f"\nMissing values after merge:")
    missing = merged_df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(missing_cols)
    else:
        print("  None")
    
    # Validate alignment
    print(f"\nAlignment validation:")
    print(f"  Timestamps match: {len(merged_df):,} common timestamps")
    
    # Check if load values from Price match Demand from NERLDC (should be similar)
    if 'L(T-1)' in merged_df.columns and 'Demand' in merged_df.columns:
        # Compare current load with demand (they should be similar)
        load_demand_diff = (merged_df['L(T-1)'] - merged_df['Demand']).abs()
        print(f"  Load vs Demand comparison:")
        print(f"    Mean absolute difference: {load_demand_diff.mean():,.2f} MW")
        print(f"    Max difference: {load_demand_diff.max():,.2f} MW")
        print(f"    Correlation: {merged_df['L(T-1)'].corr(merged_df['Demand']):.4f}")
    
    return merged_df


def create_unified_dataset(price_file='data/raw/price/iex_dam_combined.parquet', data_dir='data/raw/nerldc',
                          extended_price_file=None, extend_period=True):
    """
    Create unified dataset from IEX price data and NERLDC files
    
    Parameters:
    -----------
    price_file : str
        Path to combined IEX price parquet file (default: iex_dam_combined.parquet)
    data_dir : str
        Directory containing NERLDC files
    extended_price_file : str, optional
        Path to additional extended price data (parquet) - deprecated, use price_file instead
    extend_period : bool
        If True, load full date range from combined dataset
    
    Returns:
    --------
    pd.DataFrame
        Unified dataset ready for feature engineering
    """
    print("="*80)
    print("CREATING UNIFIED DATASET")
    print("="*80)
    
    # Load Price data (now from combined dataset)
    print("\n1. Loading IEX price data...")
    if extend_period:
        # Load up to June 24, 2025 (matching NERLDC coverage)
        price_df = load_price_data(price_file, start_date='2021-01-01', end_date='2025-06-24')
    else:
        # Load only original period
        price_df = load_price_data(price_file, start_date='2021-01-01', end_date='2023-08-31')
    
    # Note: extended_price_file parameter is deprecated since combined dataset
    # already contains the full period. Keeping for backward compatibility.
    if extended_price_file and Path(extended_price_file).exists():
        print("\n1b. Loading additional extended price data...")
        extended_price_df = load_extended_price_data(extended_price_file)
        if extended_price_df is not None:
            print("  Merging extended price data...")
            price_df = merge_price_dataframes(price_df, extended_price_df)
    
    # Load NERLDC data
    print("\n2. Loading NERLDC data...")
    nerdc_df = load_all_nerdc_files(data_dir)
    
    if nerdc_df is None:
        raise ValueError("Failed to load NERLDC data")
    
    # Limit NERLDC data to June 24, 2025 (last available date)
    nerdc_end_limit = pd.Timestamp('2025-06-24 23:00:00', tz='Asia/Kolkata')
    if nerdc_df.index.max() > nerdc_end_limit:
        print(f"\n2b. Limiting NERLDC data to June 24, 2025...")
        nerdc_df = nerdc_df[nerdc_df.index <= nerdc_end_limit]
        print(f"  ✓ Filtered to: {nerdc_df.index.min()} to {nerdc_df.index.max()}")
        print(f"  Total hours: {len(nerdc_df):,}")
    
    # Merge datasets
    print("\n3. Merging datasets...")
    merged_df = merge_price_nerdc(price_df, nerdc_df)
    
    # Select only relevant columns (remove extra SCADA columns)
    relevant_cols = [
        # Price/Load features
        'L(T-1)', 'L(T-2)', 'L(T-24)', 'L(T-48)',
        'P(T-1)', 'P(T-2)', 'P(T-24)', 'P(T-48)',
        'Day', 'Season',
        # NERLDC features
        'Thermal', 'Hydro', 'Gas', 'Nuclear', 'Wind', 'Solar', 'Demand',
        # Target
        'P(T)'
    ]
    
    # Keep only columns that exist
    available_cols = [col for col in relevant_cols if col in merged_df.columns]
    merged_df = merged_df[available_cols]
    
    print(f"\n4. Final dataset:")
    print(f"  Columns: {list(merged_df.columns)}")
    print(f"  Rows: {len(merged_df):,}")
    print(f"  Date range: {merged_df.index.min()} to {merged_df.index.max()}")
    
    return merged_df


if __name__ == "__main__":
    # Create unified dataset
    unified_df = create_unified_dataset()
    
    print("\n" + "="*80)
    print("SAMPLE DATA")
    print("="*80)
    print(unified_df.head(10))
    
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    print(unified_df.describe())
