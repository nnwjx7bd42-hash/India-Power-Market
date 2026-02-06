"""
Load and preprocess NERLDC monthly files

Data Source:
    North-Eastern Regional Load Despatch Centre (NERLDC), Grid-India
    Ministry of Power, Government of India
    
    Official description: PMU-electricity demand, solar and wind generation data
    of all 5 regional grid networks combined at 1-hour interval
    Period: September 2021 to December 2023

Note:
    Raw files contain 5-minute SCADA data with all generation types
    (Thermal, Hydro, Gas, Nuclear, Wind, Solar, Demand).
    We resample to hourly resolution (using mean) to match the official
    1-hour interval specification and for consistency with other data sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# SCADA tag to readable name mapping
SCADA_COLUMN_MAPPING = {
    'SCADA/ANALOG/044MQ068/0': 'Thermal',
    'SCADA/ANALOG/044MQ062/0': 'Hydro',
    'SCADA/ANALOG/044MQ063/0': 'Gas',
    'SCADA/ANALOG/044MQ064/0': 'Nuclear',
    'SCADA/ANALOG/044MQ070/0': 'Wind',
    'SCADA/ANALOG/044MQ206/0': 'Solar',
    'SCADA/ANALOG/044MQ067/0': 'Demand'
}


def parse_nerdc_file(filepath, sheet_name='Sheet1'):
    """
    Parse a single NERLDC monthly file
    
    Parameters:
    -----------
    filepath : str or Path
        Path to NERLDC Excel file
    sheet_name : str
        Sheet name containing data (default: 'Sheet1')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with datetime index and generation/demand columns
    """
    filepath = Path(filepath)
    print(f"  Parsing {filepath.name}...")
    
    try:
        # Read Sheet1
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        
        # Skip first row if it contains header metadata (NaT in Time column)
        if pd.isna(df.iloc[0]['Time']):
            df = df.iloc[1:].copy()
        
        # Convert Time to datetime
        df['Time'] = pd.to_datetime(df['Time'])
        
        # Remove rows with invalid timestamps
        df = df[df['Time'].notna()].copy()
        
        if len(df) == 0:
            print(f"    WARNING: No valid data rows found")
            return None
        
        # Map SCADA columns to readable names
        rename_dict = {}
        for scada_col, readable_name in SCADA_COLUMN_MAPPING.items():
            if scada_col in df.columns:
                rename_dict[scada_col] = readable_name
        
        df = df.rename(columns=rename_dict)
        
        # Convert to numeric (handle any string values)
        for col in SCADA_COLUMN_MAPPING.values():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Set Time as index
        df = df.set_index('Time')
        
        # Ensure timezone is IST
        if df.index.tz is None:
            df.index = df.index.tz_localize('Asia/Kolkata')
        else:
            df.index = df.index.tz_convert('Asia/Kolkata')
        
        # Resample from 5-minute to hourly
        # For generation: use mean (average power over the hour)
        # For demand: use mean (average demand over the hour)
        df_hourly = df.resample('h').mean()  # 'h' for hourly
        
        # Remove any rows with all NaN values
        df_hourly = df_hourly.dropna(how='all')
        
        print(f"    ✓ Parsed: {len(df_hourly):,} hourly rows")
        print(f"      Date range: {df_hourly.index.min()} to {df_hourly.index.max()}")
        
        # Check for missing columns
        missing_cols = set(SCADA_COLUMN_MAPPING.values()) - set(df_hourly.columns)
        if missing_cols:
            print(f"      WARNING: Missing columns: {missing_cols}")
        
        return df_hourly
        
    except Exception as e:
        print(f"    ERROR parsing {filepath.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_nerdc_extended_file(filepath):
    """
    Parse the extended period file (2024+) which has different format
    
    NOTE: The extended file (Jan 2024 - Jun 2025) has a different structure:
    - Only contains: Timestamp, Demand (MW), Wind (MW), Solar (MW), Total Generation (MW)
    - Does NOT contain: Hydro, Gas, Nuclear breakdown (these will be NaN)
    - Thermal is calculated as: Total Generation - Wind - Solar
    
    This is a data source limitation - the extended period file does not provide
    generation breakdown by fuel type. Missing values will be handled by the
    data cleaning pipeline.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to extended period Excel file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with datetime index
    """
    filepath = Path(filepath)
    print(f"  Parsing extended file {filepath.name}...")
    
    try:
        # Read Report sheet
        df = pd.read_excel(filepath, sheet_name='Report')
        
        # Skip header rows if they contain non-date strings
        # Find first row with valid timestamp
        for idx in range(len(df)):
            try:
                test_date = pd.to_datetime(df.iloc[idx]['Timestamp'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
                if pd.notna(test_date):
                    df = df.iloc[idx:].copy()
                    break
            except:
                continue
        
        # Convert Timestamp to datetime (try multiple formats)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        
        # Remove rows with invalid timestamps
        df = df[df['Timestamp'].notna()].copy()
        
        # Convert numeric columns
        numeric_cols = ['Demand (MW)', 'Wind (MW)', 'Solar (MW)', 'Total Generation (MW)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Rename to match standard format
        df = df.rename(columns={
            'Timestamp': 'Time',
            'Demand (MW)': 'Demand',
            'Wind (MW)': 'Wind',
            'Solar (MW)': 'Solar',
            'Total Generation (MW)': 'Total_Generation'
        })
        
        # Set Time as index
        df = df.set_index('Time')
        
        # Ensure timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('Asia/Kolkata')
        
        # Calculate Thermal as Total - Wind - Solar (approximation)
        if 'Total_Generation' in df.columns:
            df['Thermal'] = df['Total_Generation'] - df.get('Wind', 0) - df.get('Solar', 0)
            df['Thermal'] = df['Thermal'].clip(lower=0)  # Ensure non-negative
        
        # Note: Hydro, Gas, Nuclear are not available in extended file format
        # These will remain as NaN and be handled by data cleaning pipeline
        for col in ['Hydro', 'Gas', 'Nuclear']:
            if col not in df.columns:
                df[col] = np.nan
        
        print(f"    ✓ Parsed: {len(df):,} hourly rows")
        print(f"      Date range: {df.index.min()} to {df.index.max()}")
        print(f"      NOTE: Hydro, Gas, Nuclear not available in extended file format")
        
        return df
        
    except Exception as e:
        print(f"    ERROR parsing extended file: {e}")
        import traceback
        traceback.print_exc()
        return None


def estimate_missing_generation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate missing Hydro, Gas, Nuclear values for extended period (2024-2025)
    using historical patterns from Period 1 (2021-2023)
    
    Strategy:
    - Calculate average ratios by hour-of-day and day-of-week from Period 1
    - Apply ratios to Period 2 Total Generation to estimate missing values
    - Only estimates if values are missing and Total Generation is available
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined NERLDC dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with estimated values filled in
    """
    period_1_end = pd.Timestamp('2023-12-31 23:00:00', tz='Asia/Kolkata')
    period_1 = df[df.index <= period_1_end].copy()
    period_2 = df[df.index > period_1_end].copy()
    
    # Check if estimation is needed
    cols_to_estimate = ['Hydro', 'Gas', 'Nuclear']
    needs_estimation = False
    
    for col in cols_to_estimate:
        if col in period_2.columns:
            if period_2[col].isnull().sum() > 0:
                needs_estimation = True
                break
    
    if not needs_estimation:
        return df
    
    print(f"\n{'='*80}")
    print("ESTIMATING MISSING GENERATION VALUES")
    print("="*80)
    print(f"Using historical patterns from Period 1 to estimate Period 2")
    
    # Calculate Total Generation if not available
    # Use sum of all generation types
    if 'Total_Generation' not in period_1.columns or period_1['Total_Generation'].isnull().all():
        period_1['Total_Generation'] = (
            period_1['Thermal'].fillna(0) +
            period_1['Wind'].fillna(0) +
            period_1['Solar'].fillna(0) +
            period_1['Hydro'].fillna(0) +
            period_1['Gas'].fillna(0) +
            period_1['Nuclear'].fillna(0)
        )
    
    # Filter Period 1 to only rows where all values are available
    period_1_complete = period_1[
        period_1[cols_to_estimate + ['Total_Generation']].notna().all(axis=1)
    ].copy()
    
    if len(period_1_complete) == 0:
        print("  ⚠ No complete Period 1 data - cannot estimate")
        return df
    
    # Add time features
    period_1_complete['Hour'] = period_1_complete.index.hour
    period_1_complete['DayOfWeek'] = period_1_complete.index.dayofweek
    
    # Calculate average ratios by hour and day of week
    print(f"\n  Calculating historical ratios from {len(period_1_complete):,} complete hours...")
    
    ratios = {}
    for col in cols_to_estimate:
        if col in period_1_complete.columns:
            # Calculate ratio: generation_type / total_generation
            period_1_complete[f'{col}_ratio'] = (
                period_1_complete[col] / period_1_complete['Total_Generation']
            ).clip(lower=0, upper=1)  # Ensure ratio is between 0 and 1
            
            # Average ratio by hour and day of week
            ratio_by_time = period_1_complete.groupby(['Hour', 'DayOfWeek'])[f'{col}_ratio'].mean()
            ratios[col] = ratio_by_time
    
    # Apply to Period 2
    period_2_estimated = period_2.copy()
    
    # Calculate Total Generation for Period 2 if not available
    if 'Total_Generation' not in period_2_estimated.columns or period_2_estimated['Total_Generation'].isnull().any():
        period_2_estimated['Total_Generation'] = (
            period_2_estimated['Thermal'].fillna(0) +
            period_2_estimated['Wind'].fillna(0) +
            period_2_estimated['Solar'].fillna(0) +
            period_2_estimated.get('Hydro', pd.Series(0, index=period_2_estimated.index)).fillna(0) +
            period_2_estimated.get('Gas', pd.Series(0, index=period_2_estimated.index)).fillna(0) +
            period_2_estimated.get('Nuclear', pd.Series(0, index=period_2_estimated.index)).fillna(0)
        )
    
    period_2_estimated['Hour'] = period_2_estimated.index.hour
    period_2_estimated['DayOfWeek'] = period_2_estimated.index.dayofweek
    
    estimated_count = 0
    for col in cols_to_estimate:
        if col in period_2_estimated.columns:
            missing_mask = period_2_estimated[col].isnull()
            if missing_mask.sum() > 0:
                # Only estimate where Total Generation is available
                can_estimate = missing_mask & period_2_estimated['Total_Generation'].notna()
                
                if can_estimate.sum() > 0:
                    # Vectorised map approach (replaces row-by-row loop)
                    ratio_series = ratios[col]  # MultiIndex (Hour, DayOfWeek) -> ratio
                    keys = pd.MultiIndex.from_arrays(
                        [period_2_estimated['Hour'], period_2_estimated['DayOfWeek']],
                        names=['Hour', 'DayOfWeek'],
                    )
                    mapped_ratios = keys.map(
                        lambda k: ratio_series.get(k, np.nan)
                    )
                    fill_mask = can_estimate & pd.notna(mapped_ratios)
                    period_2_estimated.loc[fill_mask, col] = (
                        period_2_estimated.loc[fill_mask, 'Total_Generation']
                        * mapped_ratios[fill_mask]
                    )
                    estimated_count += int(fill_mask.sum())
                    
                    print(f"  ✓ Estimated {col}: {can_estimate.sum():,} values")
    
    # Combine back
    result_df = pd.concat([
        period_1.drop(['Hour', 'DayOfWeek'], axis=1, errors='ignore'),
        period_2_estimated.drop(['Hour', 'DayOfWeek'], axis=1, errors='ignore')
    ]).sort_index()
    
    print(f"\n  Total estimated values: {estimated_count:,}")
    print(f"  Note: Values are estimates based on historical patterns")
    
    return result_df


def load_all_nerdc_files(data_dir='.'):
    """
    Load all NERLDC monthly files and combine into single dataframe
    
    Parameters:
    -----------
    data_dir : str or Path
        Directory containing NERLDC files
    
    Returns:
    --------
    pd.DataFrame
        Combined dataframe with all NERLDC data
    """
    data_dir = Path(data_dir)
    
    # Find all NERLDC files (exclude extended parquet files)
    all_files = sorted(data_dir.glob('*.xlsx'))
    nerdc_files = [
        f for f in all_files 
        if not f.name.endswith('.parquet') 
        and '2024' not in f.name 
        and '2025' not in f.name
    ]
    
    extended_files = [
        f for f in all_files 
        if ('2024' in f.name or '2025' in f.name)
        and not f.name.startswith('~$')  # Exclude Excel temp files
    ]
    
    print(f"Found {len(nerdc_files)} standard NERLDC files")
    print(f"Found {len(extended_files)} extended period files")
    
    # Parse standard files
    dataframes = []
    for filepath in nerdc_files:
        df = parse_nerdc_file(filepath)
        if df is not None:
            dataframes.append(df)
    
    # Parse extended files
    for filepath in extended_files:
        df = parse_nerdc_extended_file(filepath)
        if df is not None:
            dataframes.append(df)
    
    if not dataframes:
        print("ERROR: No dataframes successfully parsed")
        return None
    
    # Combine all dataframes
    print(f"\nCombining {len(dataframes)} dataframes...")
    combined_df = pd.concat(dataframes, axis=0)
    
    # Sort by timestamp
    combined_df = combined_df.sort_index()
    
    # Remove duplicates (keep first occurrence)
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    
    # Estimate missing generation values (Hydro, Gas, Nuclear) for extended period
    combined_df = estimate_missing_generation(combined_df)
    
    print(f"✓ Combined NERLDC data:")
    print(f"  Total rows: {len(combined_df):,}")
    print(f"  Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"  Columns: {list(combined_df.columns)}")
    
    # Data quality summary
    print(f"\nData Quality Summary:")
    for col in combined_df.columns:
        if col in ['Time', 'Timestamp']:
            continue
        print(f"  {col}:")
        print(f"    Min: {combined_df[col].min():,.0f}")
        print(f"    Max: {combined_df[col].max():,.0f}")
        print(f"    Mean: {combined_df[col].mean():,.0f}")
        print(f"    Missing: {combined_df[col].isna().sum():,} ({100*combined_df[col].isna().sum()/len(combined_df):.2f}%)")
    
    return combined_df


if __name__ == "__main__":
    # Test loading
    print("="*80)
    print("LOADING NERLDC DATA")
    print("="*80)
    
    df = load_all_nerdc_files('.')
    
    if df is not None:
        print(f"\nSample data (first 5 rows):")
        print(df.head())
