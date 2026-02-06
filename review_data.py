#!/usr/bin/env python3
"""
Data Review Script for IEX Price Forecasting Project
Examines IEX price data and NERLDC monthly files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def review_price_data(filepath='data/raw/price/iex_dam_combined.parquet'):
    """Review IEX price data structure and content"""
    print("=" * 80)
    print("IEX PRICE DATA REVIEW")
    print("=" * 80)
    
    try:
        # Load parquet file
        df = pd.read_parquet(filepath)
        
        print(f"\n1. Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        print(f"\n2. Column Names:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")
        
        print(f"\n3. Data Types:")
        print(df.dtypes)
        
        print(f"\n4. First 10 Rows:")
        print(df.head(10).to_string())
        
        print(f"\n5. Missing Values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("   No missing values found")
        
        print(f"\n6. Basic Statistics:")
        print(df.describe())
        
        # Check for timestamp column
        print(f"\n7. Timestamp Analysis:")
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower()]
        if date_cols:
            print(f"   Found date columns: {date_cols}")
            for col in date_cols:
                print(f"   {col}: {df[col].dtype}, Range: {df[col].min()} to {df[col].max()}")
        else:
            print("   No explicit date column found - checking index")
            print(f"   Index type: {type(df.index)}")
        
        # Check for target variable P(T)
        target_cols = [col for col in df.columns if 'price' in col.lower() or col == 'P(T)' or 'P' in col]
        if target_cols:
            print(f"\n8. Target Variable Analysis:")
            for col in target_cols:
                if df[col].dtype in [np.float64, np.int64]:
                    print(f"   {col}:")
                    print(f"      Min: {df[col].min():,.2f}")
                    print(f"      Max: {df[col].max():,.2f}")
                    print(f"      Mean: {df[col].mean():,.2f}")
                    print(f"      Median: {df[col].median():,.2f}")
                    print(f"      Std: {df[col].std():,.2f}")
                    print(f"      Values > 20,000 (ceiling): {(df[col] > 20000).sum()}")
        
        # Check Season values
        if 'Season' in df.columns:
            print(f"\n9. Season Variable:")
            print(f"   Unique values: {df['Season'].unique()}")
            print(f"   Value counts:\n{df['Season'].value_counts().sort_index()}")
        
        # Check Day values
        if 'Day' in df.columns:
            print(f"\n10. Day Variable:")
            print(f"    Unique values: {df['Day'].unique()}")
            print(f"    Value counts:\n{df['Day'].value_counts().sort_index()}")
        
        return df
        
    except Exception as e:
        print(f"ERROR reading {filepath}: {e}")
        return None

def review_nerdc_file(filepath):
    """Review a single NERLDC monthly file"""
    print("\n" + "=" * 80)
    print(f"NERLDC FILE REVIEW: {Path(filepath).name}")
    print("=" * 80)
    
    try:
        # Try reading with different sheet names
        xl_file = pd.ExcelFile(filepath)
        print(f"\nAvailable Sheets: {xl_file.sheet_names}")
        
        # Read first sheet
        df = pd.read_excel(filepath, sheet_name=0)
        
        print(f"\n1. Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        print(f"\n2. Column Names:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")
        
        print(f"\n3. First 10 Rows:")
        print(df.head(10).to_string())
        
        print(f"\n4. Data Types:")
        print(df.dtypes)
        
        # Check for timestamp/date columns
        date_cols = [col for col in df.columns if any(x in str(col).lower() for x in ['date', 'time', 'timestamp', 'datetime'])]
        if date_cols:
            print(f"\n5. Date Columns Found: {date_cols}")
            for col in date_cols[:3]:  # Show first 3
                print(f"   {col}: Sample values = {df[col].head(3).tolist()}")
        
        # Look for generation/demand columns
        gen_cols = [col for col in df.columns if any(x in str(col).lower() for x in ['thermal', 'hydro', 'gas', 'nuclear', 'wind', 'solar', 'demand', 'generation'])]
        if gen_cols:
            print(f"\n6. Generation/Demand Columns Found: {gen_cols}")
            for col in gen_cols[:5]:  # Show first 5
                if df[col].dtype in [np.float64, np.int64]:
                    print(f"   {col}: Min={df[col].min():,.0f}, Max={df[col].max():,.0f}, Mean={df[col].mean():,.0f}")
        
        print(f"\n7. Missing Values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0].head(10))
        else:
            print("   No missing values found")
        
        return df
        
    except Exception as e:
        print(f"ERROR reading {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None

def list_nerdc_files():
    """List all NERLDC monthly files"""
    files = sorted(Path('.').glob('[A-Z]*.xlsx'))
    nerdc_files = [f for f in files if not f.name.endswith('.parquet')]
    return nerdc_files

if __name__ == "__main__":
    # Review IEX price data
    price_df = review_price_data('data/raw/price/iex_dam_combined.parquet')
    
    # List NERLDC files
    print("\n" + "=" * 80)
    print("NERLDC FILES INVENTORY")
    print("=" * 80)
    nerdc_files = list_nerdc_files()
    print(f"\nFound {len(nerdc_files)} NERLDC monthly files:")
    for f in nerdc_files:
        print(f"  - {f.name}")
    
    # Review sample NERLDC files
    if nerdc_files:
        print("\n" + "=" * 80)
        print("REVIEWING SAMPLE NERLDC FILES")
        print("=" * 80)
        
        # Review first file (September 2021)
        if any('September-2021' in f.name for f in nerdc_files):
            review_nerdc_file([f for f in nerdc_files if 'September-2021' in f.name][0])
        
        # Review a middle file (June 2022)
        if any('June-2022' in f.name for f in nerdc_files):
            review_nerdc_file([f for f in nerdc_files if 'June-2022' in f.name][0])
        
        # Review last file (December 2023)
        if any('December-2023' in f.name for f in nerdc_files):
            review_nerdc_file([f for f in nerdc_files if 'December-2023' in f.name][0])
        
        # Review the extended file if it exists
        extended_files = [f for f in nerdc_files if '2024' in f.name or '2025' in f.name]
        if extended_files:
            print("\n" + "=" * 80)
            print("EXTENDED PERIOD FILE FOUND")
            print("=" * 80)
            review_nerdc_file(extended_files[0])
