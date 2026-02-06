#!/usr/bin/env python3
"""
Step-by-step dataset regeneration for full period
Sep 1, 2021 - Jun 24, 2025
"""

import sys
from pathlib import Path
import pandas as pd

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'data_pipeline'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'utils'))

print("="*80)
print("REGENERATING FULL DATASET")
print("="*80)
print("Period: Sep 1, 2021 - Jun 24, 2025")
print("="*80)

output_dir = Path('data/processed')
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Create unified dataset
print("\n" + "="*80)
print("STEP 1: CREATING UNIFIED DATASET")
print("="*80)

from merge_datasets import create_unified_dataset

unified_df = create_unified_dataset(
    price_file='data/raw/price/iex_dam_combined.parquet',
    data_dir='data/raw/nerldc',
    extend_period=True
)

unified_path = output_dir / 'unified_dataset.parquet'
unified_df.to_parquet(unified_path)
print(f"\n✓ Saved unified dataset: {unified_path}")
print(f"  Shape: {unified_df.shape}")
print(f"  Date range: {unified_df.index.min()} to {unified_df.index.max()}")

# Step 2: Merge weather data
print("\n" + "="*80)
print("STEP 2: MERGING WEATHER DATA")
print("="*80)

weather_path = output_dir / 'weather_national.parquet'
if weather_path.exists():
    weather_df = pd.read_parquet(weather_path)
    print(f"Loaded weather data: {weather_df.shape}")
    print(f"  Date range: {weather_df.index.min()} to {weather_df.index.max()}")
    
    # Merge weather
    unified_sorted = unified_df.sort_index()
    weather_sorted = weather_df.sort_index()
    
    df = pd.merge_asof(
        unified_sorted,
        weather_sorted,
        left_index=True,
        right_index=True,
        direction='nearest',
        tolerance=pd.Timedelta('30min')
    )
    print(f"  After merge: {df.shape}")
    
    # Fill any missing weather values (backward fill from next available)
    weather_cols = [c for c in df.columns if 'national' in c.lower()]
    if weather_cols:
        missing_before = df[weather_cols].isnull().sum().sum()
        if missing_before > 0:
            print(f"  Filling {missing_before} missing weather values...")
            for col in weather_cols:
                df[col] = df[col].bfill().ffill()  # Backward then forward fill
            missing_after = df[weather_cols].isnull().sum().sum()
            print(f"  Missing after fill: {missing_after}")
else:
    print("⚠ Weather file not found, skipping merge")
    df = unified_df.copy()

# Step 3: Feature engineering
print("\n" + "="*80)
print("STEP 3: FEATURE ENGINEERING")
print("="*80)

from feature_engineering import create_features

df_features = create_features(df)
features_path = output_dir / 'dataset_with_features.parquet'
df_features.to_parquet(features_path)
print(f"\n✓ Saved dataset with features: {features_path}")
print(f"  Shape: {df_features.shape}")
print(f"  Columns: {len(df_features.columns)}")

# Step 4: Data cleaning
print("\n" + "="*80)
print("STEP 4: DATA CLEANING")
print("="*80)

sys.path.insert(0, str(Path(__file__).parent))
from fix_data_issues import (
    remove_redundant_features,
    handle_missing_lag_values,
    handle_missing_scada_values,
    remove_low_correlation_features
)

df_cleaned = remove_redundant_features(df_features)
df_cleaned = handle_missing_lag_values(df_cleaned)
df_cleaned = handle_missing_scada_values(df_cleaned)
df_cleaned = remove_low_correlation_features(df_cleaned)

cleaned_path = output_dir / 'dataset_cleaned.parquet'
df_cleaned.to_parquet(cleaned_path)
print(f"\n✓ Saved cleaned dataset: {cleaned_path}")
print(f"  Shape: {df_cleaned.shape}")

# Final summary
print("\n" + "="*80)
print("REGENERATION COMPLETE")
print("="*80)
print(f"\nFinal Dataset:")
print(f"  Shape: {df_cleaned.shape}")
print(f"  Date range: {df_cleaned.index.min()} to {df_cleaned.index.max()}")
print(f"  Total hours: {len(df_cleaned):,}")
print(f"  Columns: {len(df_cleaned.columns)}")
print(f"  Missing values: {df_cleaned.isnull().sum().sum()}")
