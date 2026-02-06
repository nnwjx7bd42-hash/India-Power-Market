#!/usr/bin/env python3
"""
Fix identified data issues:
1. Remove highly correlated redundant features
2. Handle missing values in lag features
"""

import pandas as pd
import numpy as np

def remove_redundant_features(df):
    """Remove redundant highly correlated features"""
    print("="*80)
    print("REMOVING REDUNDANT FEATURES")
    print("="*80)
    
    # Features to remove (keeping the more informative one)
    features_to_remove = [
        'IsFriday',  # Highly correlated with Day (keep Day)
        'Hydro_Availability',  # Highly correlated with Hydro (keep Hydro)
        'Total_Generation',  # Highly correlated with Demand (keep Demand)
        'Heat_Index',  # Highly correlated with temperature (keep temperature)
        'shortwave_radiation_national',  # Highly correlated with direct_radiation (keep direct)
        'Hour_x_NetLoad',  # Highly correlated with Hour (keep Hour and Net_Load separately)
        'Temp_x_Hour',  # Highly correlated with Hour (keep Hour and Temp separately)
        'WeekOfYear',  # Highly correlated with Month (keep Month)
        'DayOfYear',  # Highly correlated with Month (keep Month)
        'Quarter',  # Highly correlated with Month (keep Month)
    ]
    
    # Check which ones exist
    existing_to_remove = [f for f in features_to_remove if f in df.columns]
    
    print(f"\nFeatures to remove: {len(existing_to_remove)}")
    for f in existing_to_remove:
        print(f"  - {f}")
    
    df_cleaned = df.drop(columns=existing_to_remove)
    
    print(f"\nBefore: {df.shape[1]} columns")
    print(f"After: {df_cleaned.shape[1]} columns")
    print(f"Removed: {len(existing_to_remove)} columns")
    
    return df_cleaned


def handle_missing_lag_values(df):
    """Handle missing values in lag features"""
    print("\n" + "="*80)
    print("HANDLING MISSING LAG VALUES")
    print("="*80)
    
    # Lag features with missing values
    lag_features = ['P_T-168', 'L_T-168', 'P_T-3', 'P_T-4']
    
    missing_before = df[lag_features].isnull().sum()
    print(f"\nMissing values before:")
    for col in lag_features:
        if col in df.columns:
            print(f"  {col}: {missing_before[col]}")
    
    # Strategy: For T-168, fill with mean (can't forward fill from start)
    # For T-3, T-4: use mean of available values
    for col in lag_features:
        if col in df.columns:
            # Fill with mean (reasonable default for lag features)
            df[col] = df[col].fillna(df[col].mean())
    
    missing_after = df[lag_features].isnull().sum()
    print(f"\nMissing values after:")
    for col in lag_features:
        if col in df.columns:
            print(f"  {col}: {missing_after[col]}")
    
    return df


def handle_missing_scada_values(df):
    """
    Handle missing values in SCADA generation columns (Hydro, Gas, Nuclear)
    
    NOTE: These are missing for Jan 2024 - Jun 2025 period due to data source
    limitation (extended file doesn't provide breakdown). Options:
    1. Leave as NaN (model handles missing values)
    2. Fill with 0 (assumes no generation - not accurate)
    3. Estimate from historical patterns (requires implementation)
    
    Current approach: Leave as NaN - handled by model or feature selection
    """
    print("\n" + "="*80)
    print("HANDLING MISSING SCADA VALUES")
    print("="*80)
    
    scada_cols = ['Hydro', 'Gas', 'Nuclear']
    available_cols = [c for c in scada_cols if c in df.columns]
    
    if not available_cols:
        print("  No SCADA columns found")
        return df
    
    missing_before = df[available_cols].isnull().sum()
    print(f"\nMissing values before:")
    for col in available_cols:
        missing_count = missing_before[col]
        pct = (missing_count / len(df)) * 100
        print(f"  {col}: {missing_count:,} ({pct:.2f}%)")
    
    # Document but don't fill - let model handle or use available features
    # These are data source limitations, not errors
    print(f"\n  NOTE: Missing values are due to extended file format limitation")
    print(f"  (Jan 2024 - Jun 2025 file doesn't provide generation breakdown)")
    print(f"  Model will use available features (Thermal, Wind, Solar, Demand)")
    
    return df


def remove_low_correlation_features(df, threshold=0.01):
    """Remove features with very low correlation to target"""
    print("\n" + "="*80)
    print("CHECKING LOW CORRELATION FEATURES")
    print("="*80)
    
    # Sample for correlation if large
    if len(df) > 10000:
        sample_df = df.sample(10000, random_state=42)
    else:
        sample_df = df
    
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c != 'P(T)']
    
    correlations = sample_df[feature_cols].corrwith(sample_df['P(T)']).abs()
    low_corr_features = correlations[correlations < threshold].index.tolist()
    
    print(f"\nFeatures with correlation < {threshold}: {len(low_corr_features)}")
    for f in low_corr_features[:10]:
        print(f"  {f}: {correlations[f]:.6f}")
    
    # Don't remove them automatically - just report
    # Some features might be useful in interactions even if low correlation
    print("\n  Note: Keeping low-correlation features (may be useful in interactions)")
    
    return df


if __name__ == "__main__":
    print("="*80)
    print("FIXING DATA ISSUES")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_parquet('data/processed/dataset_with_features.parquet')
    print(f"  Shape: {df.shape}")
    
    # Fix issues
    df_fixed = remove_redundant_features(df)
    df_fixed = handle_missing_lag_values(df_fixed)
    df_fixed = handle_missing_scada_values(df_fixed)
    df_fixed = remove_low_correlation_features(df_fixed)
    
    # Save cleaned dataset
    output_path = 'data/processed/dataset_cleaned.parquet'
    print(f"\n{'='*80}")
    print("SAVING CLEANED DATASET")
    print("="*80)
    df_fixed.to_parquet(output_path)
    
    print(f"\n✓ Cleaned dataset saved to: {output_path}")
    print(f"  Shape: {df_fixed.shape}")
    print(f"  Missing values: {df_fixed.isnull().sum().sum()}")
    
    # Verify
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print("="*80)
    print(f"  Columns removed: {df.shape[1] - df_fixed.shape[1]}")
    print(f"  Missing values before: {df.isnull().sum().sum()}")
    print(f"  Missing values after: {df_fixed.isnull().sum().sum()}")
    print(f"\n✓ Data cleaning complete!")
