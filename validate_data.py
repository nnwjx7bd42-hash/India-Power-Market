#!/usr/bin/env python3
"""
Comprehensive data validation and assessment
Validates the feature-engineered dataset before model training
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def validate_data_completeness(df):
    """Check data completeness and missing values"""
    print("="*80)
    print("1. DATA COMPLETENESS CHECK")
    print("="*80)
    
    print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    print(f"Duration: {(df.index.max() - df.index.min()).days} days")
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    print(f"\nMissing Values Analysis:")
    if missing.sum() == 0:
        print("  ✓ No missing values")
    else:
        print(f"  ⚠️  {missing.sum():,} total missing values")
        missing_cols = missing[missing > 0].sort_values(ascending=False)
        print(f"  Columns with missing values: {len(missing_cols)}")
        print("\n  Top 10 columns with missing values:")
        for col, count in missing_cols.head(10).items():
            print(f"    {col}: {count:,} ({missing_pct[col]:.2f}%)")
    
    # Check timestamp continuity
    print(f"\nTimestamp Continuity:")
    expected_hours = (df.index.max() - df.index.min()).total_seconds() / 3600 + 1
    actual_hours = len(df)
    missing_hours = expected_hours - actual_hours
    print(f"  Expected hours: {expected_hours:,.0f}")
    print(f"  Actual hours: {actual_hours:,.0f}")
    print(f"  Missing hours: {missing_hours:,.0f} ({100*missing_hours/expected_hours:.2f}%)")
    
    if missing_hours == 0:
        print("  ✓ Continuous hourly data")
    else:
        print("  ⚠️  Gaps detected in timestamps")
        # Find gaps
        time_diffs = df.index.to_series().diff()
        gaps = time_diffs[time_diffs > pd.Timedelta(hours=1.5)]
        if len(gaps) > 0:
            print(f"  Found {len(gaps)} gaps > 1 hour")
            print(f"  Largest gap: {gaps.max()}")
    
    return {
        'missing_total': missing.sum(),
        'missing_cols': len(missing[missing > 0]),
        'missing_hours': missing_hours,
        'is_complete': missing.sum() == 0 and missing_hours == 0
    }


def validate_target_variable(df):
    """Validate target variable P(T)"""
    print("\n" + "="*80)
    print("2. TARGET VARIABLE VALIDATION")
    print("="*80)
    
    target = df['P(T)']
    
    print(f"\nBasic Statistics:")
    print(f"  Count: {len(target):,}")
    print(f"  Mean: ₹{target.mean():,.2f}")
    print(f"  Median: ₹{target.median():,.2f}")
    print(f"  Std: ₹{target.std():,.2f}")
    print(f"  Min: ₹{target.min():,.2f}")
    print(f"  Max: ₹{target.max():,.2f}")
    print(f"  25th percentile: ₹{target.quantile(0.25):,.2f}")
    print(f"  75th percentile: ₹{target.quantile(0.75):,.2f}")
    
    # Check for outliers
    print(f"\nOutlier Analysis:")
    q1, q3 = target.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = target[(target < lower_bound) | (target > upper_bound)]
    print(f"  IQR: ₹{iqr:,.2f}")
    print(f"  Outliers (IQR method): {len(outliers):,} ({100*len(outliers)/len(target):.2f}%)")
    
    # Check price ceiling
    ceiling_price = 20000
    at_ceiling = (target == ceiling_price).sum()
    print(f"\nPrice Ceiling Check (₹{ceiling_price:,}):")
    print(f"  Values at ceiling: {at_ceiling:,} ({100*at_ceiling/len(target):.2f}%)")
    
    # Check for negative prices
    negative = (target < 0).sum()
    if negative > 0:
        print(f"  ⚠️  Negative prices: {negative}")
    else:
        print(f"  ✓ No negative prices")
    
    # Check distribution by hour
    print(f"\nPrice Distribution by Hour:")
    hourly_stats = df.groupby('Hour')['P(T)'].agg(['mean', 'std', 'min', 'max'])
    print(f"  Highest mean price hour: {hourly_stats['mean'].idxmax()} ({hourly_stats['mean'].max():,.2f})")
    print(f"  Lowest mean price hour: {hourly_stats['mean'].idxmin()} ({hourly_stats['mean'].min():,.2f})")
    
    return {
        'has_outliers': len(outliers) > 0,
        'at_ceiling_pct': 100*at_ceiling/len(target),
        'has_negative': negative > 0
    }


def validate_feature_distributions(df):
    """Check feature distributions for issues"""
    print("\n" + "="*80)
    print("3. FEATURE DISTRIBUTION VALIDATION")
    print("="*80)
    
    issues = []
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = []
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            inf_cols.append(col)
    
    if inf_cols:
        print(f"\n⚠️  Infinite values found in {len(inf_cols)} columns:")
        for col in inf_cols[:10]:
            print(f"    {col}: {np.isinf(df[col]).sum()} infinite values")
        issues.append("infinite_values")
    else:
        print(f"\n✓ No infinite values")
    
    # Check for constant features
    constant_cols = []
    for col in numeric_cols:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"\n⚠️  Constant features (no variance): {len(constant_cols)}")
        for col in constant_cols[:10]:
            print(f"    {col}: {df[col].nunique()} unique values")
        issues.append("constant_features")
    else:
        print(f"\n✓ No constant features")
    
    # Check for highly correlated features (potential redundancy)
    print(f"\nFeature Correlation Check:")
    # Sample correlation for large datasets
    if len(df) > 10000:
        sample_df = df.sample(10000)
    else:
        sample_df = df
    
    corr_matrix = sample_df[numeric_cols].corr().abs()
    # Find highly correlated pairs (excluding self-correlation)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.95:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        print(f"  ⚠️  Found {len(high_corr_pairs)} highly correlated pairs (>0.95):")
        for col1, col2, corr_val in high_corr_pairs[:10]:
            print(f"    {col1} ↔ {col2}: {corr_val:.3f}")
        issues.append("high_correlation")
    else:
        print(f"  ✓ No highly correlated feature pairs")
    
    return {
        'has_infinite': len(inf_cols) > 0,
        'has_constant': len(constant_cols) > 0,
        'high_corr_pairs': len(high_corr_pairs),
        'issues': issues
    }


def validate_temporal_consistency(df):
    """Check temporal consistency and data leakage"""
    print("\n" + "="*80)
    print("4. TEMPORAL CONSISTENCY & DATA LEAKAGE CHECK")
    print("="*80)
    
    issues = []
    
    # Check if target variable appears in features (direct leakage)
    feature_cols = [c for c in df.columns if c != 'P(T)']
    if 'P(T)' in feature_cols:
        print("  ⚠️  CRITICAL: Target variable P(T) found in features!")
        issues.append("target_in_features")
    else:
        print("  ✓ Target variable not in features")
    
    # Check for future information leakage in lag features
    print(f"\nLag Feature Validation:")
    lag_features = [c for c in df.columns if 'T-' in c or 'T-' in c]
    print(f"  Found {len(lag_features)} lag features")
    
    # Check if any lag feature equals target (would be perfect predictor)
    for col in lag_features[:10]:  # Check first 10
        if col in df.columns:
            exact_matches = (df[col] == df['P(T)']).sum()
            if exact_matches > len(df) * 0.9:  # More than 90% match
                print(f"    ⚠️  {col} matches P(T) in {exact_matches} cases ({100*exact_matches/len(df):.1f}%)")
                issues.append("lag_leakage")
    
    # Check timestamp ordering
    print(f"\nTimestamp Ordering:")
    is_sorted = df.index.is_monotonic_increasing
    if is_sorted:
        print("  ✓ Timestamps are sorted")
    else:
        print("  ⚠️  Timestamps are not sorted")
        issues.append("unsorted_timestamps")
    
    # Check for duplicate timestamps
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        print(f"  ⚠️  {duplicates} duplicate timestamps found")
        issues.append("duplicate_timestamps")
    else:
        print("  ✓ No duplicate timestamps")
    
    return {
        'has_leakage': len(issues) > 0,
        'issues': issues
    }


def validate_weather_features(df):
    """Validate weather features"""
    print("\n" + "="*80)
    print("5. WEATHER FEATURES VALIDATION")
    print("="*80)
    
    weather_cols = [c for c in df.columns if 'national' in c or 'CDH' in c or 'HDH' in c or 'Heat' in c]
    
    if not weather_cols:
        print("  ⚠️  No weather features found")
        return {'has_weather': False}
    
    print(f"\nFound {len(weather_cols)} weather-related features")
    
    # Check temperature range (reasonable for India)
    if 'temperature_2m_national' in df.columns:
        temp = df['temperature_2m_national']
        print(f"\nTemperature Range:")
        print(f"  Min: {temp.min():.1f}°C")
        print(f"  Max: {temp.max():.1f}°C")
        print(f"  Mean: {temp.mean():.1f}°C")
        
        if temp.min() < -10 or temp.max() > 50:
            print("  ⚠️  Temperature values outside expected range for India")
        else:
            print("  ✓ Temperature values within expected range")
    
    # Check CDH/HDH
    if 'CDH' in df.columns:
        cdh = df['CDH']
        print(f"\nCooling Degree Hours (CDH):")
        print(f"  Mean: {cdh.mean():.2f}")
        print(f"  Max: {cdh.max():.2f}")
        print(f"  Non-zero: {(cdh > 0).sum():,} ({100*(cdh > 0).sum()/len(cdh):.1f}%)")
    
    if 'HDH' in df.columns:
        hdh = df['HDH']
        print(f"\nHeating Degree Hours (HDH):")
        print(f"  Mean: {hdh.mean():.2f}")
        print(f"  Max: {hdh.max():.2f}")
        print(f"  Non-zero: {(hdh > 0).sum():,} ({100*(hdh > 0).sum()/len(hdh):.1f}%)")
    
    return {'has_weather': True}


def validate_supply_features(df):
    """Validate supply-side features"""
    print("\n" + "="*80)
    print("6. SUPPLY-SIDE FEATURES VALIDATION")
    print("="*80)
    
    supply_cols = ['Thermal', 'Hydro', 'Gas', 'Nuclear', 'Wind', 'Solar', 'Demand']
    available_cols = [c for c in supply_cols if c in df.columns]
    
    print(f"\nAvailable supply features: {len(available_cols)}/{len(supply_cols)}")
    
    if 'RE_Penetration' in df.columns:
        re_pen = df['RE_Penetration']
        print(f"\nRenewable Penetration:")
        print(f"  Mean: {re_pen.mean():.2%}")
        print(f"  Max: {re_pen.max():.2%}")
        print(f"  Min: {re_pen.min():.2%}")
        
        # Check for invalid values
        invalid = ((re_pen < 0) | (re_pen > 1)).sum()
        if invalid > 0:
            print(f"  ⚠️  {invalid} invalid values (<0 or >1)")
        else:
            print("  ✓ All values in valid range [0, 1]")
    
    if 'Net_Load' in df.columns and 'Demand' in df.columns:
        net_load = df['Net_Load']
        demand = df['Demand']
        print(f"\nNet Load vs Demand:")
        print(f"  Net Load mean: {net_load.mean():,.0f} MW")
        print(f"  Demand mean: {demand.mean():,.0f} MW")
        
        # Net load should be <= Demand
        invalid = (net_load > demand * 1.1).sum()  # Allow 10% tolerance
        if invalid > 0:
            print(f"  ⚠️  {invalid} cases where Net_Load > Demand (unexpected)")
        else:
            print("  ✓ Net_Load <= Demand (as expected)")
    
    return {'supply_features_ok': True}


def validate_feature_target_relationships(df):
    """Check relationships between features and target"""
    print("\n" + "="*80)
    print("7. FEATURE-TARGET RELATIONSHIPS")
    print("="*80)
    
    # Sample for correlation calculation if dataset is large
    if len(df) > 10000:
        sample_df = df.sample(10000, random_state=42)
    else:
        sample_df = df
    
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c != 'P(T)']
    
    # Calculate correlations with target
    correlations = sample_df[feature_cols].corrwith(sample_df['P(T)']).abs().sort_values(ascending=False)
    
    print(f"\nTop 15 features correlated with P(T):")
    for i, (feature, corr) in enumerate(correlations.head(15).items(), 1):
        print(f"  {i:2d}. {feature}: {corr:.4f}")
    
    # Check if any features have very low correlation (might be noise)
    low_corr = correlations[correlations < 0.01]
    if len(low_corr) > 0:
        print(f"\n  ⚠️  {len(low_corr)} features have correlation < 0.01 with target")
        print(f"      (may be noise or require transformation)")
    
    return {
        'top_corr_features': correlations.head(10).to_dict(),
        'low_corr_count': len(low_corr)
    }


def generate_validation_report(df):
    """Generate comprehensive validation report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE DATA VALIDATION REPORT")
    print("="*80)
    
    results = {}
    
    # Run all validations
    results['completeness'] = validate_data_completeness(df)
    results['target'] = validate_target_variable(df)
    results['distributions'] = validate_feature_distributions(df)
    results['temporal'] = validate_temporal_consistency(df)
    results['weather'] = validate_weather_features(df)
    results['supply'] = validate_supply_features(df)
    results['relationships'] = validate_feature_target_relationships(df)
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    all_ok = True
    
    print(f"\n✓ Data Completeness: {'PASS' if results['completeness']['is_complete'] else 'ISSUES FOUND'}")
    if not results['completeness']['is_complete']:
        all_ok = False
    
    print(f"✓ Target Variable: {'PASS' if not results['target']['has_negative'] else 'ISSUES FOUND'}")
    if results['target']['has_negative']:
        all_ok = False
    
    print(f"✓ Feature Distributions: {'PASS' if len(results['distributions']['issues']) == 0 else 'ISSUES FOUND'}")
    if len(results['distributions']['issues']) > 0:
        all_ok = False
    
    print(f"✓ Temporal Consistency: {'PASS' if not results['temporal']['has_leakage'] else 'CRITICAL ISSUES'}")
    if results['temporal']['has_leakage']:
        all_ok = False
        print(f"    ⚠️  Data leakage detected! Fix before training.")
    
    print(f"✓ Weather Features: {'PASS' if results['weather'].get('has_weather', False) else 'MISSING'}")
    
    print(f"✓ Supply Features: {'PASS' if results['supply'].get('supply_features_ok', False) else 'ISSUES'}")
    
    print("\n" + "="*80)
    if all_ok:
        print("✓ OVERALL: DATA READY FOR MODEL TRAINING")
    else:
        print("⚠️  OVERALL: ISSUES FOUND - REVIEW BEFORE TRAINING")
        print("\nRecommendations:")
        if results['completeness']['missing_hours'] > 0:
            print("  - Address missing hours in dataset")
        if results['distributions']['has_infinite']:
            print("  - Remove or fix infinite values")
        if results['temporal']['has_leakage']:
            print("  - CRITICAL: Fix data leakage issues")
        if results['distributions']['high_corr_pairs'] > 0:
            print("  - Consider removing highly correlated features")
    
    return results


if __name__ == "__main__":
    # Load dataset (prefer cleaned version if available)
    import os
    if os.path.exists('data/processed/dataset_cleaned.parquet'):
        print("Loading cleaned dataset...")
        df = pd.read_parquet('data/processed/dataset_cleaned.parquet')
    else:
        print("Loading dataset...")
        df = pd.read_parquet('data/processed/dataset_with_features.parquet')
    
    # Run validation
    results = generate_validation_report(df)
    
    print(f"\n✓ Validation complete")
