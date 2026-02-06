#!/usr/bin/env python3
"""
Fetch full historical weather data for all 5 cities
Period: Sep 1, 2021 to Jun 24, 2025 (matching NERLDC coverage)

Fetches hourly weather data for:
- Delhi (NR - 30% weight)
- Mumbai (WR - 28% weight)
- Chennai (SR - 25% weight)
- Kolkata (ER - 12% weight)
- Guwahati (NER - 5% weight)

Then aggregates to national level using demand-weighted averages.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'data_pipeline'))

from load_weather_data import load_weather_data

if __name__ == "__main__":
    print("="*80)
    print("FETCHING FULL HISTORICAL WEATHER DATA")
    print("="*80)
    print("\nPeriod: Aug 31, 2021 to Jun 24, 2025 (extended to cover Sep 1 start)")
    print("Cities: Delhi, Mumbai, Chennai, Kolkata, Guwahati")
    print("\nThis will fetch hourly weather data for all 5 cities.")
    print("This may take 10-15 minutes due to API rate limiting...")
    print("="*80)
    
    # Full period matching NERLDC coverage
    # Start from Aug 31 to ensure coverage for Sep 1 00:00:00 (weather API starts at 05:30)
    start_date = '2021-08-31'
    end_date = '2025-06-24'
    
    # Fetch weather data for all cities
    weather_df = load_weather_data(
        start_date=start_date,
        end_date=end_date,
        use_cache=True  # Will use cached files if available, extend if needed
    )
    
    # Save aggregated national weather
    output_path = Path('data/processed/weather_national.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weather_df.to_parquet(output_path)
    
    print("\n" + "="*80)
    print("WEATHER DATA FETCH COMPLETE")
    print("="*80)
    print(f"\n✓ Weather data saved to: {output_path}")
    print(f"  Shape: {weather_df.shape}")
    print(f"  Date range: {weather_df.index.min()} to {weather_df.index.max()}")
    print(f"  Total hours: {len(weather_df):,}")
    print(f"  Columns: {list(weather_df.columns)}")
    
    # Verify coverage
    import pandas as pd
    expected_hours = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days * 24
    actual_hours = len(weather_df)
    coverage_pct = (actual_hours / expected_hours) * 100
    
    print(f"\nCoverage:")
    print(f"  Expected hours: {expected_hours:,}")
    print(f"  Actual hours: {actual_hours:,}")
    print(f"  Coverage: {coverage_pct:.1f}%")
    
    if coverage_pct < 95:
        print(f"\n⚠ WARNING: Coverage is less than 95%. Some data may be missing.")
