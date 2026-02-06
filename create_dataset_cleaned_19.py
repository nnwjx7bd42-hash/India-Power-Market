#!/usr/bin/env python3
"""
Create dataset_cleaned_19.parquet with only the 19 curated features + target P(T).
Uses the same index and row order as dataset_cleaned.parquet.
"""

import pandas as pd
from pathlib import Path

# 19 features from framework application (curated set)
CURATED_19_FEATURES = [
    'P(T-1)', 'P(T-2)', 'P(T-24)', 'P(T-48)', 'P_T-168', 'Price_MA_24h',
    'Hour', 'Hour_Sin', 'Hour_Cos', 'DayOfWeek', 'IsMonday', 'IsWeekend', 'Month', 'Season',
    'CDH',
    'Net_Load', 'Solar_Ramp',
    'Load_MA_24h',
    'Hour_x_CDH',
]
TARGET = 'P(T)'
COLUMNS = CURATED_19_FEATURES + [TARGET]


def main():
    src = Path('data/processed/dataset_cleaned.parquet')
    dst = Path('data/processed/dataset_cleaned_19.parquet')
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    df = pd.read_parquet(src)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in source: {missing}")
    df_19 = df[COLUMNS].copy()
    df_19.to_parquet(dst)
    print(f"Saved {dst}")
    print(f"  Shape: {df_19.shape} ({len(CURATED_19_FEATURES)} features + 1 target)")
    print(f"  Date range: {df_19.index.min()} to {df_19.index.max()}")


if __name__ == '__main__':
    main()
