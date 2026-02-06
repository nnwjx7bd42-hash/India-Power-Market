#!/usr/bin/env python3
"""Verify the unified dataset"""

import pandas as pd

df = pd.read_parquet('data/processed/unified_dataset.parquet')

print('='*80)
print('UNIFIED DATASET VERIFICATION')
print('='*80)
print(f'\n✓ Dataset loaded successfully')
print(f'  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns')
print(f'  Date range: {df.index.min()} to {df.index.max()}')
print(f'  Duration: {(df.index.max() - df.index.min()).days} days')

print(f'\nColumns ({len(df.columns)}):')
for i, col in enumerate(df.columns, 1):
    print(f'  {i:2d}. {col}')

print(f'\nSample data (first 5 rows):')
print(df.head())

print(f'\nMissing values:')
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print('  None')

print(f'\nData summary:')
print(df.describe().T)

print(f'\n✓ Data is ready for feature engineering!')
