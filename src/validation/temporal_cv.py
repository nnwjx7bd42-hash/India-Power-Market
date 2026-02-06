"""
Temporal cross-validation for time series data
Implements expanding window CV with gap buffer to prevent data leakage
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def temporal_cv_split(df, n_splits=7, gap_hours=24, test_months=3):
    """
    Generate temporal cross-validation splits with expanding window
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    n_splits : int or None
        Number of CV folds. If None, use all available test windows (auto from data).
    gap_hours : int
        Gap between train and test sets (hours)
    test_months : int
        Number of months per test fold
    
    Returns:
    --------
    list of tuples
        List of (train_indices, test_indices) for each fold
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index")
    
    # Sort by timestamp
    df_sorted = df.sort_index()
    
    # Calculate split points
    start_date = df_sorted.index.min()
    end_date = df_sorted.index.max()
    
    # Generate test period start dates (every 3 months)
    test_starts = pd.date_range(
        start=start_date + pd.DateOffset(months=4),  # First test after 4 months of training
        end=end_date - pd.DateOffset(months=test_months),
        freq=f'{test_months}MS'  # Month start, every test_months
    )
    
    # Limit to n_splits (if None, use all available test windows)
    if n_splits is not None and len(test_starts) > n_splits:
        test_starts = test_starts[:n_splits]
    
    splits = []
    
    for i, test_start in enumerate(test_starts):
        # Test period
        test_end = test_start + pd.DateOffset(months=test_months)
        
        # Training period: from start to test_start - gap
        train_end = test_start - pd.Timedelta(hours=gap_hours)
        
        # Get indices
        train_mask = (df_sorted.index >= start_date) & (df_sorted.index <= train_end)
        test_mask = (df_sorted.index >= test_start) & (df_sorted.index < test_end)
        
        train_indices = df_sorted.index[train_mask]
        test_indices = df_sorted.index[test_mask]
        
        if len(train_indices) > 0 and len(test_indices) > 0:
            splits.append((train_indices, test_indices))
    
    return splits


class TemporalCV:
    """
    Temporal cross-validation iterator
    """
    
    def __init__(self, df, n_splits=7, gap_hours=24, test_months=3):
        """
        Initialize temporal CV
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with datetime index
        n_splits : int or None
            Number of CV folds. If None, use all available test windows.
        gap_hours : int
            Gap between train and test sets
        test_months : int
            Number of months per test fold
        """
        self.df = df.sort_index()
        self.gap_hours = gap_hours
        self.test_months = test_months
        self.splits = temporal_cv_split(df, n_splits, gap_hours, test_months)
        self.n_splits = len(self.splits)
    
    def __iter__(self):
        """Iterator over CV splits"""
        for train_indices, test_indices in self.splits:
            yield train_indices, test_indices
    
    def __len__(self):
        """Number of splits"""
        return len(self.splits)
    
    def get_split_info(self, fold_idx):
        """
        Get information about a specific fold
        
        Parameters:
        -----------
        fold_idx : int
            Fold index (0-based)
        
        Returns:
        --------
        dict
            Information about the fold
        """
        if fold_idx >= len(self.splits):
            raise IndexError(f"Fold {fold_idx} does not exist")
        
        train_indices, test_indices = self.splits[fold_idx]
        
        return {
            'fold': fold_idx + 1,
            'train_start': train_indices.min(),
            'train_end': train_indices.max(),
            'test_start': test_indices.min(),
            'test_end': test_indices.max(),
            'train_size': len(train_indices),
            'test_size': len(test_indices),
            'gap_hours': self.gap_hours
        }
    
    def print_summary(self):
        """Print summary of all CV folds"""
        print("="*80)
        print("TEMPORAL CROSS-VALIDATION SPLITS")
        print("="*80)
        
        for i in range(len(self.splits)):
            info = self.get_split_info(i)
            print(f"\nFold {info['fold']}:")
            print(f"  Train: {info['train_start']} to {info['train_end']} ({info['train_size']:,} samples)")
            print(f"  Gap: {info['gap_hours']} hours")
            print(f"  Test:  {info['test_start']} to {info['test_end']} ({info['test_size']:,} samples)")
