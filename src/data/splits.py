import pandas as pd
import yaml

def split_by_date(df, config, date_col='date'):
    """Split DataFrame into train, val, backtest based on config."""
    # If df has delivery_start_ist but no date column, derive it
    if date_col not in df.columns and 'delivery_start_ist' in df.columns:
        df = df.copy()
        df[date_col] = df['delivery_start_ist'].dt.date.astype(str)
    
    # Ensure date column is string for comparison or datetime
    # Config dates are strings "YYYY-MM-DD"
    # Let's standardize on string YYYY-MM-DD for comparison safety
    
    # Convert config dates to timestamps for robust comparison
    splits_cfg = config['splits']
    
    train_start = pd.Timestamp(splits_cfg['train']['start'])
    train_end = pd.Timestamp(splits_cfg['train']['end'])
    
    val_start = pd.Timestamp(splits_cfg['validation']['start'])
    val_end = pd.Timestamp(splits_cfg['validation']['end'])
    
    backtest_start = pd.Timestamp(splits_cfg['backtest']['start'])
    backtest_end = pd.Timestamp(splits_cfg['backtest']['end'])
    
    # Convert series to timestamp if not already
    dates = pd.to_datetime(df[date_col])
    
    train_mask = (dates >= train_start) & (dates <= train_end)
    val_mask = (dates >= val_start) & (dates <= val_end)
    backtest_mask = (dates >= backtest_start) & (dates <= backtest_end)
    
    return {
        "train": df[train_mask].copy(),
        "val": df[val_mask].copy(),
        "backtest": df[backtest_mask].copy()
    }

def validate_no_leakage(splits_dict, date_col='date'):
    """Assert strict temporal separation."""
    train = splits_dict['train']
    val = splits_dict['val']
    backtest = splits_dict['backtest']
    
    # Get dates
    def get_dates(d):
        if date_col in d.columns:
            return pd.to_datetime(d[date_col])
        elif 'delivery_start_ist' in d.columns:
            return d['delivery_start_ist']
        else:
            raise ValueError(f"No date column '{date_col}' found")

    train_dates = get_dates(train)
    val_dates = get_dates(val)
    backtest_dates = get_dates(backtest)
    
    print("\nValidating Splits (Anti-Leakage):")
    print(f"Train:    {train_dates.min()} -> {train_dates.max()} ({len(train)} rows)")
    print(f"Val:      {val_dates.min()} -> {val_dates.max()} ({len(val)} rows)")
    print(f"Backtest: {backtest_dates.min()} -> {backtest_dates.max()} ({len(backtest)} rows)")
    
    # Assertions
    assert train_dates.max() < val_dates.min(), "Train overlaps with Validation!"
    assert val_dates.max() < backtest_dates.min(), "Validation overlaps with Backtest!"
    
    # Overlap check
    t_set = set(train_dates)
    v_set = set(val_dates)
    b_set = set(backtest_dates)
    
    assert len(t_set.intersection(v_set)) == 0, "Date leakage Train-Val"
    assert len(v_set.intersection(b_set)) == 0, "Date leakage Val-Backtest"
    assert len(t_set.intersection(b_set)) == 0, "Date leakage Train-Backtest"
    
    print("✓ Temporal separation passed.")
