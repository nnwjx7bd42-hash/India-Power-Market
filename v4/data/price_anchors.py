"""
Long-memory price anchor features for the v4 planning model.

All anchors reference prices >= 168 hours in the past and are therefore
fully known at week-ahead planning time.

Features
--------
P_same_hour_last_week      P(t - 168)
P_same_hour_2weeks_ago     P(t - 336)
P_weekly_avg_by_hour       Rolling 4-week mean for same hour-of-day
P_weekly_median_by_hour    Rolling 4-week median for same hour-of-day
P_weekly_std_by_hour       Rolling 4-week std for same hour-of-day
P_percentile_90_by_hour    Rolling 4-week 90th percentile for same hour-of-day
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rolling stats grouped by hour-of-day
# ---------------------------------------------------------------------------

def _rolling_stat_by_hour(price: pd.Series, hour: pd.Series,
                          window_weeks: int, stat: str) -> pd.Series:
    """
    For each row, compute a rolling statistic of *price* over the last
    ``window_weeks`` weeks, restricted to rows with the **same hour-of-day**.

    The window ends 168 hours (1 week) before the current row to guarantee
    that only information available at planning time is used.

    Parameters
    ----------
    price : pd.Series  — P(T), datetime-indexed
    hour  : pd.Series  — hour-of-day (0-23), same index
    window_weeks : int  — number of weeks in the lookback window (default 4)
    stat  : str         — one of 'mean', 'median', 'std', 'p90'
    """
    n = len(price)
    result = np.full(n, np.nan)
    # Pre-compute aligned arrays for speed
    p_arr = price.values.astype(float)
    h_arr = hour.values.astype(int)
    # window size in hours and safety offset (1 week)
    window_h = window_weeks * 168
    offset = 168  # exclude most recent week (planning constraint)

    for i in range(n):
        target_hour = h_arr[i]
        start = max(0, i - offset - window_h)
        end = i - offset  # exclusive upper bound
        if end <= start:
            continue
        # Select same-hour rows within window
        mask = h_arr[start:end] == target_hour
        vals = p_arr[start:end][mask]
        if len(vals) == 0:
            continue
        if stat == "mean":
            result[i] = np.nanmean(vals)
        elif stat == "median":
            result[i] = np.nanmedian(vals)
        elif stat == "std":
            result[i] = np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0
        elif stat == "p90":
            result[i] = np.nanpercentile(vals, 90)
    return pd.Series(result, index=price.index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_price_anchors(df: pd.DataFrame,
                      price_col: str = "P(T)",
                      window_weeks: int = 4) -> pd.DataFrame:
    """
    Add all 6 long-memory price anchor columns to *df* (in-place).

    Parameters
    ----------
    df : pd.DataFrame — must contain ``price_col`` and ``Hour``
    price_col : str
    window_weeks : int — rolling window for grouped stats (default 4)

    Returns
    -------
    pd.DataFrame with 6 new columns added.
    """
    price = df[price_col]
    hour = df["Hour"] if "Hour" in df.columns else pd.Series(df.index.hour, index=df.index)

    # Simple shift anchors
    df["P_same_hour_last_week"] = price.shift(168)
    df["P_same_hour_2weeks_ago"] = price.shift(336)

    # Rolling grouped stats (vectorised inner loop)
    print("   Computing P_weekly_avg_by_hour ...")
    df["P_weekly_avg_by_hour"] = _rolling_stat_by_hour(price, hour, window_weeks, "mean")
    print("   Computing P_weekly_median_by_hour ...")
    df["P_weekly_median_by_hour"] = _rolling_stat_by_hour(price, hour, window_weeks, "median")
    print("   Computing P_weekly_std_by_hour ...")
    df["P_weekly_std_by_hour"] = _rolling_stat_by_hour(price, hour, window_weeks, "std")
    print("   Computing P_percentile_90_by_hour ...")
    df["P_percentile_90_by_hour"] = _rolling_stat_by_hour(price, hour, window_weeks, "p90")

    return df
