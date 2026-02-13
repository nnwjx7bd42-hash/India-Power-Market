import pandas as pd
import numpy as np

def build_price_features(prices_df, market):
    """
    Build price features for ONE market.
    Input: hourly prices for ONE market, sorted by delivery_start_ist.
    Output: DataFrame indexed on delivery_start_ist, feature columns only.
    """
    # Ensure sorted by time
    df = prices_df.sort_values('delivery_start_ist').copy()
    df = df.set_index('delivery_start_ist')
    
    # Base columns to use
    mcp_col = 'mcp_rs_mwh'
    mcv_col = 'mcv_mwh'
    buy_col = 'purchase_bid_mwh'
    sell_col = 'sell_bid_mwh'
    
    features = pd.DataFrame(index=df.index)
    
    # ─── Lags (Backward-looking) ───
    features['mcp_lag_1h'] = df[mcp_col].shift(1)
    features['mcp_lag_2h'] = df[mcp_col].shift(2)
    features['mcp_lag_4h'] = df[mcp_col].shift(4)
    features['mcp_lag_24h'] = df[mcp_col].shift(24)
    features['mcp_lag_168h'] = df[mcp_col].shift(168)
    
    # ─── Rolling Stats (Backward-looking) ───
    # min_periods=window to avoid leakage at start
    features['mcp_rolling_mean_24h'] = df[mcp_col].rolling(window=24, min_periods=24).mean()
    features['mcp_rolling_std_24h'] = df[mcp_col].rolling(window=24, min_periods=24).std()
    features['mcp_rolling_mean_168h'] = df[mcp_col].rolling(window=168, min_periods=168).mean()
    
    # ─── Volume Features ───
    features['mcv_lag_1h'] = df[mcv_col].shift(1)
    features['mcv_rolling_mean_24h'] = df[mcv_col].rolling(window=24, min_periods=24).mean()
    
    # ─── Bid Pressure ───
    # Ratio: purchase / sell. Handle divide by zero.
    bid_ratio = df[buy_col] / df[sell_col]
    bid_ratio = bid_ratio.replace([np.inf, -np.inf], np.nan)
    features['bid_ratio_lag_1h'] = bid_ratio.shift(1)
    
    return features
