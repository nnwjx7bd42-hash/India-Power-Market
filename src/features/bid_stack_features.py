import pandas as pd
import numpy as np

def build_bid_stack_features(bid_stack_df, market):
    """
    Build bid stack features for ONE market.
    Input: hourly bid stack for ONE market (12 bands per hour).
    Output: DataFrame indexed on delivery_start_ist, feature columns only.
    """
    # 1. Aggregate across 12 bands per hour
    # We need to preserve delivery_start_ist index in the result
    
    # Ensure sorted
    df = bid_stack_df.sort_values(['delivery_start_ist', 'price_band_rs_mwh'])
    
    # Features relative to total
    # Group by delivery_start_ist
    grouped = df.groupby('delivery_start_ist')
    
    # Total Buy/Sell
    total_buy = grouped['buy_demand_mw'].sum()
    total_sell = grouped['sell_supply_mw'].sum()
    
    # Features Dataframe
    feats = pd.DataFrame(index=total_buy.index)
    
    feats['bs_total_buy_mw'] = total_buy
    feats['bs_total_sell_mw'] = total_sell
    
    # Ratio
    feats['bs_buy_sell_ratio'] = feats['bs_total_buy_mw'] / feats['bs_total_sell_mw']
    feats['bs_buy_sell_ratio'] = feats['bs_buy_sell_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Supply Margin (sell_supply in bands >= 8001)
    # Filter bands
    # Unique bands: '0-1000' ... '11001-12000'
    # High bands: '8001-9000', '9001-10000', '10001-11000', '11001-12000'
    high_bands = ['8001-9000', '9001-10000', '10001-11000', '11001-12000']
    
    # Filter rows with these bands
    high_supply = df[df['price_band_rs_mwh'].isin(high_bands)].groupby('delivery_start_ist')['sell_supply_mw'].sum()
    # Need to align with full index (some hours might have 0 high supply and thus missing from groupby if filtered before)
    # But since we have 12 bands always, it should be fine. 
    # Use reindex to be safe
    feats['bs_supply_margin_mw'] = high_supply.reindex(feats.index, fill_value=0)
    
    # Cheap Supply (0-3000)
    cheap_bands = ['0-1000', '1001-2000', '2001-3000']
    cheap_supply = df[df['price_band_rs_mwh'].isin(cheap_bands)].groupby('delivery_start_ist')['sell_supply_mw'].sum()
    feats['bs_cheap_supply_mw'] = cheap_supply.reindex(feats.index, fill_value=0)
    
    # Cheap Supply Share
    feats['bs_cheap_supply_share'] = feats['bs_cheap_supply_mw'] / feats['bs_total_sell_mw']
    feats['bs_cheap_supply_share'] = feats['bs_cheap_supply_share'].replace([np.inf, -np.inf], np.nan)
    
    # HHI (Concentration)
    # sum((band_share)^2)
    # Need share per band
    # Perform calculation on the original df, then sum
    
    # Map totals back to original DF for calculation
    # Only need totals for the same hour
    # Join totals
    df_with_totals = df.merge(total_buy.rename('total_buy'), on='delivery_start_ist').merge(total_sell.rename('total_sell'), on='delivery_start_ist')
    
    # HHI calculation
    # Avoid div by zero
    df_with_totals['buy_share_sq'] = (df_with_totals['buy_demand_mw'] / df_with_totals['total_buy'].replace(0, np.nan)) ** 2
    df_with_totals['sell_share_sq'] = (df_with_totals['sell_supply_mw'] / df_with_totals['total_sell'].replace(0, np.nan)) ** 2
    
    buy_hhi = df_with_totals.groupby('delivery_start_ist')['buy_share_sq'].sum()
    sell_hhi = df_with_totals.groupby('delivery_start_ist')['sell_share_sq'].sum()
    
    feats['bs_buy_hhi'] = buy_hhi
    feats['bs_sell_hhi'] = sell_hhi
    
    # Fill NA HHI with 0 (if total was 0)
    feats[['bs_buy_hhi', 'bs_sell_hhi']] = feats[['bs_buy_hhi', 'bs_sell_hhi']].fillna(0)
    
    # 2. Lag all 8 features by 1 hour
    lagged_feats = feats.shift(1).add_suffix('_lag_1h')
    
    return lagged_feats
