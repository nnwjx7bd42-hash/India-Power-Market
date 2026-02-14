import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import os

from src.data.loader import DataLoader
from src.data.splits import split_by_date, validate_no_leakage
from src.features.price_features import build_price_features
from src.features.bid_stack_features import build_bid_stack_features
from src.features.grid_features import build_grid_features
from src.features.weather_features import build_weather_features
from src.features.calendar_features import build_calendar_features

def build_all_features(config_path):
    """
    Orchestrate feature creation for DAM and RTM.
    Enforce temporal causality.
    Save parquets.
    """
    print("Initializing Pipeline...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    root_dir = Path(config_path).parent.parent
    features_dir = root_dir / config['data']['features_dir']
    
    # 2. Load + Aggregate Data
    loader = DataLoader(config_path)
    data = loader.load_all()
    
    # 3. Build Market-Independent Base Features
    print("Building base features (grid, weather, calendar)...")
    grid_feats = build_grid_features(data['grid'])
    weather_feats = build_weather_features(data['weather'])
    
    # Calendar uses delivery_start_ist from grid (which covers full range)
    calendar_feats = build_calendar_features(
        data['grid']['delivery_start_ist'], 
        data['holidays']
    )
    
    # 4. For each market
    for market in config['markets']:
        print(f"\n─── Processing Market: {market.upper()} ───")
        
        # a. Filter prices + bid stack
        prices_mkt = data['price'][data['price']['market'] == market].copy()
        bid_stack_mkt = data['bid_stack'][data['bid_stack']['market'] == market].copy()
        
        # b. Build price & bid stack features
        print("Building price & bid stack features...")
        price_feats = build_price_features(prices_mkt, market)
        bs_feats = build_bid_stack_features(bid_stack_mkt, market)
        
        # d. Cross-market features
        print("Building cross-market features...")
        cross_market_feats = pd.DataFrame(index=price_feats.index)
        other_mkt = 'rtm' if market == 'dam' else 'dam'
        other_prices = data['price'][data['price']['market'] == other_mkt]
        # Align on index
        other_prices = other_prices.set_index('delivery_start_ist').reindex(price_feats.index)
        
        # Spread lag 1h
        # (dam_mcp - rtm_mcp).shift(1)
        # But we are in 'market' loop. If market=dam, dam-rtm. If market=rtm, rtm-dam?
        # Prompt says: "dam_rtm_spread_lag_1h: DAM_mcp(t-1) - RTM_mcp(t-1)"
        # This implies direction is always DAM - RTM.
        # We need both DAM and RTM prices aligned.
        
        # Better: get DAM and RTM prices specifically
        dam_prices = data['price'][data['price']['market'] == 'dam'].set_index('delivery_start_ist')
        rtm_prices = data['price'][data['price']['market'] == 'rtm'].set_index('delivery_start_ist')
        
        # Align to current market index
        dam_aligned = dam_prices['mcp_rs_mwh'].reindex(price_feats.index)
        rtm_aligned = rtm_prices['mcp_rs_mwh'].reindex(price_feats.index)
        
        spread = dam_aligned - rtm_aligned
        cross_market_feats['cross_dam_rtm_spread_lag_1h'] = spread.shift(1)
        
        if market == 'rtm':
            # RTM only: dam_mcp_same_hour
            # Known after 13:00 D-1. Available for all hours on day D.
            # So for RTM at (D, H), taking DAM at (D, H) is safe?
            # Prompt: "dam_mcp_same_hour = DAM MCP for (same date, same hour)... valid feature for all 24 RTM hours"
            cross_market_feats['cross_dam_mcp_same_hour'] = dam_aligned
        
        # e. Join all features
        print("Joining all features...")
        # Inner join on delivery_start_ist
        all_feats = price_feats.join(bs_feats, how='inner')\
            .join(grid_feats, how='inner')\
            .join(weather_feats, how='inner')\
            .join(calendar_feats, how='inner')\
            .join(cross_market_feats, how='inner')
            
        # f. Add target
        # Target is actual MCP for this market
        # Prices_mkt already indexed by delivery_start_ist
        target_series = prices_mkt.set_index('delivery_start_ist')['mcp_rs_mwh']
        all_feats['target_mcp_rs_mwh'] = target_series
        
        # g. ENFORCE TEMPORAL CAUSALITY
        print(f"Enforcing temporal causality for {market}...")
        
        final_feats = None
        
        if market == 'rtm':
            # RTM Strategy
            # Shift passthrough columns by 1
            rtm_passthrough_cols = [
                'grid_demand_mw', 'grid_net_demand_mw', 'grid_solar_mw',
                'grid_wind_mw', 'grid_total_gen_mw', 'grid_fuel_mix_imputed',
                'grid_demand_gen_gap', 'grid_thermal_util', 'grid_renewable_share',
                'wx_national_temp', 'wx_delhi_temp', 'wx_national_shortwave',
                'wx_chennai_wind', 'wx_national_cloud',
                'wx_cooling_degree_hours', 'wx_heat_index', 'wx_temp_spread'
            ]
            
            # Verify columns exist
            valid_cols = [c for c in rtm_passthrough_cols if c in all_feats.columns]
            all_feats[valid_cols] = all_feats[valid_cols].shift(1)
            
            final_feats = all_feats
            # Add metadata columns
            final_feats['target_date'] = final_feats.index.date.astype(str) # String for safer splitting
            final_feats['target_hour'] = final_feats.index.hour
            
            # Drop incomplete days (warmup edge case)
            # We want exactly 24 rows per date
            date_counts = final_feats.groupby('target_date').size()
            valid_dates = date_counts[date_counts == 24].index
            final_feats = final_feats[final_feats['target_date'].isin(valid_dates)]
            
        elif market == 'dam':
            # DAM Strategy: Snapshot at D-1 09:00
            
            # Step 1: Raw hourly features (already in all_feats)
            # Make sure we have date/hour columns
            all_feats['date_obj'] = all_feats.index.date
            all_feats['hour'] = all_feats.index.hour
            
            # Step 2: Create snapshot mapping
            # For each unique target_date D, snapshot = (D - 1 day) at 08:00 IST
            unique_dates = pd.Series(all_feats['date_obj'].unique()).sort_values()
            
            # Calculate snapshot timestamp for each target date
            # target_date -> snapshot_ts
            # We need to be careful with timezones. existing index is tz-aware.
            # unique_dates are dates.
            
            # Convert unique_dates to D-1 08:00 IST
            # Using vectorized operations
            target_dates_dt = pd.to_datetime(unique_dates)
            snapshot_timestamps = target_dates_dt - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
            
            # Localize
            snapshot_timestamps = snapshot_timestamps.dt.tz_localize('Asia/Kolkata', ambiguous='infer')
            
            snapshot_map = pd.DataFrame({
                'target_date': unique_dates,
                'snapshot_ts': snapshot_timestamps
            })
            
            # Step 3: Extract shared features at snapshot_ts
            # Inner merge snapshot_map with all_feats on snapshot_ts
            # all_feats index is delivery_start_ist. Reset index to merge.
            
            # Columns that are shared (everything except calendar and same-hour-lag)
            # Identify columns to keep from snapshot
            exclude_cols = [
                'cal_hour', 'cal_hour_sin', 'cal_hour_cos', # vary per hour
                'cal_day_of_week', 'cal_month', 'cal_quarter', 'cal_is_weekend', 
                'cal_is_holiday', 'cal_is_monsoon', 'cal_days_to_nearest_holiday',
                'cal_month_sin', 'cal_month_cos', # technically these belong to target date D, not snapshot D-1
                'target_mcp_rs_mwh', # target
                'date_obj', 'hour'
            ]
            
            # Also exclude mcp_same_hour_yesterday if we had calculated it 
            # (but we haven't, that's done in Step 4)
            
            shared_cols = [c for c in all_feats.columns if c not in exclude_cols]
            
            # Prepare all_feats for merge
            feats_reset = all_feats.reset_index() # has 'delivery_start_ist'
            
            # Merge snapshot_map with feats_reset using snapshot_ts == delivery_start_ist
            shared_feats = pd.merge(
                snapshot_map,
                feats_reset[['delivery_start_ist'] + shared_cols],
                left_on='snapshot_ts',
                right_on='delivery_start_ist',
                how='inner'
            )
            # shared_feats has [target_date, snapshot_ts, delivery_start_ist (same), shared_cols...]
            
            # Step 4: Expand to 24 hours per target_date
            # Cross join target_dates with hours 0-23
            hours_df = pd.DataFrame({'target_hour': range(24)})
            
            # We want to join shared_feats (one per date) with hours (24 per date)
            # shared_feats has 'target_date'
            
            # Cross join shared_feats with hours_df?
            # efficiently: repeat shared_feats 24 times
            # shared_feats['key'] = 1
            # hours_df['key'] = 1
            # combined = pd.merge(shared_feats, hours_df, on='key').drop('key', axis=1)
            # This might be memory intensive if big. 1371 days * 24 = 33k rows. Small.
            
            shared_feats['key'] = 1
            hours_df['key'] = 1
            expanded = pd.merge(shared_feats, hours_df, on='key').drop('key', axis=1)
            
            # Now add per-hour features
            # 1. Calendar features for (target_date, target_hour)
            # Re-calculate or fetch?
            # Fetching is easier if we have them indexed.
            # Construct target timestamp
            target_ts = pd.to_datetime(expanded['target_date']) + pd.to_timedelta(expanded['target_hour'], unit='h')
            target_ts = target_ts.dt.tz_localize('Asia/Kolkata', ambiguous='infer', nonexistent='shift_forward')
            
            expanded['target_ts'] = target_ts
            
            # Merge calendar features from all_feats (or recalculate)
            cal_cols = [c for c in all_feats.columns if c.startswith('cal_')]
            cal_data = feats_reset[['delivery_start_ist'] + cal_cols]
            
            # DEBUG: Check merge keys
            # print(f"Target TS sample: {target_ts.head()}")
            # print(f"Cal Data TS sample: {cal_data['delivery_start_ist'].head()}")
            
            expanded = pd.merge(
                expanded,
                cal_data,
                left_on='target_ts',
                right_on='delivery_start_ist',
                how='left',
                suffixes=('', '_cal')
            )
            # Debug outcome
            # if expanded['cal_hour'].isnull().any():
            #    print("Merge failed for calendar features. Sample NaNs:")
            #    print(expanded[expanded['cal_hour'].isnull()][['target_ts']].head())
            #    print("Sample Matches:")
            #    print(expanded[~expanded['cal_hour'].isnull()][['target_ts']].head())
            
            # Drop extra delivery_start_ist_cal
            # Drop extra delivery_start_ist_cal
            if 'delivery_start_ist_cal' in expanded.columns:
                 expanded = expanded.drop('delivery_start_ist_cal', axis=1)
            
            # 2. mcp_same_hour_yesterday
            # Logic:
            # If target_hour H <= 8: mcp at D-1 hour H
            # If target_hour H >= 9: mcp at D-2 hour H
            
            # Get MCP history
            # We need MCP At (D-1, H) and (D-2, H)
            # Vectorized lookup
            
            # Construct timestamps for D-1 H and D-2 H
            ts_d1 = expanded['target_ts'] - pd.Timedelta(days=1)
            ts_d2 = expanded['target_ts'] - pd.Timedelta(days=2)
            
            # Create condition
            cond = expanded['target_hour'] <= 8
            
            # Select timestamp based on condition
            lookup_ts = np.where(cond, ts_d1, ts_d2)
            
            # Map lookup_ts to MCP
            # Create a lookup series
            # Use raw prices_mkt for lookup, as all_feats doesn't contain raw mcp (only target and lags)
            price_lookup = prices_mkt.set_index('delivery_start_ist')['mcp_rs_mwh']
            
            # We need to map lookup_ts (array of timestamps) to prices
            # Use pd.Series.map or merge? Merge is safer with timestamps
            
            lookup_df = pd.DataFrame({'lookup_ts': lookup_ts}, index=expanded.index)
                                     
            # Merge to get value
            # price_lookup needs to be DataFrame for merge
            price_lookup_df = price_lookup.reset_index()
            
            merged_lookup = pd.merge(
                lookup_df, 
                price_lookup_df, 
                left_on='lookup_ts', 
                right_on='delivery_start_ist', 
                how='left'
            )
            
            expanded['mcp_same_hour_yesterday'] = merged_lookup['mcp_rs_mwh']
            
            # Step 5: Add Target
            # Target is MCP at target_ts
            # Merge
            expanded = pd.merge(
                expanded,
                price_lookup_df.rename(columns={'mcp_rs_mwh': 'target_mcp_rs_mwh'}),
                left_on='target_ts',
                right_on='delivery_start_ist',
                how='left'
            )
            
            # Cleanup
            # Set index to target_ts (which is delivery_start_ist for the prediction)
            final_feats = expanded.set_index('target_ts')
            final_feats.index.name = 'delivery_start_ist'
            
            # Ensure target_date is string
            final_feats['target_date'] = final_feats['target_date'].astype(str)
            
            # Remove intermediate columns
            drop_cols = ['snapshot_ts', 'delivery_start_ist_x', 'delivery_start_ist_y', 'lookup_ts']
            final_feats = final_feats.drop([c for c in drop_cols if c in final_feats.columns], axis=1)

            # ═══════════════════════════════════════════════════════
            # DAM Day+1 Feature Construction
            # Same D-1 08:00 snapshot, shifted target forward by 1 day.
            # Causality: all features from D-1, targets from D+1.
            # ═══════════════════════════════════════════════════════
            print("\n─── Building DAM Day+1 Features ───")

            # Start from the Day D expanded frame (before target merge)
            # Re-use 'expanded' which has shared feats + calendar + mcp_same_hour_yesterday + target
            expanded_d1 = expanded.copy()

            # Shift target_date forward by 1 day
            expanded_d1['target_date'] = (
                pd.to_datetime(expanded_d1['target_date']) + pd.Timedelta(days=1)
            ).dt.strftime('%Y-%m-%d')

            # Reconstruct target_ts for D+1
            target_ts_d1 = (
                pd.to_datetime(expanded_d1['target_date'])
                + pd.to_timedelta(expanded_d1['target_hour'], unit='h')
            )
            target_ts_d1 = target_ts_d1.dt.tz_localize('Asia/Kolkata', ambiguous='infer', nonexistent='shift_forward')
            expanded_d1['target_ts'] = target_ts_d1

            # Re-merge calendar features for D+1 dates
            cal_cols_to_drop = [c for c in expanded_d1.columns if c.startswith('cal_')]
            expanded_d1 = expanded_d1.drop(columns=cal_cols_to_drop)
            expanded_d1 = pd.merge(
                expanded_d1, cal_data,
                left_on='target_ts', right_on='delivery_start_ist',
                how='left'
            )
            if 'delivery_start_ist' in expanded_d1.columns and 'target_ts' in expanded_d1.columns:
                expanded_d1 = expanded_d1.drop(columns=['delivery_start_ist'], errors='ignore')

            # mcp_same_hour_yesterday for D+1:
            # At D-1 09:00, the most recent same-hour price is (D-1, H) for ALL hours.
            # D+1 - 2 days = D-1, so lookup_ts = target_ts - 2 days.
            ts_d_minus_1 = expanded_d1['target_ts'] - pd.Timedelta(days=2)
            lookup_d1 = pd.DataFrame({'lookup_ts': ts_d_minus_1}, index=expanded_d1.index)
            merged_d1 = pd.merge(
                lookup_d1, price_lookup_df,
                left_on='lookup_ts', right_on='delivery_start_ist', how='left'
            )
            expanded_d1['mcp_same_hour_yesterday'] = merged_d1['mcp_rs_mwh'].values

            # Target for D+1: MCP at (D+1, H)
            # Drop any existing target column first
            expanded_d1 = expanded_d1.drop(columns=['target_mcp_rs_mwh'], errors='ignore')
            expanded_d1 = pd.merge(
                expanded_d1,
                price_lookup_df.rename(columns={'mcp_rs_mwh': 'target_mcp_rs_mwh'}),
                left_on='target_ts', right_on='delivery_start_ist',
                how='left'
            )

            # Cleanup D+1
            final_feats_d1 = expanded_d1.set_index('target_ts')
            final_feats_d1.index.name = 'delivery_start_ist'
            final_feats_d1['target_date'] = final_feats_d1['target_date'].astype(str)
            drop_cols_d1 = ['snapshot_ts', 'delivery_start_ist_x', 'delivery_start_ist_y',
                            'delivery_start_ist', 'lookup_ts']
            final_feats_d1 = final_feats_d1.drop(
                [c for c in drop_cols_d1 if c in final_feats_d1.columns], axis=1
            )

            # Drop NaN, enforce complete days, split, validate, save — same as Day D
            null_counts_d1 = final_feats_d1.isnull().sum()
            if null_counts_d1.sum() > 0:
                nonnull = null_counts_d1[null_counts_d1 > 0]
                print(f"D+1 NaN counts: {dict(nonnull)}")
            before_d1 = len(final_feats_d1)
            final_feats_d1 = final_feats_d1.dropna()
            print(f"D+1: Dropped {before_d1 - len(final_feats_d1)} rows due to warmup/NaNs")

            date_counts_d1 = final_feats_d1.groupby('target_date').size()
            valid_dates_d1 = date_counts_d1[date_counts_d1 == 24].index
            final_feats_d1 = final_feats_d1[final_feats_d1['target_date'].isin(valid_dates_d1)]

            print("Splitting D+1...")
            splits_d1 = split_by_date(final_feats_d1, config, date_col='target_date')
            validate_no_leakage(splits_d1, date_col='target_date')

            print("Saving D+1 Parquets...")
            for split_name, df_split in splits_d1.items():
                filename = f"dam_d1_features_{split_name}.parquet"
                out_path = features_dir / filename
                df_split.to_parquet(out_path)
                print(f"Saved {filename}: {df_split.shape}")

        # h. Drop NaN (Warmup)
        print("Dropping NaNs...")
        # DEBUG: Check which columns have NaNs
        null_counts = final_feats.isnull().sum()
        if null_counts.sum() > 0:
            print("NaN Counts per column before drop:")
            print(null_counts[null_counts > 0])
            
        before_len = len(final_feats)
        final_feats = final_feats.dropna()
        after_len = len(final_feats)
        print(f"Dropped {before_len - after_len} rows due to warmup/NaNs")
        
        # Enforce complete days (24 rows)
        # This handles the RTM edge case where warmup consumes partial first day
        date_counts = final_feats.groupby('target_date').size()
        valid_dates = date_counts[date_counts == 24].index
        rows_before_filter = len(final_feats)
        final_feats = final_feats[final_feats['target_date'].isin(valid_dates)]
        print(f"Dropped {rows_before_filter - len(final_feats)} rows to enforce complete days")
        
        # i. Split by date
        print("Splitting...")
        splits = split_by_date(final_feats, config, date_col='target_date')
        
        # j. Validate splits
        validate_no_leakage(splits, date_col='target_date')
        
        # k. Save
        print("Saving Parquets...")
        for split_name, df_split in splits.items():
            filename = f"{market}_features_{split_name}.parquet"
            out_path = features_dir / filename
            df_split.to_parquet(out_path)
            print(f"Saved {filename}: {df_split.shape}")

    print("\nPipeline Complete.")

if __name__ == "__main__":
    # For testing import
    pass
