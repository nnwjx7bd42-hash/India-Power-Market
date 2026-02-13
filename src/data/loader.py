import pandas as pd
import yaml
import os
from pathlib import Path

class DataLoader:
    def __init__(self, config_path):
        """Initialize DataLoader with config path."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.root_dir = Path(config_path).parent.parent
        self.cleaned_dir = self.root_dir / self.config['data']['cleaned_dir']

    def load_all(self):
        """Load all data and aggregate to hourly resolution."""
        print("Loading and aggregating data...")
        
        price_df = self._load_prices()
        bid_stack_df = self._load_bid_stack()
        grid_df = self._load_grid()
        weather_df = self._load_weather()
        holidays_df = self._load_holidays()
        
        data = {
            'price': price_df,
            'bid_stack': bid_stack_df,
            'grid': grid_df,
            'weather': weather_df,
            'holidays': holidays_df
        }
        
        self._print_summary(data)
        return data

    def _load_prices(self):
        """Load prices and aggregate 15-min -> hourly."""
        path = self.cleaned_dir / self.config['data']['price_file']
        df = pd.read_parquet(path)
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Use existing 'hour' col if present, else derive from 'time_block'
        if 'hour' not in df.columns:
            if 'time_block' in df.columns:
                df['hour'] = (df['time_block'] - 1) // 4
            elif 'block' in df.columns:
                 df['hour'] = (df['block'] - 1) // 4
        
        # Aggregation logic
        # mcp_rs_mwh: volume-weighted mean -> sum(mcp * mcv) / sum(mcv)
        # Columns in file: mcp_rs_mwh, mcv_mwh, purchase_bid_mwh, sell_bid_mwh, weighted_mcp_rs_mwh
        
        # 1. Calculate value * volume
        # Handle potential missing columns gracefully, but we know they exist from inspection
        if 'mcp_rs_mwh' in df.columns and 'mcv_mwh' in df.columns:
             df['mcp_x_volume'] = df['mcp_rs_mwh'] * df['mcv_mwh']
        else:
             # Fallback or error?
             # If mcv is missing, we can't volume weight.
             if 'mcp_rs_mwh' in df.columns:
                 df['mcp_x_volume'] = df['mcp_rs_mwh'] # Just to have something, but weighting will fail
             else:
                 pass # Error likely later
        
        agg_rules = {
            'mcp_x_volume': 'sum',
            'mcv_mwh': 'sum',
            'purchase_bid_mwh': 'sum',
            'sell_bid_mwh': 'sum',
            'mcp_rs_mwh': 'mean', # Fallback
            'weighted_mcp_rs_mwh': 'mean',
            'final_scheduled_volume_mwh': 'sum'
        }
        
        # Filter agg_rules to only cols present
        agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}
        
        grouped = df.groupby(['date', 'market', 'hour'])
        agg_df = grouped.agg(agg_rules).reset_index()
        
        # Calculate volume weighted MCP
        if 'mcp_x_volume' in agg_df.columns and 'mcv_mwh' in agg_df.columns:
            mask = agg_df['mcv_mwh'] > 0
            agg_df.loc[mask, 'mcp_rs_mwh'] = agg_df.loc[mask, 'mcp_x_volume'] / agg_df.loc[mask, 'mcv_mwh']
        
        # Construct delivery_start_ist
        # We know inputs have it, but for aggregation result, we must reconstruct it for the hour
        # because the aggregation "collapses" the 15-min rows.
        # Vectorized creation
        dates = pd.to_datetime(agg_df['date'])
        hours = pd.to_timedelta(agg_df['hour'], unit='h')
        timestamps = dates + hours
        # Localize
        agg_df['delivery_start_ist'] = timestamps.dt.tz_localize('Asia/Kolkata', ambiguous='infer', nonexistent='shift_forward')
        
        return agg_df

    def _load_bid_stack(self):
        """Load bid stack and aggregate 15-min -> hourly."""
        path = self.cleaned_dir / self.config['data']['bid_stack_file']
        df = pd.read_parquet(path)
        
        df['date'] = pd.to_datetime(df['date'])
        
        if 'hour' not in df.columns:
             if 'time_block' in df.columns:
                 df['hour'] = (df['time_block'] - 1) // 4
        
        # Columns: buy_demand_mw, sell_supply_mw
        agg_cols = {}
        if 'buy_demand_mw' in df.columns: agg_cols['buy_demand_mw'] = 'mean'
        if 'sell_supply_mw' in df.columns: agg_cols['sell_supply_mw'] = 'mean'
        
        grouped = df.groupby(['date', 'market', 'hour', 'price_band_rs_mwh'])
        agg_df = grouped.agg(agg_cols).reset_index()
        
        # Add delivery_start_ist 
        dates = pd.to_datetime(agg_df['date'])
        hours = pd.to_timedelta(agg_df['hour'], unit='h')
        timestamps = dates + hours
        agg_df['delivery_start_ist'] = timestamps.dt.tz_localize('Asia/Kolkata', ambiguous='infer', nonexistent='shift_forward')
        
        return agg_df

    def _load_grid(self):
        """Load grid data (already hourly)."""
        path = self.cleaned_dir / self.config['data']['grid_file']
        df = pd.read_parquet(path)
        
        # Ensure delivery_start_ist is correct
        # Should be IST-aware
        if 'delivery_start' in df.columns and 'delivery_start_ist' not in df.columns:
             # Convert
             df['delivery_start_ist'] = pd.to_datetime(df['delivery_start']).dt.tz_localize('Asia/Kolkata', ambiguous='infer')
        elif 'delivery_start_ist' in df.columns:
            # Ensure it is timezone aware
            if df['delivery_start_ist'].dt.tz is None:
                df['delivery_start_ist'] = df['delivery_start_ist'].dt.tz_localize('Asia/Kolkata')
                
        return df

    def _load_weather(self):
        """Load weather and aggregate 5 cities -> national."""
        path = self.cleaned_dir / self.config['data']['weather_file']
        df = pd.read_parquet(path)
        
        # Weights
        # If 'weight' column exists, use it. Otherwise map.
        if 'weight' not in df.columns:
            weights = {
                'Delhi': 0.30,
                'Mumbai': 0.28,
                'Chennai': 0.25,
                'Kolkata': 0.12,
                'Guwahati': 0.05
            }
            df['weight'] = df['city'].map(weights)
        
        # Columns to weight
        # temp, humidity, shortwave (radiation), cloud_cover
        cols_to_weight = ['temperature_2m', 'relative_humidity_2m', 'shortwave_radiation', 'cloud_cover']
        cols_present = [c for c in cols_to_weight if c in df.columns]

        for col in cols_present:
            df[f'{col}_weighted'] = df[col] * df['weight']
            
        # Group by delivery_start_ist
        if 'delivery_start_ist' in df.columns:
             group_col = 'delivery_start_ist'
        elif 'timestamp' in df.columns:
             group_col = 'timestamp' # Fallback
        else:
             # creating from date+hour failed, so must have one of above
             raise ValueError("Weather file missing delivery_start_ist or timestamp")

        # Aggregation
        agg_rules = {f'{col}_weighted': 'sum' for col in cols_present}
        
        grouped = df.groupby(group_col)
        national = grouped.agg(agg_rules).reset_index()
        
        # Rename national columns
        rename_map = {
            'temperature_2m_weighted': 'national_temperature',
            'relative_humidity_2m_weighted': 'national_humidity',
            'shortwave_radiation_weighted': 'national_shortwave',
            'cloud_cover_weighted': 'national_cloud_cover'
        }
        national = national.rename(columns=rename_map)
        
        # Delhi Temp
        delhi = df[df['city'] == 'Delhi'][[group_col, 'temperature_2m']].rename(
            columns={'temperature_2m': 'delhi_temperature'}
        )
        
        # Chennai Wind
        # Assuming 'wind_speed_10m' exists
        chennai_col = 'wind_speed_10m' if 'wind_speed_10m' in df.columns else 'wind_speed'
        chennai = df[df['city'] == 'Chennai'][[group_col, chennai_col]].rename(
            columns={chennai_col: 'chennai_wind_speed'}
        )
        
        # Merge
        # Rename group_col to delivery_start_ist for consistency if needed
        if group_col != 'delivery_start_ist':
             national = national.rename(columns={group_col: 'delivery_start_ist'})
             delhi = delhi.rename(columns={group_col: 'delivery_start_ist'})
             chennai = chennai.rename(columns={group_col: 'delivery_start_ist'})
        
        final_df = national.merge(delhi, on='delivery_start_ist', how='left').merge(chennai, on='delivery_start_ist', how='left')
        
        # Ensure delivery_start_ist is IST-aware
        # If it was 'timestamp' which might be UTC, check.
        # Cleaned/weather data usually UTC or IST. 
        # Inspection showed '00:00:00+05:30' for price. Weather likely same.
        # But if not, localize.
        
        # Safe localize
        if final_df['delivery_start_ist'].dt.tz is None:
             final_df['delivery_start_ist'] = final_df['delivery_start_ist'].dt.tz_localize('Asia/Kolkata', ambiguous='infer')
        else:
             final_df['delivery_start_ist'] = final_df['delivery_start_ist'].dt.tz_convert('Asia/Kolkata')
        
        return final_df

    def _load_holidays(self):
        """Load holidays CSV."""
        path = self.root_dir / self.config['data']['holiday_file']
        if not path.exists():
            print(f"Warning: Holiday file not found at {path}. Returning empty DataFrame.")
            return pd.DataFrame(columns=['date', 'holiday_name'])
            
        try:
            df = pd.read_csv(path)
            # Parse date
            # Assuming 'Date' column exists
            date_col = 'Date' if 'Date' in df.columns else 'date'
            if date_col in df.columns:
                df['date'] = pd.to_datetime(df[date_col], dayfirst=True).dt.date # Indian format often DD/MM/YYYY
                # Rename to standard
                if date_col != 'date':
                    df = df.rename(columns={date_col: 'date'})
            return df
        except Exception as e:
             print(f"Error loading holidays: {e}")
             return pd.DataFrame(columns=['date', 'holiday_name'])

    def _print_summary(self, data):
        print("\n=== Data Load Summary ===")
        for name, df in data.items():
            if name == 'holidays': # it's a list or df
                print(f"{name}: {len(df)} rows")
                continue
                
            print(f"{name}: {df.shape}")
            if 'delivery_start_ist' in df.columns:
                 start = df['delivery_start_ist'].min()
                 end = df['delivery_start_ist'].max()
                 print(f"  Range: {start} -> {end}")
            
            # Null counts
            nulls = df.isnull().sum().sum()
            if nulls > 0:
                print(f"  Nulls: {nulls}")
