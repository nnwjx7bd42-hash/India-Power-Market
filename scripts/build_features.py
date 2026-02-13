import argparse
import sys
import os
import yaml
import pandas as pd
from pathlib import Path
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.features.pipeline import build_all_features

def main():
    parser = argparse.ArgumentParser(description="Build feature engineering pipeline")
    parser.add_argument('--config', default='config/backtest_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)
        
    start_time = time.time()
    
    try:
        build_all_features(config_path)
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    end_time = time.time()
    print(f"\nTime Elapsed: {end_time - start_time:.2f} seconds")
    
    # Print Summary
    print("\n=== FINAL SUMMARY ===")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    features_dir = Path(config_path).parent.parent / config['data']['features_dir']
    
    for market in config['markets']:
        print(f"\n{market.upper()} Features:")
        for split in ['train', 'val', 'backtest']:
            fpath = features_dir / f"{market}_features_{split}.parquet"
            if fpath.exists():
                df = pd.read_parquet(fpath)
                print(f"  {split.upper()}: {df.shape} rows x cols")
                print(f"    Date Range: {df['target_date'].min()} -> {df['target_date'].max()}")
                
                # Check NaNs
                nulls = df.isnull().sum().sum()
                if nulls > 0:
                     print(f"    WARNING: {nulls} NaN values found!")
                else:
                     print(f"    NaNs: 0 (OK)")
                
                # Target Stats
                tgt = df['target_mcp_rs_mwh']
                print(f"    Target (MCP): Min {tgt.min():.2f}, Median {tgt.median():.2f}, Max {tgt.max():.2f}")
            else:
                print(f"  {split.upper()}: File not found!")

if __name__ == "__main__":
    main()
