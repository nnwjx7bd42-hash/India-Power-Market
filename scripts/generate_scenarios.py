import argparse
import sys
import yaml
import pandas as pd
import numpy as np
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.scenarios.dam_copula import DAMCopulaGenerator
from src.scenarios.rtm_rollout import RTMRolloutGenerator
from src.models.quantile_lgbm import QuantileLGBM
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_scenarios(config_path):
    start_time = time.time()
    config = load_config(config_path)
    
    predictions_dir = Path(config['paths']['predictions_dir'])
    models_dir = Path(config['paths']['models_dir'])
    copula_dir = models_dir / "copula"
    copula_dir.mkdir(exist_ok=True)
    
    # === DAM Scenarios (Copula) ===
    print("\n=== Generating DAM Scenarios (Copula) ===")
    
    # 1. Fit Copula
    print("Fitting Copula...")
    dam_train_feats = pd.read_parquet("Data/Features/dam_features_train.parquet")
    
    # Load q50 model
    with open(models_dir / "dam" / "best_params.json") as f:
        dam_params = json.load(f)
    dam_q50 = QuantileLGBM.load(str(models_dir / "dam" / "q50.txt"), alpha=0.50) # Load trained model
    
    # Identify features
    with open(models_dir / "dam" / "feature_columns.json") as f:
        dam_cols = json.load(f)
    
    X_train = dam_train_feats[dam_cols]
    y_train = dam_train_feats['target_mcp_rs_mwh'].values
    
    # Compute Log Residuals
    y_pred_log = dam_q50.predict_log(X_train)
    y_true_log = np.log1p(y_train)
    residuals = y_true_log - y_pred_log
    
    # Reshape (n_days, 24)
    # Ensure sorted by date and hour
    dam_train_feats['resid'] = residuals
    # Check completeness
    # Group by date
    pivoted = dam_train_feats.pivot(index='target_date', columns='target_hour', values='resid')
    # Drop incomplete days if any
    pivoted = pivoted.dropna()
    resid_matrix = pivoted.values
    
    copula = DAMCopulaGenerator(seed=config['scenarios']['dam']['seed'])
    copula.fit(resid_matrix)
    copula.save(str(copula_dir / "dam_correlation_24x24.npy"))
    
    print("Copula fitted.")
    print(f"Correlation Matrix shape: {copula.correlation_matrix.shape}")
    
    # 2. Generate Scenarios for Backtest
    print("Generating paths...")
    dam_preds = pd.read_parquet(predictions_dir / "dam_quantiles_backtest.parquet")
    
    # Group by date
    unique_dates = dam_preds['target_date'].unique()
    all_scenarios = []
    
    n_scenarios = config['scenarios']['dam']['n_scenarios']
    
    for date in unique_dates:
        # Get quantiles for this date (24 hours)
        day_preds = dam_preds[dam_preds['target_date'] == date].sort_values('target_hour')
        
        if len(day_preds) != 24:
            print(f"Warning: Date {date} has {len(day_preds)} rows. Skipping scenario gen.")
            continue
            
        q_preds = {}
        for q in config['quantiles']:
            col = f"q{int(q*100)}"
            q_preds[q] = day_preds[col].values
            
        # Generate (n_scenarios, 24)
        paths = copula.generate(q_preds, n_scenarios=n_scenarios)
        
        # Format output
        # Columns: target_date, scenario_id, h00..h23
        for s_idx in range(n_scenarios):
            row = {'target_date': date, 'scenario_id': s_idx}
            for h in range(24):
                row[f"h{h:02d}"] = paths[s_idx, h]
            all_scenarios.append(row)
            
    dam_scenarios_df = pd.DataFrame(all_scenarios)
    dam_scenarios_df.to_parquet(predictions_dir / "dam_scenarios_backtest.parquet")
    print(f"Saved DAM Scenarios: {dam_scenarios_df.shape}")

    # === RTM Scenarios (Rollout) ===
    print("\n=== Generating RTM Scenarios (Rollout) ===")
    
    # Load Models
    rtm_models = {}
    with open(models_dir / "rtm" / "best_params.json") as f:
        rtm_params = json.load(f)
        
    for q in config['quantiles']:
        rtm_models[q] = QuantileLGBM.load(str(models_dir / "rtm" / f"q{int(q*100)}.txt"), alpha=q)
        
    with open(models_dir / "rtm" / "feature_columns.json") as f:
        rtm_cols = json.load(f)
        
    # Load Backtest Features
    rtm_backtest_feats = pd.read_parquet("Data/Features/rtm_features_backtest.parquet")
    
    # We need to start rolling out from Hour 0 of each day.
    # Identify unique days.
    # Note: 'target_date' is string.
    unique_dates_rtm = rtm_backtest_feats['target_date'].unique()
    
    all_rtm_scenarios = []
    n_scenarios_rtm = config['scenarios']['rtm']['n_scenarios']
    
    generator = RTMRolloutGenerator(
        quantile_models=rtm_models,
        feature_columns=rtm_cols,
        updatable_features=config['scenarios']['rtm']['updatable_features'],
        seed=config['scenarios']['rtm']['seed']
    )
    
    for date in unique_dates_rtm:
        # Get feature row for Hour 0
        day_df = rtm_backtest_feats[rtm_backtest_feats['target_date'] == date]
        
        # Check if Hour 0 exists
        # RTM should be complete days now.
        if 0 not in day_df['target_hour'].values:
            print(f"Warning: RTM Date {date} missing Hour 0. Skipping.")
            continue
            
        start_row = day_df[day_df['target_hour'] == 0].iloc[0]
        
        # Generate
        # (n_scenarios, 24)
        paths = generator.generate(start_row, n_scenarios=n_scenarios_rtm, n_steps=24)
        
        # Format
        for s_idx in range(n_scenarios_rtm):
            row = {'target_date': date, 'scenario_id': s_idx}
            for h in range(24):
                row[f"h{h:02d}"] = paths[s_idx, h]
            all_rtm_scenarios.append(row)
            
    rtm_scenarios_df = pd.DataFrame(all_rtm_scenarios)
    rtm_scenarios_df.to_parquet(predictions_dir / "rtm_scenarios_backtest.parquet")
    print(f"Saved RTM Scenarios: {rtm_scenarios_df.shape}")
    
    print(f"\nScenario Generation Complete. Time: {time.time() - start_time:.2f}s")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/model_config.yaml')
    args = parser.parse_args()
    
    generate_scenarios(args.config)
