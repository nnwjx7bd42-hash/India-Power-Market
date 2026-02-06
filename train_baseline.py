#!/usr/bin/env python3
"""
Main training script for baseline model
End-to-end pipeline: load data, train with temporal CV, evaluate, save results
"""

import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.baseline import BaselineModel, get_baseline_features
from validation.temporal_cv import TemporalCV
from validation.metrics import calculate_metrics, calculate_metrics_by_group
from validation.diagnostics import ResidualDiagnostics


def load_config(config_path='config/model_config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_baseline_model(dataset_path='data/processed/dataset_cleaned.parquet'):
    """Main training pipeline. Baseline always uses the fixed baseline feature set (see get_baseline_features); extra columns in the dataset are ignored."""
    print("="*80)
    print("BASELINE MODEL TRAINING")
    print("="*80)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config()
    
    # Load dataset
    print("\n2. Loading dataset...")
    df = pd.read_parquet(dataset_path)
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # Optional holdout: reserve last N hours for inference test (not used in CV)
    holdout_hours = config['cv'].get('holdout_hours', 0)
    if holdout_hours and holdout_hours > 0:
        holdout_start = df.index.max() - pd.Timedelta(hours=holdout_hours)
        df = df[df.index <= holdout_start]
        print(f"   Holdout: last {holdout_hours} hours reserved (CV uses data up to {holdout_start})")
        print(f"   Shape after holdout: {df.shape}")
    
    # Baseline uses a fixed feature set: get_baseline_features(df) returns only baseline
    # feature names that exist in df; X = df[baseline_features] so extra columns are ignored.
    print("\n3. Selecting baseline features...")
    baseline_features = get_baseline_features(df)
    print(f"   Selected {len(baseline_features)} features:")
    for i, feat in enumerate(baseline_features, 1):
        print(f"     {i:2d}. {feat}")
    
    # Prepare features and target
    X = df[baseline_features].copy()
    y = df['P(T)'].copy()
    
    # Temporal cross-validation
    print("\n4. Setting up temporal cross-validation...")
    cv = TemporalCV(
        df,
        n_splits=config['cv'].get('n_splits'),
        gap_hours=config['cv']['gap_hours'],
        test_months=config['cv']['test_months']
    )
    cv.print_summary()
    
    # Store results
    cv_results = []
    all_predictions = []
    all_residuals = []
    feature_importances = []
    
    # Train and evaluate for each fold
    print("\n5. Training models for each fold...")
    print("="*80)
    
    for fold_idx, (train_indices, test_indices) in enumerate(cv):
        fold_info = cv.get_split_info(fold_idx)
        print(f"\nFold {fold_info['fold']}:")
        print(f"  Train: {fold_info['train_start'].date()} to {fold_info['train_end'].date()} ({fold_info['train_size']:,} samples)")
        print(f"  Test:  {fold_info['test_start'].date()} to {fold_info['test_end'].date()} ({fold_info['test_size']:,} samples)")
        
        # Split data
        X_train = X.loc[train_indices]
        y_train = y.loc[train_indices]
        X_test = X.loc[test_indices]
        y_test = y.loc[test_indices]
        
        # Create validation set (last 20% of training data)
        val_size = int(len(X_train) * 0.2)
        X_train_fit = X_train.iloc[:-val_size]
        y_train_fit = y_train.iloc[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]
        
        # Train model
        model = BaselineModel()
        model.train(X_train_fit, y_train_fit, X_val, y_val)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test.values, y_pred)
        metrics['fold'] = fold_info['fold']
        metrics['train_size'] = fold_info['train_size']
        metrics['test_size'] = fold_info['test_size']
        metrics['train_start'] = fold_info['train_start']
        metrics['train_end'] = fold_info['train_end']
        metrics['test_start'] = fold_info['test_start']
        metrics['test_end'] = fold_info['test_end']
        
        cv_results.append(metrics)
        
        print(f"  RMSE: ₹{metrics['RMSE']:,.2f}")
        print(f"  MAE:  ₹{metrics['MAE']:,.2f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²:   {metrics['R2']:.4f}")
        print(f"  Dir. Acc: {metrics['Directional_Accuracy']:.2f}%")
        
        # Store predictions and residuals
        fold_predictions = pd.DataFrame({
            'fold': fold_info['fold'],
            'timestamp': test_indices,
            'y_true': y_test.values,
            'y_pred': y_pred
        })
        all_predictions.append(fold_predictions)
        
        # Feature importance
        importance = model.get_feature_importance()
        importance['fold'] = fold_info['fold']
        feature_importances.append(importance)
        
        # Save model for last fold
        if fold_idx == len(cv) - 1:
            model.save('models/training/baseline_model.pkl')
            print(f"  Model saved to models/training/baseline_model.pkl")
    
    # Aggregate results
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(cv_results)
    
    # Calculate averages
    metric_cols = ['RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']
    summary = {
        'Metric': metric_cols,
        'Mean': [results_df[col].mean() for col in metric_cols],
        'Std': [results_df[col].std() for col in metric_cols],
        'Min': [results_df[col].min() for col in metric_cols],
        'Max': [results_df[col].max() for col in metric_cols]
    }
    summary_df = pd.DataFrame(summary)
    
    print("\nOverall Performance:")
    print(summary_df.to_string(index=False))
    
    # Check success criteria
    print("\n" + "="*80)
    print("SUCCESS CRITERIA CHECK")
    print("="*80)
    
    mean_mape = results_df['MAPE'].mean()
    mean_rmse = results_df['RMSE'].mean()
    mean_dir_acc = results_df['Directional_Accuracy'].mean()
    
    criteria = {
        'MAPE < 15%': mean_mape < 15,
        'RMSE < ₹2,000': mean_rmse < 2000,
        'Directional Accuracy > 60%': mean_dir_acc > 60
    }
    
    for criterion, passed in criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {criterion}: {status}")
        if criterion == 'MAPE < 15%':
            print(f"    Actual: {mean_mape:.2f}%")
        elif criterion == 'RMSE < ₹2,000':
            print(f"    Actual: ₹{mean_rmse:,.2f}")
        elif criterion == 'Directional Accuracy > 60%':
            print(f"    Actual: {mean_dir_acc:.2f}%")
    
    # Error analysis by hour (using last fold)
    print("\n" + "="*80)
    print("ERROR ANALYSIS BY HOUR (Last Fold)")
    print("="*80)
    
    last_fold_pred = all_predictions[-1]
    last_fold_pred = last_fold_pred.set_index('timestamp')
    last_fold_pred['Hour'] = last_fold_pred.index.hour
    
    hourly_metrics = calculate_metrics_by_group(
        last_fold_pred['y_true'],
        last_fold_pred['y_pred'],
        last_fold_pred['Hour']
    )
    hourly_metrics = hourly_metrics.sort_values('MAPE')
    
    print("\nWorst performing hours (by MAPE):")
    print(hourly_metrics.head(5)[['group', 'MAPE', 'RMSE', 'n_samples']].to_string(index=False))
    
    print("\nBest performing hours (by MAPE):")
    print(hourly_metrics.tail(5)[['group', 'MAPE', 'RMSE', 'n_samples']].to_string(index=False))
    
    # Evening peak check (18:00-20:00)
    evening_hours = hourly_metrics[hourly_metrics['group'].isin([18, 19, 20])]
    evening_mape = evening_hours['MAPE'].mean()
    print(f"\nEvening Peak (18:00-20:00) MAPE: {evening_mape:.2f}%")
    if evening_mape < 12:
        print("  ✓ PASS (target: < 12%)")
    else:
        print("  ✗ FAIL (target: < 12%)")
    
    # Feature importance (average across folds)
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE (Average across folds)")
    print("="*80)
    
    all_importance = pd.concat(feature_importances)
    avg_importance = all_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    # Map feature indices to names if needed
    feature_map = {f'f{i}': feat for i, feat in enumerate(baseline_features, 0)}
    avg_importance_mapped = avg_importance.copy()
    avg_importance_mapped.index = avg_importance_mapped.index.map(
        lambda x: feature_map.get(x, x)
    )
    avg_importance_mapped = avg_importance_mapped.sort_values(ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for i, (feature, importance) in enumerate(avg_importance_mapped.head(10).items(), 1):
        print(f"  {i:2d}. {feature:20s}: {importance:10.2f}")
    
    # Update feature importance dataframe with mapped names
    avg_importance_df = pd.DataFrame({
        'feature': avg_importance_mapped.index,
        'importance': avg_importance_mapped.values
    })
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    Path('results/training').mkdir(parents=True, exist_ok=True)
    
    # CV results
    results_df.to_csv('results/training/baseline_cv_results.csv', index=False)
    print("  ✓ Saved: results/training/baseline_cv_results.csv")
    
    # Summary
    summary_df.to_csv('results/training/baseline_summary.csv', index=False)
    print("  ✓ Saved: results/training/baseline_summary.csv")
    
    # Feature importance
    avg_importance_df = pd.DataFrame({
        'feature': avg_importance.index,
        'importance': avg_importance.values
    })
    avg_importance_df.to_csv('results/training/baseline_feature_importance.csv', index=False)
    print("  ✓ Saved: results/training/baseline_feature_importance.csv")
    
    # Predictions
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_predictions_df.to_parquet('results/training/baseline_predictions.parquet', index=False)
    print("  ✓ Saved: results/training/baseline_predictions.parquet")
    
    # Residuals for diagnostics
    all_predictions_df['residual'] = all_predictions_df['y_true'] - all_predictions_df['y_pred']
    all_predictions_df.to_parquet('results/training/baseline_residuals.parquet', index=False)
    print("  ✓ Saved: results/training/baseline_residuals.parquet")
    
    # Generate diagnostic plots (using last fold)
    print("\n6. Generating diagnostic plots...")
    last_fold_df = all_predictions_df[all_predictions_df['fold'] == len(cv)].copy()
    last_fold_df = last_fold_df.set_index('timestamp')
    
    # Merge with original dataframe for metadata
    last_fold_df = last_fold_df.join(df[['Hour', 'DayOfWeek', 'Month']], how='left')
    
    diagnostics = ResidualDiagnostics(
        last_fold_df['y_true'].values,
        last_fold_df['y_pred'].values,
        last_fold_df.index
    )
    # Set df with all metadata already included
    diagnostics.df = last_fold_df.copy()
    diagnostics.generate_all_plots(output_dir='results/training/diagnostics')
    print("  ✓ Diagnostic plots saved to results/training/diagnostics/")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    return results_df, avg_importance_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train baseline model (always uses fixed baseline feature set)')
    parser.add_argument('--dataset', default='data/processed/dataset_cleaned.parquet',
                        help='Path to parquet dataset (default: data/processed/dataset_cleaned.parquet)')
    args = parser.parse_args()
    results, importance = train_baseline_model(dataset_path=args.dataset)
