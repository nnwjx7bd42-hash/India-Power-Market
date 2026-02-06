#!/usr/bin/env python3
"""
Main training script for enhanced model
Trains enhanced model with all features in dataset (64 or curated 19), quantile regression, and compares with baseline
"""

import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.enhanced import EnhancedModel, get_enhanced_features
from models.quantile_regression import QuantileModel
from models.model_comparison import (
    compare_metrics, compare_by_hour, compare_by_price_regime,
    plot_comparison, generate_comparison_report
)
from validation.temporal_cv import TemporalCV
from validation.metrics import calculate_metrics, calculate_metrics_by_group
from validation.hyperparameter_tuning import tune_with_baseline_comparison
from validation.diagnostics import ResidualDiagnostics


def load_config(config_path='config/model_config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_enhanced_model(dataset_path='data/processed/dataset_cleaned.parquet', no_tune=False):
    """Main training pipeline for enhanced model"""
    print("="*80)
    print("ENHANCED MODEL TRAINING")
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
    
    # Select enhanced features (all non-target columns in dataset)
    print("\n3. Selecting enhanced features...")
    enhanced_features = get_enhanced_features(df)
    print(f"   Selected {len(enhanced_features)} features")
    print(f"   Feature categories:")
    print(f"     Price/Load lags: {len([f for f in enhanced_features if 'P(' in f or 'L(' in f or 'P_T' in f or 'L_T' in f or 'Price_' in f or 'Load_' in f])}")
    print(f"     Weather: {len([f for f in enhanced_features if 'national' in f or 'CDH' in f or 'HDH' in f or 'Temp' in f or 'Humidity' in f])}")
    print(f"     Calendar: {len([f for f in enhanced_features if 'Hour' in f or 'Day' in f or 'Month' in f])}")
    print(f"     Supply-side: {len([f for f in enhanced_features if 'RE' in f or 'Net_Load' in f or 'Thermal' in f or 'Hydro' in f or 'Gas' in f or 'Nuclear' in f or 'Wind' in f or 'Solar' in f or 'Demand' in f])}")
    print(f"     Interactions: {len([f for f in enhanced_features if '_x_' in f])}")
    
    # Prepare features and target
    X = df[enhanced_features].copy()
    y = df['P(T)'].copy()
    
    # Temporal cross-validation
    print("\n4. Setting up temporal cross-validation...")
    cv = TemporalCV(
        df,
        n_splits=config['cv'].get('n_splits'),
        gap_hours=config['cv']['gap_hours'],
        test_months=config['cv']['test_months']
    )
    
    # Hyperparameter tuning (optional - skip with --no-tune to use same params as baseline)
    print("\n5. Hyperparameter tuning...")
    use_tuning = not no_tune
    if use_tuning:
        baseline_params = config['baseline'].copy()
        # Remove non-param keys
        baseline_params = {k: v for k, v in baseline_params.items() 
                          if k not in ['model_type', 'eval_metric']}
        
        best_params = tune_with_baseline_comparison(
            X, y, list(cv),
            baseline_params,
            n_iter=20  # Reduced for faster training
        )
        enhanced_config = best_params
    else:
        # Use baseline hyperparameters (same as baseline model)
        enhanced_config = config['baseline'].copy()
        enhanced_config = {k: v for k, v in enhanced_config.items() 
                          if k not in ['model_type', 'eval_metric']}
        print("   Using baseline hyperparameters (--no-tune)")
    
    # Store results
    cv_results = []
    all_predictions = []
    feature_importances = []
    
    # Train and evaluate for each fold
    print("\n6. Training enhanced models for each fold...")
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
        
        # Create validation set
        val_size = int(len(X_train) * 0.2)
        X_train_fit = X_train.iloc[:-val_size]
        y_train_fit = y_train.iloc[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]
        
        # Train enhanced model
        model = EnhancedModel()
        # Merge enhanced config with defaults
        default_config = config['enhanced'].copy()
        default_config.update(enhanced_config)
        model.config = default_config
        model.train(X_train_fit, y_train_fit, X_val, y_val)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test.values, y_pred)
        metrics['fold'] = fold_info['fold']
        metrics['train_size'] = fold_info['train_size']
        metrics['test_size'] = fold_info['test_size']
        
        cv_results.append(metrics)
        
        print(f"  RMSE: ₹{metrics['RMSE']:,.2f}")
        print(f"  MAE:  ₹{metrics['MAE']:,.2f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²:   {metrics['R2']:.4f}")
        print(f"  Dir. Acc: {metrics['Directional_Accuracy']:.2f}%")
        
        # Store predictions
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
            model.save('models/training/enhanced_model.pkl')
            print(f"  Model saved to models/training/enhanced_model.pkl")
    
    # Aggregate results
    print("\n" + "="*80)
    print("ENHANCED MODEL CV RESULTS SUMMARY")
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
    mean_r2 = results_df['R2'].mean()
    mean_dir_acc = results_df['Directional_Accuracy'].mean()
    
    criteria = {
        'MAPE < 10%': mean_mape < 10,
        'RMSE < ₹1,000': mean_rmse < 1000,
        'R² > 0.90': mean_r2 > 0.90,
        'Directional Accuracy > 75%': mean_dir_acc > 75
    }
    
    for criterion, passed in criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {criterion}: {status}")
    
    # Feature importance (average across folds)
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE (Average across folds)")
    print("="*80)
    
    all_importance = pd.concat(feature_importances)
    avg_importance = all_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    # Map feature indices to names if needed
    feature_map = {f'f{i}': feat for i, feat in enumerate(enhanced_features, 0)}
    avg_importance_mapped = avg_importance.copy()
    avg_importance_mapped.index = avg_importance_mapped.index.map(
        lambda x: feature_map.get(x, x)
    )
    avg_importance_mapped = avg_importance_mapped.sort_values(ascending=False)
    
    print("\nTop 15 Most Important Features:")
    for i, (feature, importance) in enumerate(avg_importance_mapped.head(15).items(), 1):
        print(f"  {i:2d}. {feature:30s}: {importance:12.2f}")
    
    # Update for saving
    avg_importance = avg_importance_mapped
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    Path('results/training').mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv('results/training/enhanced_cv_results.csv', index=False)
    print("  ✓ Saved: results/training/enhanced_cv_results.csv")
    
    summary_df.to_csv('results/training/enhanced_summary.csv', index=False)
    print("  ✓ Saved: results/training/enhanced_summary.csv")
    
    avg_importance_df = pd.DataFrame({
        'feature': avg_importance.index,
        'importance': avg_importance.values
    })
    avg_importance_df.to_csv('results/training/enhanced_feature_importance.csv', index=False)
    print("  ✓ Saved: results/training/enhanced_feature_importance.csv")
    
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_predictions_df.to_parquet('results/training/enhanced_predictions.parquet', index=False)
    print("  ✓ Saved: results/training/enhanced_predictions.parquet")
    
    return results_df, avg_importance_df, all_predictions_df


def train_quantile_models(df, enhanced_features):
    """Train quantile regression models"""
    print("\n" + "="*80)
    print("QUANTILE REGRESSION TRAINING")
    print("="*80)
    
    config = load_config()
    
    X = df[enhanced_features].copy()
    y = df['P(T)'].copy()
    
    # Use last fold for quantile training (most data)
    cv = TemporalCV(df, n_splits=config['cv'].get('n_splits'),
                   gap_hours=config['cv']['gap_hours'],
                   test_months=config['cv']['test_months'])
    
    # Get last fold
    train_indices, test_indices = list(cv)[-1]
    
    X_train = X.loc[train_indices]
    y_train = y.loc[train_indices]
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]
    
    val_size = int(len(X_train) * 0.2)
    X_train_fit = X_train.iloc[:-val_size]
    y_train_fit = y_train.iloc[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    
    # Train quantile model
    print("\nTraining quantile models (P10, P50, P90)...")
    quantile_config = config['enhanced'].copy()
    quantile_config = {k: v for k, v in quantile_config.items() 
                      if k not in ['model_type', 'eval_metric']}
    
    quantile_model = QuantileModel(
        quantiles=config['quantile']['quantiles'],
        config=quantile_config
    )
    quantile_model.train(X_train_fit, y_train_fit, X_val, y_val)
    
    # Evaluate
    print("\nEvaluating quantile models...")
    quantile_metrics = quantile_model.evaluate(X_test, y_test)
    
    print("\nQuantile Model Performance:")
    for metric, value in quantile_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Make predictions
    quantile_predictions = quantile_model.predict_intervals(X_test)
    
    # Create prediction dataframe
    quantile_df = pd.DataFrame({
        'timestamp': test_indices,
        'y_true': y_test.values,
        'p10': quantile_predictions[0.1],
        'p50': quantile_predictions[0.5],
        'p90': quantile_predictions[0.9]
    })
    
    # Save
    quantile_model.save('models/training/quantile_models.pkl')
    quantile_df.to_parquet('results/training/quantile_predictions.parquet', index=False)
    print("\n  ✓ Quantile models saved")
    print("  ✓ Predictions saved to results/training/quantile_predictions.parquet")
    
    return quantile_model, quantile_df, quantile_metrics


def compare_with_baseline():
    """Compare enhanced model with baseline"""
    print("\n" + "="*80)
    print("MODEL COMPARISON: BASELINE vs ENHANCED")
    print("="*80)
    
    # Load results
    baseline_results = pd.read_csv('results/training/baseline_cv_results.csv')
    enhanced_results = pd.read_csv('results/training/enhanced_cv_results.csv')
    
    baseline_importance = pd.read_csv('results/training/baseline_feature_importance.csv')
    enhanced_importance = pd.read_csv('results/training/enhanced_feature_importance.csv')
    
    baseline_predictions = pd.read_parquet('results/training/baseline_predictions.parquet')
    enhanced_predictions = pd.read_parquet('results/training/enhanced_predictions.parquet')
    
    # Compare metrics
    comparison = compare_metrics(baseline_results, enhanced_results)
    
    print("\nMetric Comparison:")
    print(comparison.to_string(index=False))
    
    # Save comparison
    comparison.to_csv('results/training/model_comparison.csv', index=False)
    print("\n  ✓ Saved: results/training/model_comparison.csv")
    
    # Generate plots
    plot_comparison(baseline_results, enhanced_results, output_dir='results/training/comparison_plots')
    
    # Generate report
    generate_comparison_report(
        baseline_results, enhanced_results,
        baseline_importance, enhanced_importance,
        output_path='results/training/model_comparison_report.md'
    )
    
    # Compare by hour
    df = pd.read_parquet('data/processed/dataset_cleaned.parquet')
    hourly_comparison = compare_by_hour(
        baseline_predictions,
        enhanced_predictions,
        df[['Hour']]
    )
    hourly_comparison.to_csv('results/training/comparison_by_hour.csv', index=False)
    print("  ✓ Saved: results/training/comparison_by_hour.csv")
    
    # Compare by price regime
    regime_comparison = compare_by_price_regime(
        baseline_predictions,
        enhanced_predictions
    )
    regime_comparison.to_csv('results/training/comparison_by_regime.csv', index=False)
    print("  ✓ Saved: results/training/comparison_by_regime.csv")
    
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train enhanced model')
    parser.add_argument('--dataset', default='data/processed/dataset_cleaned.parquet',
                        help='Path to parquet dataset (default: data/processed/dataset_cleaned.parquet)')
    parser.add_argument('--no-tune', action='store_true',
                        help='Use baseline hyperparameters (no tuning); same params as baseline for fair comparison')
    args = parser.parse_args()
    # Train enhanced model
    enhanced_results, enhanced_importance, enhanced_predictions = train_enhanced_model(
        dataset_path=args.dataset, no_tune=args.no_tune
    )
    
    # Train quantile models (use same holdout as main training)
    config = load_config()
    df = pd.read_parquet(args.dataset)
    holdout_hours = config['cv'].get('holdout_hours', 0)
    if holdout_hours and holdout_hours > 0:
        holdout_start = df.index.max() - pd.Timedelta(hours=holdout_hours)
        df = df[df.index <= holdout_start]
    enhanced_features = get_enhanced_features(df)
    quantile_model, quantile_predictions, quantile_metrics = train_quantile_models(
        df, enhanced_features
    )
    
    # Compare with baseline
    comparison = compare_with_baseline()
    
    print("\n" + "="*80)
    print("ENHANCED MODEL TRAINING COMPLETE")
    print("="*80)
    print("\nAll models trained and compared successfully!")
