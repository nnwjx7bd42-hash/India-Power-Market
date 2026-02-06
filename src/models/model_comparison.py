"""
Model comparison utilities
Compare baseline vs enhanced model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple


def compare_metrics(baseline_results, enhanced_results):
    """
    Compare baseline and enhanced model metrics
    
    Parameters:
    -----------
    baseline_results : pd.DataFrame
        Baseline CV results (from baseline_cv_results.csv)
    enhanced_results : pd.DataFrame
        Enhanced CV results
    
    Returns:
    --------
    pd.DataFrame
        Comparison table with improvements
    """
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']
    
    comparison = []
    for metric in metrics:
        baseline_mean = baseline_results[metric].mean()
        enhanced_mean = enhanced_results[metric].mean()
        
        # Improvement (positive = better for most metrics)
        if metric in ['RMSE', 'MAE', 'MAPE']:
            # Lower is better
            improvement_pct = ((baseline_mean - enhanced_mean) / baseline_mean) * 100
        else:
            # Higher is better (R2, Directional_Accuracy)
            improvement_pct = ((enhanced_mean - baseline_mean) / baseline_mean) * 100
        
        # Statistical significance (paired t-test)
        baseline_values = baseline_results[metric].values
        enhanced_values = enhanced_results[metric].values
        
        if len(baseline_values) == len(enhanced_values):
            t_stat, p_value = stats.ttest_rel(baseline_values, enhanced_values)
            significant = p_value < 0.05
        else:
            t_stat, p_value = np.nan, np.nan
            significant = False
        
        comparison.append({
            'Metric': metric,
            'Baseline_Mean': baseline_mean,
            'Enhanced_Mean': enhanced_mean,
            'Improvement_%': improvement_pct,
            'P_Value': p_value,
            'Significant': significant
        })
    
    return pd.DataFrame(comparison)


def compare_by_hour(baseline_pred, enhanced_pred, df_with_hour):
    """
    Compare model performance by hour of day
    
    Parameters:
    -----------
    baseline_pred : pd.DataFrame
        Baseline predictions with 'y_true', 'y_pred', 'timestamp'
    enhanced_pred : pd.DataFrame
        Enhanced predictions
    df_with_hour : pd.DataFrame
        DataFrame with Hour column
    
    Returns:
    --------
    pd.DataFrame
        Comparison by hour
    """
    # Merge with hour information
    baseline_with_hour = baseline_pred.merge(
        df_with_hour[['Hour']],
        left_on='timestamp',
        right_index=True,
        how='left'
    )
    enhanced_with_hour = enhanced_pred.merge(
        df_with_hour[['Hour']],
        left_on='timestamp',
        right_index=True,
        how='left'
    )
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from validation.metrics import calculate_metrics_by_group
    
    baseline_hourly = calculate_metrics_by_group(
        baseline_with_hour['y_true'],
        baseline_with_hour['y_pred'],
        baseline_with_hour['Hour']
    )
    enhanced_hourly = calculate_metrics_by_group(
        enhanced_with_hour['y_true'],
        enhanced_with_hour['y_pred'],
        enhanced_with_hour['Hour']
    )
    
    comparison = baseline_hourly.merge(
        enhanced_hourly,
        on='group',
        suffixes=('_baseline', '_enhanced')
    )
    
    comparison['MAPE_Improvement'] = (
        (comparison['MAPE_baseline'] - comparison['MAPE_enhanced']) 
        / comparison['MAPE_baseline'] * 100
    )
    
    return comparison


def compare_by_price_regime(baseline_pred, enhanced_pred, threshold=10000):
    """
    Compare model performance by price regime
    
    Parameters:
    -----------
    baseline_pred : pd.DataFrame
        Baseline predictions
    enhanced_pred : pd.DataFrame
        Enhanced predictions
    threshold : float
        Price threshold for regime split
    
    Returns:
    --------
    pd.DataFrame
        Comparison by regime
    """
    baseline_pred = baseline_pred.copy()
    enhanced_pred = enhanced_pred.copy()
    
    baseline_pred['regime'] = baseline_pred['y_true'].apply(
        lambda x: 'Scarcity' if x >= threshold else 'Normal'
    )
    enhanced_pred['regime'] = enhanced_pred['y_true'].apply(
        lambda x: 'Scarcity' if x >= threshold else 'Normal'
    )
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from validation.metrics import calculate_metrics_by_group
    
    baseline_regime = calculate_metrics_by_group(
        baseline_pred['y_true'],
        baseline_pred['y_pred'],
        baseline_pred['regime']
    )
    enhanced_regime = calculate_metrics_by_group(
        enhanced_pred['y_true'],
        enhanced_pred['y_pred'],
        enhanced_pred['regime']
    )
    
    comparison = baseline_regime.merge(
        enhanced_regime,
        on='group',
        suffixes=('_baseline', '_enhanced')
    )
    
    return comparison


def plot_comparison(baseline_results, enhanced_results, output_dir='results/comparison_plots'):
    """
    Generate comparison plots
    
    Parameters:
    -----------
    baseline_results : pd.DataFrame
        Baseline CV results
    enhanced_results : pd.DataFrame
        Enhanced CV results
    output_dir : str
        Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']
    
    # Comparison bar plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        baseline_vals = baseline_results[metric].values
        enhanced_vals = enhanced_results[metric].values
        n_folds = min(len(baseline_vals), len(enhanced_vals))
        baseline_vals = baseline_vals[:n_folds]
        enhanced_vals = enhanced_vals[:n_folds]
        
        x_pos = np.arange(n_folds)
        width = 0.35
        
        ax.bar(x_pos - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
        ax.bar(x_pos + width/2, enhanced_vals, width, label='Enhanced', alpha=0.8)
        
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Fold {j+1}' for j in range(n_folds)])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(output_path / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Box plot comparison
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        n_folds = min(len(baseline_results[metric]), len(enhanced_results[metric]))
        data = pd.DataFrame({
            'Baseline': baseline_results[metric].values[:n_folds],
            'Enhanced': enhanced_results[metric].values[:n_folds]
        })
        data.boxplot(ax=ax)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'metrics_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {output_dir}")


def generate_comparison_report(baseline_results, enhanced_results, 
                               baseline_importance, enhanced_importance,
                               output_path='results/model_comparison_report.md'):
    """
    Generate comprehensive comparison report
    
    Parameters:
    -----------
    baseline_results : pd.DataFrame
        Baseline CV results
    enhanced_results : pd.DataFrame
        Enhanced CV results
    baseline_importance : pd.DataFrame
        Baseline feature importance
    enhanced_importance : pd.DataFrame
        Enhanced feature importance
    output_path : str
        Path to save report
    """
    comparison = compare_metrics(baseline_results, enhanced_results)
    
    report = f"""# Model Comparison Report
## Baseline vs Enhanced Model

**Date:** Generated after enhanced model training

---

## Executive Summary

### Overall Performance Comparison

| Metric | Baseline | Enhanced | Improvement | Significant |
|--------|----------|----------|-------------|-------------|
"""
    
    for _, row in comparison.iterrows():
        sig = "✓" if row['Significant'] else "✗"
        report += f"| {row['Metric']} | {row['Baseline_Mean']:.4f} | {row['Enhanced_Mean']:.4f} | {row['Improvement_%']:.2f}% | {sig} |\n"
    
    report += f"""
---

## Detailed Analysis

### Performance Metrics

"""
    
    for _, row in comparison.iterrows():
        report += f"""
#### {row['Metric']}
- **Baseline:** {row['Baseline_Mean']:.4f}
- **Enhanced:** {row['Enhanced_Mean']:.4f}
- **Improvement:** {row['Improvement_%']:.2f}%
- **P-value:** {row['P_Value']:.4f} {'(Significant)' if row['Significant'] else '(Not Significant)'}
"""
    
    report += f"""
---

## Feature Importance Comparison

### Top 10 Baseline Features
"""
    
    for i, row in baseline_importance.head(10).iterrows():
        report += f"{i+1}. {row['feature']}: {row['importance']:.2f}\n"
    
    report += f"\n### Top 10 Enhanced Features\n"
    
    for i, row in enhanced_importance.head(10).iterrows():
        report += f"{i+1}. {row['feature']}: {row['importance']:.2f}\n"
    
    report += f"""
---

## Conclusions

"""
    
    # Determine if enhanced is better
    mape_improved = comparison[comparison['Metric'] == 'MAPE']['Improvement_%'].values[0] > 0
    rmse_improved = comparison[comparison['Metric'] == 'RMSE']['Improvement_%'].values[0] > 0
    
    if mape_improved and rmse_improved:
        report += "✅ **Enhanced model shows improvement over baseline**\n"
        report += f"- MAPE improved by {comparison[comparison['Metric'] == 'MAPE']['Improvement_%'].values[0]:.2f}%\n"
        report += f"- RMSE improved by {comparison[comparison['Metric'] == 'RMSE']['Improvement_%'].values[0]:.2f}%\n"
    else:
        report += "⚠️ **Enhanced model performance needs review**\n"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Comparison report saved to {output_path}")
