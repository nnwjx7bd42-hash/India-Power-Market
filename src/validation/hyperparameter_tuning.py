"""
Hyperparameter tuning for time series models
Uses temporal cross-validation with random search or Bayesian optimization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform, randint
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def get_param_distributions():
    """
    Define hyperparameter search space
    
    Returns:
    --------
    dict
        Parameter distributions for random search
    """
    return {
        'max_depth': randint(4, 9),  # 4 to 8
        'learning_rate': uniform(0.01, 0.09),  # 0.01 to 0.1
        'n_estimators': randint(300, 701),  # 300 to 700
        'subsample': uniform(0.7, 0.2),  # 0.7 to 0.9
        'colsample_bytree': uniform(0.7, 0.2),  # 0.7 to 0.9
        'min_child_weight': randint(1, 6),  # 1 to 5
        'gamma': uniform(0, 0.2),  # 0 to 0.2
        'reg_alpha': uniform(0, 1),  # L1 regularization
        'reg_lambda': uniform(0, 2)  # L2 regularization
    }


def evaluate_params(X_train, y_train, X_val, y_val, params, metric='mape'):
    """
    Evaluate hyperparameters on validation set
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target
    params : dict
        Hyperparameters to evaluate
    metric : str
        Metric to optimize ('mape', 'rmse', 'mae')
    
    Returns:
    --------
    float
        Metric value (lower is better)
    """
    # Prepare parameters
    model_params = {
        'objective': 'reg:squarederror',
        'random_state': 42,
        'eval_metric': 'rmse',
        **params
    }
    
    # Create and train model
    model = xgb.XGBRegressor(**model_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate metric
    if metric == 'mape':
        from .metrics import calculate_mape
        return calculate_mape(y_val.values, y_pred)
    elif metric == 'rmse':
        from .metrics import calculate_rmse
        return calculate_rmse(y_val.values, y_pred)
    elif metric == 'mae':
        from .metrics import calculate_mae
        return calculate_mae(y_val.values, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def tune_hyperparameters(X, y, cv_splits, n_iter=30, metric='mape', n_jobs=1):
    """
    Tune hyperparameters using temporal cross-validation
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    cv_splits : list
        List of (train_indices, test_indices) tuples
    n_iter : int
        Number of random search iterations
    metric : str
        Metric to optimize
    n_jobs : int
        Number of parallel jobs
    
    Returns:
    --------
    dict
        Best hyperparameters
    """
    print("="*80)
    print("HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Using {len(cv_splits)} CV folds for tuning")
    print(f"Random search iterations: {n_iter}")
    print(f"Optimizing: {metric.upper()}")
    
    # Use first 3 folds for tuning (faster)
    tuning_folds = cv_splits[:min(3, len(cv_splits))]
    
    param_distributions = get_param_distributions()
    results = []
    
    # Random search
    param_samples = list(ParameterSampler(
        param_distributions,
        n_iter=n_iter,
        random_state=42
    ))
    
    print(f"\nEvaluating {len(param_samples)} parameter combinations...")
    
    for i, params in enumerate(param_samples):
        # Convert to proper types
        params = {
            'max_depth': int(params['max_depth']),
            'learning_rate': float(params['learning_rate']),
            'n_estimators': int(params['n_estimators']),
            'subsample': float(params['subsample']),
            'colsample_bytree': float(params['colsample_bytree']),
            'min_child_weight': int(params['min_child_weight']),
            'gamma': float(params['gamma']),
            'reg_alpha': float(params['reg_alpha']),
            'reg_lambda': float(params['reg_lambda'])
        }
        
        # Evaluate on tuning folds
        fold_scores = []
        for train_idx, val_idx in tuning_folds:
            X_train_fold = X.loc[train_idx]
            y_train_fold = y.loc[train_idx]
            X_val_fold = X.loc[val_idx]
            y_val_fold = y.loc[val_idx]
            
            # Create validation split from training data
            val_size = int(len(X_train_fold) * 0.2)
            X_train_fit = X_train_fold.iloc[:-val_size]
            y_train_fit = y_train_fold.iloc[:-val_size]
            X_val_fit = X_train_fold.iloc[-val_size:]
            y_val_fit = y_train_fold.iloc[-val_size:]
            
            try:
                score = evaluate_params(
                    X_train_fit, y_train_fit,
                    X_val_fit, y_val_fit,
                    params, metric
                )
                fold_scores.append(score)
            except Exception as e:
                print(f"  Warning: Error evaluating params {i+1}: {e}")
                fold_scores.append(np.inf)
        
        avg_score = np.mean(fold_scores)
        results.append({
            'params': params,
            'score': avg_score,
            'fold_scores': fold_scores
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{len(param_samples)} iterations")
    
    # Find best parameters
    best_result = min(results, key=lambda x: x['score'])
    best_params = best_result['params']
    best_score = best_result['score']
    
    print(f"\n✓ Tuning complete")
    print(f"  Best {metric.upper()}: {best_score:.4f}")
    print(f"\n  Best parameters:")
    for key, value in sorted(best_params.items()):
        print(f"    {key}: {value}")
    
    return best_params


def tune_with_baseline_comparison(X, y, cv_splits, baseline_params, n_iter=30):
    """
    Tune hyperparameters and compare with baseline
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    cv_splits : list
        CV splits
    baseline_params : dict
        Baseline model parameters
    n_iter : int
        Number of iterations
    
    Returns:
    --------
    dict
        Best parameters and comparison
    """
    # Evaluate baseline params
    print("\nEvaluating baseline parameters...")
    baseline_score = evaluate_params_on_cv(X, y, cv_splits[:3], baseline_params)
    print(f"  Baseline MAPE: {baseline_score:.4f}")
    
    # Tune
    best_params = tune_hyperparameters(X, y, cv_splits, n_iter=n_iter)
    
    # Evaluate best params
    print("\nEvaluating tuned parameters...")
    tuned_score = evaluate_params_on_cv(X, y, cv_splits[:3], best_params)
    print(f"  Tuned MAPE: {tuned_score:.4f}")
    
    improvement = ((baseline_score - tuned_score) / baseline_score) * 100
    print(f"  Improvement: {improvement:.2f}%")
    
    if tuned_score < baseline_score:
        print("  ✓ Tuned parameters are better")
        return best_params
    else:
        print("  ⚠️  Baseline parameters are better, using baseline")
        return baseline_params


def evaluate_params_on_cv(X, y, cv_splits, params):
    """Evaluate parameters on CV folds"""
    scores = []
    for train_idx, val_idx in cv_splits:
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_val = X.loc[val_idx]
        y_val = y.loc[val_idx]
        
        val_size = int(len(X_train) * 0.2)
        X_train_fit = X_train.iloc[:-val_size]
        y_train_fit = y_train.iloc[:-val_size]
        X_val_fit = X_train.iloc[-val_size:]
        y_val_fit = y_train.iloc[-val_size:]
        
        score = evaluate_params(
            X_train_fit, y_train_fit,
            X_val_fit, y_val_fit,
            params, 'mape'
        )
        scores.append(score)
    
    return np.mean(scores)
