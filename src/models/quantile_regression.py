"""
Quantile regression models for prediction intervals
Trains separate models for P10, P50, P90 quantiles
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from typing import Dict, List, Optional


def pinball_loss(y_true, y_pred, quantile):
    """
    Calculate pinball loss for quantile regression
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted quantile values
    quantile : float
        Quantile level (0-1)
    
    Returns:
    --------
    float
        Pinball loss
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1) * error)
    return np.mean(loss)


def calculate_coverage(y_true, y_lower, y_upper):
    """
    Calculate coverage of prediction interval
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_lower : array-like
        Lower bound predictions
    y_upper : array-like
        Upper bound predictions
    
    Returns:
    --------
    float
        Coverage percentage (0-100)
    """
    y_true = np.array(y_true)
    y_lower = np.array(y_lower)
    y_upper = np.array(y_upper)
    
    covered = np.sum((y_true >= y_lower) & (y_true <= y_upper))
    return (covered / len(y_true)) * 100


class QuantileModel:
    """
    Quantile regression model for prediction intervals
    """
    
    def __init__(self, quantiles=[0.1, 0.5, 0.9], config=None):
        """
        Initialize quantile model
        
        Parameters:
        -----------
        quantiles : list
            List of quantiles to predict (default: [0.1, 0.5, 0.9])
        config : dict
            Model configuration (hyperparameters)
        """
        self.quantiles = quantiles
        self.config = config or {}
        self.models = {}
        self.feature_names = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train models for each quantile
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.array
            Training features
        y_train : pd.Series or np.array
            Training target
        X_val : pd.DataFrame or np.array, optional
            Validation features
        y_val : pd.Series or np.array, optional
            Validation target
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
        
        base_params = {
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.05),
            'n_estimators': self.config.get('n_estimators', 500),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'random_state': self.config.get('random_state', 42),
            'eval_metric': 'rmse'
        }
        
        for quantile in self.quantiles:
            print(f"  Training quantile {quantile} model...")
            
            # For P50, can use standard regression or quantile
            if quantile == 0.5:
                # Use standard regression for median
                params = {
                    **base_params,
                    'objective': 'reg:squarederror'
                }
            else:
                # Use quantile objective
                params = {
                    **base_params,
                    'objective': 'reg:quantileerror',
                    'quantile_alpha': quantile
                }
            
            model = xgb.XGBRegressor(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train, verbose=False)
            
            self.models[quantile] = model
    
    def predict_intervals(self, X):
        """
        Predict quantile intervals
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature matrix
        
        Returns:
        --------
        dict
            Dictionary of quantile predictions
        """
        predictions = {}
        for quantile, model in self.models.items():
            predictions[quantile] = model.predict(X)
        
        return predictions
    
    def predict(self, X):
        """
        Predict median (P50) as point estimate
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature matrix
        
        Returns:
        --------
        np.array
            Median predictions
        """
        if 0.5 in self.models:
            return self.models[0.5].predict(X)
        else:
            raise ValueError("P50 model not trained")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate quantile models
        
        Parameters:
        -----------
        X_test : pd.DataFrame or np.array
            Test features
        y_test : pd.Series or np.array
            Test target
        
        Returns:
        --------
        dict
            Evaluation metrics
        """
        predictions = self.predict_intervals(X_test)
        y_true = np.array(y_test)
        
        results = {}
        
        # Pinball loss for each quantile
        for quantile, y_pred in predictions.items():
            results[f'pinball_loss_q{int(quantile*100)}'] = pinball_loss(
                y_true, y_pred, quantile
            )
        
        # Coverage for P10-P90 interval
        if 0.1 in predictions and 0.9 in predictions:
            coverage = calculate_coverage(
                y_true,
                predictions[0.1],
                predictions[0.9]
            )
            results['coverage_p10_p90'] = coverage
        
        # Interval width statistics
        if 0.1 in predictions and 0.9 in predictions:
            interval_width = predictions[0.9] - predictions[0.1]
            results['mean_interval_width'] = np.mean(interval_width)
            results['median_interval_width'] = np.median(interval_width)
        
        return results
    
    def save(self, filepath):
        """
        Save quantile models
        
        Parameters:
        -----------
        filepath : str
            Path to save models
        """
        model_data = {
            'models': self.models,
            'quantiles': self.quantiles,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"Quantile models saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load quantile models
        
        Parameters:
        -----------
        filepath : str
            Path to model file
        
        Returns:
        --------
        QuantileModel
            Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        instance = cls(
            quantiles=model_data['quantiles'],
            config=model_data.get('config', {})
        )
        instance.models = model_data['models']
        instance.feature_names = model_data.get('feature_names')
        
        return instance
