"""
Baseline XGBoost model for IEX price forecasting
Uses original 10 features + basic calendar features
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
import yaml
from typing import List, Optional


def get_baseline_features(df):
    """
    Select baseline feature set
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with all features
    
    Returns:
    --------
    list
        List of feature names for baseline model
    """
    original_features = [
        'L(T-1)', 'L(T-2)', 'L(T-24)', 'L(T-48)',
        'P(T-1)', 'P(T-2)', 'P(T-24)', 'P(T-48)',
        'Day', 'Season'
    ]
    calendar_features = ['Hour', 'DayOfWeek', 'Month']
    
    # Check which features exist in dataframe
    all_features = original_features + calendar_features
    available_features = [f for f in all_features if f in df.columns]
    
    missing = set(all_features) - set(available_features)
    if missing:
        print(f"Warning: Missing features: {missing}")
    
    return available_features


class BaselineModel:
    """
    Baseline XGBoost model for price forecasting
    """
    
    def __init__(self, config_path='config/model_config.yaml'):
        """
        Initialize baseline model
        
        Parameters:
        -----------
        config_path : str
            Path to model configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.feature_names = None
        self.training_date = None
    
    def _load_config(self):
        """Load model configuration"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['baseline']
    
    def prepare_features(self, df, features=None):
        """
        Prepare features for training/prediction
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        features : list, optional
            Feature names to use (default: baseline features)
        
        Returns:
        --------
        pd.DataFrame
            Feature matrix
        """
        if features is None:
            features = get_baseline_features(df)
        
        self.feature_names = features
        
        # Select features
        X = df[features].copy()
        
        # Ensure no missing values
        if X.isnull().sum().sum() > 0:
            print(f"Warning: {X.isnull().sum().sum()} missing values found, filling with median")
            X = X.fillna(X.median())
        
        return X
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train baseline model
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.array
            Training features
        y_train : pd.Series or np.array
            Training target
        X_val : pd.DataFrame or np.array, optional
            Validation features for early stopping
        y_val : pd.Series or np.array, optional
            Validation target
        """
        # Prepare model parameters
        params = {
            'objective': self.config['objective'],
            'max_depth': self.config['max_depth'],
            'learning_rate': self.config['learning_rate'],
            'n_estimators': self.config['n_estimators'],
            'subsample': self.config['subsample'],
            'colsample_bytree': self.config['colsample_bytree'],
            'random_state': self.config['random_state'],
            'eval_metric': self.config.get('eval_metric', 'rmse')
        }
        
        # Create model
        self.model = xgb.XGBRegressor(**params)
        
        # Fit model
        # Note: XGBoost 3.x has different API - using eval_set for monitoring only
        if X_val is not None and y_val is not None:
            # Use eval_set for monitoring (early stopping handled by n_estimators)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)
        
        self.training_date = pd.Timestamp.now()
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature matrix
        
        Returns:
        --------
        np.array
            Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        return self.model.predict(X)
    
    def get_feature_importance(self, importance_type='gain'):
        """
        Get feature importance
        
        Parameters:
        -----------
        importance_type : str
            Type of importance ('gain', 'cover', 'weight')
        
        Returns:
        --------
        pd.DataFrame
            Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Convert to DataFrame
        if self.feature_names:
            # Map XGBoost's f0, f1, ... indices to actual feature names
            # XGBoost uses f0, f1, f2, ... as keys
            feature_map = {f'f{i}': feat for i, feat in enumerate(self.feature_names)}
            
            # Create DataFrame with mapped names
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': [importance.get(f'f{i}', 0) for i in range(len(self.feature_names))]
            })
        else:
            feature_importance = pd.DataFrame({
                'feature': list(importance.keys()),
                'importance': list(importance.values())
            })
        
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save(self, filepath):
        """
        Save model to file
        
        Parameters:
        -----------
        filepath : str
            Path to save model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_date': self.training_date,
            'config': self.config
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load model from file
        
        Parameters:
        -----------
        filepath : str
            Path to model file
        
        Returns:
        --------
        BaselineModel
            Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        instance = cls()
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.training_date = model_data['training_date']
        instance.config = model_data.get('config', {})
        
        return instance
