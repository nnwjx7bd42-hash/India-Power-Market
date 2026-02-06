"""
Enhanced XGBoost model for IEX price forecasting
Uses all 64 features including NERLDC supply-side and weather data
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
import yaml
from typing import List, Optional
from .baseline import BaselineModel


def get_enhanced_features(df):
    """
    Select all features except target for enhanced model
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with all features
    
    Returns:
    --------
    list
        List of all feature names (excluding target)
    """
    exclude_cols = ['P(T)']
    features = [c for c in df.columns if c not in exclude_cols]
    return features


class EnhancedModel(BaselineModel):
    """
    Enhanced XGBoost model using all 64 features
    Extends BaselineModel functionality
    """
    
    def __init__(self, config_path='config/model_config.yaml'):
        """
        Initialize enhanced model
        
        Parameters:
        -----------
        config_path : str
            Path to model configuration file
        """
        super().__init__(config_path)
        # Override config section if enhanced config exists
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            if 'enhanced' in full_config:
                self.config = full_config['enhanced']
            else:
                # Use baseline config as starting point
                self.config = full_config['baseline'].copy()
                # Adjust for more features (may need different regularization)
                self.config['colsample_bytree'] = 0.7  # Lower for more features
    
    def prepare_features(self, df, features=None):
        """
        Prepare features for training/prediction
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        features : list, optional
            Feature names to use (default: all enhanced features)
        
        Returns:
        --------
        pd.DataFrame
            Feature matrix
        """
        if features is None:
            features = get_enhanced_features(df)
        
        self.feature_names = features
        
        # Select features
        X = df[features].copy()
        
        # Ensure no missing values
        if X.isnull().sum().sum() > 0:
            print(f"Warning: {X.isnull().sum().sum()} missing values found, filling with median")
            X = X.fillna(X.median())
        
        return X
