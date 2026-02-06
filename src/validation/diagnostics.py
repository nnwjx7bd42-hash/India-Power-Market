"""
Residual diagnostics for model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats
from pathlib import Path


class ResidualDiagnostics:
    """
    Residual diagnostics and visualization
    """
    
    def __init__(self, y_true, y_pred, timestamps=None):
        """
        Initialize diagnostics
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        timestamps : array-like, optional
            Timestamps for time series analysis
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.residuals = self.y_true - self.y_pred
        self.timestamps = timestamps
        
        if timestamps is not None:
            self.df = pd.DataFrame({
                'timestamp': timestamps,
                'y_true': self.y_true,
                'y_pred': self.y_pred,
                'residual': self.residuals
            })
            self.df = self.df.set_index('timestamp')
        else:
            self.df = pd.DataFrame({
                'y_true': self.y_true,
                'y_pred': self.y_pred,
                'residual': self.residuals
            })
    
    def plot_residuals_vs_time(self, figsize=(15, 5), save_path=None):
        """
        Plot residuals vs time
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        if self.timestamps is None:
            print("Warning: No timestamps provided, skipping time plot")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.df.index, self.df['residual'], alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Residual')
        ax.set_title('Residuals vs Time')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_residuals_vs_hour(self, hour_col='Hour', figsize=(12, 6), save_path=None):
        """
        Plot residuals by hour of day
        
        Parameters:
        -----------
        hour_col : str
            Column name for hour
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        if hour_col not in self.df.columns:
            # Try to extract hour from index if datetime
            if isinstance(self.df.index, pd.DatetimeIndex):
                self.df['Hour'] = self.df.index.hour
                hour_col = 'Hour'
            else:
                print(f"Warning: {hour_col} not found and cannot extract from index")
                return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Box plot by hour
        hourly_residuals = [self.df[self.df[hour_col] == h]['residual'].values 
                           for h in sorted(self.df[hour_col].unique())]
        ax1.boxplot(hourly_residuals, labels=sorted(self.df[hour_col].unique()))
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Residual')
        ax1.set_title('Residual Distribution by Hour')
        ax1.grid(True, alpha=0.3)
        
        # Mean residual by hour
        hourly_mean = self.df.groupby(hour_col)['residual'].mean()
        ax2.plot(hourly_mean.index, hourly_mean.values, marker='o')
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Mean Residual')
        ax2.set_title('Mean Residual by Hour')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_residuals_vs_actual(self, figsize=(10, 6), save_path=None):
        """
        Plot residuals vs actual price
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.df['y_true'], self.df['residual'], alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax.set_xlabel('Actual Price (₹)')
        ax.set_ylabel('Residual')
        ax.set_title('Residuals vs Actual Price')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_residual_distribution(self, figsize=(12, 5), save_path=None):
        """
        Plot residual distribution
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(self.residuals, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='r', linestyle='--', linewidth=1)
        ax1.set_xlabel('Residual')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Residual Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(self.residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Check)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_acf(self, lags=48, figsize=(12, 5), save_path=None):
        """
        Plot autocorrelation function of residuals
        
        Parameters:
        -----------
        lags : int
            Number of lags to plot
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # ACF
        acf_values = acf(self.residuals, nlags=lags, fft=True)
        ax1.plot(range(len(acf_values)), acf_values, marker='o', markersize=4)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax1.axhline(y=0.05, color='r', linestyle='--', linewidth=1, label='95% CI')
        ax1.axhline(y=-0.05, color='r', linestyle='--', linewidth=1)
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('ACF')
        ax1.set_title('Autocorrelation Function of Residuals')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # PACF
        pacf_values = pacf(self.residuals, nlags=lags)
        ax2.plot(range(len(pacf_values)), pacf_values, marker='o', markersize=4)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.axhline(y=0.05, color='r', linestyle='--', linewidth=1, label='95% CI')
        ax2.axhline(y=-0.05, color='r', linestyle='--', linewidth=1)
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('PACF')
        ax2.set_title('Partial Autocorrelation Function of Residuals')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_all_plots(self, df_with_metadata=None, output_dir='results/training/diagnostics'):
        """
        Generate all diagnostic plots
        
        Parameters:
        -----------
        df_with_metadata : pd.DataFrame, optional
            DataFrame with additional columns (Hour, etc.)
        output_dir : str
            Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Merge metadata if provided
        if df_with_metadata is not None:
            if isinstance(df_with_metadata.index, pd.DatetimeIndex):
                # Only add columns that don't already exist
                new_cols = [c for c in df_with_metadata.columns if c not in self.df.columns]
                if new_cols:
                    self.df = self.df.join(df_with_metadata[new_cols], how='left')
                else:
                    # If columns exist, update them
                    for col in df_with_metadata.columns:
                        if col in self.df.columns:
                            self.df[col] = df_with_metadata[col]
        
        print("Generating diagnostic plots...")
        
        # Residuals vs time
        if self.timestamps is not None:
            self.plot_residuals_vs_time(
                save_path=output_path / 'residuals_vs_time.png'
            )
            plt.close()
        
        # Residuals vs hour
        self.plot_residuals_vs_hour(
            save_path=output_path / 'residuals_vs_hour.png'
        )
        plt.close()
        
        # Residuals vs actual
        self.plot_residuals_vs_actual(
            save_path=output_path / 'residuals_vs_actual.png'
        )
        plt.close()
        
        # Residual distribution
        self.plot_residual_distribution(
            save_path=output_path / 'residual_distribution.png'
        )
        plt.close()
        
        # ACF/PACF
        self.plot_acf(
            save_path=output_path / 'residual_acf.png'
        )
        plt.close()
        
        print(f"All plots saved to {output_dir}")
