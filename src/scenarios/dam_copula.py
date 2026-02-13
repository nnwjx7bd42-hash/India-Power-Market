import numpy as np
from scipy import stats
from .utils import inverse_cdf

class DAMCopulaGenerator:
    """
    Gaussian Copula Scenario Generator for DAM.
    Models correlation structure of residuals across 24 hours.
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.correlation_matrix = None
        self.rng = np.random.default_rng(seed)

    def fit(self, residuals_by_day: np.ndarray):
        """
        Fit Gaussian Copula on residuals.
        residuals_by_day: (n_days, 24)
        """
        n_days, n_hours = residuals_by_day.shape
        if n_hours != 24:
            raise ValueError("Residuals must have 24 columns.")

        # 1. Rank transform to uniform
        # argsort.argsort gives ranks (0 to n-1)
        ranks = np.argsort(np.argsort(residuals_by_day, axis=0), axis=0)
        u = (ranks + 1) / (n_days + 1)
        
        # 2. Convert to normal scores
        # Clip to avoid inf at 0 or 1 (though n_days+1 denom prevents exact 0/1)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        z = stats.norm.ppf(u)
        
        # 3. Compute Correlation
        self.correlation_matrix = np.corrcoef(z.T)
        
        # 4. Valid Positive Semi-Definite
        eigenvalues, eigenvectors = np.linalg.eigh(self.correlation_matrix)
        if np.any(eigenvalues < 0):
            print("Warning: Correlation matrix not PSD, fixing...")
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            # Reconstruct
            self.correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            # Re-normalize diagonal to 1
            d = np.diag(self.correlation_matrix)
            self.correlation_matrix = self.correlation_matrix / np.sqrt(d[:, None] * d[None, :])

    def generate(self, quantile_predictions: dict, n_scenarios: int) -> np.ndarray:
        """
        Generate scenarios.
        quantile_predictions: {alpha: array_24h_scalars}
        """
        if self.correlation_matrix is None:
            raise ValueError("Model not fitted.")
            
        # 1. Sample z ~ N(0, Sigma)
        # shape (n_scenarios, 24)
        z = self.rng.multivariate_normal(
            mean=np.zeros(24), 
            cov=self.correlation_matrix, 
            size=n_scenarios
        )
        
        # 2. Convert to uniform
        u = stats.norm.cdf(z)
        
        # 3. Inverse CDF
        # Output shape (n_scenarios, 24)
        scenarios = np.zeros((n_scenarios, 24))
        
        for h in range(24):
            # Extract quantiles for this hour h
            q_vals_h = {q: preds[h] for q, preds in quantile_predictions.items()}
            
            # Map column u[:, h] to price
            scenarios[:, h] = inverse_cdf(u[:, h], q_vals_h)
            
        # 4. Floor at 0
        return np.maximum(0.0, scenarios)

    def save(self, path: str):
        np.save(path, self.correlation_matrix)

    @classmethod
    def load(cls, path: str, seed=42):
        instance = cls(seed)
        instance.correlation_matrix = np.load(path)
        return instance
