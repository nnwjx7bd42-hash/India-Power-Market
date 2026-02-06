"""
Multi-quantile XGBoost forecaster for the v4 planning model.

Uses XGBoost >= 2.0 native ``reg:quantileerror`` with a ``quantile_alpha``
array so a **single** booster outputs all quantiles simultaneously.
Post-prediction monotonicity enforcement via isotonic rearrangement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import xgboost as xgb


class QuantileForecaster:
    """XGBoost multi-quantile forecaster (single booster, many quantiles)."""

    def __init__(
        self,
        quantiles: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.40,
                                       0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
        *,
        max_depth: int = 6,
        learning_rate: float = 0.04,
        n_estimators: int = 500,
        min_child_weight: int = 50,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 20,
        random_state: int = 42,
        tree_method: str = "hist",
    ):
        self.quantiles = np.asarray(sorted(quantiles))
        self.params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": self.quantiles,
            "tree_method": tree_method,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "seed": random_state,
        }
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.booster: Optional[xgb.Booster] = None
        self.feature_names: Optional[List[str]] = None
        self.best_iteration: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict:
        """Train with early stopping on validation average pinball loss."""
        self.feature_names = feature_names

        dtrain = xgb.QuantileDMatrix(X_train, label=y_train,
                                     feature_names=feature_names)
        dval = xgb.QuantileDMatrix(X_val, label=y_val, ref=dtrain,
                                   feature_names=feature_names)

        self.booster = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            early_stopping_rounds=self.early_stopping_rounds,
            evals=[(dtrain, "Train"), (dval, "Val")],
            verbose_eval=50 if verbose else False,
        )
        self.best_iteration = int(self.booster.best_iteration)

        # Evaluate on validation
        q_preds = self.predict(X_val, feature_names=feature_names)
        metrics = self.evaluate(y_val, q_preds)
        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        X: np.ndarray,
        *,
        feature_names: Optional[List[str]] = None,
        enforce_monotonicity: bool = True,
    ) -> np.ndarray:
        """
        Predict quantiles.

        Returns
        -------
        np.ndarray of shape (n_samples, n_quantiles)
        """
        if self.booster is None:
            raise RuntimeError("Model not trained — call .train() first.")
        fnames = feature_names or self.feature_names
        dm = xgb.DMatrix(X, feature_names=fnames)
        preds = self.booster.predict(dm)  # (n, n_quantiles)
        if preds.ndim == 1:
            # Single quantile edge-case
            preds = preds.reshape(-1, 1)
        if enforce_monotonicity:
            preds = np.sort(preds, axis=1)
        return preds

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def evaluate(self, y_true: np.ndarray, q_preds: np.ndarray) -> Dict:
        """Compute probabilistic metrics (pinball, CRPS, coverage, Winkler)."""
        # Lazy import — resolve path relative to this file
        import sys as _sys
        _v4 = str(Path(__file__).resolve().parent.parent)
        if _v4 not in _sys.path:
            _sys.path.insert(0, _v4)
        from evaluation.pinball_loss import (
            avg_pinball_loss,
            crps_from_quantiles,
            empirical_coverage,
            median_mape,
            winkler_score,
        )
        y_true = np.asarray(y_true)
        metrics: Dict = {}
        metrics["avg_pinball"] = float(avg_pinball_loss(y_true, q_preds, self.quantiles))
        metrics["crps"] = float(crps_from_quantiles(y_true, q_preds, self.quantiles))
        metrics["median_mape"] = float(median_mape(y_true, q_preds, self.quantiles))

        # Coverage for symmetric intervals
        pairs = [(0.05, 0.95), (0.10, 0.90), (0.20, 0.80)]
        for lo, hi in pairs:
            lo_idx = int(np.argmin(np.abs(self.quantiles - lo)))
            hi_idx = int(np.argmin(np.abs(self.quantiles - hi)))
            cov = float(empirical_coverage(y_true, q_preds[:, lo_idx], q_preds[:, hi_idx]))
            width = float(np.mean(q_preds[:, hi_idx] - q_preds[:, lo_idx]))
            nominal = hi - lo
            metrics[f"coverage_{int(nominal*100)}pct"] = cov
            metrics[f"width_{int(nominal*100)}pct"] = width
            wink = float(winkler_score(y_true, q_preds[:, lo_idx], q_preds[:, hi_idx], nominal))
            metrics[f"winkler_{int(nominal*100)}pct"] = wink

        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "quantile_model.json"))
        meta = {
            "quantiles": self.quantiles.tolist(),
            "feature_names": self.feature_names,
            "best_iteration": self.best_iteration,
            "params": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in self.params.items()},
        }
        with open(directory / "quantile_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "QuantileForecaster":
        directory = Path(directory)
        with open(directory / "quantile_meta.json") as f:
            meta = json.load(f)
        obj = cls(quantiles=meta["quantiles"])
        obj.feature_names = meta.get("feature_names")
        obj.best_iteration = meta.get("best_iteration", 0)
        obj.params = meta.get("params", obj.params)
        # Restore numpy array for quantile_alpha
        if "quantile_alpha" in obj.params and isinstance(obj.params["quantile_alpha"], list):
            obj.params["quantile_alpha"] = np.array(obj.params["quantile_alpha"])
        obj.booster = xgb.Booster()
        obj.booster.load_model(str(directory / "quantile_model.json"))
        return obj
