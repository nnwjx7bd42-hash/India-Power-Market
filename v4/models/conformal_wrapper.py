"""
Adaptive Conformal Inference (ACI) wrapper for time-series calibration.

Wraps around *any* quantile forecaster to provide coverage guarantees by
adaptively widening/narrowing prediction intervals based on recent misses.

References
----------
- Gibbs & Candes (2021): Adaptive Conformal Inference Under Distribution Shift
- Angelopoulos et al. (2023): Conformal PID Control for Time Series
- O'Connor et al. (2025): CP for battery trading in electricity markets
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


class ConformalCalibrator:
    """
    Adaptive Conformal Prediction wrapper for quantile forecasts.

    Given a calibration set of (actual, predicted-quantiles), learns
    adaptive correction factors that ensure long-run coverage.

    Usage
    -----
    1. ``fit(y_cal, q_cal)``     — calibrate on held-out data
    2. ``adjust(q_new)``         — widen/narrow new quantile forecasts
    3. ``update(y_new, q_adj)``  — online update after observing actual
    """

    def __init__(
        self,
        quantiles: Sequence[float],
        target_coverage: float = 0.90,
        learning_rate: float = 0.05,
    ):
        self.quantiles = np.asarray(sorted(quantiles))
        self.target_coverage = target_coverage
        self.lr = learning_rate

        # Identify symmetric interval pairs for calibration
        self.pairs = self._find_pairs()

        # Per-pair adaptive correction (additive shift in price units)
        self.corrections: Dict[Tuple[float, float], float] = {
            pair: 0.0 for pair in self.pairs
        }

        # Calibration residual quantiles (static baseline)
        self._residual_q: Dict[Tuple[float, float], float] = {}

    # ------------------------------------------------------------------
    def _find_pairs(self):
        """Find symmetric (lo, hi) pairs around the median."""
        pairs = []
        qs = self.quantiles
        for i in range(len(qs) // 2):
            lo, hi = qs[i], qs[-(i + 1)]
            if lo < 0.5 < hi:
                pairs.append((float(lo), float(hi)))
        return pairs

    def _pair_indices(self, lo: float, hi: float):
        lo_idx = int(np.argmin(np.abs(self.quantiles - lo)))
        hi_idx = int(np.argmin(np.abs(self.quantiles - hi)))
        return lo_idx, hi_idx

    # ------------------------------------------------------------------
    # Fit (static calibration)
    # ------------------------------------------------------------------
    def fit(self, y_cal: np.ndarray, q_cal: np.ndarray) -> Dict:
        """
        Calibrate on a held-out set.

        Parameters
        ----------
        y_cal  : (n,) actual values
        q_cal  : (n, n_quantiles) predicted quantiles

        Returns
        -------
        dict with pre- and post-calibration coverage per pair.
        """
        y_cal = np.asarray(y_cal)
        q_cal = np.asarray(q_cal)
        n = len(y_cal)
        report: Dict = {}

        for lo, hi in self.pairs:
            lo_idx, hi_idx = self._pair_indices(lo, hi)
            nominal = hi - lo

            lower = q_cal[:, lo_idx]
            upper = q_cal[:, hi_idx]

            # Pre-calibration coverage
            inside = (y_cal >= lower) & (y_cal <= upper)
            pre_cov = float(np.mean(inside))

            # Compute signed nonconformity scores
            # Positive score = actual is outside interval
            scores = np.maximum(lower - y_cal, y_cal - upper)
            # The (1-alpha) quantile of scores gives the correction
            alpha = 1.0 - nominal
            q_level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
            correction = float(np.quantile(scores, q_level))
            correction = max(correction, 0.0)  # only widen, never narrow below raw
            self._residual_q[(lo, hi)] = correction
            self.corrections[(lo, hi)] = correction

            # Post-calibration coverage (on cal set itself)
            inside_post = (y_cal >= lower - correction) & (y_cal <= upper + correction)
            post_cov = float(np.mean(inside_post))

            report[f"coverage_{int(nominal*100)}pct_pre"] = pre_cov
            report[f"coverage_{int(nominal*100)}pct_post"] = post_cov
            report[f"correction_{int(nominal*100)}pct"] = correction

        return report

    # ------------------------------------------------------------------
    # Adjust new forecasts
    # ------------------------------------------------------------------
    def adjust(self, q_new: np.ndarray) -> np.ndarray:
        """
        Apply current corrections to new quantile predictions.

        Returns adjusted quantiles (same shape).
        """
        q_adj = q_new.copy()
        for lo, hi in self.pairs:
            lo_idx, hi_idx = self._pair_indices(lo, hi)
            c = self.corrections[(lo, hi)]
            q_adj[:, lo_idx] -= c
            q_adj[:, hi_idx] += c
        # Re-enforce monotonicity after adjustment
        q_adj = np.sort(q_adj, axis=1)
        return q_adj

    # ------------------------------------------------------------------
    # Online adaptive update
    # ------------------------------------------------------------------
    def update(self, y_new: np.ndarray, q_adj: np.ndarray) -> None:
        """
        After observing actuals, adaptively update corrections.

        Uses a simple PID-style rule:
            correction += lr * (miss_rate - target_miss_rate)

        Where miss_rate is computed over the new batch.
        """
        y_new = np.asarray(y_new)
        q_adj = np.asarray(q_adj)

        for lo, hi in self.pairs:
            lo_idx, hi_idx = self._pair_indices(lo, hi)
            nominal = hi - lo
            target_miss = 1.0 - nominal

            lower = q_adj[:, lo_idx]
            upper = q_adj[:, hi_idx]
            inside = (y_new >= lower) & (y_new <= upper)
            empirical_miss = 1.0 - float(np.mean(inside))

            # PID update: widen if missing too often, tighten if covering too much
            delta = self.lr * (empirical_miss - target_miss)
            self.corrections[(lo, hi)] = max(0.0, self.corrections[(lo, hi)] + delta)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict:
        return {
            "quantiles": self.quantiles.tolist(),
            "target_coverage": self.target_coverage,
            "learning_rate": self.lr,
            "corrections": {f"{lo},{hi}": v for (lo, hi), v in self.corrections.items()},
            "residual_q": {f"{lo},{hi}": v for (lo, hi), v in self._residual_q.items()},
        }

    @classmethod
    def from_state_dict(cls, d: Dict) -> "ConformalCalibrator":
        obj = cls(
            quantiles=d["quantiles"],
            target_coverage=d.get("target_coverage", 0.90),
            learning_rate=d.get("learning_rate", 0.05),
        )
        for key, v in d.get("corrections", {}).items():
            lo, hi = map(float, key.split(","))
            obj.corrections[(lo, hi)] = v
        for key, v in d.get("residual_q", {}).items():
            lo, hi = map(float, key.split(","))
            obj._residual_q[(lo, hi)] = v
        return obj
