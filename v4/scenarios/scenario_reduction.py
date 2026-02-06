"""
Forward scenario reduction.

Reduces a large set of scenarios to a small representative set while
preserving the probability distribution (especially tails for CVaR).

Algorithm (Dupacova, Growe-Kuska, Romisch 2003):
    1. Start with all N scenarios, each with probability 1/N.
    2. Compute pairwise L2 distances between scenarios.
    3. Iteratively remove the scenario with the smallest
       (probability × distance-to-nearest-neighbour) product.
    4. Redistribute its probability to its nearest surviving neighbour.
    5. Stop at the target number of scenarios.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist


def forward_reduction(
    scenarios: np.ndarray,
    n_keep: int = 10,
    distance_metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce scenarios to *n_keep* representative paths.

    Parameters
    ----------
    scenarios : (N, n_hours) — full set of price paths
    n_keep    : int — number of scenarios to keep
    distance_metric : str — metric for cdist (default: euclidean / L2)

    Returns
    -------
    reduced  : (n_keep, n_hours) — selected representative scenarios
    weights  : (n_keep,) — probability weights summing to 1.0
    """
    N = scenarios.shape[0]
    if n_keep >= N:
        return scenarios.copy(), np.full(N, 1.0 / N)

    # Pairwise distances
    D = cdist(scenarios, scenarios, metric=distance_metric)
    np.fill_diagonal(D, np.inf)

    alive = np.ones(N, dtype=bool)
    probs = np.full(N, 1.0 / N)

    for _ in range(N - n_keep):
        # For each alive scenario, find distance to nearest alive neighbour
        alive_idx = np.where(alive)[0]
        D_alive = D[np.ix_(alive_idx, alive_idx)]
        np.fill_diagonal(D_alive, np.inf)
        min_dists = D_alive.min(axis=1)
        nearest = alive_idx[D_alive.argmin(axis=1)]

        # Weighted distance: prob * min_dist
        weighted = probs[alive_idx] * min_dists
        # Remove the scenario with smallest weighted distance
        remove_local = np.argmin(weighted)
        remove_global = alive_idx[remove_local]
        neighbour_global = nearest[remove_local]

        # Redistribute probability
        probs[neighbour_global] += probs[remove_global]
        probs[remove_global] = 0.0
        alive[remove_global] = False

    kept_idx = np.where(alive)[0]
    reduced = scenarios[kept_idx]
    weights = probs[kept_idx]
    # Normalise (should already sum to 1, but guard against float drift)
    weights = weights / weights.sum()

    return reduced, weights
