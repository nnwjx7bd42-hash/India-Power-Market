"""v4.scenarios — Copula-based scenario generation and reduction."""

from .copula_generator import (
    build_weekly_residuals,
    estimate_rank_correlation,
    generate_scenarios,
)
from .scenario_reduction import forward_reduction

__all__ = [
    "build_weekly_residuals",
    "estimate_rank_correlation",
    "generate_scenarios",
    "forward_reduction",
]
