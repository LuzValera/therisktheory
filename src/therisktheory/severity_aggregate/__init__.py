"""Claim severity, aggregate loss models, approximations, and simulation."""

from therisktheory.severity_aggregate.monte_carlo import (
    aggregate_loss_monte_carlo_metrics,
    benchmark_compound_poisson_exponential,
    simulate_aggregate_loss,
)
from therisktheory.severity_aggregate.reinsurance_stability import (
    insurer_reciprocity_metric,
    optimal_reinsurance,
    technical_result_variance,
)

__all__ = [
    "aggregate_loss_monte_carlo_metrics",
    "benchmark_compound_poisson_exponential",
    "insurer_reciprocity_metric",
    "optimal_reinsurance",
    "simulate_aggregate_loss",
    "technical_result_variance",
]
