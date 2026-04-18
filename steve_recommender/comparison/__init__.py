"""Comparison pipeline for tool recommendation benchmarks."""

from .config import (
    ComparisonCandidateSpec,
    ComparisonConfig,
    ResolvedCandidate,
    comparison_config_from_dict,
    load_comparison_config,
)
from .pipeline import resolve_candidates, run_comparison

__all__ = [
    "ComparisonCandidateSpec",
    "ComparisonConfig",
    "ResolvedCandidate",
    "comparison_config_from_dict",
    "load_comparison_config",
    "resolve_candidates",
    "run_comparison",
]
