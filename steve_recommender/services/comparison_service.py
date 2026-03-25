"""High-level comparison entrypoints for UI/CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from steve_recommender.comparison import (
    ComparisonConfig,
    ResolvedCandidate,
    comparison_config_from_dict,
    load_comparison_config,
    resolve_candidates as _resolve_candidates,
    run_comparison as _run_comparison,
)


def load_comparison(path: str | Path) -> ComparisonConfig:
    return load_comparison_config(path)


def comparison_from_dict(payload: Dict[str, Any]) -> ComparisonConfig:
    return comparison_config_from_dict(payload)


def resolve_candidates(cfg: ComparisonConfig) -> List[ResolvedCandidate]:
    return _resolve_candidates(cfg)


def run_comparison(cfg: ComparisonConfig) -> Path:
    return _run_comparison(cfg)
