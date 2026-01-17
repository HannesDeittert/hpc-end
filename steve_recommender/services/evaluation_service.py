"""High-level evaluation entrypoints for UI/CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from steve_recommender.evaluation.config import EvaluationConfig, config_from_dict, load_config
from steve_recommender.evaluation.pipeline import run_evaluation as _run_evaluation


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    return load_config(path)


def evaluation_config_from_dict(payload: Dict[str, Any]) -> EvaluationConfig:
    return config_from_dict(payload)


def run_evaluation(cfg: EvaluationConfig) -> Path:
    return _run_evaluation(cfg)
