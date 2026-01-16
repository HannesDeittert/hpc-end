"""Evaluation pipeline for trained RL agents.

This package is intentionally kept outside upstream stEVE repos so it can be imported
both from the CLI and from the UI code without modifying upstream dependencies.
"""

from .config import EvaluationConfig, load_config
from .pipeline import run_evaluation

__all__ = ["EvaluationConfig", "load_config", "run_evaluation"]
