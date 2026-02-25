"""Evaluation pipeline for trained RL agents.

This package is intentionally kept outside upstream stEVE repos so it can be imported
both from the CLI and from the UI code without modifying upstream dependencies.
"""

from .config import EvaluationConfig, load_config

__all__ = ["EvaluationConfig", "load_config", "run_evaluation"]


def __getattr__(name: str):
    if name == "run_evaluation":
        from .pipeline import run_evaluation

        return run_evaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
