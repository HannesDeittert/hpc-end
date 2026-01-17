"""High-level training entrypoints for UI/CLI."""

from __future__ import annotations

from pathlib import Path

from steve_recommender.domain import TrainingConfig
from steve_recommender.rl import train_paper_arch


def run_training(cfg: TrainingConfig) -> Path:
    """Run a training job synchronously and return the run folder."""

    return train_paper_arch.run_training(cfg)
