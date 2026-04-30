"""Local agent construction for train_v2."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import eve
import torch

from steve_recommender.training.bench_agents import BenchAgentSynchron

from ..config import TrainingRunConfig


def build_agent(
    *,
    config: TrainingRunConfig,
    env_train: eve.Env,
    env_eval: eve.Env,
) -> BenchAgentSynchron:
    """Build the train_v2 multi-worker agent."""

    return BenchAgentSynchron(
        torch.device(config.trainer_device),
        torch.device(config.worker_device),
        config.learning_rate,
        config.lr_end_factor,
        config.lr_linear_end_steps,
        list(config.hidden_layers),
        config.embedder_nodes,
        config.embedder_layers,
        config.gamma,
        config.batch_size,
        config.reward_scaling,
        config.replay_buffer_size,
        env_train,
        env_eval,
        config.consecutive_action_steps,
        config.worker_count,
        config.stochastic_eval,
        False,
    )


def maybe_resume_agent(
    *,
    agent: BenchAgentSynchron,
    checkpoint_path: Optional[Path],
) -> None:
    """Load one checkpoint when the CLI requested resume semantics."""

    if checkpoint_path is None:
        return
    agent.load_checkpoint(str(checkpoint_path))
