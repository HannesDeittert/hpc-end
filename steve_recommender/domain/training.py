"""Configuration models for training runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainingConfig:
    tool: str
    model: Optional[str] = None
    name: str = "paper_run"
    n_worker: int = 2
    trainer_device: str = "cpu"
    worker_device: str = "cpu"
    replay_device: str = "cpu"
    out_root: Path = Path("results/paper_runs")
    learning_rate: float = 3.2e-4
    hidden_layers: List[int] = field(
        default_factory=lambda: [900, 900, 900, 900]
    )
    embedder_nodes: int = 500
    embedder_layers: int = 1
    heatup_steps: float = 5e5
    training_steps: float = 2e7
    eval_every: float = 2.5e5
    eval_episodes: int = 1
    explore_episodes_between_updates: int = 100
    update_per_explore_step: float = 1 / 20
    consecutive_action_steps: int = 1
    batch_size: int = 32
    replay_buffer_size: float = 1e4
    log_interval_s: float = 600.0
    omp_threads: int = 1
    torch_threads: int = 1
    torch_interop_threads: int = 1
    stochastic_eval: bool = False
    tensorboard: bool = False
    tb_logdir: Optional[Path] = None
    resume_from: Optional[Path] = None
    register_agent: bool = True
    agent_name: Optional[str] = None
    stdout: bool = False
