"""Configuration models for training runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


ARCHVARIETY_EVAL_SEEDS = [
    1,
    2,
    3,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    13,
    14,
    16,
    17,
    18,
    21,
    22,
    23,
    27,
    31,
    34,
    35,
    37,
    39,
    42,
    43,
    44,
    47,
    48,
    50,
    52,
    55,
    56,
    58,
    61,
    62,
    63,
    68,
    69,
    70,
    71,
    73,
    79,
    80,
    81,
    84,
    89,
    91,
    92,
    93,
    95,
    97,
    102,
    103,
    108,
    109,
    110,
    115,
    116,
    117,
    118,
    120,
    122,
    123,
    124,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    134,
    136,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    147,
    148,
    149,
    150,
    151,
    152,
    154,
    155,
    156,
    158,
    159,
    161,
    162,
    167,
    168,
    171,
    175,
]


@dataclass
class TrainingConfig:
    tool: str
    model: Optional[str] = None
    name: str = "paper_run"
    n_worker: int = 2
    trainer_device: str = "cpu"
    worker_device: str = "cpu"
    replay_device: str = "cpu"
    out_root: Path = Path("results/eve_paper/neurovascular/aorta/gw_only/arch_vmr_94")
    learning_rate: float = 0.0003217978434614328
    hidden_layers: List[int] = field(default_factory=lambda: [400, 400, 400])
    embedder_nodes: int = 700
    embedder_layers: int = 1
    heatup_steps: float = 5e5
    training_steps: float = 2e7
    eval_every: float = 2.5e5
    eval_episodes: int = 1
    eval_seeds: Optional[List[int]] = field(
        default_factory=lambda: ARCHVARIETY_EVAL_SEEDS.copy()
    )
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
