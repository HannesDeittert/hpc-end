"""Configuration models for train_v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple


DEFAULT_RESULTS_ROOT = Path("results/train_v2")
DEFAULT_POLICY_DEVICE = "cpu"
DEFAULT_WORKER_DEVICE = "cpu"
DEFAULT_REPLAY_DEVICE = "cpu"
DEFAULT_HEATUP_STEPS = int(5e5)
DEFAULT_TRAINING_STEPS = int(2e7)
DEFAULT_EVAL_EVERY = int(2.5e5)
DEFAULT_EXPLORE_EPISODES_BETWEEN_UPDATES = 100
DEFAULT_CONSECUTIVE_ACTION_STEPS = 1
DEFAULT_BATCH_SIZE = 32
DEFAULT_REPLAY_BUFFER_SIZE = int(1e4)
DEFAULT_UPDATE_PER_EXPLORE_STEP = 1 / 20
DEFAULT_LEARNING_RATE = 0.0003217978434614328
DEFAULT_HIDDEN_LAYERS = (400, 400, 400)
DEFAULT_EMBEDDER_NODES = 900
DEFAULT_EMBEDDER_LAYERS = 1
DEFAULT_GAMMA = 0.99
DEFAULT_REWARD_SCALING = 1.0
DEFAULT_LR_END_FACTOR = 0.15
DEFAULT_LR_LINEAR_END_STEPS = int(6e6)
DEFAULT_OMP_THREADS = 1
DEFAULT_TORCH_THREADS = 1
DEFAULT_TORCH_INTEROP_THREADS = 1
DEFAULT_LOG_INTERVAL_S = 600.0
DEFAULT_TARGET_BRANCHES = ("lcca",)
DEFAULT_FRICTION_MU = 0.01
DEFAULT_FLUORO_FREQUENCY_HZ = 7.5
DEFAULT_FLUORO_ROT_ZX_DEG = (20.0, 5.0)
DEFAULT_FORCE_PENALTY_FACTOR = 0.0
DEFAULT_FORCE_THRESHOLD_N = 0.85
DEFAULT_FORCE_DIVISOR = 1000.0
DEFAULT_FORCE_TELEMETRY_MODE = "constraint_projected_si_validated"
DEFAULT_REWARD_PROFILE = "default"

RewardProfile = Literal[
    "default",
    "default_plus_force_penalty",
    "default_plus_excess_force_penalty",
]
ForceTelemetryMode = Literal[
    "passive", "intrusive_lcp", "constraint_projected_si_validated"
]


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_positive_int(value: int, *, field_name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{field_name} must be > 0")


def _require_non_negative_float(value: float, *, field_name: str) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{field_name} must be >= 0")


def parse_eval_seeds(text: str) -> Optional[Tuple[int, ...]]:
    """Parse one CLI seed string into a stable tuple of ints."""

    cleaned = str(text).strip()
    if not cleaned or cleaned.lower() == "none":
        return None
    return tuple(int(part.strip()) for part in cleaned.split(",") if part.strip())


@dataclass(frozen=True)
class RewardSpec:
    """Reward construction parameters for one train_v2 run."""

    profile: RewardProfile = DEFAULT_REWARD_PROFILE
    target_reached_factor: float = 1.0
    step_factor: float = -0.005
    path_delta_factor: float = 0.001
    force_penalty_factor: float = DEFAULT_FORCE_PENALTY_FACTOR
    force_threshold_N: float = DEFAULT_FORCE_THRESHOLD_N
    force_divisor: float = DEFAULT_FORCE_DIVISOR
    force_tip_only: bool = False
    force_telemetry_mode: ForceTelemetryMode = DEFAULT_FORCE_TELEMETRY_MODE

    def __post_init__(self) -> None:
        _require_non_negative_float(
            self.target_reached_factor, field_name="target_reached_factor"
        )
        _require_non_negative_float(
            self.path_delta_factor, field_name="path_delta_factor"
        )
        _require_non_negative_float(
            self.force_penalty_factor, field_name="force_penalty_factor"
        )
        _require_non_negative_float(
            self.force_threshold_N, field_name="force_threshold_N"
        )
        if self.force_divisor <= 0.0:
            raise ValueError("force_divisor must be > 0")
        if (
            self.profile == "default_plus_force_penalty"
            and self.force_penalty_factor <= 0.0
        ):
            raise ValueError(
                "force_penalty_factor must be > 0 for profile=default_plus_force_penalty"
            )
        if (
            self.profile == "default_plus_excess_force_penalty"
            and self.force_divisor <= 0.0
        ):
            raise ValueError(
                "force_divisor must be > 0 for profile=default_plus_excess_force_penalty"
            )


@dataclass(frozen=True)
class RuntimeSpec:
    """Intervention/runtime parameters for one run."""

    tool_ref: str
    anatomy_id: Optional[str] = None
    tool_module: Optional[str] = None
    tool_class: Optional[str] = None
    target_branches: Tuple[str, ...] = DEFAULT_TARGET_BRANCHES
    friction_mu: float = DEFAULT_FRICTION_MU
    fluoroscopy_frequency_hz: float = DEFAULT_FLUORO_FREQUENCY_HZ
    fluoroscopy_rot_zx_deg: Tuple[float, float] = DEFAULT_FLUORO_ROT_ZX_DEG

    def __post_init__(self) -> None:
        _require_non_empty(self.tool_ref, field_name="tool_ref")
        if not self.target_branches:
            raise ValueError("target_branches must not be empty")


@dataclass(frozen=True)
class DoctorConfig:
    """Inputs for the train_v2 preflight command."""

    runtime: RuntimeSpec
    reward: RewardSpec = field(default_factory=RewardSpec)
    trainer_device: str = DEFAULT_POLICY_DEVICE
    output_root: Path = DEFAULT_RESULTS_ROOT
    resume_from: Optional[Path] = None
    resume_replay_buffer_from: Optional[Path] = None
    strict: bool = False
    boot_env: bool = True


@dataclass(frozen=True)
class TrainingRunConfig:
    """Full CLI configuration for one train_v2 training run."""

    name: str
    runtime: RuntimeSpec
    reward: RewardSpec = field(default_factory=RewardSpec)
    trainer_device: str = DEFAULT_POLICY_DEVICE
    worker_device: str = DEFAULT_WORKER_DEVICE
    replay_device: str = DEFAULT_REPLAY_DEVICE
    output_root: Path = DEFAULT_RESULTS_ROOT
    worker_count: int = 2
    heatup_steps: int = DEFAULT_HEATUP_STEPS
    training_steps: int = DEFAULT_TRAINING_STEPS
    eval_every: int = DEFAULT_EVAL_EVERY
    eval_episodes: int = 1
    eval_seeds: Optional[Tuple[int, ...]] = None
    explore_episodes_between_updates: int = DEFAULT_EXPLORE_EPISODES_BETWEEN_UPDATES
    consecutive_action_steps: int = DEFAULT_CONSECUTIVE_ACTION_STEPS
    batch_size: int = DEFAULT_BATCH_SIZE
    replay_buffer_size: int = DEFAULT_REPLAY_BUFFER_SIZE
    update_per_explore_step: float = DEFAULT_UPDATE_PER_EXPLORE_STEP
    learning_rate: float = DEFAULT_LEARNING_RATE
    hidden_layers: Tuple[int, ...] = DEFAULT_HIDDEN_LAYERS
    embedder_nodes: int = DEFAULT_EMBEDDER_NODES
    embedder_layers: int = DEFAULT_EMBEDDER_LAYERS
    gamma: float = DEFAULT_GAMMA
    reward_scaling: float = DEFAULT_REWARD_SCALING
    lr_end_factor: float = DEFAULT_LR_END_FACTOR
    lr_linear_end_steps: int = DEFAULT_LR_LINEAR_END_STEPS
    resume_from: Optional[Path] = None
    resume_skip_heatup: bool = False
    save_latest_replay_buffer: bool = False
    resume_replay_buffer_from: Optional[Path] = None
    train_max_steps: Optional[int] = None
    eval_max_steps: Optional[int] = None
    log_interval_s: float = DEFAULT_LOG_INTERVAL_S
    omp_threads: int = DEFAULT_OMP_THREADS
    torch_threads: int = DEFAULT_TORCH_THREADS
    torch_interop_threads: int = DEFAULT_TORCH_INTEROP_THREADS
    stochastic_eval: bool = False
    preflight: bool = True
    preflight_only: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(self.name, field_name="name")
        _require_positive_int(self.worker_count, field_name="worker_count")
        _require_positive_int(self.heatup_steps, field_name="heatup_steps")
        _require_positive_int(self.training_steps, field_name="training_steps")
        _require_positive_int(self.eval_every, field_name="eval_every")
        _require_positive_int(
            self.explore_episodes_between_updates,
            field_name="explore_episodes_between_updates",
        )
        _require_positive_int(self.batch_size, field_name="batch_size")
        _require_positive_int(self.replay_buffer_size, field_name="replay_buffer_size")
        _require_positive_int(self.embedder_nodes, field_name="embedder_nodes")
        _require_positive_int(self.embedder_layers, field_name="embedder_layers")
        if not self.hidden_layers:
            raise ValueError("hidden_layers must not be empty")
        _require_non_negative_float(
            self.update_per_explore_step, field_name="update_per_explore_step"
        )
        _require_non_negative_float(self.learning_rate, field_name="learning_rate")
        _require_non_negative_float(self.gamma, field_name="gamma")
        _require_non_negative_float(self.reward_scaling, field_name="reward_scaling")
        _require_non_negative_float(self.lr_end_factor, field_name="lr_end_factor")
        _require_positive_int(
            self.lr_linear_end_steps, field_name="lr_linear_end_steps"
        )


def build_doctor_config(
    *,
    tool_ref: str,
    anatomy_id: Optional[str] = None,
    tool_module: Optional[str] = None,
    tool_class: Optional[str] = None,
    reward_profile: RewardProfile = DEFAULT_REWARD_PROFILE,
    force_penalty_factor: float = DEFAULT_FORCE_PENALTY_FACTOR,
    force_threshold_N: float = DEFAULT_FORCE_THRESHOLD_N,
    force_divisor: float = DEFAULT_FORCE_DIVISOR,
    force_tip_only: bool = False,
    resume_from: Optional[Path] = None,
    resume_replay_buffer_from: Optional[Path] = None,
    trainer_device: str = DEFAULT_POLICY_DEVICE,
    output_root: Path = DEFAULT_RESULTS_ROOT,
    strict: bool = False,
    boot_env: bool = True,
    target_branches: Sequence[str] = DEFAULT_TARGET_BRANCHES,
) -> DoctorConfig:
    """Assemble a validated doctor config from CLI-adjacent inputs."""

    return DoctorConfig(
        runtime=RuntimeSpec(
            tool_ref=tool_ref,
            anatomy_id=anatomy_id,
            tool_module=tool_module,
            tool_class=tool_class,
            target_branches=tuple(target_branches),
        ),
        reward=RewardSpec(
            profile=reward_profile,
            force_penalty_factor=force_penalty_factor,
            force_threshold_N=force_threshold_N,
            force_divisor=force_divisor,
            force_tip_only=force_tip_only,
        ),
        trainer_device=trainer_device,
        output_root=Path(output_root),
        resume_from=Path(resume_from) if resume_from is not None else None,
        resume_replay_buffer_from=(
            Path(resume_replay_buffer_from)
            if resume_replay_buffer_from is not None
            else None
        ),
        strict=strict,
        boot_env=boot_env,
    )


def build_training_config(
    *,
    name: str,
    tool_ref: str,
    anatomy_id: Optional[str] = None,
    tool_module: Optional[str] = None,
    tool_class: Optional[str] = None,
    reward_profile: RewardProfile = DEFAULT_REWARD_PROFILE,
    force_penalty_factor: float = DEFAULT_FORCE_PENALTY_FACTOR,
    force_threshold_N: float = DEFAULT_FORCE_THRESHOLD_N,
    force_divisor: float = DEFAULT_FORCE_DIVISOR,
    force_tip_only: bool = False,
    trainer_device: str = DEFAULT_POLICY_DEVICE,
    worker_device: str = DEFAULT_WORKER_DEVICE,
    replay_device: str = DEFAULT_REPLAY_DEVICE,
    output_root: Path = DEFAULT_RESULTS_ROOT,
    worker_count: int = 2,
    heatup_steps: int = DEFAULT_HEATUP_STEPS,
    training_steps: int = DEFAULT_TRAINING_STEPS,
    eval_every: int = DEFAULT_EVAL_EVERY,
    eval_episodes: int = 1,
    eval_seeds: Optional[Tuple[int, ...]] = None,
    explore_episodes_between_updates: int = DEFAULT_EXPLORE_EPISODES_BETWEEN_UPDATES,
    consecutive_action_steps: int = DEFAULT_CONSECUTIVE_ACTION_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    replay_buffer_size: int = DEFAULT_REPLAY_BUFFER_SIZE,
    update_per_explore_step: float = DEFAULT_UPDATE_PER_EXPLORE_STEP,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    hidden_layers: Sequence[int] = DEFAULT_HIDDEN_LAYERS,
    embedder_nodes: int = DEFAULT_EMBEDDER_NODES,
    embedder_layers: int = DEFAULT_EMBEDDER_LAYERS,
    gamma: float = DEFAULT_GAMMA,
    reward_scaling: float = DEFAULT_REWARD_SCALING,
    lr_end_factor: float = DEFAULT_LR_END_FACTOR,
    lr_linear_end_steps: int = DEFAULT_LR_LINEAR_END_STEPS,
    resume_from: Optional[Path] = None,
    resume_skip_heatup: bool = False,
    save_latest_replay_buffer: bool = False,
    resume_replay_buffer_from: Optional[Path] = None,
    train_max_steps: Optional[int] = None,
    eval_max_steps: Optional[int] = None,
    preflight: bool = True,
    preflight_only: bool = False,
    stochastic_eval: bool = False,
    target_branches: Sequence[str] = DEFAULT_TARGET_BRANCHES,
) -> TrainingRunConfig:
    """Assemble a validated training config from CLI-adjacent inputs."""

    return TrainingRunConfig(
        name=name,
        runtime=RuntimeSpec(
            tool_ref=tool_ref,
            anatomy_id=anatomy_id,
            tool_module=tool_module,
            tool_class=tool_class,
            target_branches=tuple(target_branches),
        ),
        reward=RewardSpec(
            profile=reward_profile,
            force_penalty_factor=force_penalty_factor,
            force_threshold_N=force_threshold_N,
            force_divisor=force_divisor,
            force_tip_only=force_tip_only,
        ),
        trainer_device=trainer_device,
        worker_device=worker_device,
        replay_device=replay_device,
        output_root=Path(output_root),
        worker_count=worker_count,
        heatup_steps=heatup_steps,
        training_steps=training_steps,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        eval_seeds=eval_seeds,
        explore_episodes_between_updates=explore_episodes_between_updates,
        consecutive_action_steps=consecutive_action_steps,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        update_per_explore_step=update_per_explore_step,
        learning_rate=learning_rate,
        hidden_layers=tuple(int(v) for v in hidden_layers),
        embedder_nodes=embedder_nodes,
        embedder_layers=embedder_layers,
        gamma=gamma,
        reward_scaling=reward_scaling,
        lr_end_factor=lr_end_factor,
        lr_linear_end_steps=lr_linear_end_steps,
        resume_from=Path(resume_from) if resume_from is not None else None,
        resume_skip_heatup=resume_skip_heatup,
        save_latest_replay_buffer=save_latest_replay_buffer,
        resume_replay_buffer_from=(
            Path(resume_replay_buffer_from)
            if resume_replay_buffer_from is not None
            else None
        ),
        train_max_steps=train_max_steps,
        eval_max_steps=eval_max_steps,
        preflight=preflight,
        preflight_only=preflight_only,
        stochastic_eval=stochastic_eval,
    )
