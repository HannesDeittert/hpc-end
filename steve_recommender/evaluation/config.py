from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import yaml


@dataclass(frozen=True)
class AgentSpec:
    """One trained agent + the tool it was trained to control."""

    name: str
    tool: str
    checkpoint: str


@dataclass(frozen=True)
class AorticArchSpec:
    """Aortic arch anatomy configuration (stEVE built-in generator)."""

    type: Literal["aortic_arch"] = "aortic_arch"
    arch_type: str = "I"
    seed: int = 30
    rotation_yzx_deg: Optional[Sequence[float]] = None
    scaling_xyzd: Optional[Sequence[float]] = None
    omit_axis: Optional[str] = None

    # Target definition: for now we support 'branch_end' on one or more branches.
    target_mode: Literal["branch_end"] = "branch_end"
    target_branches: List[str] = field(default_factory=lambda: ["lcca"])
    target_threshold_mm: float = 5.0

    # Simulator/fluoro parameters (kept explicit so evaluation is reproducible).
    image_frequency_hz: float = 7.5
    image_rot_zx_deg: Sequence[float] = (20.0, 5.0)
    friction: float = 0.001


AnatomySpec = AorticArchSpec


@dataclass(frozen=True)
class ScoringConfig:
    """How to compute a per-trial score and aggregate it per agent.

    The goal is a *relative* score for comparing agents on the same benchmark.
    Force units can vary with scene/unit conventions, so the scale parameters
    are configurable in the YAML.
    """

    mode: Literal["default_v1"] = "default_v1"

    # Weights for the default score components. Higher is better.
    # If `normalize_weights` is True the final score is divided by the sum of weights.
    w_success: float = 2.0
    w_efficiency: float = 1.0
    w_safety: float = 1.0
    w_smoothness: float = 0.25
    normalize_weights: bool = True

    # Scale factors used by the default scoring:
    # - safety uses exp(-force/force_scale) and exp(-lcp/lcp_scale)
    # - smoothness uses exp(-tip_speed_max/speed_scale_mm_s)
    force_scale: float = 1.0
    lcp_scale: float = 1.0
    speed_scale_mm_s: float = 50.0


@dataclass(frozen=True)
class EvaluationConfig:
    """Top-level evaluation configuration.

    This is designed to be:
    - human editable (YAML/JSON)
    - easy to create from the UI (dict → dataclass)
    - stable to extend over time (add optional fields with defaults)
    """

    name: str
    agents: List[AgentSpec]
    anatomy: AnatomySpec = field(default_factory=AorticArchSpec)

    # How many evaluation trials per agent.
    n_trials: int = 10
    base_seed: int = 123

    # Environment settings.
    max_episode_steps: int = 1000

    # Where results are stored.
    output_root: str = "results/eval_runs"

    # Policy device for inference. Use "cuda" if you want fast policy inference.
    policy_device: str = "cuda"

    # If True, we run a non-multiprocessing SOFA simulation so we can read forces.
    # If False, SOFA will run in a separate process (faster/safer) but forces are
    # not accessible with the current upstream API.
    use_non_mp_sim: bool = True

    # Optional scoring configuration for post-processing.
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


def _require(mapping: Dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required key: {key}")
    return mapping[key]


def _to_agent_specs(raw_agents: Sequence[Dict[str, Any]]) -> List[AgentSpec]:
    agents: List[AgentSpec] = []
    for i, entry in enumerate(raw_agents):
        if not isinstance(entry, dict):
            raise TypeError(f"agents[{i}] must be a mapping, got {type(entry)}")
        agents.append(
            AgentSpec(
                name=str(_require(entry, "name")),
                tool=str(_require(entry, "tool")),
                checkpoint=str(_require(entry, "checkpoint")),
            )
        )
    if not agents:
        raise ValueError("agents must not be empty")
    return agents


def _to_anatomy_spec(raw: Dict[str, Any]) -> AnatomySpec:
    typ = str(raw.get("type", "aortic_arch"))
    if typ != "aortic_arch":
        raise ValueError(f"Unsupported anatomy type: {typ} (only 'aortic_arch' for now)")

    return AorticArchSpec(
        arch_type=str(raw.get("arch_type", "I")),
        seed=int(raw.get("seed", 30)),
        rotation_yzx_deg=raw.get("rotation_yzx_deg"),
        scaling_xyzd=raw.get("scaling_xyzd"),
        omit_axis=raw.get("omit_axis"),
        target_mode=str(raw.get("target_mode", "branch_end")),
        target_branches=list(raw.get("target_branches", ["lcca"])),
        target_threshold_mm=float(raw.get("target_threshold_mm", 5.0)),
        image_frequency_hz=float(raw.get("image_frequency_hz", 7.5)),
        image_rot_zx_deg=tuple(raw.get("image_rot_zx_deg", (20.0, 5.0))),
        friction=float(raw.get("friction", 0.001)),
    )


def _to_scoring_config(raw: Optional[Dict[str, Any]]) -> ScoringConfig:
    if raw is None:
        return ScoringConfig()
    if not isinstance(raw, dict):
        raise TypeError(f"scoring must be a mapping, got {type(raw)}")

    mode = str(raw.get("mode", "default_v1"))
    if mode != "default_v1":
        raise ValueError(f"Unsupported scoring.mode: {mode} (only 'default_v1' for now)")

    return ScoringConfig(
        mode="default_v1",
        w_success=float(raw.get("w_success", 2.0)),
        w_efficiency=float(raw.get("w_efficiency", 1.0)),
        w_safety=float(raw.get("w_safety", 1.0)),
        w_smoothness=float(raw.get("w_smoothness", 0.25)),
        normalize_weights=bool(raw.get("normalize_weights", True)),
        force_scale=float(raw.get("force_scale", 1.0)),
        lcp_scale=float(raw.get("lcp_scale", 1.0)),
        speed_scale_mm_s=float(raw.get("speed_scale_mm_s", 50.0)),
    )


def config_from_dict(raw: Dict[str, Any]) -> EvaluationConfig:
    """Convert a plain dict (YAML/JSON/UI) into a validated EvaluationConfig."""

    agents = _to_agent_specs(_require(raw, "agents"))
    anatomy = _to_anatomy_spec(raw.get("anatomy", {}))
    scoring = _to_scoring_config(raw.get("scoring"))

    return EvaluationConfig(
        name=str(_require(raw, "name")),
        agents=agents,
        anatomy=anatomy,
        n_trials=int(raw.get("n_trials", 10)),
        base_seed=int(raw.get("base_seed", 123)),
        max_episode_steps=int(raw.get("max_episode_steps", 1000)),
        output_root=str(raw.get("output_root", "results/eval_runs")),
        policy_device=str(raw.get("policy_device", "cuda")),
        use_non_mp_sim=bool(raw.get("use_non_mp_sim", True)),
        scoring=scoring,
    )


def load_config(path: str | Path) -> EvaluationConfig:
    """Load an evaluation config from YAML (recommended) or JSON-like YAML."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Config root must be a mapping, got {type(raw)}")
    return config_from_dict(raw)
