from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from steve_recommender.evaluation.config import AorticArchSpec, ScoringConfig


@dataclass(frozen=True)
class ComparisonCandidateSpec:
    """One comparison candidate.

    A candidate can be expressed in two ways:
    - registry-driven via ``agent_ref`` (`model/wire:agent`)
    - explicit via ``tool`` + ``checkpoint``
    """

    name: Optional[str] = None
    agent_ref: Optional[str] = None
    tool: Optional[str] = None
    checkpoint: Optional[str] = None
    checkpoint_override: Optional[str] = None


@dataclass(frozen=True)
class ResolvedCandidate:
    """Resolved tool/checkpoint pair that is ready for evaluation."""

    name: str
    tool: str
    checkpoint: Path
    agent_ref: Optional[str]
    source: str


@dataclass(frozen=True)
class ComparisonConfig:
    """Top-level comparison configuration used by CLI and UI."""

    name: str
    candidates: List[ComparisonCandidateSpec]
    anatomy: AorticArchSpec = field(default_factory=AorticArchSpec)
    n_trials: int = 10
    base_seed: int = 123
    seeds: Optional[List[int]] = None
    max_episode_steps: int = 1000
    output_root: str = "results/eval_runs"
    policy_device: str = "cuda"
    use_non_mp_sim: bool = True
    stochastic_eval: bool = False
    visualize: bool = False
    visualize_trials_per_agent: int = 1
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


def _require(mapping: Dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required key: {key}")
    return mapping[key]


def _to_anatomy_spec(raw: Dict[str, Any]) -> AorticArchSpec:
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


def _to_candidates(raw_candidates: Any) -> List[ComparisonCandidateSpec]:
    if not isinstance(raw_candidates, list):
        raise TypeError(f"candidates must be a list, got {type(raw_candidates)}")

    candidates: List[ComparisonCandidateSpec] = []
    for i, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            raise TypeError(f"candidates[{i}] must be a mapping, got {type(raw)}")

        spec = ComparisonCandidateSpec(
            name=str(raw["name"]) if "name" in raw and raw["name"] is not None else None,
            agent_ref=str(raw["agent_ref"]) if raw.get("agent_ref") else None,
            tool=str(raw["tool"]) if raw.get("tool") else None,
            checkpoint=str(raw["checkpoint"]) if raw.get("checkpoint") else None,
            checkpoint_override=(
                str(raw["checkpoint_override"])
                if raw.get("checkpoint_override")
                else None
            ),
        )

        if not spec.agent_ref and not (spec.tool and spec.checkpoint):
            raise ValueError(
                f"candidates[{i}] must provide either agent_ref or tool+checkpoint"
            )
        candidates.append(spec)

    if not candidates:
        raise ValueError("candidates must not be empty")
    return candidates


def comparison_config_from_dict(raw: Dict[str, Any]) -> ComparisonConfig:
    """Convert a plain dict (YAML/JSON/UI) into a validated ComparisonConfig."""

    candidates = _to_candidates(_require(raw, "candidates"))
    anatomy = _to_anatomy_spec(raw.get("anatomy", {}))
    scoring = _to_scoring_config(raw.get("scoring"))

    seeds_raw = raw.get("seeds")
    seeds: Optional[List[int]]
    if seeds_raw is None:
        seeds = None
    elif isinstance(seeds_raw, list):
        seeds = [int(s) for s in seeds_raw]
    elif isinstance(seeds_raw, str):
        parsed = [s.strip() for s in seeds_raw.split(",") if s.strip()]
        seeds = [int(s) for s in parsed]
    else:
        raise TypeError(f"seeds must be list[str/int], comma-string or null, got {type(seeds_raw)}")

    return ComparisonConfig(
        name=str(_require(raw, "name")),
        candidates=candidates,
        anatomy=anatomy,
        n_trials=int(raw.get("n_trials", 10)),
        base_seed=int(raw.get("base_seed", 123)),
        seeds=seeds,
        max_episode_steps=int(raw.get("max_episode_steps", 1000)),
        output_root=str(raw.get("output_root", "results/eval_runs")),
        policy_device=str(raw.get("policy_device", "cuda")),
        use_non_mp_sim=bool(raw.get("use_non_mp_sim", True)),
        stochastic_eval=bool(raw.get("stochastic_eval", False)),
        visualize=bool(raw.get("visualize", False)),
        visualize_trials_per_agent=int(raw.get("visualize_trials_per_agent", 1)),
        scoring=scoring,
    )


def load_comparison_config(path: str | Path) -> ComparisonConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Config root must be a mapping, got {type(raw)}")
    return comparison_config_from_dict(raw)
