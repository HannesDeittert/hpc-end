"""Reward factory for train_v2."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from eve.reward import PathLengthDelta, Step, TargetReached

from ..config import RewardSpec
from ..telemetry.force_runtime import ForceRuntime
from .force import NormalForcePenaltyReward
from .tracker import RewardTracker


def build_reward(
    *,
    intervention,
    pathfinder,
    reward_spec: RewardSpec,
    telemetry: Optional[ForceRuntime] = None,
    terminal=None,
    truncation=None,
    csv_path: Optional[Path] = None,
) -> RewardTracker:
    """Build a tracked reward combination from the local train_v2 reward spec."""

    components = [
        (
            "target",
            TargetReached(
                intervention,
                factor=reward_spec.target_reached_factor,
                final_only_after_all_interim=False,
            ),
        ),
        ("path_delta", PathLengthDelta(pathfinder, reward_spec.path_delta_factor)),
        ("step", Step(factor=reward_spec.step_factor)),
    ]

    if reward_spec.profile == "default_plus_normal_force_penalty":
        if telemetry is None:
            raise ValueError(
                "telemetry is required for profile=default_plus_normal_force_penalty"
            )
        if terminal is None or truncation is None:
            raise ValueError(
                "terminal and truncation are required for profile=default_plus_normal_force_penalty"
            )
        components.append(
            (
                "force",
                NormalForcePenaltyReward(
                    intervention=intervention,
                    telemetry=telemetry,
                    terminal=terminal,
                    truncation=truncation,
                    alpha=reward_spec.force_alpha,
                    beta=reward_spec.force_beta,
                    force_region=reward_spec.force_region,
                ),
            )
        )

    return RewardTracker(components, csv_path=csv_path)
