"""Reward factory for train_v2."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from eve.reward import PathLengthDelta, Step, TargetReached

from ..config import RewardSpec
from ..telemetry.force_runtime import ForceRuntime
from .force import ExcessForcePenaltyReward, ForcePenaltyReward
from .tracker import RewardTracker


def build_reward(
    *,
    intervention,
    pathfinder,
    reward_spec: RewardSpec,
    telemetry: Optional[ForceRuntime] = None,
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

    if reward_spec.profile == "default_plus_force_penalty":
        if telemetry is None:
            raise ValueError(
                "telemetry is required for profile=default_plus_force_penalty"
            )
        components.append(
            (
                "force",
                ForcePenaltyReward(
                    intervention=intervention,
                    telemetry=telemetry,
                    factor=reward_spec.force_penalty_factor,
                    tip_only=reward_spec.force_tip_only,
                ),
            )
        )
    elif reward_spec.profile == "default_plus_excess_force_penalty":
        if telemetry is None:
            raise ValueError(
                "telemetry is required for profile=default_plus_excess_force_penalty"
            )
        components.append(
            (
                "force",
                ExcessForcePenaltyReward(
                    intervention=intervention,
                    telemetry=telemetry,
                    threshold_N=reward_spec.force_threshold_N,
                    divisor=reward_spec.force_divisor,
                    tip_only=reward_spec.force_tip_only,
                ),
            )
        )

    return RewardTracker(components, csv_path=csv_path)
