"""Default reward stack construction for train_v2."""

from __future__ import annotations

from eve.reward import (
    Combination,
    PathLengthDelta,
    Step,
    TargetReached,
)

from ..config import RewardSpec


def build_default_reward(
    *,
    intervention,
    pathfinder,
    reward_spec: RewardSpec,
    extras: list | None = None,
):
    """Build the default ArchVar-style reward combination."""

    rewards = [
        TargetReached(
            intervention,
            factor=reward_spec.target_reached_factor,
            final_only_after_all_interim=False,
        ),
        PathLengthDelta(pathfinder, reward_spec.path_delta_factor),
        Step(factor=reward_spec.step_factor),
    ]
    if extras:
        rewards.extend(extras)
    return Combination(rewards)
