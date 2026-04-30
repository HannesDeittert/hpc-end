"""Local environment factory for train_v2."""

from __future__ import annotations

from eve.env import Env
from eve.info import (
    AverageTranslationSpeed,
    Combination as InfoCombination,
    PathRatio,
    Steps,
    TargetReached as TargetReachedInfo,
    TrajectoryLength,
)
from eve.observation import LastAction, ObsDict, Target2D, Tracking2D
from eve.observation.wrapper import (
    Memory,
    MemoryResetMode,
    Normalize,
    NormalizeTracking2DEpisode,
)
from eve.pathfinder import BruteForceBFS
from eve.start import InsertionPoint
from eve.terminal import TargetReached as TargetReachedTerminal
from eve.truncation import (
    Combination as TruncationCombination,
    MaxSteps,
    SimError,
    VesselEnd,
)

from pathlib import Path
from typing import Optional

from ..config import RewardSpec
from ..rewards.factory import build_reward
from ..rewards.tracker import RewardComponentInfo
from ..telemetry.force_runtime import ForceRuntime

TRACKING_POINT_COUNT = 3
TRACKING_RESOLUTION_MM = 2
TRACKING_MEMORY_STEPS = 2


def build_env(
    *,
    intervention,
    reward_spec: RewardSpec,
    mode: str,
    n_max_steps: int = 1000,
    reward_csv_path: Optional[Path] = None,
    track_reward_components: bool = False,
):
    """Build one train_v2 env with injectable reward construction."""

    start = InsertionPoint(intervention)
    pathfinder = BruteForceBFS(intervention=intervention)

    tracking = Tracking2D(
        intervention,
        n_points=TRACKING_POINT_COUNT,
        resolution=TRACKING_RESOLUTION_MM,
    )
    tracking = NormalizeTracking2DEpisode(tracking, intervention)
    tracking = Memory(tracking, TRACKING_MEMORY_STEPS, MemoryResetMode.FILL)
    target_state = Target2D(intervention)
    target_state = NormalizeTracking2DEpisode(target_state, intervention)
    last_action = LastAction(intervention)
    last_action = Normalize(last_action)
    observation = ObsDict(
        {
            "tracking": tracking,
            "target": target_state,
            "last_action": last_action,
        }
    )

    telemetry = None
    if reward_spec.profile in (
        "default_plus_force_penalty",
        "default_plus_excess_force_penalty",
    ):
        telemetry = ForceRuntime(
            reward_spec=reward_spec,
            action_dt_s=1.0 / 7.5,
        )
        telemetry.ensure_runtime(intervention=intervention)
    reward = build_reward(
        intervention=intervention,
        pathfinder=pathfinder,
        reward_spec=reward_spec,
        telemetry=telemetry,
        csv_path=reward_csv_path,
    )

    terminal = TargetReachedTerminal(intervention)
    max_steps = MaxSteps(n_max_steps)
    vessel_end = VesselEnd(intervention)
    sim_error = SimError(intervention)
    truncation = (
        TruncationCombination([max_steps, vessel_end, sim_error])
        if mode == "train"
        else max_steps
    )

    info_components = [
        TargetReachedInfo(intervention, name="success"),
        PathRatio(pathfinder),
        Steps(),
        AverageTranslationSpeed(intervention),
        TrajectoryLength(intervention),
    ]
    if track_reward_components:
        info_components.append(RewardComponentInfo(reward))
    info = InfoCombination(info_components)

    intervention.make_mp()
    return Env(
        intervention,
        observation,
        reward,
        terminal,
        truncation=truncation,
        start=start,
        pathfinder=pathfinder,
        visualisation=None,
        info=info,
        interim_target=None,
    )
