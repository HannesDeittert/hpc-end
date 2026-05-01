"""Local environment factory for train_v2."""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
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

from ..config import RewardSpec
from ..rewards.factory import build_reward
from ..rewards.tracker import RewardComponentInfo, RewardTracker
from ..telemetry.force_runtime import ForceRuntime
from ..telemetry.step_trace import StepTraceRecorder

TRACKING_POINT_COUNT = 3
TRACKING_RESOLUTION_MM = 2
TRACKING_MEMORY_STEPS = 2
logger = logging.getLogger(__name__)


class TrainV2Env(Env):
    """Local env wrapper with reward evaluated after terminal/truncation."""

    def __init__(
        self,
        *args,
        step_trace: Optional[StepTraceRecorder] = None,
        force_telemetry: Optional[ForceRuntime] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._step_trace = step_trace
        self._force_telemetry = force_telemetry

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.intervention.step(action)
        self.pathfinder.step()
        self.interim_target.step()
        self.observation.step()
        self.terminal.step()
        self.truncation.step()
        self.reward.step()
        self.info.step()
        if self._step_trace is not None and isinstance(self.reward, RewardTracker):
            self._step_trace.record_step(
                step_index=int(self.reward._step_count - 1),
                terminated=bool(self.terminal.terminal),
                truncated=bool(self.truncation.truncated),
                reward_snapshot=self.reward.debug_snapshot(),
                info_snapshot=dict(self.info.info),
            )
        return (
            deepcopy(self.observation()),
            deepcopy(self.reward.reward),
            deepcopy(self.terminal.terminal),
            deepcopy(self.truncation.truncated),
            deepcopy(self.info.info),
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self._step_trace is not None:
            self._step_trace.reset(episode_nr=int(self.episode_number))
        result = super().reset(seed=seed, options=options)
        if self._force_telemetry is not None:
            status = self._force_telemetry.ensure_runtime(
                intervention=self.intervention
            )
            if not status.configured:
                raise RuntimeError(
                    "train_v2 force telemetry failed to bind after env.reset: "
                    f"{status.source} ({status.error})"
                )
        return result

    def close(self):
        if self._step_trace is not None:
            self._step_trace.close()
        super().close()


def build_env(
    *,
    intervention,
    reward_spec: RewardSpec,
    mode: str,
    n_max_steps: int = 1000,
    reward_csv_path: Optional[Path] = None,
    track_reward_components: bool = False,
    step_trace_path: Optional[Path] = None,
    step_trace_every_n_steps: int = 10,
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
    terminal = TargetReachedTerminal(intervention)
    max_steps = MaxSteps(n_max_steps)
    vessel_end = VesselEnd(intervention)
    sim_error = SimError(intervention)
    truncation = (
        TruncationCombination([max_steps, vessel_end, sim_error])
        if mode == "train"
        else max_steps
    )

    needs_direct_simulation = reward_spec.profile == "default_plus_normal_force_penalty"
    if needs_direct_simulation:
        telemetry = ForceRuntime(
            reward_spec=reward_spec,
            action_dt_s=1.0 / 7.5,
        )
    reward = build_reward(
        intervention=intervention,
        pathfinder=pathfinder,
        reward_spec=reward_spec,
        telemetry=telemetry,
        terminal=terminal,
        truncation=truncation,
        csv_path=reward_csv_path,
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

    step_trace = None
    if step_trace_path is not None:
        step_trace = StepTraceRecorder(
            base_path=step_trace_path,
            mode=mode,
            every_n_steps=step_trace_every_n_steps,
        )

    if needs_direct_simulation:
        logger.info(
            "[train_v2 env] using direct simulation because force telemetry needs a live Sofa root"
        )
        intervention.make_non_mp()
    else:
        intervention.make_mp()
    return TrainV2Env(
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
        step_trace=step_trace,
        force_telemetry=telemetry,
    )
