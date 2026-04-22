from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
from third_party.stEVE.eve.env import Env
from third_party.stEVE.eve.info import (
    AverageTranslationSpeed,
    Combination as InfoCombination,
    Info,
    Steps,
    TargetReached as TargetReachedInfo,
    TrajectoryLength,
)
from third_party.stEVE.eve.observation import LastAction, ObsDict, Observation, Tracking2D
from third_party.stEVE.eve.observation.wrapper import (
    Memory,
    MemoryResetMode,
    Normalize,
    NormalizeTracking2DEpisode,
)
from third_party.stEVE.eve.pathfinder import BruteForceBFS
from third_party.stEVE.eve.reward import (
    Combination as RewardCombination,
    PathLengthDelta,
    Step,
    TargetReached as TargetReachedReward,
)
from third_party.stEVE.eve.start import InsertionPoint
from third_party.stEVE.eve.terminal import TargetReached as TargetReachedTerminal
from third_party.stEVE.eve.truncation import (
    Combination as TruncationCombination,
    MaxSteps,
    SimError,
    VesselEnd,
)
from third_party.stEVE.eve.util.coordtransform import tracking3d_to_2d

from .models import (
    ExecutionPlan,
    ForceTelemetrySummary,
    ScoringSpec,
    TrialArtifactPaths,
    TrialResult,
    TrialTelemetrySummary,
)
from .runtime import PreparedEvaluationRuntime, safe_reset_intervention
from .scoring import score_trial
from .visualization import TrialVisualisation, build_trial_visualisation


TRACKING_POINT_COUNT = 3
TRACKING_RESOLUTION_MM = 2.0
TRACKING_MEMORY_STEPS = 2


class TargetState2D(Observation):
    """Target observation compatible with branch and manual targets."""

    def __init__(self, runtime: PreparedEvaluationRuntime, name: str = "target2d") -> None:
        self.name = name
        self.runtime = runtime
        self.obs: np.ndarray | None = None

    @property
    def space(self):  # type: ignore[override]
        return self.runtime.intervention.fluoroscopy.tracking2d_space

    def step(self) -> None:
        target = self.runtime.intervention.target
        coordinates2d = getattr(target, "coordinates2d", None)
        if coordinates2d is None:
            coordinates2d = tracking3d_to_2d(target.coordinates3d)
        self.obs = np.asarray(coordinates2d, dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        _ = episode_nr
        self.step()


class SafePathRatio(Info):
    """Path-ratio info that tolerates zero initial path length."""

    def __init__(self, pathfinder: BruteForceBFS, name: str = "path_ratio") -> None:
        super().__init__(name)
        self.pathfinder = pathfinder
        self.initial_pathlength = 1.0

    @property
    def info(self) -> Dict[str, Any]:
        if not math.isfinite(self.initial_pathlength) or self.initial_pathlength <= 0.0:
            value = float("nan")
        else:
            value = 1.0 - (self.pathfinder.path_length / self.initial_pathlength)
        return {self.name: value}

    def reset(self, episode_nr: int = 0) -> None:
        _ = episode_nr
        self.initial_pathlength = float(self.pathfinder.path_length)


def build_single_trial_env(
    runtime: PreparedEvaluationRuntime,
    *,
    max_episode_steps: int,
    visualisation: TrialVisualisation | None = None,
) -> Env:
    """Build one clean-room stEVE `Env` around a prepared eval_v2 runtime."""

    intervention = runtime.intervention
    start = InsertionPoint(intervention)
    pathfinder = BruteForceBFS(intervention=intervention)

    tracking = Tracking2D(
        intervention,
        n_points=TRACKING_POINT_COUNT,
        resolution=TRACKING_RESOLUTION_MM,
    )
    tracking = NormalizeTracking2DEpisode(tracking, intervention)
    tracking = Memory(tracking, TRACKING_MEMORY_STEPS, MemoryResetMode.FILL)

    target_state = TargetState2D(runtime)
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

    target_reward = TargetReachedReward(
        intervention,
        factor=1.0,
        final_only_after_all_interim=False,
    )
    step_reward = Step(factor=-0.005)
    path_delta_reward = PathLengthDelta(pathfinder, 0.001)
    reward = RewardCombination([target_reward, path_delta_reward, step_reward])

    terminal = TargetReachedTerminal(intervention)
    truncation = TruncationCombination(
        [
            MaxSteps(max_episode_steps),
            VesselEnd(intervention),
            SimError(intervention),
        ]
    )

    info = InfoCombination(
        [
            TargetReachedInfo(intervention, name="success"),
            SafePathRatio(pathfinder, name="path_ratio"),
            Steps(name="steps"),
            AverageTranslationSpeed(
                intervention,
                name="average_translation_speed",
            ),
            TrajectoryLength(intervention, name="trajectory_length"),
        ]
    )

    return Env(
        intervention=intervention,
        observation=observation,
        reward=reward,
        terminal=terminal,
        truncation=truncation,
        info=info,
        start=start,
        pathfinder=pathfinder,
        interim_target=None,
        visualisation=visualisation,
    )


def _reset_single_trial_env(
    env: Env,
    *,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    episode_number = env.episode_number
    safe_reset_intervention(
        env.intervention,
        episode_number=episode_number,
        seed=seed,
    )
    env.start.reset(episode_number)
    env.pathfinder.reset(episode_number)
    env.interim_target.reset(episode_number)
    env.observation.reset(episode_number)
    env.reward.reset(episode_number)
    env.terminal.reset(episode_number)
    env.truncation.reset(episode_number)
    env.info.reset(episode_number)
    env.visualisation.reset(episode_number)
    env.episode_number += 1
    return deepcopy(env.observation()), deepcopy(env.info())


def _flatten_observation(obs: object) -> np.ndarray:
    if isinstance(obs, np.ndarray):
        return obs.flatten()
    if isinstance(obs, list):
        return np.concatenate([np.asarray(item).flatten() for item in obs], axis=0)
    if isinstance(obs, Mapping):
        return np.concatenate(
            [np.asarray(item).flatten() for item in obs.values()],
            axis=0,
        )
    raise TypeError(f"Unsupported observation type: {type(obs)!r}")


def _translation_speeds_from_action(
    action: np.ndarray,
    *,
    action_shape: Tuple[int, ...],
) -> List[float]:
    reshaped = np.asarray(action, dtype=np.float32).reshape(action_shape)
    return [float(abs(value)) for value in reshaped[:, 0]]


def _optional_finite(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _to_rgb_frame(
    frame: np.ndarray,
    *,
    source_is_bgr: bool,
) -> np.ndarray:
    """Normalize raw frame buffers to contiguous RGB uint8 arrays."""

    image = np.require(np.asarray(frame), requirements=["C"])
    if image.ndim == 2:
        gray = np.ascontiguousarray(image)
        if not np.issubdtype(gray.dtype, np.uint8):
            gray = np.nan_to_num(gray, nan=0.0, posinf=255.0, neginf=0.0)
            if np.issubdtype(gray.dtype, np.floating) and gray.size > 0 and float(np.max(gray)) <= 1.0:
                gray = gray * 255.0
            gray = np.clip(gray, 0.0, 255.0).astype(np.uint8)
        return np.repeat(gray[:, :, np.newaxis], 3, axis=2)

    if image.ndim != 3:
        raise ValueError(f"Unsupported frame shape: {image.shape!r}")

    channels = image.shape[2]
    if channels == 1:
        single = np.ascontiguousarray(image[:, :, 0])
        if not np.issubdtype(single.dtype, np.uint8):
            single = np.nan_to_num(single, nan=0.0, posinf=255.0, neginf=0.0)
            if np.issubdtype(single.dtype, np.floating) and single.size > 0 and float(np.max(single)) <= 1.0:
                single = single * 255.0
            single = np.clip(single, 0.0, 255.0).astype(np.uint8)
        return np.repeat(single[:, :, np.newaxis], 3, axis=2)

    rgb = np.ascontiguousarray(image[:, :, :3])
    if not np.issubdtype(rgb.dtype, np.uint8):
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
        if np.issubdtype(rgb.dtype, np.floating) and rgb.size > 0 and float(np.max(rgb)) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    if source_is_bgr:
        return np.ascontiguousarray(rgb[:, :, ::-1])
    return rgb


def _build_force_telemetry_summary(runtime: PreparedEvaluationRuntime) -> ForceTelemetrySummary:
    force_spec = runtime.scenario.force_telemetry
    if force_spec.required:
        raise NotImplementedError(
            "Force telemetry is marked as required, but the clean-room runner does not collect it yet"
        )
    if force_spec.mode != "passive":
        raise NotImplementedError(
            f"Force telemetry mode {force_spec.mode!r} is not implemented yet"
        )
    return ForceTelemetrySummary(
        available_for_score=False,
        validation_status="not_collected",
        source="eval_v2_runner",
        channel="none",
        quality_tier="unavailable",
    )


def _render_trial_if_enabled(
    env: Env,
    *,
    visualisation: TrialVisualisation | None,
) -> np.ndarray | None:
    if visualisation is not None:
        rendered = env.render()
        if rendered is not None:
            return np.asarray(rendered)
    return None


def _close_trial_visualisation(visualisation: TrialVisualisation | None) -> None:
    if visualisation is not None:
        visualisation.close()


def _select_action(
    runtime: PreparedEvaluationRuntime,
    *,
    flat_state: np.ndarray,
    execution: ExecutionPlan,
) -> np.ndarray:
    if execution.policy_mode == "deterministic":
        return np.asarray(runtime.play_policy.get_eval_action(flat_state), dtype=np.float32)
    exploration_action = getattr(runtime.play_policy, "get_exploration_action", None)
    if not callable(exploration_action):
        raise NotImplementedError(
            "Stochastic policy mode requires a play policy with get_exploration_action(...)"
        )
    return np.asarray(exploration_action(flat_state), dtype=np.float32)


def _to_env_action(
    action: np.ndarray,
    *,
    env: Env,
    normalize_action: bool,
) -> np.ndarray:
    """Convert policy output into the action representation expected by `env.step()`.

    When `normalize_action=True`, legacy checkpoints emit normalized actions in
    `[-1, 1]`. Mapping through `env.action_space` preserves compatibility with
    current stEVE behavior and becomes an identity transform if upstream stEVE
    exposes an already-normalized action space in the future.
    """

    env_action = np.asarray(action, dtype=np.float32).reshape(env.action_space.shape)
    if not normalize_action:
        return env_action
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    return (env_action + 1.0) / 2.0 * (action_high - action_low) + action_low


def run_single_trial(
    *,
    runtime: PreparedEvaluationRuntime,
    trial_index: int,
    seed: int,
    execution: ExecutionPlan,
    scoring: ScoringSpec,
    frame_callback: Optional[Callable[[np.ndarray], None]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> TrialResult:
    """Execute one candidate/scenario/seed rollout and normalize it to `TrialResult`."""

    visualisation = build_trial_visualisation(
        runtime,
        execution=execution,
        trial_index=trial_index,
        hidden_window=frame_callback is not None,
    )
    env = build_single_trial_env(
        runtime,
        max_episode_steps=execution.max_episode_steps,
        visualisation=visualisation,
    )
    try:
        observation, info = _reset_single_trial_env(env, seed=seed)
        runtime.play_policy.reset()
        if progress_callback is not None:
            progress_callback(
                f"trial_start index={trial_index} seed={seed} scenario={runtime.scenario.name} candidate={runtime.candidate.name}"
            )

        translation_speeds_mm_s: list[float] = []
        episode_reward = 0.0
        last_info = dict(info)
        steps_to_success: int | None = None
        wall_time_start = time.perf_counter()

        for step_index in range(execution.max_episode_steps):
            flat_state = _flatten_observation(observation)
            action = _select_action(runtime, flat_state=flat_state, execution=execution)
            env_action = _to_env_action(
                action,
                env=env,
                normalize_action=runtime.scenario.normalize_action,
            )
            translation_speeds_mm_s.extend(
                _translation_speeds_from_action(
                    env_action,
                    action_shape=runtime.intervention.velocity_limits.shape,
                )
            )
            observation, reward, terminated, truncated, info = env.step(env_action)
            rendered_frame = _render_trial_if_enabled(env, visualisation=visualisation)
            if frame_callback is not None:
                raw_frame = (
                    rendered_frame
                    if rendered_frame is not None
                    else np.asarray(env.intervention.fluoroscopy.image)
                )
                frame = _to_rgb_frame(
                    raw_frame,
                    source_is_bgr=rendered_frame is None,
                )
                frame_callback(np.copy(frame))
            episode_reward += float(reward)
            last_info = dict(info)
            if progress_callback is not None:
                progress_callback(
                    f"trial_step index={trial_index} step={step_index + 1} terminated={int(bool(terminated))} truncated={int(bool(truncated))}"
                )
            if terminated and steps_to_success is None:
                steps_to_success = step_index + 1
            if terminated or truncated:
                break

        wall_time_s = time.perf_counter() - wall_time_start
        steps_total = int(last_info.get("steps", len(translation_speeds_mm_s)))
        tip_speed_max_mm_s = (
            max(translation_speeds_mm_s) if translation_speeds_mm_s else 0.0
        )
        tip_speed_mean_mm_s = (
            float(np.mean(translation_speeds_mm_s)) if translation_speeds_mm_s else 0.0
        )
        telemetry = TrialTelemetrySummary(
            success=bool(last_info.get("success", steps_to_success is not None)),
            steps_total=steps_total,
            steps_to_success=steps_to_success,
            episode_reward=float(episode_reward),
            wall_time_s=wall_time_s,
            sim_time_s=float(steps_total) * runtime.scenario.action_dt_s,
            path_ratio_last=_optional_finite(last_info.get("path_ratio")),
            trajectory_length_last=_optional_finite(last_info.get("trajectory_length")),
            average_translation_speed_last=_optional_finite(
                last_info.get("average_translation_speed")
            ),
            tip_speed_max_mm_s=float(tip_speed_max_mm_s),
            tip_speed_mean_mm_s=float(tip_speed_mean_mm_s),
            forces=_build_force_telemetry_summary(runtime),
        )
        score = score_trial(
            telemetry=telemetry,
            max_episode_steps=execution.max_episode_steps,
            scoring=scoring,
        )
        if progress_callback is not None:
            progress_callback(
                f"trial_end index={trial_index} success={int(bool(telemetry.success))} steps={telemetry.steps_total}"
            )
        return TrialResult(
            scenario_name=runtime.scenario.name,
            candidate_name=runtime.candidate.name,
            execution_wire=runtime.candidate.execution_wire,
            policy=runtime.candidate.policy,
            trial_index=trial_index,
            seed=seed,
            score=score,
            telemetry=telemetry,
            artifacts=TrialArtifactPaths(),
        )
    finally:
        _close_trial_visualisation(visualisation)


__all__ = ["build_single_trial_env", "run_single_trial"]
