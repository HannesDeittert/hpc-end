from __future__ import annotations

import math
import random
import time
from copy import deepcopy
import json
import logging
from pathlib import Path
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
from third_party.stEVE.eve.observation import (
    LastAction,
    ObsDict,
    Observation,
    Tracking2D,
)
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

from .force_telemetry import EvalV2ForceTelemetryCollector
from .force_trace_persistence import (
    SceneStaticState,
    ScenarioConfig,
    StepData,
    TriangleContactRecord,
    TrialTraceRecorder,
    WireContactRecord,
)
from .models import (
    ExecutionPlan,
    ScoringSpec,
    TrialArtifactPaths,
    TrialResult,
    TrialTelemetrySummary,
)
from .runtime import PreparedEvaluationRuntime, safe_reset_intervention
from .scoring import force_within_safety_threshold, score_trial, valid_for_ranking
from .visualization import TrialVisualisation, build_trial_visualisation


TRACKING_POINT_COUNT = 3
TRACKING_RESOLUTION_MM = 2.0
TRACKING_MEMORY_STEPS = 2
IDENTITY_QUATERNION = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
logger = logging.getLogger(__name__)


def configure_cpu_eval_threads(thread_count: int = 1) -> None:
    """Keep CPU eval rollouts comparable across serial and worker processes."""

    try:
        import torch

        torch.set_num_threads(max(1, int(thread_count)))
    except Exception:
        pass


def _seed_trial_random_generators(seed: int) -> None:
    normalized_seed = int(seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed % (2**32))
    try:
        import torch

        torch.manual_seed(normalized_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(normalized_seed)
    except Exception:
        pass


def _capture_trial_random_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return state


def _restore_trial_random_state(state: Mapping[str, Any]) -> None:
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    try:
        import torch

        torch_cpu_state = state.get("torch_cpu")
        if torch_cpu_state is not None:
            torch.set_rng_state(torch_cpu_state)
        torch_cuda_state = state.get("torch_cuda")
        if torch_cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(torch_cuda_state)
    except Exception:
        pass


def _build_seeded_random_state(seed: int) -> dict[str, Any]:
    baseline = _capture_trial_random_state()
    try:
        _seed_trial_random_generators(seed)
        return _capture_trial_random_state()
    finally:
        _restore_trial_random_state(baseline)


def _reset_play_policy(play_policy: Any) -> None:
    """Reset play policy state, including recurrent components hidden by SAC wrappers."""

    reset = getattr(play_policy, "reset", None)
    if callable(reset):
        reset()

    policy = getattr(getattr(play_policy, "model", None), "policy", None)
    for component_name in ("head", "body"):
        component_reset = getattr(getattr(policy, component_name, None), "reset", None)
        if callable(component_reset):
            component_reset()


class TargetState2D(Observation):
    """Target observation compatible with branch and manual targets."""

    def __init__(
        self, runtime: PreparedEvaluationRuntime, name: str = "target2d"
    ) -> None:
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


def _trial_end_reason(env: Env, *, terminated: bool, truncated: bool) -> str:
    if terminated:
        return "target_reached"
    reasons: list[str] = []
    if truncated:
        truncations = getattr(env.truncation, "truncations", (env.truncation,))
        for truncation in truncations:
            try:
                is_active = bool(truncation.truncated)
            except Exception:
                is_active = False
            if not is_active:
                continue
            if isinstance(truncation, MaxSteps):
                reasons.append("max_steps")
            elif isinstance(truncation, VesselEnd):
                reasons.append("vessel_end")
            elif isinstance(truncation, SimError):
                reasons.append("simulation_error")
            else:
                reasons.append(type(truncation).__name__)
    if reasons:
        return "+".join(dict.fromkeys(reasons))
    return "loop_exhausted"


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
            if (
                np.issubdtype(gray.dtype, np.floating)
                and gray.size > 0
                and float(np.max(gray)) <= 1.0
            ):
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
            if (
                np.issubdtype(single.dtype, np.floating)
                and single.size > 0
                and float(np.max(single)) <= 1.0
            ):
                single = single * 255.0
            single = np.clip(single, 0.0, 255.0).astype(np.uint8)
        return np.repeat(single[:, :, np.newaxis], 3, axis=2)

    rgb = np.ascontiguousarray(image[:, :, :3])
    if not np.issubdtype(rgb.dtype, np.uint8):
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
        if (
            np.issubdtype(rgb.dtype, np.floating)
            and rgb.size > 0
            and float(np.max(rgb)) <= 1.0
        ):
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    if source_is_bgr:
        return np.ascontiguousarray(rgb[:, :, ::-1])
    return rgb


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
        return np.asarray(
            runtime.play_policy.get_eval_action(flat_state), dtype=np.float32
        )
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


def _trial_trace_component(value: object) -> str:
    text = str(value)
    sanitized = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in text
    )
    return sanitized.strip("_") or "value"


def _trial_trace_path(
    *,
    output_dir: Path,
    candidate_name: str,
    env_seed: int,
    policy_seed: int | None,
) -> Path:
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    policy_seed_component = "none" if policy_seed is None else str(policy_seed)
    return traces_dir / (
        "trial_{candidate}_{env_seed}_{policy_seed}.h5".format(
            candidate=_trial_trace_component(candidate_name),
            env_seed=env_seed,
            policy_seed=_trial_trace_component(policy_seed_component),
        )
    )


def _read_collision_positions_mm(intervention: Any) -> np.ndarray:
    simulation = getattr(intervention, "simulation", None)
    root = getattr(simulation, "root", None)
    collision_obj = None
    if root is not None:
        try:
            collision_obj = getattr(
                root, "InstrumentCombined"
            ).CollisionModel.CollisionDOFs
        except Exception:
            collision_obj = getattr(root, "CollisionDOFs", None)
    if collision_obj is not None:
        position_data = getattr(getattr(collision_obj, "position", None), "value", None)
        if position_data is not None:
            collision_positions = np.asarray(position_data, dtype=np.float32).reshape(
                (-1, 3)
            )
            if collision_positions.size > 0:
                return collision_positions
    return np.asarray(intervention.simulation.dof_positions, dtype=np.float32).reshape(
        (-1, 3)
    )


def _distal_tip_position_mm(intervention: Any) -> np.ndarray:
    try:
        positions = np.asarray(
            intervention.simulation.dof_positions, dtype=np.float32
        ).reshape((-1, 3))
    except AttributeError:
        return np.zeros((3,), dtype=np.float32)
    if positions.shape[0] == 0:
        return np.zeros((3,), dtype=np.float32)
    return np.asarray(positions[-1], dtype=np.float32)


def _tip_motion_metrics(
    tip_positions_mm: list[np.ndarray],
    *,
    dt_s: float,
) -> dict[str, float | None]:
    if not tip_positions_mm:
        return {
            "tip_total_distance_mm": None,
            "tip_speed_max_mm_s": None,
            "tip_speed_mean_mm_s": None,
            "tip_acc_p95": None,
            "tip_acc_max": None,
            "tip_jerk_p95": None,
            "tip_jerk_max": None,
        }
    positions = np.asarray(tip_positions_mm, dtype=np.float64).reshape((-1, 3))
    if positions.shape[0] == 1:
        return {
            "tip_total_distance_mm": 0.0,
            "tip_speed_max_mm_s": 0.0,
            "tip_speed_mean_mm_s": 0.0,
            "tip_acc_p95": None,
            "tip_acc_max": None,
            "tip_jerk_p95": None,
            "tip_jerk_max": None,
        }
    dt = float(max(dt_s, 1e-9))
    deltas = np.diff(positions, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    velocities = deltas / dt
    speed_norms = np.linalg.norm(velocities, axis=1)
    accelerations = np.diff(velocities, axis=0) / dt if velocities.shape[0] >= 2 else np.zeros((0, 3), dtype=np.float64)
    acc_norms = np.linalg.norm(accelerations, axis=1)
    jerks = np.diff(accelerations, axis=0) / dt if accelerations.shape[0] >= 2 else np.zeros((0, 3), dtype=np.float64)
    jerk_norms = np.linalg.norm(jerks, axis=1)
    return {
        "tip_total_distance_mm": float(np.sum(distances)),
        "tip_speed_max_mm_s": float(np.max(speed_norms)) if speed_norms.size else 0.0,
        "tip_speed_mean_mm_s": float(np.mean(speed_norms)) if speed_norms.size else 0.0,
        "tip_acc_p95": float(np.percentile(acc_norms, 95.0)) if acc_norms.size else None,
        "tip_acc_max": float(np.max(acc_norms)) if acc_norms.size else None,
        "tip_jerk_p95": float(np.percentile(jerk_norms, 95.0)) if jerk_norms.size else None,
        "tip_jerk_max": float(np.max(jerk_norms)) if jerk_norms.size else None,
    }


def _triangle_contacts_from_records(
    records: List[dict[str, Any]],
) -> tuple[TriangleContactRecord, ...]:
    return tuple(
        TriangleContactRecord(
            timestep=int(record.get("timestep", 0)),
            triangle_id=int(record.get("triangle_id", -1)),
            fx_N=float(record.get("fx_N", 0.0)),
            fy_N=float(record.get("fy_N", 0.0)),
            fz_N=float(record.get("fz_N", 0.0)),
            norm_N=float(record.get("norm_N", 0.0)),
            contributing_rows=int(record.get("contributing_rows", 0)),
            mapped=bool(record.get("mapped", True)),
        )
        for record in records
    )


def _wire_contacts_from_records(
    records: List[dict[str, Any]],
) -> tuple[WireContactRecord, ...]:
    return tuple(
        WireContactRecord(
            timestep=int(record.get("timestep", 0)),
            wire_collision_dof=int(record.get("wire_collision_dof", -1)),
            row_idx=int(record.get("row_idx", -1)),
            fx_N=float(record.get("fx_N", 0.0)),
            fy_N=float(record.get("fy_N", 0.0)),
            fz_N=float(record.get("fz_N", 0.0)),
            norm_N=float(record.get("norm_N", 0.0)),
            arc_length_from_distal_mm=record.get("arc_length_from_distal_mm"),
            is_tip=bool(record.get("is_tip", False)),
            world_pos=record.get("world_pos"),
            fx_scene=record.get("fx_scene"),
            fy_scene=record.get("fy_scene"),
            fz_scene=record.get("fz_scene"),
            norm_scene=record.get("norm_scene"),
            mapped=bool(record.get("mapped", True)),
        )
        for record in records
    )


def _force_norm_from_contacts(
    wire_contacts: tuple[WireContactRecord, ...],
    triangle_contacts: tuple[TriangleContactRecord, ...],
) -> float:
    if triangle_contacts:
        total_vector = np.asarray(
            [[record.fx_N, record.fy_N, record.fz_N] for record in triangle_contacts],
            dtype=np.float32,
        ).sum(axis=0)
    elif wire_contacts:
        total_vector = np.asarray(
            [[record.fx_N, record.fy_N, record.fz_N] for record in wire_contacts],
            dtype=np.float32,
        ).sum(axis=0)
    else:
        total_vector = np.zeros((3,), dtype=np.float32)
    return float(np.linalg.norm(total_vector))


def _tip_force_norm_from_wire_contacts(
    wire_contacts: tuple[WireContactRecord, ...],
) -> float:
    tip_records = tuple(record for record in wire_contacts if record.is_tip)
    if not tip_records:
        return 0.0
    total_vector = np.asarray(
        [[record.fx_N, record.fy_N, record.fz_N] for record in tip_records],
        dtype=np.float32,
    ).sum(axis=0)
    return float(np.linalg.norm(total_vector))


def _build_trial_trace_scenario(
    *,
    runtime: PreparedEvaluationRuntime,
    seed: int,
    policy_seed: int | None,
    execution: ExecutionPlan,
) -> ScenarioConfig:
    anatomy = runtime.scenario.anatomy
    anatomy_id = anatomy.record_id or runtime.scenario.name
    return ScenarioConfig(
        anatomy_id=anatomy_id,
        wire_id=runtime.candidate.execution_wire.tool_ref,
        target_spec_json=json.dumps(
            runtime.scenario.target.__dict__, sort_keys=True, default=str
        ),
        env_seed=seed,
        policy_seed=policy_seed,
        dt_s=runtime.scenario.action_dt_s,
        friction_mu=runtime.scenario.friction,
        tip_threshold_mm=runtime.scenario.force_telemetry.tip_threshold_mm,
        max_episode_steps=execution.max_episode_steps,
        mesh_ref=f"../meshes/anatomy_{anatomy_id}.h5",
        eval_v2_sha="unknown",
        sofa_version="unknown",
    )


def _build_scene_static_state(runtime: PreparedEvaluationRuntime) -> SceneStaticState:
    return SceneStaticState(
        wire_initial_position_mm=np.asarray(
            runtime.intervention.simulation.dof_positions, dtype=np.float32
        ).reshape((-1, 3)),
        wire_initial_rotation_quat=np.array(IDENTITY_QUATERNION, copy=True),
    )


def run_single_trial(
    *,
    runtime: PreparedEvaluationRuntime,
    trial_index: int,
    seed: int,
    execution: ExecutionPlan,
    scoring: ScoringSpec,
    output_dir: Optional[Path] = None,
    frame_callback: Optional[Callable[[np.ndarray], None]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> TrialResult:
    """Execute one candidate/scenario/seed rollout and normalize it to `TrialResult`.

    `seed` remains the environment seed for backward compatibility. When the
    execution plan uses stochastic policy sampling, the policy RNG is reseeded
    after the environment reset with the trial-specific policy seed so callers
    can independently control initial conditions and action sampling.
    """

    if str(getattr(runtime.play_policy, "device", "")).lower() == "cpu":
        configure_cpu_eval_threads(1)
    _seed_trial_random_generators(seed)
    policy_seed = execution.policy_seed_for_trial(trial_index)

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
        _reset_play_policy(runtime.play_policy)
        trial_warnings: list[str] = []
        policy_rng_state = (
            None if policy_seed is None else _build_seeded_random_state(policy_seed)
        )
        force_collector = EvalV2ForceTelemetryCollector(
            spec=runtime.scenario.force_telemetry,
            action_dt_s=runtime.scenario.action_dt_s,
            anatomy_mesh_path=runtime.scenario.anatomy.simulation_mesh_path,
        )
        force_status = force_collector.ensure_runtime(intervention=runtime.intervention)
        if runtime.scenario.force_telemetry.required and not force_status.configured:
            raise RuntimeError(
                "Force telemetry is required but could not be configured: "
                f"{force_status.source} ({force_status.error})"
            )
        recorder: TrialTraceRecorder | None = None
        trace_path: Path | None = None
        if runtime.scenario.force_telemetry.write_full_trace and output_dir is not None:
            trace_path = _trial_trace_path(
                output_dir=Path(output_dir),
                candidate_name=runtime.candidate.name,
                env_seed=seed,
                policy_seed=policy_seed,
            )
            try:
                recorder = TrialTraceRecorder(
                    path=trace_path,
                    scenario=_build_trial_trace_scenario(
                        runtime=runtime,
                        seed=seed,
                        policy_seed=policy_seed,
                        execution=execution,
                    ),
                    scene_static=_build_scene_static_state(runtime),
                )
                recorder.__enter__()
            except Exception as exc:
                warning = (
                    f"trial_trace_warning trial={trial_index} "
                    f"candidate={runtime.candidate.name} "
                    f"scenario={runtime.scenario.name} error={exc}"
                )
                logger.warning(warning)
                trial_warnings.append(warning)
                recorder = None

        if progress_callback is not None:
            progress_callback(
                f"trial_start index={trial_index} seed={seed} scenario={runtime.scenario.name} candidate={runtime.candidate.name}"
            )

        translation_speeds_mm_s: list[float] = []
        tip_positions_mm: list[np.ndarray] = [_distal_tip_position_mm(runtime.intervention)]
        episode_reward = 0.0
        last_info = dict(info)
        steps_to_success: int | None = None
        last_terminated = False
        last_truncated = False
        wall_time_start = time.perf_counter()
        wire_record_cursor = 0
        triangle_record_cursor = 0

        for step_index in range(execution.max_episode_steps):
            flat_state = _flatten_observation(observation)
            if policy_rng_state is None:
                action = _select_action(
                    runtime, flat_state=flat_state, execution=execution
                )
            else:
                rollout_rng_state = _capture_trial_random_state()
                _restore_trial_random_state(policy_rng_state)
                try:
                    action = _select_action(
                        runtime, flat_state=flat_state, execution=execution
                    )
                    policy_rng_state = _capture_trial_random_state()
                finally:
                    _restore_trial_random_state(rollout_rng_state)
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
            last_terminated = bool(terminated)
            last_truncated = bool(truncated)
            tip_positions_mm.append(_distal_tip_position_mm(runtime.intervention))
            force_collector.capture_step(
                intervention=runtime.intervention,
                step_index=step_index + 1,
            )
            if recorder is not None:
                wire_records_raw = list(
                    force_collector._wire_force_records[wire_record_cursor:]
                )
                triangle_records_raw = list(
                    force_collector._triangle_force_records[triangle_record_cursor:]
                )
                wire_record_cursor = len(force_collector._wire_force_records)
                triangle_record_cursor = len(force_collector._triangle_force_records)
                wire_contacts = _wire_contacts_from_records(wire_records_raw)
                triangle_contacts = _triangle_contacts_from_records(
                    triangle_records_raw
                )
                try:
                    recorder.add_step(
                        StepData(
                            step_index=step_index,
                            sim_time_s=float(
                                (step_index + 1) * runtime.scenario.action_dt_s
                            ),
                            wire_positions_mm=np.asarray(
                                runtime.intervention.simulation.dof_positions,
                                dtype=np.float32,
                            ).reshape((-1, 3)),
                            wire_collision_positions_mm=_read_collision_positions_mm(
                                runtime.intervention
                            ),
                            action=np.asarray(env_action, dtype=np.float32).reshape(
                                (-1,)
                            ),
                            total_wall_force_N=_force_norm_from_contacts(
                                wire_contacts,
                                triangle_contacts,
                            ),
                            tip_force_norm_N=_tip_force_norm_from_wire_contacts(
                                wire_contacts
                            ),
                            contact_count=len(wire_contacts) + len(triangle_contacts),
                            scoreable=bool(force_status.configured),
                            wire_contacts=wire_contacts,
                            triangle_contacts=triangle_contacts,
                        )
                    )
                except Exception as exc:
                    warning = (
                        f"trial_trace_warning trial={trial_index} "
                        f"candidate={runtime.candidate.name} "
                        f"scenario={runtime.scenario.name} error={exc}"
                    )
                    logger.warning(warning)
                    trial_warnings.append(warning)
                    recorder.__exit__(type(exc), exc, exc.__traceback__)
                    recorder = None
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
        tip_motion = _tip_motion_metrics(
            tip_positions_mm,
            dt_s=runtime.scenario.action_dt_s,
        )
        telemetry = TrialTelemetrySummary(
            success=bool(last_info.get("success", steps_to_success is not None)),
            steps_total=steps_total,
            steps_to_success=steps_to_success,
            episode_reward=float(episode_reward),
            end_reason=_trial_end_reason(
                env,
                terminated=last_terminated,
                truncated=last_truncated,
            ),
            wall_time_s=wall_time_s,
            sim_time_s=float(steps_total) * runtime.scenario.action_dt_s,
            path_ratio_last=_optional_finite(last_info.get("path_ratio")),
            trajectory_length_last=_optional_finite(last_info.get("trajectory_length")),
            average_translation_speed_last=_optional_finite(
                last_info.get("average_translation_speed")
            ),
            tip_speed_max_mm_s=max(translation_speeds_mm_s) if translation_speeds_mm_s else 0.0,
            tip_speed_mean_mm_s=(
                sum(translation_speeds_mm_s) / len(translation_speeds_mm_s)
                if translation_speeds_mm_s
                else 0.0
            ),
            tip_total_distance_mm=tip_motion["tip_total_distance_mm"],
            tip_acc_p95=tip_motion["tip_acc_p95"],
            tip_acc_max=tip_motion["tip_acc_max"],
            tip_jerk_p95=tip_motion["tip_jerk_p95"],
            tip_jerk_max=tip_motion["tip_jerk_max"],
            forces=force_collector.build_summary(),
        )
        score = score_trial(
            telemetry=telemetry,
            max_episode_steps=execution.max_episode_steps,
            scoring=scoring,
        )
        trial_valid_for_ranking = valid_for_ranking(
            telemetry=telemetry,
            max_episode_steps=execution.max_episode_steps,
            scoring=scoring,
        )
        trial_force_within_threshold = force_within_safety_threshold(
            telemetry=telemetry,
            scoring=scoring,
        )
        if recorder is not None:
            recorder.__exit__(None, None, None)
            recorder = None
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
            policy_seed=policy_seed,
            score=score,
            telemetry=telemetry,
            valid_for_ranking=trial_valid_for_ranking,
            force_within_safety_threshold=trial_force_within_threshold,
            artifacts=TrialArtifactPaths(
                trace_h5_path=(
                    trace_path
                    if trace_path is not None and trace_path.exists()
                    else None
                )
            ),
            warnings=tuple(trial_warnings),
        )
    except Exception as exc:
        if "recorder" in locals() and recorder is not None:
            recorder.__exit__(type(exc), exc, exc.__traceback__)
        raise
    finally:
        _close_trial_visualisation(visualisation)


__all__ = ["build_single_trial_env", "run_single_trial"]
