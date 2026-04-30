from __future__ import annotations

import concurrent.futures
import csv
import json
import logging
import math
import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence, Tuple

import h5py
import numpy as np

from .discovery import (
    DEFAULT_WIRE_REGISTRY_PATH,
    FileBasedAnatomyDiscovery,
    FileBasedExplicitPolicyDiscovery,
    FileBasedWireRegistryDiscovery,
    _read_json_file,
)
from .models import (
    AgentRef,
    AnatomyBranch,
    AorticArchAnatomy,
    CandidateSummary,
    EvaluationArtifacts,
    EvaluationCandidate,
    EvaluationJob,
    EvaluationReport,
    HistoricalReportSummary,
    ExecutionPlan,
    EvaluationScenario,
    ScoringSpec,
    VisualizationSpec,
    TrialIndicatorSpec,
    ForceScoringSpec,
    SafetyScoreSpec,
    EfficiencyScoreSpec,
    CandidateScoreSpec,
    SmoothnessScoreSpec,
    PolicySpec,
    ScoreBreakdown,
    TargetModeDescriptor,
    TrialArtifactPaths,
    TrialResult,
    TrialTelemetrySummary,
    ForceTelemetrySummary,
    WireRef,
)
from .force_trace_persistence import write_anatomy_mesh
from .runner import configure_cpu_eval_threads, run_single_trial
from .runtime import PreparedEvaluationRuntime, prepare_evaluation_runtime
from .scoring import soft_score_total
from .target_discovery import AnatomyTargetDiscovery
from third_party.stEVE.eve.intervention.vesseltree.util.meshing import load_mesh


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ParallelTrialTask:
    scenario_index: int
    candidate_index: int
    trial_index: int
    seed: int
    policy_seed: int | None
    scenario: EvaluationScenario
    candidate: EvaluationCandidate
    execution: ExecutionPlan
    scoring: ScoringSpec
    registry_path: Path
    policy_device: str
    output_dir: Path


@dataclass(frozen=True)
class _ParallelTrialOutcome:
    scenario_index: int
    candidate_index: int
    trial_index: int
    trial: TrialResult


def _run_parallel_trial_task(task: _ParallelTrialTask) -> _ParallelTrialOutcome:
    configure_cpu_eval_threads(1)

    runtime = prepare_evaluation_runtime(
        candidate=task.candidate,
        scenario=task.scenario,
        registry_path=task.registry_path,
        policy_device=task.policy_device,
    )
    try:
        trial = run_single_trial(
            runtime=runtime,
            trial_index=task.trial_index,
            seed=task.seed,
            execution=task.execution,
            scoring=task.scoring,
            output_dir=task.output_dir,
            frame_callback=None,
            progress_callback=None,
        )
    finally:
        _maybe_close(runtime.play_policy)
        _maybe_close(runtime.intervention)

    return _ParallelTrialOutcome(
        scenario_index=task.scenario_index,
        candidate_index=task.candidate_index,
        trial_index=task.trial_index,
        trial=trial,
    )


def _run_parallel_trial_tasks_process(
    tasks: Sequence[_ParallelTrialTask],
    *,
    worker_count: int,
    task_runner: Callable[
        [_ParallelTrialTask], _ParallelTrialOutcome
    ] = _run_parallel_trial_task,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[_ParallelTrialOutcome, ...]:
    if not tasks:
        return ()

    max_workers = max(1, min(int(worker_count), len(tasks)))
    if progress_callback is not None:
        progress_callback(f"parallel_start workers={max_workers} total={len(tasks)}")

    outcomes: list[_ParallelTrialOutcome] = []
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
    ) as pool:
        futures = [pool.submit(task_runner, task) for task in tasks]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            completed += 1
            if progress_callback is not None:
                progress_callback(
                    "parallel_trial_done completed={completed} total={total} "
                    "scenario={scenario} candidate={candidate} trial_index={trial_index}".format(
                        completed=completed,
                        total=len(tasks),
                        scenario=outcome.trial.scenario_name,
                        candidate=outcome.trial.candidate_name,
                        trial_index=outcome.trial.trial_index,
                    )
                )

    outcomes.sort(
        key=lambda item: (
            item.scenario_index,
            item.candidate_index,
            item.trial_index,
        )
    )
    if progress_callback is not None:
        progress_callback(f"parallel_end completed={len(outcomes)} total={len(tasks)}")
    return tuple(outcomes)


def _generated_at_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_timestamp(generated_at: str) -> str:
    return (
        str(generated_at)
        .replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace(".", "_")
    )


def _job_output_dir(*, output_root: Path, job_name: str, generated_at: str) -> Path:
    return Path(output_root) / f"{job_name}_{_artifact_timestamp(generated_at)}"


def _finite_mean(values: Tuple[float | None, ...]) -> Optional[float]:
    finite = tuple(
        float(value)
        for value in values
        if value is not None and float(value) == float(value)
    )
    if not finite:
        return None
    return sum(finite) / len(finite)


def _finite_std(values: Tuple[float | None, ...]) -> Optional[float]:
    finite = tuple(
        float(value)
        for value in values
        if value is not None and float(value) == float(value)
    )
    if not finite:
        return None
    if len(finite) == 1:
        return 0.0
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return variance**0.5


def _maybe_close(obj: object) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        close()


def _anatomy_mesh_output_path(output_dir: Path, anatomy: AorticArchAnatomy) -> Path:
    anatomy_id = anatomy.record_id or anatomy.arch_type
    return output_dir / "meshes" / f"anatomy_{anatomy_id}.h5"


def _load_mesh_arrays(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = load_mesh(str(mesh_path)).extract_surface().triangulate()
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape((-1, 4))
    if faces.shape[1] != 4 or np.any(faces[:, 0] != 3):
        raise ValueError(f"mesh must contain triangle faces only: {mesh_path}")
    triangle_indices = np.asarray(faces[:, 1:4], dtype=np.int32)
    vertex_positions = np.asarray(mesh.points, dtype=np.float32).reshape((-1, 3))
    return triangle_indices, vertex_positions


def pre_write_meshes_for_job(job: EvaluationJob, output_dir: Path) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_paths: list[Path] = []
    written_anatomy_ids: set[str] = set()
    for scenario in job.scenarios:
        anatomy = scenario.anatomy
        anatomy_id = anatomy.record_id or anatomy.arch_type
        if anatomy_id in written_anatomy_ids:
            continue
        written_anatomy_ids.add(anatomy_id)
        source_mesh_path = anatomy.simulation_mesh_path
        if source_mesh_path is None:
            continue
        triangle_indices, vertex_positions = _load_mesh_arrays(Path(source_mesh_path))
        mesh_output_path = _anatomy_mesh_output_path(output_dir, anatomy)
        try:
            write_anatomy_mesh(
                mesh_output_path,
                triangle_indices=triangle_indices,
                vertex_positions=vertex_positions,
                anatomy_id=anatomy_id,
            )
        except FileExistsError:
            logger.debug("mesh trace already exists for anatomy %s", anatomy_id)
        mesh_paths.append(mesh_output_path)
    return tuple(mesh_paths)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _scoring_spec_payload(scoring: ScoringSpec) -> dict[str, Any]:
    return {
        "trial_indicator": _jsonable(scoring.trial_indicator),
        "force": _jsonable(scoring.force),
        "safety_score": _jsonable(scoring.safety_score),
        "efficiency_score": _jsonable(scoring.efficiency_score),
        "candidate_score": {
            "lambda": float(scoring.candidate_score.lambda_),
            "beta": float(scoring.candidate_score.beta),
            "default_weights": {
                str(key): float(value)
                for key, value in scoring.candidate_score.default_weights.items()
            },
            "active_components": list(scoring.candidate_score.active_components),
        },
        "smoothness_score": _jsonable(scoring.smoothness_score),
    }


def _summary_row(summary: CandidateSummary) -> dict[str, Any]:
    return {
        "scenario_name": summary.scenario_name,
        "candidate_name": summary.candidate_name,
        "execution_wire": summary.execution_wire.tool_ref,
        "trained_on_wire": (
            summary.trained_on_wire.tool_ref if summary.trained_on_wire is not None else ""
        ),
        "trial_count": summary.trial_count,
        "success_rate": summary.success_rate,
        "valid_rate": summary.valid_rate,
        "soft_score_mean_valid": summary.soft_score_mean_valid,
        "soft_score_std_valid": summary.soft_score_std_valid,
        "candidate_score_final": summary.candidate_score_final,
        "score_mean": summary.score_mean,
        "score_std": summary.score_std,
        "steps_total_mean": summary.steps_total_mean,
        "steps_to_success_mean": summary.steps_to_success_mean,
        "tip_speed_max_mean_mm_s": summary.tip_speed_max_mean_mm_s,
        "wall_force_max_mean": summary.wall_force_max_mean,
        "wall_force_max_mean_newton": summary.wall_force_max_mean_newton,
        "force_available_rate": summary.force_available_rate,
    }


def _trial_rows(trials: Tuple[TrialResult, ...], *, max_episode_steps: int, tip_length_mm: float) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        forces = trial.telemetry.forces
        rows.append(
            {
                "scenario_name": trial.scenario_name,
                "candidate_name": trial.candidate_name,
                "execution_wire": trial.execution_wire.tool_ref,
                "trained_on_wire": (
                    trial.policy.trained_on_wire.tool_ref
                    if trial.policy.trained_on_wire is not None
                    else ""
                ),
                "trial_index": int(trial.trial_index),
                "env_seed": int(trial.seed),
                "policy_seed": trial.policy_seed,
                "success": bool(trial.telemetry.success),
                "valid_for_ranking": bool(trial.valid_for_ranking),
                "force_within_safety_threshold": bool(trial.force_within_safety_threshold),
                "steps_total": int(trial.telemetry.steps_total),
                "steps_to_success": trial.telemetry.steps_to_success,
                "max_episode_steps": int(max_episode_steps),
                "episode_reward": float(trial.telemetry.episode_reward),
                "sim_time_s": trial.telemetry.sim_time_s,
                "wall_time_s": trial.telemetry.wall_time_s,
                "tip_speed_max_mm_s": trial.telemetry.tip_speed_max_mm_s,
                "tip_speed_mean_mm_s": trial.telemetry.tip_speed_mean_mm_s,
                "tip_total_distance_mm": trial.telemetry.tip_total_distance_mm,
                "force_available_for_score": bool(
                    forces is not None and forces.available_for_score
                ),
                "force_total_norm_max_N": (
                    None if forces is None else forces.total_force_norm_max_newton
                ),
                "force_total_norm_mean_N": (
                    None if forces is None else forces.total_force_norm_mean_newton
                ),
                "tip_force_peak_normal_N": (
                    None if forces is None else forces.tip_force_peak_normal_N
                ),
                "tip_force_total_mean_N": (
                    None if forces is None else forces.tip_force_total_mean_N
                ),
                "tip_length_mm": float(tip_length_mm),
                "tip_acc_p95": trial.telemetry.tip_acc_p95,
                "tip_acc_max": trial.telemetry.tip_acc_max,
                "tip_jerk_p95": trial.telemetry.tip_jerk_p95,
                "tip_jerk_max": trial.telemetry.tip_jerk_max,
                "score_success": float(trial.score.success),
                "score_efficiency": float(trial.score.efficiency),
                "score_safety": float(trial.score.safety),
                "score_smoothness": trial.score.smoothness,
                "score_total": float(trial.score.total),
                "trace_h5_path": (
                    None
                    if trial.artifacts.trace_h5_path is None
                    else str(trial.artifacts.trace_h5_path)
                ),
            }
        )
    return tuple(rows)


def _write_trials_h5(path: Path, rows: Tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    columns = (
        "scenario_name",
        "candidate_name",
        "execution_wire",
        "trained_on_wire",
        "trial_index",
        "env_seed",
        "policy_seed",
        "success",
        "valid_for_ranking",
        "force_within_safety_threshold",
        "steps_total",
        "steps_to_success",
        "max_episode_steps",
        "episode_reward",
        "sim_time_s",
        "wall_time_s",
        "tip_speed_max_mm_s",
        "tip_speed_mean_mm_s",
        "tip_total_distance_mm",
        "force_available_for_score",
        "force_total_norm_max_N",
        "force_total_norm_mean_N",
        "tip_force_peak_normal_N",
        "tip_force_total_mean_N",
        "tip_length_mm",
        "tip_acc_p95",
        "tip_acc_max",
        "tip_jerk_p95",
        "tip_jerk_max",
        "score_success",
        "score_efficiency",
        "score_safety",
        "score_smoothness",
        "score_total",
        "trace_h5_path",
    )
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 1
        group = handle.create_group("trials")
        for column in columns:
            values = [row.get(column) for row in rows]
            if column in {
                "scenario_name",
                "candidate_name",
                "execution_wire",
                "trained_on_wire",
                "trace_h5_path",
            }:
                group.create_dataset(
                    column,
                    data=np.asarray(
                        ["" if value is None else str(value) for value in values],
                        dtype=object,
                    ),
                    dtype=string_dtype,
                )
            elif column in {
                "success",
                "valid_for_ranking",
                "force_within_safety_threshold",
                "force_available_for_score",
            }:
                group.create_dataset(
                    column,
                    data=np.asarray([bool(value) for value in values], dtype=np.bool_),
                )
            elif column in {
                "trial_index",
                "env_seed",
                "steps_total",
                "max_episode_steps",
            }:
                group.create_dataset(
                    column,
                    data=np.asarray([int(value) for value in values], dtype=np.int64),
                )
            elif column in {"policy_seed", "steps_to_success"}:
                group.create_dataset(
                    column,
                    data=np.asarray(
                        [
                            np.nan if value is None else float(value)
                            for value in values
                        ],
                        dtype=np.float64,
                    ),
                )
            else:
                group.create_dataset(
                    column,
                    data=np.asarray(
                        [
                            np.nan if value is None else float(value)
                            for value in values
                        ],
                        dtype=np.float64,
                    ),
                )


def _write_report_artifacts(
    *,
    job: EvaluationJob,
    report: EvaluationReport,
) -> EvaluationArtifacts:
    output_dir = (
        report.artifacts.output_dir
        if report.artifacts is not None
        else _job_output_dir(
            output_root=job.output_root,
            job_name=job.name,
            generated_at=report.generated_at,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_summaries_csv_path = output_dir / "candidate_summaries.csv"
    candidate_summaries_json_path = output_dir / "candidate_summaries.json"
    manifest_json_path = output_dir / "manifest.json"
    trials_h5_path = output_dir / "trials.h5"
    report_markdown_path = output_dir / "report.md"

    summary_fieldnames = list(_summary_row(report.summaries[0]).keys()) if report.summaries else []
    with candidate_summaries_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        for summary in report.summaries:
            writer.writerow(_summary_row(summary))

    candidate_summaries_json_path.write_text(
        json.dumps([_summary_row(summary) for summary in report.summaries], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    trial_rows = _trial_rows(
        report.trials,
        max_episode_steps=job.execution.max_episode_steps,
        tip_length_mm=job.scoring.force.tip_length_mm,
    )
    _write_trials_h5(trials_h5_path, trial_rows)

    with manifest_json_path.open("w", encoding="utf-8") as handle:
        anatomy_metadata = [
            {
                "name": scenario.name,
                "record_id": scenario.anatomy.record_id,
                "arch_type": scenario.anatomy.arch_type,
                "seed": scenario.anatomy.seed,
            }
            for scenario in job.scenarios
        ]
        summary_stats = {
            "success_rate_mean": _finite_mean(tuple(summary.success_rate for summary in report.summaries)),
            "candidate_score_final_mean": _finite_mean(tuple(summary.candidate_score_final for summary in report.summaries)),
            "valid_rate_mean": _finite_mean(tuple(summary.valid_rate for summary in report.summaries)),
        }
        payload = {
            "schema_version": 1,
            "job_name": report.job_name,
            "generated_time": report.generated_at,
            "anatomy_metadata": anatomy_metadata,
            "targets": [_jsonable(scenario.target) for scenario in job.scenarios],
            "wires": [_jsonable(candidate.execution_wire) for candidate in job.candidates],
            "candidates": [_jsonable(candidate) for candidate in job.candidates],
            "execution_plan": _jsonable(job.execution),
            "seed_schedule": [
                {
                    "trial_index": index,
                    "env_seed": env_seed,
                    "policy_seed": policy_seed,
                }
                for index, (env_seed, policy_seed) in enumerate(job.execution.trial_seed_pairs)
            ],
            "counts": {
                "n_candidates": len(job.candidates),
                "n_trials_total": len(report.trials),
            },
            "artifact_paths": {
                "candidate_summaries_csv": str(candidate_summaries_csv_path),
                "candidate_summaries_json": str(candidate_summaries_json_path),
                "trials_h5": str(trials_h5_path),
                "traces_dir": str(output_dir / "traces"),
                "meshes_dir": str(output_dir / "meshes"),
                "report_md": str(report_markdown_path),
            },
            "scoring_spec": _scoring_spec_payload(job.scoring),
            "summary_stats": summary_stats,
            "summaries": [_summary_row(summary) for summary in report.summaries],
            "tested_wires": sorted(
                {candidate.execution_wire.tool_ref for candidate in job.candidates}
            ),
        }
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    lines = [
        f"# {report.job_name}",
        "",
                f"Generated at: {report.generated_at}",
                f"Output dir: {output_dir}",
                f"Trials: {len(report.trials)}",
                f"Summaries: {len(report.summaries)}",
                "",
    ]
    for summary in report.summaries:
        lines.extend(
            [
                f"## {summary.scenario_name} / {summary.candidate_name}",
                f"- execution wire: {summary.execution_wire.tool_ref}",
                f"- trained on: {summary.trained_on_wire.tool_ref if summary.trained_on_wire is not None else 'unknown'}",
                f"- trials: {summary.trial_count}",
                f"- success rate: {summary.success_rate if summary.success_rate is not None else 'n/a'}",
                f"- valid rate: {summary.valid_rate if summary.valid_rate is not None else 'n/a'}",
                f"- candidate score: {summary.candidate_score_final if summary.candidate_score_final is not None else 'n/a'}",
                "",
            ]
        )
    report_markdown_path.write_text("\n".join(lines), encoding="utf-8")

    return EvaluationArtifacts(
        output_dir=output_dir,
        candidate_summaries_csv_path=candidate_summaries_csv_path,
        candidate_summaries_json_path=candidate_summaries_json_path,
        manifest_json_path=manifest_json_path,
        trials_h5_path=trials_h5_path,
        report_markdown_path=report_markdown_path,
        traces_dir=output_dir / "traces",
        meshes_dir=output_dir / "meshes",
    )


def summarize_trials(
    trials: Tuple[TrialResult, ...],
    *,
    scoring: ScoringSpec,
) -> CandidateSummary:
    """Aggregate all trials for one candidate/scenario pair."""

    if not trials:
        raise ValueError("trials must not be empty")

    first = trials[0]
    valid_soft_scores = tuple(
        soft_score_total(breakdown=trial.score, scoring=scoring)
        if trial.valid_for_ranking
        else None
        for trial in trials
    )
    valid_rate = _finite_mean(
        tuple(1.0 if trial.valid_for_ranking else 0.0 for trial in trials)
    )
    soft_score_mean_valid = _finite_mean(valid_soft_scores)
    soft_score_std_valid = _finite_std(valid_soft_scores)
    n_valid = sum(1 for trial in trials if trial.valid_for_ranking)
    if n_valid == 0 or soft_score_mean_valid is None:
        candidate_score_final = 0.0
    else:
        sigma = 0.0 if soft_score_std_valid is None else float(soft_score_std_valid)
        p_w = 0.0 if valid_rate is None else float(valid_rate)
        candidate_score_final = (p_w ** float(scoring.candidate_score.lambda_)) * max(
            0.0,
            float(soft_score_mean_valid) - float(scoring.candidate_score.beta) * sigma,
        )

    return CandidateSummary(
        scenario_name=first.scenario_name,
        candidate_name=first.candidate_name,
        execution_wire=first.execution_wire,
        trained_on_wire=first.policy.trained_on_wire,
        trial_count=len(trials),
        success_rate=_finite_mean(
            tuple(1.0 if trial.telemetry.success else 0.0 for trial in trials)
        ),
        valid_rate=valid_rate,
        soft_score_mean_valid=soft_score_mean_valid,
        soft_score_std_valid=soft_score_std_valid,
        candidate_score_final=candidate_score_final,
        score_mean=_finite_mean(tuple(trial.score.total for trial in trials)),
        score_std=_finite_std(tuple(trial.score.total for trial in trials)),
        steps_total_mean=_finite_mean(
            tuple(float(trial.telemetry.steps_total) for trial in trials)
        ),
        steps_to_success_mean=_finite_mean(
            tuple(
                (
                    None
                    if trial.telemetry.steps_to_success is None
                    else float(trial.telemetry.steps_to_success)
                )
                for trial in trials
            )
        ),
        tip_speed_max_mean_mm_s=_finite_mean(
            tuple(trial.telemetry.tip_speed_max_mm_s for trial in trials)
        ),
        wall_force_max_mean=_finite_mean(
            tuple(
                (
                    None
                    if trial.telemetry.forces is None
                    else trial.telemetry.forces.total_force_norm_max
                )
                for trial in trials
            )
        ),
        wall_force_max_mean_newton=_finite_mean(
            tuple(
                (
                    None
                    if trial.telemetry.forces is None
                    else trial.telemetry.forces.total_force_norm_max_newton
                )
                for trial in trials
            )
        ),
        force_available_rate=_finite_mean(
            tuple(
                (
                    None
                    if trial.telemetry.forces is None
                    else 1.0 if trial.telemetry.forces.available_for_score else 0.0
                )
                for trial in trials
            )
        ),
    )


class AnatomyDiscoveryPort(Protocol):
    """Discovery port for anatomy records available to eval_v2."""

    def list_anatomies(
        self,
        *,
        registry_path: Optional[Path] = None,
    ) -> Tuple[AorticArchAnatomy, ...]:
        """Return all discoverable anatomies from the configured registry file."""

    def get_anatomy(
        self,
        *,
        record_id: str,
        registry_path: Optional[Path] = None,
    ) -> AorticArchAnatomy:
        """Return one resolved anatomy by its stable registry record id."""


class PolicyDiscoveryPort(Protocol):
    """Discovery port for execution wires and evaluable policies."""

    def list_execution_wires(self) -> Tuple[WireRef, ...]:
        """Return all wires available as physical execution devices."""

    def list_startable_wires(self) -> Tuple[WireRef, ...]:
        """Return only wires with at least one loadable agent checkpoint."""

    def list_registry_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        """Return policies discoverable from the local registry-backed agents."""

    def list_explicit_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        """Return policies discoverable from explicit checkpoint sources."""

    def resolve_policy_from_agent_ref(self, agent_ref: AgentRef) -> PolicySpec:
        """Resolve one registry agent reference into a concrete policy artifact."""


class TargetDiscoveryPort(Protocol):
    """Discovery port for anatomy-specific target-selection options."""

    def list_branches(self, anatomy: AorticArchAnatomy) -> Tuple[AnatomyBranch, ...]:
        """Return all branch descriptors for one resolved anatomy."""

    def get_branch(
        self,
        anatomy: AorticArchAnatomy,
        *,
        branch_name: str,
    ) -> AnatomyBranch:
        """Return one branch descriptor for one resolved anatomy."""

    def list_target_modes(self) -> Tuple[TargetModeDescriptor, ...]:
        """Return the supported target-construction modes."""


class EvaluationRunnerPort(Protocol):
    """Execution port for fully-resolved eval_v2 jobs."""

    def run_evaluation_job(
        self,
        job: EvaluationJob,
        *,
        frame_callback: Optional[Callable[[np.ndarray], None]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> EvaluationReport:
        """Execute one fully-resolved evaluation job and return its report."""


class EvaluationService(ABC):
    """Single service-layer API boundary for eval_v2."""

    @abstractmethod
    def list_anatomies(
        self,
        *,
        registry_path: Optional[Path] = None,
    ) -> Tuple[AorticArchAnatomy, ...]:
        """Return discoverable anatomies as resolved domain models."""

    @abstractmethod
    def get_anatomy(
        self,
        *,
        record_id: str,
        registry_path: Optional[Path] = None,
    ) -> AorticArchAnatomy:
        """Return one anatomy selected by its stable registry record id."""

    @abstractmethod
    def list_branches(self, anatomy: AorticArchAnatomy) -> Tuple[AnatomyBranch, ...]:
        """Return branch descriptors available for target selection."""

    @abstractmethod
    def get_branch(
        self,
        anatomy: AorticArchAnatomy,
        *,
        branch_name: str,
    ) -> AnatomyBranch:
        """Return one branch descriptor by name."""

    @abstractmethod
    def list_target_modes(self) -> Tuple[TargetModeDescriptor, ...]:
        """Return target modes offered to adapters."""

    @abstractmethod
    def list_execution_wires(self) -> Tuple[WireRef, ...]:
        """Return wires that can be mounted as the execution device."""

    @abstractmethod
    def list_startable_wires(self) -> Tuple[WireRef, ...]:
        """Return wires that currently have at least one loadable agent."""

    @abstractmethod
    def list_registry_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        """Return registry-backed policies that can be offered for evaluation."""

    @abstractmethod
    def list_explicit_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        """Return non-registry policies discoverable from explicit checkpoints."""

    @abstractmethod
    def resolve_policy_from_agent_ref(self, agent_ref: AgentRef) -> PolicySpec:
        """Resolve one stable registry agent reference into a concrete policy."""


class ExplicitPolicyDiscoveryPort(Protocol):
    """Discovery port for explicit checkpoint-backed policies."""

    def list_explicit_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        """Return policies discoverable from explicit checkpoint sources."""

    def resolve_policy_from_agent_ref(self, agent_ref: AgentRef) -> PolicySpec:
        """Resolve one explicit policy by its stable agent reference."""

    @abstractmethod
    def build_candidate(
        self,
        *,
        name: str,
        execution_wire: WireRef,
        policy: PolicySpec,
    ) -> EvaluationCandidate:
        """Construct one candidate from a selected execution wire and policy."""

    @abstractmethod
    def list_candidates(
        self,
        *,
        execution_wire: WireRef,
        include_cross_wire: bool = True,
    ) -> Tuple[EvaluationCandidate, ...]:
        """Return discoverable candidate options for one execution wire."""

    @abstractmethod
    def run_evaluation_job(
        self,
        job: EvaluationJob,
        *,
        frame_callback: Optional[Callable[[np.ndarray], None]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> EvaluationReport:
        """Execute one evaluation job and return the normalized service report."""

    @abstractmethod
    def list_historical_reports(
        self,
        *,
        output_root: Path = Path("results/eval_runs"),
    ) -> Tuple[HistoricalReportSummary, ...]:
        """Return lightweight metadata for persisted manifests."""

    @abstractmethod
    def load_manifest_from_disk(self, manifest_json_path: Path) -> EvaluationReport:
        """Load one full persisted evaluation run from disk."""

    @abstractmethod
    def save_clinical_feedback(
        self,
        *,
        report_id: str,
        feedback_data: dict[str, Any],
    ) -> Path:
        """Persist clinician ground-truth feedback for one report output directory."""


class LocalEvaluationRunner:
    """Concrete job runner built on the clean-room eval_v2 runtime and runner."""

    def __init__(
        self,
        *,
        registry_path: Path = DEFAULT_WIRE_REGISTRY_PATH,
        runtime_factory: Callable[
            ..., PreparedEvaluationRuntime
        ] = prepare_evaluation_runtime,
        trial_runner: Callable[..., TrialResult] = run_single_trial,
        parallel_task_runner: Callable[
            ..., Tuple[_ParallelTrialOutcome, ...]
        ] = _run_parallel_trial_tasks_process,
        generated_at_factory: Callable[[], str] = _generated_at_utc,
    ) -> None:
        self._registry_path = Path(registry_path)
        self._runtime_factory = runtime_factory
        self._trial_runner = trial_runner
        self._parallel_task_runner = parallel_task_runner
        self._generated_at_factory = generated_at_factory

    def run_evaluation_job(
        self,
        job: EvaluationJob,
        *,
        frame_callback: Optional[Callable[[np.ndarray], None]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> EvaluationReport:
        generated_at = self._generated_at_factory()
        output_dir = _job_output_dir(
            output_root=job.output_root,
            job_name=job.name,
            generated_at=generated_at,
        )
        pre_write_meshes_for_job(job, output_dir)
        worker_count = max(1, int(job.execution.worker_count))
        use_parallel = (
            worker_count > 1
            and not job.execution.visualization.enabled
            and frame_callback is None
            and self._runtime_factory is prepare_evaluation_runtime
            and self._trial_runner is run_single_trial
        )
        if worker_count > 1 and job.execution.visualization.enabled:
            raise ValueError("worker_count > 1 is only supported for headless runs")
        if use_parallel and str(job.execution.policy_device).lower() != "cpu":
            raise ValueError("parallel eval_v2 workers require policy_device='cpu'")
        if use_parallel:
            return self._run_evaluation_job_parallel(
                job,
                generated_at=generated_at,
                output_dir=output_dir,
                worker_count=worker_count,
                progress_callback=progress_callback,
            )

        isolate_headless_trials = (
            not job.execution.visualization.enabled
            and frame_callback is None
            and self._runtime_factory is prepare_evaluation_runtime
            and self._trial_runner is run_single_trial
        )
        trials: list[TrialResult] = []
        for scenario in job.scenarios:
            for candidate in job.candidates:
                if isolate_headless_trials:
                    for trial_index, (seed, _policy_seed) in enumerate(
                        job.execution.trial_seed_pairs
                    ):
                        if progress_callback is not None:
                            progress_callback(
                                "runtime_prepare "
                                f"scenario={scenario.name} candidate={candidate.name} "
                                f"trial_index={trial_index}"
                            )
                        if str(job.execution.policy_device).lower() == "cpu":
                            configure_cpu_eval_threads(1)
                        runtime = self._runtime_factory(
                            candidate=candidate,
                            scenario=scenario,
                            registry_path=self._registry_path,
                            policy_device=job.execution.policy_device,
                        )
                        try:
                            trial = self._trial_runner(
                                runtime=runtime,
                                trial_index=trial_index,
                                seed=seed,
                                execution=job.execution,
                                scoring=job.scoring,
                                output_dir=output_dir,
                                frame_callback=frame_callback,
                                progress_callback=progress_callback,
                            )
                            trials.append(trial)
                        finally:
                            _maybe_close(runtime.play_policy)
                            _maybe_close(runtime.intervention)
                    continue

                if progress_callback is not None:
                    progress_callback(
                        f"runtime_prepare scenario={scenario.name} candidate={candidate.name}"
                    )
                if str(job.execution.policy_device).lower() == "cpu":
                    configure_cpu_eval_threads(1)
                runtime = self._runtime_factory(
                    candidate=candidate,
                    scenario=scenario,
                    registry_path=self._registry_path,
                    policy_device=job.execution.policy_device,
                )
                try:
                    for trial_index, (seed, _policy_seed) in enumerate(
                        job.execution.trial_seed_pairs
                    ):
                        trial = self._trial_runner(
                            runtime=runtime,
                            trial_index=trial_index,
                            seed=seed,
                            execution=job.execution,
                            scoring=job.scoring,
                            output_dir=output_dir,
                            frame_callback=frame_callback,
                            progress_callback=progress_callback,
                        )
                        trials.append(trial)
                finally:
                    _maybe_close(runtime.play_policy)
                    _maybe_close(runtime.intervention)

        summaries = tuple(
            summarize_trials(
                tuple(
                    trial
                    for trial in trials
                    if trial.scenario_name == scenario.name
                    and trial.candidate_name == candidate.name
                    and trial.execution_wire == candidate.execution_wire
                ),
                scoring=job.scoring,
            )
            for scenario in job.scenarios
            for candidate in job.candidates
        )
        return EvaluationReport(
            job_name=job.name,
            generated_at=generated_at,
            summaries=summaries,
            trials=tuple(trials),
            execution_plan=job.execution,
            scoring_spec=job.scoring,
            artifacts=EvaluationArtifacts(output_dir=output_dir),
        )

    def _run_evaluation_job_parallel(
        self,
        job: EvaluationJob,
        *,
        generated_at: str,
        output_dir: Path,
        worker_count: int,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> EvaluationReport:
        tasks = tuple(
            _ParallelTrialTask(
                scenario_index=scenario_index,
                candidate_index=candidate_index,
                trial_index=trial_index,
                seed=seed,
                policy_seed=policy_seed,
                scenario=scenario,
                candidate=candidate,
                execution=job.execution,
                scoring=job.scoring,
                registry_path=self._registry_path,
                policy_device=job.execution.policy_device,
                output_dir=output_dir,
            )
            for scenario_index, scenario in enumerate(job.scenarios)
            for candidate_index, candidate in enumerate(job.candidates)
            for trial_index, (seed, policy_seed) in enumerate(
                job.execution.trial_seed_pairs
            )
        )
        outcomes = self._parallel_task_runner(
            tasks,
            worker_count=worker_count,
            progress_callback=progress_callback,
        )
        outcomes = tuple(
            sorted(
                outcomes,
                key=lambda item: (
                    item.scenario_index,
                    item.candidate_index,
                    item.trial_index,
                ),
            )
        )
        trials = tuple(outcome.trial for outcome in outcomes)
        summaries = tuple(
            summarize_trials(
                tuple(
                    trial
                    for trial in trials
                    if trial.scenario_name == scenario.name
                    and trial.candidate_name == candidate.name
                    and trial.execution_wire == candidate.execution_wire
                ),
                scoring=job.scoring,
            )
            for scenario in job.scenarios
            for candidate in job.candidates
        )
        return EvaluationReport(
            job_name=job.name,
            generated_at=generated_at,
            summaries=summaries,
            trials=trials,
            execution_plan=job.execution,
            scoring_spec=job.scoring,
            artifacts=EvaluationArtifacts(output_dir=output_dir),
        )


class DefaultEvaluationService(EvaluationService):
    """Default concrete service composed from file-based discovery and local execution."""

    def __init__(
        self,
        *,
        anatomy_discovery: Optional[AnatomyDiscoveryPort] = None,
        policy_discovery: Optional[PolicyDiscoveryPort] = None,
        explicit_policy_discovery: Optional[ExplicitPolicyDiscoveryPort] = None,
        target_discovery: Optional[TargetDiscoveryPort] = None,
        evaluation_runner: Optional[EvaluationRunnerPort] = None,
    ) -> None:
        self._anatomy_discovery = anatomy_discovery or FileBasedAnatomyDiscovery()
        self._policy_discovery = policy_discovery or FileBasedWireRegistryDiscovery()
        self._explicit_policy_discovery = (
            explicit_policy_discovery or FileBasedExplicitPolicyDiscovery()
        )
        self._target_discovery = target_discovery or AnatomyTargetDiscovery()
        self._evaluation_runner = evaluation_runner or LocalEvaluationRunner()

    def list_anatomies(
        self,
        *,
        registry_path: Optional[Path] = None,
    ) -> Tuple[AorticArchAnatomy, ...]:
        return self._anatomy_discovery.list_anatomies(registry_path=registry_path)

    def get_anatomy(
        self,
        *,
        record_id: str,
        registry_path: Optional[Path] = None,
    ) -> AorticArchAnatomy:
        return self._anatomy_discovery.get_anatomy(
            record_id=record_id,
            registry_path=registry_path,
        )

    def list_branches(self, anatomy: AorticArchAnatomy) -> Tuple[AnatomyBranch, ...]:
        return self._target_discovery.list_branches(anatomy)

    def get_branch(
        self,
        anatomy: AorticArchAnatomy,
        *,
        branch_name: str,
    ) -> AnatomyBranch:
        return self._target_discovery.get_branch(anatomy, branch_name=branch_name)

    def list_target_modes(self) -> Tuple[TargetModeDescriptor, ...]:
        return self._target_discovery.list_target_modes()

    def list_execution_wires(self) -> Tuple[WireRef, ...]:
        return self._policy_discovery.list_execution_wires()

    def list_startable_wires(self) -> Tuple[WireRef, ...]:
        return self._policy_discovery.list_startable_wires()

    def list_registry_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        return self._policy_discovery.list_registry_policies(
            execution_wire=execution_wire
        )

    def list_explicit_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        return self._explicit_policy_discovery.list_explicit_policies(
            execution_wire=execution_wire
        )

    def resolve_policy_from_agent_ref(self, agent_ref: AgentRef) -> PolicySpec:
        matches = tuple(
            policy
            for policy in (
                self._policy_discovery.list_registry_policies()
                + self._explicit_policy_discovery.list_explicit_policies()
            )
            if policy.registry_agent == agent_ref
        )
        if not matches:
            raise KeyError(agent_ref.agent_ref)
        if len(matches) > 1:
            checkpoint_paths = {policy.checkpoint_path for policy in matches}
            if len(checkpoint_paths) > 1:
                raise ValueError(
                    f"Policy agent ref {agent_ref.agent_ref!r} is ambiguous across multiple checkpoints; use --policy-checkpoint instead"
                )
        return matches[0]

    def build_candidate(
        self,
        *,
        name: str,
        execution_wire: WireRef,
        policy: PolicySpec,
    ) -> EvaluationCandidate:
        return EvaluationCandidate(
            name=name,
            execution_wire=execution_wire,
            policy=policy,
        )

    def list_candidates(
        self,
        *,
        execution_wire: WireRef,
        include_cross_wire: bool = True,
    ) -> Tuple[EvaluationCandidate, ...]:
        policies = (
            self._policy_discovery.list_registry_policies()
            + self._explicit_policy_discovery.list_explicit_policies()
        )
        if not include_cross_wire:
            policies = tuple(
                policy
                for policy in policies
                if policy.trained_on_wire is None
                or policy.trained_on_wire == execution_wire
            )

        candidates: list[EvaluationCandidate] = []
        used_names: dict[str, int] = {}
        for policy in policies:
            base_name = self._candidate_name(policy, execution_wire=execution_wire)
            count = used_names.get(base_name, 0)
            used_names[base_name] = count + 1
            name = base_name if count == 0 else f"{base_name} #{count + 1}"
            candidates.append(
                self.build_candidate(
                    name=name,
                    execution_wire=execution_wire,
                    policy=policy,
                )
            )
        return tuple(candidates)

    def run_evaluation_job(
        self,
        job: EvaluationJob,
        *,
        frame_callback: Optional[Callable[[np.ndarray], None]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> EvaluationReport:
        report = self._evaluation_runner.run_evaluation_job(
            job,
            frame_callback=frame_callback,
            progress_callback=progress_callback,
        )
        artifacts = _write_report_artifacts(job=job, report=report)
        return replace(report, artifacts=artifacts)

    def list_historical_reports(
        self,
        *,
        output_root: Path = Path("results/eval_runs"),
    ) -> Tuple[HistoricalReportSummary, ...]:
        root = Path(output_root)
        if not root.exists():
            return ()

        summaries: list[HistoricalReportSummary] = []
        for manifest_json in sorted(root.rglob("manifest.json")):
            try:
                payload = _read_json_file(manifest_json)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            anatomy_metadata = payload.get("anatomy_metadata", ())
            anatomy = "unknown"
            if isinstance(anatomy_metadata, list) and anatomy_metadata:
                anatomy = ", ".join(
                    str(item.get("record_id") or item.get("name") or "unknown")
                    for item in anatomy_metadata
                    if isinstance(item, dict)
                ) or "unknown"

            job_name = str(payload.get("job_name", manifest_json.parent.name))
            generated_at = str(payload.get("generated_time", ""))
            tested_wires_raw = payload.get("tested_wires", ())
            tested_wires = (
                tuple(str(wire) for wire in tested_wires_raw)
                if isinstance(tested_wires_raw, list)
                else ()
            )

            summaries.append(
                HistoricalReportSummary(
                    job_name=job_name,
                    generated_at=generated_at,
                    anatomy=anatomy,
                    tested_wires=tested_wires,
                    manifest_json_path=manifest_json,
                    output_dir=manifest_json.parent,
                )
            )

        summaries.sort(key=lambda summary: summary.generated_at, reverse=True)
        return tuple(summaries)

    def load_manifest_from_disk(self, manifest_json_path: Path) -> EvaluationReport:
        payload = _read_json_file(Path(manifest_json_path))
        if not isinstance(payload, dict):
            raise TypeError("manifest.json payload must be an object")

        summaries = tuple(_parse_candidate_summary(item) for item in payload.get("summaries", ()))
        artifact_paths = payload.get("artifact_paths", {})
        if not isinstance(artifact_paths, dict):
            artifact_paths = {}
        trials_h5_path = Path(
            artifact_paths.get("trials_h5", Path(manifest_json_path).parent / "trials.h5")
        )
        trials = _load_trials_from_h5(trials_h5_path)

        artifacts = EvaluationArtifacts(
            output_dir=Path(manifest_json_path).parent,
            candidate_summaries_csv_path=Path(
                artifact_paths.get(
                    "candidate_summaries_csv",
                    Path(manifest_json_path).parent / "candidate_summaries.csv",
                )
            ),
            candidate_summaries_json_path=Path(
                artifact_paths.get(
                    "candidate_summaries_json",
                    Path(manifest_json_path).parent / "candidate_summaries.json",
                )
            ),
            manifest_json_path=Path(manifest_json_path),
            trials_h5_path=trials_h5_path,
            report_markdown_path=Path(
                artifact_paths.get("report_md", Path(manifest_json_path).parent / "report.md")
            ),
            traces_dir=Path(
                artifact_paths.get("traces_dir", Path(manifest_json_path).parent / "traces")
            ),
            meshes_dir=Path(
                artifact_paths.get("meshes_dir", Path(manifest_json_path).parent / "meshes")
            ),
        )
        return EvaluationReport(
            job_name=str(payload.get("job_name", Path(manifest_json_path).parent.name)),
            generated_at=str(payload.get("generated_time", "")),
            summaries=summaries,
            trials=trials,
            execution_plan=_parse_execution_plan(payload.get("execution_plan", {})),
            scoring_spec=_parse_scoring_spec(payload.get("scoring_spec", {})),
            artifacts=artifacts,
        )

    def save_clinical_feedback(
        self,
        *,
        report_id: str,
        feedback_data: dict[str, Any],
    ) -> Path:
        output_dir = Path(report_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        feedback_path = output_dir / "feedback.json"
        payload = dict(feedback_data)
        payload["saved_at"] = _generated_at_utc()
        with feedback_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return feedback_path

    @staticmethod
    def _candidate_name(
        policy: PolicySpec,
        *,
        execution_wire: WireRef,
    ) -> str:
        trained_on_wire = policy.trained_on_wire
        if trained_on_wire is None or trained_on_wire == execution_wire:
            return policy.name
        return (
            f"{policy.name} [{trained_on_wire.tool_ref} -> {execution_wire.tool_ref}]"
        )


__all__ = [
    "AnatomyDiscoveryPort",
    "DefaultEvaluationService",
    "ExplicitPolicyDiscoveryPort",
    "EvaluationRunnerPort",
    "EvaluationService",
    "LocalEvaluationRunner",
    "PolicyDiscoveryPort",
    "TargetDiscoveryPort",
    "summarize_trials",
]


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _parse_vector3(value: Any) -> tuple[float, float, float]:
    if value is None:
        return (0.0, 0.0, 0.0)
    items = tuple(value)
    if len(items) != 3:
        return (0.0, 0.0, 0.0)
    return (float(items[0]), float(items[1]), float(items[2]))


def _parse_wire_ref(value: Any) -> WireRef:
    if isinstance(value, dict):
        return WireRef(
            model=str(value.get("model", "")), wire=str(value.get("wire", ""))
        )
    text = str(value)
    model, wire = text.split("/", maxsplit=1)
    return WireRef(model=model, wire=wire)


def _parse_policy_spec(payload: dict[str, Any]) -> PolicySpec:
    trained_on_wire = payload.get("trained_on_wire")
    return PolicySpec(
        name=str(payload.get("name", "policy")),
        checkpoint_path=Path(payload.get("checkpoint_path", "")),
        source=str(payload.get("source", "explicit")),
        trained_on_wire=_parse_wire_ref(trained_on_wire) if trained_on_wire else None,
    )


def _parse_execution_plan(payload: Any) -> ExecutionPlan:
    if not isinstance(payload, dict):
        return ExecutionPlan()
    visualization_payload = payload.get("visualization", {})
    if not isinstance(visualization_payload, dict):
        visualization_payload = {}
    visualization = VisualizationSpec(
        enabled=bool(visualization_payload.get("enabled", False)),
        rendered_trials_per_candidate=int(
            visualization_payload.get("rendered_trials_per_candidate", 1)
        ),
        force_debug_overlay=bool(
            visualization_payload.get("force_debug_overlay", False)
        ),
        force_debug_top_k_segments=int(
            visualization_payload.get("force_debug_top_k_segments", 5)
        ),
    )
    return ExecutionPlan(
        trials_per_candidate=int(payload.get("trials_per_candidate", 10)),
        base_seed=int(payload.get("base_seed", 123)),
        explicit_seeds=tuple(int(seed) for seed in payload.get("explicit_seeds", ())),
        policy_base_seed=int(payload.get("policy_base_seed", 1000)),
        policy_explicit_seeds=tuple(
            int(seed) for seed in payload.get("policy_explicit_seeds", ())
        ),
        max_episode_steps=int(payload.get("max_episode_steps", 1000)),
        policy_device=str(payload.get("policy_device", "cuda")),
        policy_mode=str(payload.get("policy_mode", "deterministic")),
        stochastic_environment_mode=str(
            payload.get("stochastic_environment_mode", "random_start")
        ),
        simulation_backend=str(payload.get("simulation_backend", "single_process")),
        visualization=visualization,
        worker_count=int(payload.get("worker_count", 1)),
    )


def _parse_scoring_spec(payload: Any) -> ScoringSpec:
    if not isinstance(payload, dict):
        return ScoringSpec()
    trial_indicator_payload = payload.get("trial_indicator", {})
    force_payload = payload.get("force", {})
    safety_payload = payload.get("safety_score", {})
    efficiency_payload = payload.get("efficiency_score", {})
    candidate_payload = payload.get("candidate_score", {})
    smoothness_payload = payload.get("smoothness_score", {})
    if not isinstance(trial_indicator_payload, dict):
        trial_indicator_payload = {}
    if not isinstance(force_payload, dict):
        force_payload = {}
    if not isinstance(safety_payload, dict):
        safety_payload = {}
    if not isinstance(efficiency_payload, dict):
        efficiency_payload = {}
    if not isinstance(candidate_payload, dict):
        candidate_payload = {}
    if not isinstance(smoothness_payload, dict):
        smoothness_payload = {}

    return ScoringSpec(
        mode=str(payload.get("mode", "ranking_v1")),
        trial_indicator=TrialIndicatorSpec(
            requires_success=bool(
                trial_indicator_payload.get("requires_success", True)
            ),
            requires_steps_within_episode_limit=bool(
                trial_indicator_payload.get(
                    "requires_steps_within_episode_limit", True
                )
            ),
            requires_force_available=bool(
                trial_indicator_payload.get("requires_force_available", True)
            ),
            requires_force_within_safety_threshold=bool(
                trial_indicator_payload.get(
                    "requires_force_within_safety_threshold", True
                )
            ),
        ),
        force=ForceScoringSpec(
            default_safety_force_source=str(
                force_payload.get(
                    "default_safety_force_source", "force_total_norm_max_N"
                )
            ),
            force_max_N=float(force_payload.get("force_max_N", 2.0)),
            tip_length_mm=float(force_payload.get("tip_length_mm", 3.0)),
            tip_force_definition=str(
                force_payload.get(
                    "tip_force_definition",
                    "maximum compressive normal contact force within distal tip region",
                )
            ),
            whole_wire_force_definition=str(
                force_payload.get(
                    "whole_wire_force_definition",
                    "maximum contact force norm along the entire guidewire during the trial",
                )
            ),
        ),
        safety_score=SafetyScoreSpec(
            type=str(safety_payload.get("type", "nonlinear_product")),
            c=float(safety_payload.get("c", 0.30)),
            p=float(safety_payload.get("p", 2.0)),
            k=float(safety_payload.get("k", 10.0)),
            F50_N=float(safety_payload.get("F50_N", 1.55)),
            F_max_N=float(safety_payload.get("F_max_N", 2.0)),
        ),
        efficiency_score=EfficiencyScoreSpec(
            type=str(
                efficiency_payload.get(
                    "type", "steps_to_success_normalized_by_max_episode_steps"
                )
            )
        ),
        candidate_score=CandidateScoreSpec(
            lambda_=float(candidate_payload.get("lambda", 1.0)),
            beta=float(candidate_payload.get("beta", 0.0)),
            default_weights={
                str(key): float(value)
                for key, value in dict(
                    candidate_payload.get(
                        "default_weights",
                        {
                            "score_safety": 0.5,
                            "score_efficiency": 0.5,
                        },
                    )
                ).items()
            },
            active_components=tuple(
                str(item)
                for item in candidate_payload.get(
                    "active_components",
                    ("score_safety", "score_efficiency"),
                )
            ),
        ),
        smoothness_score=SmoothnessScoreSpec(
            jerk_scale_mm_s3=_to_optional_float(
                smoothness_payload.get("jerk_scale_mm_s3")
            )
        ),
    )


def _parse_candidate_summary(payload: dict[str, Any]) -> CandidateSummary:
    return CandidateSummary(
        scenario_name=str(payload.get("scenario_name", "")),
        candidate_name=str(payload.get("candidate_name", "")),
        execution_wire=_parse_wire_ref(payload.get("execution_wire", "")),
        trained_on_wire=(
            _parse_wire_ref(payload["trained_on_wire"])
            if payload.get("trained_on_wire")
            else None
        ),
        trial_count=int(payload.get("trial_count", 0)),
        success_rate=_to_optional_float(payload.get("success_rate")),
        valid_rate=_to_optional_float(payload.get("valid_rate")),
        soft_score_mean_valid=_to_optional_float(payload.get("soft_score_mean_valid")),
        soft_score_std_valid=_to_optional_float(payload.get("soft_score_std_valid")),
        candidate_score_final=_to_optional_float(payload.get("candidate_score_final")),
        score_mean=_to_optional_float(payload.get("score_mean")),
        score_std=_to_optional_float(payload.get("score_std")),
        steps_total_mean=_to_optional_float(payload.get("steps_total_mean")),
        steps_to_success_mean=_to_optional_float(payload.get("steps_to_success_mean")),
        tip_speed_max_mean_mm_s=_to_optional_float(payload.get("tip_speed_max_mean_mm_s")),
        wall_force_max_mean=_to_optional_float(payload.get("wall_force_max_mean")),
        wall_force_max_mean_newton=_to_optional_float(
            payload.get("wall_force_max_mean_newton")
        ),
        force_available_rate=_to_optional_float(payload.get("force_available_rate")),
    )


def _parse_force_telemetry(
    payload: Optional[dict[str, Any]]
) -> Optional[ForceTelemetrySummary]:
    if not payload:
        return None
    return ForceTelemetrySummary(
        available_for_score=bool(payload.get("available_for_score", False)),
        validation_status=str(payload.get("validation_status", "unknown")),
        validation_error=payload.get("validation_error"),
        source=str(payload.get("source", "")),
        channel=str(payload.get("channel", "")),
        quality_tier=str(payload.get("quality_tier", "unavailable")),
        association_method=str(payload.get("association_method", "none")),
        association_explicit_ratio=_to_optional_float(
            payload.get("association_explicit_ratio")
        ),
        association_coverage=_to_optional_float(payload.get("association_coverage")),
        association_explicit_force_coverage=_to_optional_float(
            payload.get("association_explicit_force_coverage")
        ),
        ordering_stable=bool(payload.get("ordering_stable", False)),
        active_constraint_any=bool(payload.get("active_constraint_any", False)),
        contact_detected_any=bool(payload.get("contact_detected_any", False)),
        contact_count_max=int(payload.get("contact_count_max", 0)),
        segment_count_max=int(payload.get("segment_count_max", 0)),
        lcp_max_abs_max=_to_optional_float(payload.get("lcp_max_abs_max")),
        lcp_sum_abs_mean=_to_optional_float(payload.get("lcp_sum_abs_mean")),
        wire_force_norm_max=_to_optional_float(payload.get("wire_force_norm_max")),
        wire_force_norm_mean=_to_optional_float(payload.get("wire_force_norm_mean")),
        collision_force_norm_max=_to_optional_float(
            payload.get("collision_force_norm_max")
        ),
        collision_force_norm_mean=_to_optional_float(
            payload.get("collision_force_norm_mean")
        ),
        total_force_norm_max=_to_optional_float(payload.get("total_force_norm_max")),
        total_force_norm_mean=_to_optional_float(payload.get("total_force_norm_mean")),
        total_force_norm_max_newton=_to_optional_float(
            payload.get("total_force_norm_max_newton")
        ),
        total_force_norm_mean_newton=_to_optional_float(
            payload.get("total_force_norm_mean_newton")
        ),
        peak_segment_force_norm=_to_optional_float(
            payload.get("peak_segment_force_norm")
        ),
        peak_segment_force_norm_newton=_to_optional_float(
            payload.get("peak_segment_force_norm_newton")
        ),
        peak_segment_force_step=(
            None
            if payload.get("peak_segment_force_step") is None
            else int(payload.get("peak_segment_force_step"))
        ),
        peak_segment_force_segment_id=(
            None
            if payload.get("peak_segment_force_segment_id") is None
            else int(payload.get("peak_segment_force_segment_id"))
        ),
        peak_segment_force_time_s=_to_optional_float(
            payload.get("peak_segment_force_time_s")
        ),
        gap_active_projected_count_sum=int(
            payload.get("gap_active_projected_count_sum", 0)
        ),
        gap_explicit_mapped_count_sum=int(
            payload.get("gap_explicit_mapped_count_sum", 0)
        ),
        gap_unmapped_count_sum=int(payload.get("gap_unmapped_count_sum", 0)),
        gap_unmapped_ratio=_to_optional_float(payload.get("gap_unmapped_ratio")),
        gap_dominant_class=str(payload.get("gap_dominant_class", "none")),
        gap_contact_mode=str(payload.get("gap_contact_mode", "none")),
        tip_force_available=bool(payload.get("tip_force_available", False)),
        tip_force_validation_status=str(
            payload.get("tip_force_validation_status", "unmapped")
        ),
        tip_force_records=tuple(payload.get("tip_force_records", ())),
        tip_force_total_vector_N=_parse_vector3(
            payload.get("tip_force_total_vector_N")
        ),
        tip_force_total_norm_N=float(payload.get("tip_force_total_norm_N", 0.0)),
        tip_force_peak_normal_N=_to_optional_float(
            payload.get("tip_force_peak_normal_N")
        ),
        tip_force_total_mean_N=_to_optional_float(
            payload.get("tip_force_total_mean_N")
        ),
        lcp_mapped_wall_row_count_max=int(
            payload.get("lcp_mapped_wall_row_count_max", 0)
        ),
        lcp_contact_export_coverage=_to_optional_float(
            payload.get("lcp_contact_export_coverage")
        ),
    )


def _h5_string_column(group: h5py.Group, name: str) -> list[str]:
    values = group[name][...]
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _load_trials_from_h5(path: Path) -> Tuple[TrialResult, ...]:
    with h5py.File(path, "r") as handle:
        group = handle["trials"]
        n_rows = int(group["trial_index"].shape[0])
        scenario_names = _h5_string_column(group, "scenario_name")
        candidate_names = _h5_string_column(group, "candidate_name")
        execution_wires = _h5_string_column(group, "execution_wire")
        trained_on_wires = _h5_string_column(group, "trained_on_wire")
        trace_paths = _h5_string_column(group, "trace_h5_path")
        rows: list[TrialResult] = []
        for idx in range(n_rows):
            force_available = bool(group["force_available_for_score"][idx])
            force_total_norm_max_N = _to_optional_float(group["force_total_norm_max_N"][idx])
            force_total_norm_mean_N = _to_optional_float(group["force_total_norm_mean_N"][idx])
            tip_force_peak_normal_N = _to_optional_float(group["tip_force_peak_normal_N"][idx])
            tip_force_total_mean_N = _to_optional_float(group["tip_force_total_mean_N"][idx])
            trace_path_text = trace_paths[idx]
            forces = ForceTelemetrySummary(
                available_for_score=force_available,
                validation_status="loaded_from_trials_h5",
                total_force_norm_max_newton=force_total_norm_max_N,
                total_force_norm_mean_newton=force_total_norm_mean_N,
                total_force_norm_max=force_total_norm_max_N,
                total_force_norm_mean=force_total_norm_mean_N,
                tip_force_peak_normal_N=tip_force_peak_normal_N,
                tip_force_total_mean_N=tip_force_total_mean_N,
            )
            rows.append(
                TrialResult(
                    scenario_name=scenario_names[idx],
                    candidate_name=candidate_names[idx],
                    execution_wire=_parse_wire_ref(execution_wires[idx]),
                    policy=PolicySpec(
                        name=f"{candidate_names[idx]}_policy",
                        checkpoint_path=Path(),
                        source="explicit",
                        trained_on_wire=(
                            _parse_wire_ref(trained_on_wires[idx])
                            if trained_on_wires[idx]
                            else None
                        ),
                    ),
                    trial_index=int(group["trial_index"][idx]),
                    seed=int(group["env_seed"][idx]),
                    policy_seed=(
                        None
                        if math.isnan(float(group["policy_seed"][idx]))
                        else int(group["policy_seed"][idx])
                    ),
                    score=ScoreBreakdown(
                        total=float(group["score_total"][idx]),
                        success=float(group["score_success"][idx]),
                        efficiency=float(group["score_efficiency"][idx]),
                        safety=float(group["score_safety"][idx]),
                        smoothness=_to_optional_float(group["score_smoothness"][idx]),
                    ),
                    telemetry=TrialTelemetrySummary(
                        success=bool(group["success"][idx]),
                        steps_total=int(group["steps_total"][idx]),
                        steps_to_success=(
                            None
                            if math.isnan(float(group["steps_to_success"][idx]))
                            else int(group["steps_to_success"][idx])
                        ),
                        episode_reward=float(group["episode_reward"][idx]),
                        wall_time_s=_to_optional_float(group["wall_time_s"][idx]),
                        sim_time_s=_to_optional_float(group["sim_time_s"][idx]),
                        tip_speed_max_mm_s=_to_optional_float(group["tip_speed_max_mm_s"][idx]),
                        tip_speed_mean_mm_s=_to_optional_float(group["tip_speed_mean_mm_s"][idx]),
                        tip_total_distance_mm=_to_optional_float(group["tip_total_distance_mm"][idx]),
                        tip_acc_p95=_to_optional_float(group["tip_acc_p95"][idx]),
                        tip_acc_max=_to_optional_float(group["tip_acc_max"][idx]),
                        tip_jerk_p95=_to_optional_float(group["tip_jerk_p95"][idx]),
                        tip_jerk_max=_to_optional_float(group["tip_jerk_max"][idx]),
                        forces=forces,
                    ),
                    valid_for_ranking=bool(group["valid_for_ranking"][idx]),
                    force_within_safety_threshold=bool(
                        group["force_within_safety_threshold"][idx]
                    ),
                    artifacts=TrialArtifactPaths(
                        trace_h5_path=Path(trace_path_text) if trace_path_text else None
                    ),
                )
            )
        return tuple(rows)


def _parse_trial_result(payload: dict[str, Any]) -> TrialResult:
    telemetry_payload = payload.get("telemetry", {})
    forces_payload = (
        telemetry_payload.get("forces") if isinstance(telemetry_payload, dict) else None
    )
    return TrialResult(
        scenario_name=str(payload.get("scenario_name", "")),
        candidate_name=str(payload.get("candidate_name", "")),
        execution_wire=_parse_wire_ref(payload.get("execution_wire", "")),
        policy=_parse_policy_spec(payload.get("policy", {})),
        trial_index=int(payload.get("trial_index", 0)),
        seed=int(payload.get("seed", 0)),
        policy_seed=(
            None
            if payload.get("policy_seed") is None
            else int(payload.get("policy_seed"))
        ),
        score=ScoreBreakdown(
            total=float(payload.get("score", {}).get("total", 0.0)),
            success=float(payload.get("score", {}).get("success", 0.0)),
            efficiency=float(payload.get("score", {}).get("efficiency", 0.0)),
            safety=float(payload.get("score", {}).get("safety", 0.0)),
            smoothness=_to_optional_float(payload.get("score", {}).get("smoothness")),
        ),
        telemetry=TrialTelemetrySummary(
            success=bool(telemetry_payload.get("success", False)),
            steps_total=int(telemetry_payload.get("steps_total", 0)),
            steps_to_success=(
                None
                if telemetry_payload.get("steps_to_success") is None
                else int(telemetry_payload.get("steps_to_success"))
            ),
            episode_reward=float(telemetry_payload.get("episode_reward", 0.0)),
            wall_time_s=_to_optional_float(telemetry_payload.get("wall_time_s")),
            sim_time_s=_to_optional_float(telemetry_payload.get("sim_time_s")),
            path_ratio_last=_to_optional_float(
                telemetry_payload.get("path_ratio_last")
            ),
            trajectory_length_last=_to_optional_float(
                telemetry_payload.get("trajectory_length_last")
            ),
            average_translation_speed_last=_to_optional_float(
                telemetry_payload.get("average_translation_speed_last")
            ),
            tip_speed_max_mm_s=_to_optional_float(
                telemetry_payload.get("tip_speed_max_mm_s")
            ),
            tip_speed_mean_mm_s=_to_optional_float(
                telemetry_payload.get("tip_speed_mean_mm_s")
            ),
            tip_total_distance_mm=_to_optional_float(
                telemetry_payload.get("tip_total_distance_mm")
            ),
            tip_acc_p95=_to_optional_float(telemetry_payload.get("tip_acc_p95")),
            tip_acc_max=_to_optional_float(telemetry_payload.get("tip_acc_max")),
            tip_jerk_p95=_to_optional_float(telemetry_payload.get("tip_jerk_p95")),
            tip_jerk_max=_to_optional_float(telemetry_payload.get("tip_jerk_max")),
            forces=_parse_force_telemetry(forces_payload),
        ),
        valid_for_ranking=bool(payload.get("valid_for_ranking", False)),
        force_within_safety_threshold=bool(
            payload.get("force_within_safety_threshold", False)
        ),
        artifacts=TrialArtifactPaths(
            trace_npz_path=None,
            trace_h5_path=(
                None
                if payload.get("artifacts", {}).get("trace_h5_path") is None
                else Path(payload.get("artifacts", {}).get("trace_h5_path"))
            ),
        ),
        warnings=tuple(str(item) for item in payload.get("warnings", ())),
    )
