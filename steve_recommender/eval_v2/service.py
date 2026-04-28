from __future__ import annotations

import concurrent.futures
import csv
import json
import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence, Tuple

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
    PolicySpec,
    ScoreBreakdown,
    TargetModeDescriptor,
    TrialArtifactPaths,
    TrialResult,
    TrialTelemetrySummary,
    ForceTelemetrySummary,
    WireRef,
)
from .runner import configure_cpu_eval_threads, run_single_trial
from .runtime import PreparedEvaluationRuntime, prepare_evaluation_runtime
from .target_discovery import AnatomyTargetDiscovery


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
    task_runner: Callable[[_ParallelTrialTask], _ParallelTrialOutcome] = _run_parallel_trial_task,
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
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
    return variance**0.5


def _maybe_close(obj: object) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        close()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _write_report_artifacts(
    *,
    job: EvaluationJob,
    report: EvaluationReport,
) -> EvaluationArtifacts:
    output_dir = (
        report.artifacts.output_dir
        if report.artifacts is not None
        else job.output_root / job.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = output_dir / "summary.csv"
    report_json_path = output_dir / "report.json"
    report_markdown_path = output_dir / "report.md"

    summary_fieldnames = [
        "scenario_name",
        "candidate_name",
        "execution_wire",
        "trained_on_wire",
        "trial_count",
        "success_rate",
        "score_mean",
        "score_std",
        "steps_total_mean",
        "steps_to_success_mean",
        "tip_speed_max_mean_mm_s",
        "wall_force_max_mean",
        "wall_force_max_mean_newton",
        "force_available_rate",
    ]
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        for summary in report.summaries:
            writer.writerow(
                {
                    "scenario_name": summary.scenario_name,
                    "candidate_name": summary.candidate_name,
                    "execution_wire": summary.execution_wire.tool_ref,
                    "trained_on_wire": (
                        summary.trained_on_wire.tool_ref
                        if summary.trained_on_wire is not None
                        else ""
                    ),
                    "trial_count": summary.trial_count,
                    "success_rate": summary.success_rate,
                    "score_mean": summary.score_mean,
                    "score_std": summary.score_std,
                    "steps_total_mean": summary.steps_total_mean,
                    "steps_to_success_mean": summary.steps_to_success_mean,
                    "tip_speed_max_mean_mm_s": summary.tip_speed_max_mean_mm_s,
                    "wall_force_max_mean": summary.wall_force_max_mean,
                    "wall_force_max_mean_newton": summary.wall_force_max_mean_newton,
                    "force_available_rate": summary.force_available_rate,
                }
            )

    with report_json_path.open("w", encoding="utf-8") as handle:
        archive_metadata = {
            "anatomy": ", ".join(scenario.name for scenario in job.scenarios),
            "tested_wires": sorted(
                {candidate.execution_wire.tool_ref for candidate in job.candidates}
            ),
            "output_dir": str(output_dir),
        }
        payload = _jsonable(report)
        payload["archive_metadata"] = archive_metadata
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
                f"- score mean: {summary.score_mean if summary.score_mean is not None else 'n/a'}",
                "",
            ]
        )
    report_markdown_path.write_text("\n".join(lines), encoding="utf-8")

    return EvaluationArtifacts(
        output_dir=output_dir,
        summary_csv_path=summary_csv_path,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
    )


def summarize_trials(trials: Tuple[TrialResult, ...]) -> CandidateSummary:
    """Aggregate all trials for one candidate/scenario pair."""

    if not trials:
        raise ValueError("trials must not be empty")

    first = trials[0]
    return CandidateSummary(
        scenario_name=first.scenario_name,
        candidate_name=first.candidate_name,
        execution_wire=first.execution_wire,
        trained_on_wire=first.policy.trained_on_wire,
        trial_count=len(trials),
        success_rate=_finite_mean(
            tuple(1.0 if trial.telemetry.success else 0.0 for trial in trials)
        ),
        score_mean=_finite_mean(tuple(trial.score.total for trial in trials)),
        score_std=_finite_std(tuple(trial.score.total for trial in trials)),
        steps_total_mean=_finite_mean(
            tuple(float(trial.telemetry.steps_total) for trial in trials)
        ),
        steps_to_success_mean=_finite_mean(
            tuple(
                None
                if trial.telemetry.steps_to_success is None
                else float(trial.telemetry.steps_to_success)
                for trial in trials
            )
        ),
        tip_speed_max_mean_mm_s=_finite_mean(
            tuple(trial.telemetry.tip_speed_max_mm_s for trial in trials)
        ),
        wall_force_max_mean=_finite_mean(
            tuple(
                None
                if trial.telemetry.forces is None
                else trial.telemetry.forces.total_force_norm_max
                for trial in trials
            )
        ),
        wall_force_max_mean_newton=_finite_mean(
            tuple(
                None
                if trial.telemetry.forces is None
                else trial.telemetry.forces.total_force_norm_max_newton
                for trial in trials
            )
        ),
        force_available_rate=_finite_mean(
            tuple(
                None
                if trial.telemetry.forces is None
                else 1.0 if trial.telemetry.forces.available_for_score else 0.0
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
        """Return lightweight metadata for persisted reports."""

    @abstractmethod
    def load_report_from_disk(self, report_json_path: Path) -> EvaluationReport:
        """Load one full persisted report from disk."""

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
        runtime_factory: Callable[..., PreparedEvaluationRuntime] = prepare_evaluation_runtime,
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
                    for trial_index, (seed, _policy_seed) in enumerate(job.execution.trial_seed_pairs):
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
                    for trial_index, (seed, _policy_seed) in enumerate(job.execution.trial_seed_pairs):
                        trial = self._trial_runner(
                            runtime=runtime,
                            trial_index=trial_index,
                            seed=seed,
                            execution=job.execution,
                            scoring=job.scoring,
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
                )
            )
            for scenario in job.scenarios
            for candidate in job.candidates
        )
        return EvaluationReport(
            job_name=job.name,
            generated_at=self._generated_at_factory(),
            summaries=summaries,
            trials=tuple(trials),
            artifacts=EvaluationArtifacts(output_dir=job.output_root / job.name),
        )

    def _run_evaluation_job_parallel(
        self,
        job: EvaluationJob,
        *,
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
            )
            for scenario_index, scenario in enumerate(job.scenarios)
            for candidate_index, candidate in enumerate(job.candidates)
            for trial_index, (seed, policy_seed) in enumerate(job.execution.trial_seed_pairs)
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
                )
            )
            for scenario in job.scenarios
            for candidate in job.candidates
        )
        return EvaluationReport(
            job_name=job.name,
            generated_at=self._generated_at_factory(),
            summaries=summaries,
            trials=trials,
            artifacts=EvaluationArtifacts(output_dir=job.output_root / job.name),
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
                if policy.trained_on_wire is None or policy.trained_on_wire == execution_wire
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
        for report_json in sorted(root.rglob("report.json")):
            try:
                payload = _read_json_file(report_json)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            archive_metadata = payload.get("archive_metadata", {})
            if not isinstance(archive_metadata, dict):
                archive_metadata = {}

            job_name = str(payload.get("job_name", report_json.parent.name))
            generated_at = str(payload.get("generated_at", ""))
            anatomy = str(archive_metadata.get("anatomy", "unknown"))
            tested_wires_raw = archive_metadata.get("tested_wires", ())
            tested_wires = tuple(str(wire) for wire in tested_wires_raw) if isinstance(tested_wires_raw, list) else ()

            summaries.append(
                HistoricalReportSummary(
                    job_name=job_name,
                    generated_at=generated_at,
                    anatomy=anatomy,
                    tested_wires=tested_wires,
                    report_json_path=report_json,
                    output_dir=report_json.parent,
                )
            )

        summaries.sort(key=lambda summary: summary.generated_at, reverse=True)
        return tuple(summaries)

    def load_report_from_disk(self, report_json_path: Path) -> EvaluationReport:
        payload = _read_json_file(Path(report_json_path))
        if not isinstance(payload, dict):
            raise TypeError("report.json payload must be an object")

        summaries = tuple(
            CandidateSummary(
                scenario_name=str(item["scenario_name"]),
                candidate_name=str(item["candidate_name"]),
                execution_wire=_parse_wire_ref(item["execution_wire"]),
                trained_on_wire=(
                    _parse_wire_ref(item["trained_on_wire"])
                    if item.get("trained_on_wire")
                    else None
                ),
                trial_count=int(item["trial_count"]),
                success_rate=_to_optional_float(item.get("success_rate")),
                score_mean=_to_optional_float(item.get("score_mean")),
                score_std=_to_optional_float(item.get("score_std")),
                steps_total_mean=_to_optional_float(item.get("steps_total_mean")),
                steps_to_success_mean=_to_optional_float(item.get("steps_to_success_mean")),
                tip_speed_max_mean_mm_s=_to_optional_float(item.get("tip_speed_max_mean_mm_s")),
                wall_force_max_mean=_to_optional_float(item.get("wall_force_max_mean")),
                wall_force_max_mean_newton=_to_optional_float(item.get("wall_force_max_mean_newton")),
                force_available_rate=_to_optional_float(item.get("force_available_rate")),
            )
            for item in payload.get("summaries", ())
        )

        trials = tuple(_parse_trial_result(item) for item in payload.get("trials", ()))

        artifacts = EvaluationArtifacts(
            output_dir=Path(report_json_path).parent,
            summary_csv_path=Path(report_json_path).parent / "summary.csv",
            report_json_path=Path(report_json_path),
            report_markdown_path=Path(report_json_path).parent / "report.md",
        )
        return EvaluationReport(
            job_name=str(payload.get("job_name", Path(report_json_path).parent.name)),
            generated_at=str(payload.get("generated_at", "")),
            summaries=summaries,
            trials=trials,
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
        return f"{policy.name} [{trained_on_wire.tool_ref} -> {execution_wire.tool_ref}]"


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
        return WireRef(model=str(value.get("model", "")), wire=str(value.get("wire", "")))
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


def _parse_force_telemetry(payload: Optional[dict[str, Any]]) -> Optional[ForceTelemetrySummary]:
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
        association_explicit_ratio=_to_optional_float(payload.get("association_explicit_ratio")),
        association_coverage=_to_optional_float(payload.get("association_coverage")),
        association_explicit_force_coverage=_to_optional_float(payload.get("association_explicit_force_coverage")),
        ordering_stable=bool(payload.get("ordering_stable", False)),
        active_constraint_any=bool(payload.get("active_constraint_any", False)),
        contact_detected_any=bool(payload.get("contact_detected_any", False)),
        contact_count_max=int(payload.get("contact_count_max", 0)),
        segment_count_max=int(payload.get("segment_count_max", 0)),
        lcp_max_abs_max=_to_optional_float(payload.get("lcp_max_abs_max")),
        lcp_sum_abs_mean=_to_optional_float(payload.get("lcp_sum_abs_mean")),
        wire_force_norm_max=_to_optional_float(payload.get("wire_force_norm_max")),
        wire_force_norm_mean=_to_optional_float(payload.get("wire_force_norm_mean")),
        collision_force_norm_max=_to_optional_float(payload.get("collision_force_norm_max")),
        collision_force_norm_mean=_to_optional_float(payload.get("collision_force_norm_mean")),
        total_force_norm_max=_to_optional_float(payload.get("total_force_norm_max")),
        total_force_norm_mean=_to_optional_float(payload.get("total_force_norm_mean")),
        total_force_norm_max_newton=_to_optional_float(payload.get("total_force_norm_max_newton")),
        total_force_norm_mean_newton=_to_optional_float(payload.get("total_force_norm_mean_newton")),
        peak_segment_force_norm=_to_optional_float(payload.get("peak_segment_force_norm")),
        peak_segment_force_norm_newton=_to_optional_float(payload.get("peak_segment_force_norm_newton")),
        peak_segment_force_step=(None if payload.get("peak_segment_force_step") is None else int(payload.get("peak_segment_force_step"))),
        peak_segment_force_segment_id=(None if payload.get("peak_segment_force_segment_id") is None else int(payload.get("peak_segment_force_segment_id"))),
        peak_segment_force_time_s=_to_optional_float(payload.get("peak_segment_force_time_s")),
        gap_active_projected_count_sum=int(payload.get("gap_active_projected_count_sum", 0)),
        gap_explicit_mapped_count_sum=int(payload.get("gap_explicit_mapped_count_sum", 0)),
        gap_unmapped_count_sum=int(payload.get("gap_unmapped_count_sum", 0)),
        gap_unmapped_ratio=_to_optional_float(payload.get("gap_unmapped_ratio")),
        gap_dominant_class=str(payload.get("gap_dominant_class", "none")),
        gap_contact_mode=str(payload.get("gap_contact_mode", "none")),
        tip_force_available=bool(payload.get("tip_force_available", False)),
        tip_force_validation_status=str(payload.get("tip_force_validation_status", "unmapped")),
        tip_force_records=tuple(payload.get("tip_force_records", ())),
        tip_force_total_vector_N=_parse_vector3(payload.get("tip_force_total_vector_N")),
        tip_force_total_norm_N=float(payload.get("tip_force_total_norm_N", 0.0)),
        lcp_mapped_wall_row_count_max=int(payload.get("lcp_mapped_wall_row_count_max", 0)),
        lcp_contact_export_coverage=_to_optional_float(payload.get("lcp_contact_export_coverage")),
    )


def _parse_trial_result(payload: dict[str, Any]) -> TrialResult:
    telemetry_payload = payload.get("telemetry", {})
    forces_payload = telemetry_payload.get("forces") if isinstance(telemetry_payload, dict) else None
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
            safety=_to_optional_float(payload.get("score", {}).get("safety")),
            smoothness=float(payload.get("score", {}).get("smoothness", 0.0)),
        ),
        telemetry=TrialTelemetrySummary(
            success=bool(telemetry_payload.get("success", False)),
            steps_total=int(telemetry_payload.get("steps_total", 0)),
            steps_to_success=(None if telemetry_payload.get("steps_to_success") is None else int(telemetry_payload.get("steps_to_success"))),
            episode_reward=float(telemetry_payload.get("episode_reward", 0.0)),
            wall_time_s=_to_optional_float(telemetry_payload.get("wall_time_s")),
            sim_time_s=_to_optional_float(telemetry_payload.get("sim_time_s")),
            path_ratio_last=_to_optional_float(telemetry_payload.get("path_ratio_last")),
            trajectory_length_last=_to_optional_float(telemetry_payload.get("trajectory_length_last")),
            average_translation_speed_last=_to_optional_float(telemetry_payload.get("average_translation_speed_last")),
            tip_speed_max_mm_s=_to_optional_float(telemetry_payload.get("tip_speed_max_mm_s")),
            tip_speed_mean_mm_s=_to_optional_float(telemetry_payload.get("tip_speed_mean_mm_s")),
            forces=_parse_force_telemetry(forces_payload),
        ),
        artifacts=TrialArtifactPaths(),
    )
