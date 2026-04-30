from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv

from steve_recommender.eval_v2.models import (
    AnatomyBranch,
    AorticArchAnatomy,
    BranchEndTarget,
    CandidateSummary,
    EvaluationCandidate,
    EvaluationJob,
    EvaluationReport,
    EvaluationScenario,
    ExecutionPlan,
    ForceTelemetrySummary,
    PolicySpec,
    ScoreBreakdown,
    ScoringSpec,
    TargetModeDescriptor,
    TrialResult,
    TrialTelemetrySummary,
    WireRef,
)
from steve_recommender.eval_v2.service import (
    DefaultEvaluationService,
    LocalEvaluationRunner,
    _ParallelTrialTask,
    _ParallelTrialOutcome,
    _run_parallel_trial_tasks_process,
    pre_write_meshes_for_job,
)


def _wire(model: str, wire: str) -> WireRef:
    return WireRef(model=model, wire=wire)


def _policy(name: str, trained_on_wire: WireRef | None) -> PolicySpec:
    return PolicySpec(
        name=name,
        checkpoint_path=Path(f"/tmp/{name}.everl"),
        trained_on_wire=trained_on_wire,
    )


def _candidate(
    name: str, execution_wire: WireRef, policy: PolicySpec
) -> EvaluationCandidate:
    return EvaluationCandidate(
        name=name,
        execution_wire=execution_wire,
        policy=policy,
    )


def _scenario(name: str = "scenario_a") -> EvaluationScenario:
    return EvaluationScenario(
        name=name,
        anatomy=AorticArchAnatomy(arch_type="II", seed=42),
        target=BranchEndTarget(threshold_mm=5.0, branches=("lcca",)),
    )


def _write_source_mesh(path: Path) -> None:
    mesh = pv.PolyData(
        np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        np.asarray([3, 0, 1, 2], dtype=np.int64),
    )
    mesh.save(path)


def _scenario_with_mesh(name: str, mesh_path: Path) -> EvaluationScenario:
    return EvaluationScenario(
        name=name,
        anatomy=AorticArchAnatomy(
            arch_type="II",
            seed=42,
            record_id="Tree_00",
            simulation_mesh_path=mesh_path,
        ),
        target=BranchEndTarget(threshold_mm=5.0, branches=("lcca",)),
    )


def _trial(
    *,
    scenario_name: str,
    candidate_name: str,
    execution_wire: WireRef,
    policy: PolicySpec,
    trial_index: int,
    seed: int,
    policy_seed: int | None = None,
    success: bool,
    score_total: float,
    steps_total: int,
    steps_to_success: int | None,
    tip_speed_max_mm_s: float | None,
    force_available_for_score: bool,
    wall_force_max: float | None = None,
    wall_force_max_newton: float | None = None,
) -> TrialResult:
    from steve_recommender.eval_v2.models import ForceTelemetrySummary

    forces = ForceTelemetrySummary(
        available_for_score=force_available_for_score,
        validation_status="stub",
        total_force_norm_max=wall_force_max,
        total_force_norm_max_newton=wall_force_max_newton,
    )
    return TrialResult(
        scenario_name=scenario_name,
        candidate_name=candidate_name,
        execution_wire=execution_wire,
        policy=policy,
        trial_index=trial_index,
        seed=seed,
        policy_seed=policy_seed,
        score=ScoreBreakdown(
            total=score_total,
            success=1.0 if success else 0.0,
            efficiency=1.0 if steps_to_success == 1 else 0.0,
            safety=None if not force_available_for_score else 1.0,
            smoothness=1.0,
        ),
        telemetry=TrialTelemetrySummary(
            success=success,
            steps_total=steps_total,
            steps_to_success=steps_to_success,
            episode_reward=score_total,
            tip_speed_max_mm_s=tip_speed_max_mm_s,
            forces=forces,
        ),
    )


@dataclass
class _RuntimeStub:
    candidate: EvaluationCandidate
    scenario: EvaluationScenario
    play_policy: object
    intervention: object
    device: object = object()


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _parallel_smoke_task_runner(task: _ParallelTrialTask) -> _ParallelTrialOutcome:
    trial = _trial(
        scenario_name=task.scenario.name,
        candidate_name=task.candidate.name,
        execution_wire=task.candidate.execution_wire,
        policy=task.candidate.policy,
        trial_index=task.trial_index,
        seed=task.seed,
        policy_seed=task.policy_seed,
        success=True,
        score_total=float(task.trial_index + 1),
        steps_total=task.trial_index + 1,
        steps_to_success=task.trial_index + 1,
        tip_speed_max_mm_s=1.0,
        force_available_for_score=False,
    )
    return _ParallelTrialOutcome(
        scenario_index=task.scenario_index,
        candidate_index=task.candidate_index,
        trial_index=task.trial_index,
        trial=trial,
    )


class _AnatomyDiscoveryStub:
    def __init__(self, anatomies: tuple[AorticArchAnatomy, ...]) -> None:
        self._anatomies = anatomies

    def list_anatomies(self, *, registry_path=None):
        _ = registry_path
        return self._anatomies

    def get_anatomy(self, *, record_id: str, registry_path=None):
        _ = registry_path
        for anatomy in self._anatomies:
            if anatomy.record_id == record_id:
                return anatomy
        raise KeyError(record_id)


class _PolicyDiscoveryStub:
    def __init__(
        self,
        *,
        execution_wires: tuple[WireRef, ...],
        startable_wires: tuple[WireRef, ...],
        registry_policies: tuple[PolicySpec, ...],
        explicit_policies: tuple[PolicySpec, ...],
    ) -> None:
        self._execution_wires = execution_wires
        self._startable_wires = startable_wires
        self._registry_policies = registry_policies
        self._explicit_policies = explicit_policies

    def list_execution_wires(self):
        return self._execution_wires

    def list_startable_wires(self):
        return self._startable_wires

    def list_registry_policies(self, *, execution_wire=None):
        if execution_wire is None:
            return self._registry_policies
        return tuple(
            policy
            for policy in self._registry_policies
            if policy.trained_on_wire == execution_wire
        )

    def list_explicit_policies(self, *, execution_wire=None):
        _ = execution_wire
        return self._explicit_policies

    def resolve_policy_from_agent_ref(self, agent_ref):
        raise KeyError(agent_ref)


class _ExplicitPolicyDiscoveryStub:
    def __init__(self, explicit_policies: tuple[PolicySpec, ...]) -> None:
        self._explicit_policies = explicit_policies

    def list_explicit_policies(self, *, execution_wire=None):
        if execution_wire is None:
            return self._explicit_policies
        return tuple(
            policy
            for policy in self._explicit_policies
            if policy.trained_on_wire == execution_wire
        )

    def resolve_policy_from_agent_ref(self, agent_ref):
        for policy in self._explicit_policies:
            if policy.registry_agent == agent_ref:
                return policy
        raise KeyError(agent_ref)


class _TargetDiscoveryStub:
    def __init__(self) -> None:
        self._branches = (
            AnatomyBranch(
                name="lcca",
                centerline_points_vessel_cs=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                length_mm=1.0,
            ),
        )
        self._modes = (
            TargetModeDescriptor(
                kind="branch_end",
                label="Branch End",
                description="Terminal branch endpoint",
                requires_branch_selection=True,
                requires_index_selection=False,
                allows_multi_branch_selection=True,
                requires_manual_points=False,
            ),
        )

    def list_branches(self, anatomy: AorticArchAnatomy):
        _ = anatomy
        return self._branches

    def get_branch(self, anatomy: AorticArchAnatomy, *, branch_name: str):
        _ = anatomy
        for branch in self._branches:
            if branch.name == branch_name:
                return branch
        raise KeyError(branch_name)

    def list_target_modes(self):
        return self._modes


class _RunnerStub:
    def __init__(self, report: EvaluationReport) -> None:
        self.report = report
        self.jobs: list[EvaluationJob] = []

    def run_evaluation_job(
        self,
        job: EvaluationJob,
        *,
        frame_callback=None,
        progress_callback=None,
    ) -> EvaluationReport:
        _ = frame_callback, progress_callback
        self.jobs.append(job)
        return self.report


class LocalEvaluationRunnerTests(unittest.TestCase):
    def test_parallel_task_runner_process_pool_smoke(self) -> None:
        execution_wire = _wire("steve_default", "standard_j")
        policy = _policy("policy_a", execution_wire)
        candidate = _candidate("candidate_a", execution_wire, policy)
        scenario = _scenario("scenario_a")
        execution = ExecutionPlan(
            trials_per_candidate=3,
            base_seed=500,
            policy_device="cpu",
            worker_count=2,
        )
        tasks = tuple(
            _ParallelTrialTask(
                scenario_index=0,
                candidate_index=0,
                trial_index=trial_index,
                seed=seed,
                policy_seed=policy_seed,
                scenario=scenario,
                candidate=candidate,
                execution=execution,
                scoring=ScoringSpec(),
                registry_path=Path("/tmp/unused_registry.json"),
                policy_device="cpu",
                output_dir=Path("/tmp/eval_outputs/job_parallel"),
            )
            for trial_index, (seed, policy_seed) in enumerate(
                execution.trial_seed_pairs
            )
        )
        progress_events: list[str] = []

        outcomes = _run_parallel_trial_tasks_process(
            tasks,
            worker_count=2,
            task_runner=_parallel_smoke_task_runner,
            progress_callback=progress_events.append,
        )

        self.assertEqual(len(outcomes), 3)
        self.assertEqual([item.trial_index for item in outcomes], [0, 1, 2])
        self.assertEqual([item.trial.seed for item in outcomes], [500, 501, 502])
        self.assertIn("parallel_start workers=2 total=3", progress_events)
        self.assertIn("parallel_end completed=3 total=3", progress_events)
        self.assertEqual(
            len(
                [
                    event
                    for event in progress_events
                    if event.startswith("parallel_trial_done")
                ]
            ),
            3,
        )

    def test_run_evaluation_job_aggregates_trials_into_candidate_summaries(
        self,
    ) -> None:
        execution_wire = _wire("steve_default", "standard_j")
        policy = _policy("policy_a", execution_wire)
        candidate = _candidate("candidate_a", execution_wire, policy)
        scenario = _scenario("scenario_a")
        job = EvaluationJob(
            name="job_a",
            scenarios=(scenario,),
            candidates=(candidate,),
            execution=ExecutionPlan(
                trials_per_candidate=2,
                base_seed=100,
                policy_device="cpu",
            ),
            scoring=ScoringSpec(),
            output_root=Path("/tmp/eval_outputs"),
        )

        runtime_factory_calls: list[
            tuple[EvaluationCandidate, EvaluationScenario, str]
        ] = []
        trial_runner_calls: list[tuple[int, int]] = []
        play_policy = _Closable()
        intervention = _Closable()

        def runtime_factory(
            *,
            candidate,
            scenario,
            simulation=None,
            registry_path=None,
            policy_device="cpu",
        ):
            _ = simulation, registry_path
            runtime_factory_calls.append((candidate, scenario, policy_device))
            return _RuntimeStub(
                candidate=candidate,
                scenario=scenario,
                play_policy=play_policy,
                intervention=intervention,
            )

        def trial_runner(
            *,
            runtime,
            trial_index,
            seed,
            execution,
            scoring,
            output_dir=None,
            frame_callback=None,
            progress_callback=None,
        ):
            _ = (
                runtime,
                execution,
                scoring,
                output_dir,
                frame_callback,
                progress_callback,
            )
            trial_runner_calls.append((trial_index, seed))
            if trial_index == 0:
                return _trial(
                    scenario_name="scenario_a",
                    candidate_name="candidate_a",
                    execution_wire=execution_wire,
                    policy=policy,
                    trial_index=trial_index,
                    seed=seed,
                    success=True,
                    score_total=0.75,
                    steps_total=4,
                    steps_to_success=4,
                    tip_speed_max_mm_s=20.0,
                    force_available_for_score=True,
                    wall_force_max=0.4,
                    wall_force_max_newton=0.2,
                )
            return _trial(
                scenario_name="scenario_a",
                candidate_name="candidate_a",
                execution_wire=execution_wire,
                policy=policy,
                trial_index=trial_index,
                seed=seed,
                success=False,
                score_total=0.25,
                steps_total=7,
                steps_to_success=None,
                tip_speed_max_mm_s=40.0,
                force_available_for_score=False,
                wall_force_max=None,
                wall_force_max_newton=None,
            )

        runner = LocalEvaluationRunner(
            registry_path=Path("/tmp/wire_registry/index.json"),
            runtime_factory=runtime_factory,
            trial_runner=trial_runner,
            generated_at_factory=lambda: "2026-04-20T12:00:00+00:00",
        )

        report = runner.run_evaluation_job(job)

        self.assertEqual(len(runtime_factory_calls), 1)
        self.assertEqual(trial_runner_calls, [(0, 100), (1, 101)])
        self.assertTrue(play_policy.closed)
        self.assertTrue(intervention.closed)

        self.assertEqual(report.job_name, "job_a")
        self.assertEqual(report.generated_at, "2026-04-20T12:00:00+00:00")
        self.assertEqual(len(report.trials), 2)
        self.assertEqual(report.artifacts.output_dir.parent, Path("/tmp/eval_outputs"))
        self.assertTrue(report.artifacts.output_dir.name.startswith("job_a_"))
        self.assertEqual(len(report.summaries), 1)

        summary = report.summaries[0]
        self.assertIsInstance(summary, CandidateSummary)
        self.assertEqual(summary.scenario_name, "scenario_a")
        self.assertEqual(summary.candidate_name, "candidate_a")
        self.assertEqual(summary.execution_wire, execution_wire)
        self.assertEqual(summary.trained_on_wire, execution_wire)
        self.assertEqual(summary.trial_count, 2)
        self.assertAlmostEqual(summary.success_rate, 0.5)
        self.assertAlmostEqual(summary.score_mean, 0.5)
        self.assertAlmostEqual(summary.score_std, 0.35355339059)
        self.assertAlmostEqual(summary.steps_total_mean, 5.5)
        self.assertAlmostEqual(summary.steps_to_success_mean, 4.0)
        self.assertAlmostEqual(summary.tip_speed_max_mean_mm_s, 30.0)
        self.assertAlmostEqual(summary.wall_force_max_mean, 0.4)
        self.assertAlmostEqual(summary.wall_force_max_mean_newton, 0.2)
        self.assertAlmostEqual(summary.force_available_rate, 0.5)

    def test_run_evaluation_job_reuses_same_seed_schedule_for_all_candidates(
        self,
    ) -> None:
        execution_wire_a = _wire("steve_default", "standard_j")
        execution_wire_b = _wire("steve_default", "tight_j")
        policy_a = _policy("policy_a", execution_wire_a)
        policy_b = _policy("policy_b", execution_wire_b)
        candidate_a = _candidate("candidate_a", execution_wire_a, policy_a)
        candidate_b = _candidate("candidate_b", execution_wire_b, policy_b)
        scenario = _scenario("scenario_a")
        job = EvaluationJob(
            name="job_seed_schedule",
            scenarios=(scenario,),
            candidates=(candidate_a, candidate_b),
            execution=ExecutionPlan(
                trials_per_candidate=3,
                base_seed=123,
                policy_mode="stochastic",
                policy_base_seed=1000,
                stochastic_environment_mode="fixed_start",
                policy_device="cpu",
            ),
        )

        trial_runner_calls: list[tuple[str, int, int, int | None]] = []
        play_policy = _Closable()
        intervention = _Closable()

        def runtime_factory(
            *,
            candidate,
            scenario,
            simulation=None,
            registry_path=None,
            policy_device="cpu",
        ):
            _ = simulation, registry_path, policy_device
            return _RuntimeStub(
                candidate=candidate,
                scenario=scenario,
                play_policy=play_policy,
                intervention=intervention,
            )

        def trial_runner(
            *,
            runtime,
            trial_index,
            seed,
            execution,
            scoring,
            output_dir=None,
            frame_callback=None,
            progress_callback=None,
        ):
            _ = scoring, output_dir, frame_callback, progress_callback
            policy_seed = execution.policy_seeds[trial_index]
            trial_runner_calls.append(
                (runtime.candidate.name, trial_index, seed, policy_seed)
            )
            return _trial(
                scenario_name=runtime.scenario.name,
                candidate_name=runtime.candidate.name,
                execution_wire=runtime.candidate.execution_wire,
                policy=runtime.candidate.policy,
                trial_index=trial_index,
                seed=seed,
                policy_seed=policy_seed,
                success=True,
                score_total=1.0,
                steps_total=5,
                steps_to_success=5,
                tip_speed_max_mm_s=10.0,
                force_available_for_score=False,
            )

        runner = LocalEvaluationRunner(
            runtime_factory=runtime_factory,
            trial_runner=trial_runner,
        )
        report = runner.run_evaluation_job(job)

        self.assertEqual(
            trial_runner_calls,
            [
                ("candidate_a", 0, 123, 1000),
                ("candidate_a", 1, 123, 1001),
                ("candidate_a", 2, 123, 1002),
                ("candidate_b", 0, 123, 1000),
                ("candidate_b", 1, 123, 1001),
                ("candidate_b", 2, 123, 1002),
            ],
        )
        self.assertEqual(
            [(trial.seed, trial.policy_seed) for trial in report.trials],
            [
                (123, 1000),
                (123, 1001),
                (123, 1002),
                (123, 1000),
                (123, 1001),
                (123, 1002),
            ],
        )

    def test_run_evaluation_job_closes_runtime_resources_on_trial_failure(self) -> None:
        execution_wire = _wire("steve_default", "standard_j")
        policy = _policy("policy_a", execution_wire)
        candidate = _candidate("candidate_a", execution_wire, policy)
        scenario = _scenario("scenario_a")
        job = EvaluationJob(
            name="job_fail",
            scenarios=(scenario,),
            candidates=(candidate,),
            execution=ExecutionPlan(
                trials_per_candidate=1, base_seed=200, policy_device="cpu"
            ),
        )
        play_policy = _Closable()
        intervention = _Closable()

        def runtime_factory(
            *,
            candidate,
            scenario,
            simulation=None,
            registry_path=None,
            policy_device="cpu",
        ):
            _ = candidate, scenario, simulation, registry_path, policy_device
            return _RuntimeStub(
                candidate=candidate,
                scenario=scenario,
                play_policy=play_policy,
                intervention=intervention,
            )

        def failing_trial_runner(
            *,
            runtime,
            trial_index,
            seed,
            execution,
            scoring,
            output_dir=None,
            frame_callback=None,
            progress_callback=None,
        ):
            _ = (
                runtime,
                trial_index,
                seed,
                execution,
                scoring,
                output_dir,
                frame_callback,
                progress_callback,
            )
            raise RuntimeError("boom")

        runner = LocalEvaluationRunner(
            runtime_factory=runtime_factory,
            trial_runner=failing_trial_runner,
        )

        with self.assertRaises(RuntimeError):
            runner.run_evaluation_job(job)

        self.assertTrue(play_policy.closed)
        self.assertTrue(intervention.closed)

    def test_run_evaluation_job_forwards_frame_and_progress_callbacks_to_trial_runner(
        self,
    ) -> None:
        execution_wire = _wire("steve_default", "standard_j")
        policy = _policy("policy_a", execution_wire)
        candidate = _candidate("candidate_a", execution_wire, policy)
        scenario = _scenario("scenario_a")
        job = EvaluationJob(
            name="job_callbacks",
            scenarios=(scenario,),
            candidates=(candidate,),
            execution=ExecutionPlan(
                trials_per_candidate=1, base_seed=300, policy_device="cpu"
            ),
        )

        play_policy = _Closable()
        intervention = _Closable()
        frame_events: list[object] = []
        progress_events: list[str] = []

        def runtime_factory(
            *,
            candidate,
            scenario,
            simulation=None,
            registry_path=None,
            policy_device="cpu",
        ):
            _ = candidate, scenario, simulation, registry_path, policy_device
            return _RuntimeStub(
                candidate=candidate,
                scenario=scenario,
                play_policy=play_policy,
                intervention=intervention,
            )

        def trial_runner(
            *,
            runtime,
            trial_index,
            seed,
            execution,
            scoring,
            output_dir=None,
            frame_callback=None,
            progress_callback=None,
        ):
            _ = runtime, trial_index, seed, execution, scoring, output_dir
            if frame_callback is not None:
                frame_callback("frame")
            if progress_callback is not None:
                progress_callback("progress")
            return _trial(
                scenario_name="scenario_a",
                candidate_name="candidate_a",
                execution_wire=execution_wire,
                policy=policy,
                trial_index=0,
                seed=300,
                success=True,
                score_total=0.5,
                steps_total=5,
                steps_to_success=5,
                tip_speed_max_mm_s=10.0,
                force_available_for_score=False,
            )

        runner = LocalEvaluationRunner(
            runtime_factory=runtime_factory,
            trial_runner=trial_runner,
        )

        report = runner.run_evaluation_job(
            job,
            frame_callback=frame_events.append,
            progress_callback=progress_events.append,
        )

        self.assertEqual(report.job_name, "job_callbacks")
        self.assertEqual(frame_events, ["frame"])
        self.assertEqual(
            progress_events,
            ["runtime_prepare scenario=scenario_a candidate=candidate_a", "progress"],
        )

    def test_run_evaluation_job_parallel_path_aggregates_worker_outcomes(self) -> None:
        execution_wire = _wire("steve_default", "standard_j")
        policy = _policy("policy_a", execution_wire)
        candidate = _candidate("candidate_a", execution_wire, policy)
        scenario = _scenario("scenario_a")
        job = EvaluationJob(
            name="job_parallel",
            scenarios=(scenario,),
            candidates=(candidate,),
            execution=ExecutionPlan(
                trials_per_candidate=3,
                base_seed=400,
                policy_device="cpu",
                worker_count=2,
            ),
            scoring=ScoringSpec(),
            output_root=Path("/tmp/eval_outputs"),
        )
        seen_worker_count: list[int] = []
        progress_events: list[str] = []

        def parallel_task_runner(tasks, *, worker_count, progress_callback=None):
            seen_worker_count.append(worker_count)
            if progress_callback is not None:
                progress_callback(
                    f"parallel_start workers={worker_count} total={len(tasks)}"
                )
            outcomes = []
            for task in tasks:
                trial = _trial(
                    scenario_name=task.scenario.name,
                    candidate_name=task.candidate.name,
                    execution_wire=task.candidate.execution_wire,
                    policy=task.candidate.policy,
                    trial_index=task.trial_index,
                    seed=task.seed,
                    policy_seed=task.policy_seed,
                    success=(task.trial_index % 2 == 0),
                    score_total=float(task.trial_index + 1),
                    steps_total=task.trial_index + 5,
                    steps_to_success=task.trial_index + 5,
                    tip_speed_max_mm_s=10.0,
                    force_available_for_score=False,
                )
                outcomes.append(
                    _ParallelTrialOutcome(
                        scenario_index=task.scenario_index,
                        candidate_index=task.candidate_index,
                        trial_index=task.trial_index,
                        trial=trial,
                    )
                )
            if progress_callback is not None:
                progress_callback(
                    f"parallel_end completed={len(outcomes)} total={len(tasks)}"
                )
            return tuple(outcomes)

        runner = LocalEvaluationRunner(
            parallel_task_runner=parallel_task_runner,
            generated_at_factory=lambda: "2026-04-20T12:00:00+00:00",
        )

        report = runner.run_evaluation_job(
            job, progress_callback=progress_events.append
        )

        self.assertEqual(seen_worker_count, [2])
        self.assertEqual(len(report.trials), 3)
        self.assertEqual([trial.seed for trial in report.trials], [400, 401, 402])
        self.assertEqual(len(report.summaries), 1)
        self.assertEqual(report.summaries[0].trial_count, 3)
        self.assertIn("parallel_start workers=2 total=3", progress_events)
        self.assertIn("parallel_end completed=3 total=3", progress_events)

    def test_run_evaluation_job_parallel_path_preserves_policy_seed_schedule(
        self,
    ) -> None:
        execution_wire = _wire("steve_default", "standard_j")
        policy = _policy("policy_a", execution_wire)
        candidate = _candidate("candidate_a", execution_wire, policy)
        scenario = _scenario("scenario_a")
        job = EvaluationJob(
            name="job_parallel_seed_schedule",
            scenarios=(scenario,),
            candidates=(candidate,),
            execution=ExecutionPlan(
                trials_per_candidate=3,
                base_seed=123,
                policy_mode="stochastic",
                policy_base_seed=1000,
                stochastic_environment_mode="fixed_start",
                policy_device="cpu",
                worker_count=2,
            ),
        )

        seen_pairs: list[tuple[int, int, int | None]] = []

        def parallel_task_runner(tasks, *, worker_count, progress_callback=None):
            _ = worker_count, progress_callback
            outcomes = []
            for task in tasks:
                seen_pairs.append((task.trial_index, task.seed, task.policy_seed))
                outcomes.append(
                    _ParallelTrialOutcome(
                        scenario_index=task.scenario_index,
                        candidate_index=task.candidate_index,
                        trial_index=task.trial_index,
                        trial=_trial(
                            scenario_name=task.scenario.name,
                            candidate_name=task.candidate.name,
                            execution_wire=task.candidate.execution_wire,
                            policy=task.candidate.policy,
                            trial_index=task.trial_index,
                            seed=task.seed,
                            policy_seed=task.policy_seed,
                            success=True,
                            score_total=1.0,
                            steps_total=5,
                            steps_to_success=5,
                            tip_speed_max_mm_s=10.0,
                            force_available_for_score=False,
                        ),
                    )
                )
            return tuple(outcomes)

        runner = LocalEvaluationRunner(parallel_task_runner=parallel_task_runner)
        report = runner.run_evaluation_job(job)

        self.assertEqual(seen_pairs, [(0, 123, 1000), (1, 123, 1001), (2, 123, 1002)])
        self.assertEqual(
            [(trial.seed, trial.policy_seed) for trial in report.trials],
            [(123, 1000), (123, 1001), (123, 1002)],
        )

    def test_run_evaluation_job_parallel_rejects_cuda_policy_device(self) -> None:
        execution_wire = _wire("steve_default", "standard_j")
        policy = _policy("policy_a", execution_wire)
        candidate = _candidate("candidate_a", execution_wire, policy)
        job = EvaluationJob(
            name="job_parallel_cuda",
            scenarios=(_scenario("scenario_a"),),
            candidates=(candidate,),
            execution=ExecutionPlan(
                trials_per_candidate=1,
                policy_device="cuda",
                worker_count=2,
            ),
        )
        runner = LocalEvaluationRunner()

        with self.assertRaises(ValueError):
            runner.run_evaluation_job(job)

    def test_full_job_produces_n_trace_files_for_n_trials(self) -> None:
        execution_wire_a = _wire("steve_default", "standard_j")
        execution_wire_b = _wire("steve_default", "tight_j")
        policy_a = _policy("policy_a", execution_wire_a)
        policy_b = _policy("policy_b", execution_wire_b)
        candidate_a = _candidate("candidate_a", execution_wire_a, policy_a)
        candidate_b = _candidate("candidate_b", execution_wire_b, policy_b)
        scenario = _scenario("scenario_a")

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            job = EvaluationJob(
                name="job_trace_count",
                scenarios=(scenario,),
                candidates=(candidate_a, candidate_b),
                execution=ExecutionPlan(
                    trials_per_candidate=2,
                    base_seed=123,
                    policy_device="cpu",
                ),
                output_root=output_root,
            )

            play_policy = _Closable()
            intervention = _Closable()

            def runtime_factory(
                *,
                candidate,
                scenario,
                simulation=None,
                registry_path=None,
                policy_device="cpu",
            ):
                _ = simulation, registry_path, policy_device
                return _RuntimeStub(
                    candidate=candidate,
                    scenario=scenario,
                    play_policy=play_policy,
                    intervention=intervention,
                )

            def trial_runner(
                *,
                runtime,
                trial_index,
                seed,
                execution,
                scoring,
                output_dir=None,
                frame_callback=None,
                progress_callback=None,
            ):
                _ = execution, scoring, frame_callback, progress_callback
                assert output_dir is not None
                trace_dir = Path(output_dir) / "traces"
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = (
                    trace_dir
                    / f"trial_{runtime.candidate.name}_{seed}_{trial_index}.h5"
                )
                trace_path.write_bytes(b"trace")
                return _trial(
                    scenario_name=runtime.scenario.name,
                    candidate_name=runtime.candidate.name,
                    execution_wire=runtime.candidate.execution_wire,
                    policy=runtime.candidate.policy,
                    trial_index=trial_index,
                    seed=seed,
                    success=True,
                    score_total=1.0,
                    steps_total=5,
                    steps_to_success=5,
                    tip_speed_max_mm_s=10.0,
                    force_available_for_score=False,
                )

            runner = LocalEvaluationRunner(
                runtime_factory=runtime_factory,
                trial_runner=trial_runner,
            )

            report = runner.run_evaluation_job(job)

            trace_files = sorted((report.artifacts.output_dir / "traces").glob("*.h5"))
            self.assertEqual(len(trace_files), 4)

    def test_service_pre_writes_meshes_before_workers(self) -> None:
        execution_wire = _wire("steve_default", "standard_j")
        policy = _policy("policy_a", execution_wire)
        candidate = _candidate("candidate_a", execution_wire, policy)

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            source_mesh_path = Path(tmp) / "source_mesh.vtp"
            _write_source_mesh(source_mesh_path)
            scenario = _scenario_with_mesh("scenario_mesh", source_mesh_path)
            job = EvaluationJob(
                name="job_mesh_prewrite",
                scenarios=(scenario,),
                candidates=(candidate,),
                execution=ExecutionPlan(
                    trials_per_candidate=1,
                    base_seed=123,
                    policy_device="cpu",
                    worker_count=2,
                ),
                output_root=output_root,
            )
            observed_mesh_state: list[tuple[bool, float]] = []

            def parallel_task_runner(tasks, *, worker_count, progress_callback=None):
                _ = worker_count, progress_callback
                mesh_path = tasks[0].output_dir / "meshes" / "anatomy_Tree_00.h5"
                observed_mesh_state.append(
                    (mesh_path.exists(), mesh_path.stat().st_mtime)
                )
                return tuple(
                    _ParallelTrialOutcome(
                        scenario_index=task.scenario_index,
                        candidate_index=task.candidate_index,
                        trial_index=task.trial_index,
                        trial=_trial(
                            scenario_name=task.scenario.name,
                            candidate_name=task.candidate.name,
                            execution_wire=task.candidate.execution_wire,
                            policy=task.candidate.policy,
                            trial_index=task.trial_index,
                            seed=task.seed,
                            policy_seed=task.policy_seed,
                            success=True,
                            score_total=1.0,
                            steps_total=5,
                            steps_to_success=5,
                            tip_speed_max_mm_s=10.0,
                            force_available_for_score=False,
                        ),
                    )
                    for task in tasks
                )

            runner = LocalEvaluationRunner(parallel_task_runner=parallel_task_runner)
            runner.run_evaluation_job(job)

            self.assertEqual(len(observed_mesh_state), 1)
            self.assertTrue(observed_mesh_state[0][0])

    def test_meshes_written_once_per_anatomy_not_per_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "job_mesh_once"
            source_mesh_path = Path(tmp) / "source_mesh.vtp"
            _write_source_mesh(source_mesh_path)
            scenario = _scenario_with_mesh("scenario_mesh", source_mesh_path)
            job = EvaluationJob(
                name="job_mesh_once",
                scenarios=(scenario,),
                candidates=(
                    _candidate(
                        "candidate_a",
                        _wire("steve_default", "standard_j"),
                        _policy("policy_a", _wire("steve_default", "standard_j")),
                    ),
                ),
                execution=ExecutionPlan(
                    trials_per_candidate=4,
                    base_seed=123,
                    policy_device="cpu",
                ),
                output_root=Path(tmp),
            )

            written = pre_write_meshes_for_job(job, output_dir)

            self.assertEqual(len(written), 1)
            self.assertEqual(len(list((output_dir / "meshes").glob("*.h5"))), 1)


class DefaultEvaluationServiceTests(unittest.TestCase):
    def test_service_delegates_discovery_target_selection_candidate_building_and_execution(
        self,
    ) -> None:
        execution_wire = _wire("steve_default", "standard_j")
        cross_wire = _wire("universal_ii", "standard_j")
        anatomy = AorticArchAnatomy(
            arch_type="II",
            seed=42,
            record_id="Tree_00",
        )
        same_wire_policy = _policy("same_wire_policy", execution_wire)
        cross_wire_policy = _policy("cross_wire_policy", cross_wire)
        explicit_policy = _policy("explicit_policy", None)
        report = EvaluationReport(
            job_name="job_x",
            generated_at="2026-04-20T12:30:00+00:00",
            summaries=(),
            trials=(),
        )

        anatomy_discovery = _AnatomyDiscoveryStub((anatomy,))
        policy_discovery = _PolicyDiscoveryStub(
            execution_wires=(execution_wire, cross_wire),
            startable_wires=(execution_wire,),
            registry_policies=(same_wire_policy, cross_wire_policy),
            explicit_policies=(explicit_policy,),
        )
        explicit_policy_discovery = _ExplicitPolicyDiscoveryStub((explicit_policy,))
        target_discovery = _TargetDiscoveryStub()
        runner = _RunnerStub(report)
        service = DefaultEvaluationService(
            anatomy_discovery=anatomy_discovery,
            policy_discovery=policy_discovery,
            explicit_policy_discovery=explicit_policy_discovery,
            target_discovery=target_discovery,
            evaluation_runner=runner,
        )

        self.assertEqual(service.list_anatomies(), (anatomy,))
        self.assertEqual(service.get_anatomy(record_id="Tree_00"), anatomy)
        self.assertEqual(service.list_execution_wires(), (execution_wire, cross_wire))
        self.assertEqual(service.list_startable_wires(), (execution_wire,))
        self.assertEqual(
            service.list_registry_policies(execution_wire=execution_wire),
            (same_wire_policy,),
        )
        self.assertEqual(service.list_explicit_policies(), (explicit_policy,))
        self.assertEqual(
            service.list_branches(anatomy), target_discovery.list_branches(anatomy)
        )
        self.assertEqual(
            service.list_target_modes(), target_discovery.list_target_modes()
        )

        built = service.build_candidate(
            name="manual_candidate",
            execution_wire=execution_wire,
            policy=same_wire_policy,
        )
        self.assertEqual(built.name, "manual_candidate")
        self.assertEqual(built.execution_wire, execution_wire)
        self.assertEqual(built.policy, same_wire_policy)

        all_candidates = service.list_candidates(execution_wire=execution_wire)
        no_cross_wire = service.list_candidates(
            execution_wire=execution_wire,
            include_cross_wire=False,
        )

        self.assertEqual(len(all_candidates), 3)
        self.assertEqual(len(no_cross_wire), 2)
        self.assertTrue(
            any(candidate.policy == cross_wire_policy for candidate in all_candidates)
        )
        self.assertFalse(
            any(candidate.policy == cross_wire_policy for candidate in no_cross_wire)
        )

        job = EvaluationJob(
            name="job_x",
            scenarios=(_scenario("scenario_x"),),
            candidates=(built,),
        )
        returned_report = service.run_evaluation_job(job)

        self.assertEqual(returned_report.job_name, report.job_name)
        self.assertIsNotNone(returned_report.artifacts)
        assert returned_report.artifacts is not None
        self.assertTrue(returned_report.artifacts.output_dir.exists())
        self.assertEqual(runner.jobs, [job])

    def test_run_evaluation_job_writes_artifacts_to_output_root(self) -> None:
        execution_wire = _wire("steve_default", "standard_j")
        policy = _policy("policy_a", execution_wire)
        candidate = _candidate("candidate_a", execution_wire, policy)
        scenario = _scenario("scenario_a")
        report = EvaluationReport(
            job_name="job_outputs",
            generated_at="2026-04-20T12:30:00+00:00",
            summaries=(
                CandidateSummary(
                    scenario_name="scenario_a",
                    candidate_name="candidate_a",
                    execution_wire=execution_wire,
                    trained_on_wire=execution_wire,
                    trial_count=1,
                    success_rate=1.0,
                    valid_rate=1.0,
                    soft_score_mean_valid=0.9,
                    soft_score_std_valid=0.0,
                    candidate_score_final=0.9,
                    score_mean=0.9,
                    score_std=0.0,
                    steps_total_mean=3.0,
                    steps_to_success_mean=1.0,
                    tip_speed_max_mean_mm_s=12.5,
                    wall_force_max_mean=None,
                    wall_force_max_mean_newton=None,
                    force_available_rate=0.0,
                ),
            ),
            trials=(),
        )

        anatomy_discovery = _AnatomyDiscoveryStub((scenario.anatomy,))
        policy_discovery = _PolicyDiscoveryStub(
            execution_wires=(execution_wire,),
            startable_wires=(execution_wire,),
            registry_policies=(policy,),
            explicit_policies=(),
        )
        explicit_policy_discovery = _ExplicitPolicyDiscoveryStub(())
        target_discovery = _TargetDiscoveryStub()
        runner = _RunnerStub(report)

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            job = EvaluationJob(
                name="job_outputs",
                scenarios=(scenario,),
                candidates=(candidate,),
                output_root=output_root,
            )
            service = DefaultEvaluationService(
                anatomy_discovery=anatomy_discovery,
                policy_discovery=policy_discovery,
                explicit_policy_discovery=explicit_policy_discovery,
                target_discovery=target_discovery,
                evaluation_runner=runner,
            )

            returned_report = service.run_evaluation_job(job)

            artifacts = returned_report.artifacts
            self.assertIsNotNone(artifacts)
            assert artifacts is not None
            self.assertTrue(artifacts.output_dir.exists())
            self.assertTrue(artifacts.candidate_summaries_csv_path.exists())
            self.assertTrue(artifacts.candidate_summaries_json_path.exists())
            self.assertTrue(artifacts.manifest_json_path.exists())
            self.assertTrue(artifacts.trials_h5_path.exists())
            self.assertTrue(artifacts.report_markdown_path.exists())

            self.assertTrue(artifacts.output_dir.parent == output_root)
            self.assertTrue(artifacts.output_dir.name.startswith("job_outputs_"))

            with artifacts.candidate_summaries_csv_path.open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["candidate_name"], "candidate_a")
            self.assertEqual(rows[0]["candidate_score_final"], "0.9")

            payload = json.loads(artifacts.manifest_json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["job_name"], "job_outputs")
            self.assertEqual(len(payload["summaries"]), 1)
            self.assertEqual(payload["schema_version"], 1)
            with h5py.File(artifacts.trials_h5_path, "r") as handle:
                self.assertIn("trials", handle)

    def test_list_historical_reports_discovers_metadata_from_output_root(self) -> None:
        execution_wire = _wire("steve_default", "standard_j")
        policy = _policy("policy_a", execution_wire)
        candidate = _candidate("candidate_a", execution_wire, policy)
        scenario = _scenario("scenario_archive")
        report = EvaluationReport(
            job_name="archive_job",
            generated_at="2026-04-21T10:00:00+00:00",
            summaries=(
                CandidateSummary(
                    scenario_name="scenario_archive",
                    candidate_name="candidate_a",
                    execution_wire=execution_wire,
                    trained_on_wire=execution_wire,
                    trial_count=1,
                    success_rate=1.0,
                    valid_rate=1.0,
                    soft_score_mean_valid=0.8,
                    soft_score_std_valid=0.0,
                    candidate_score_final=0.8,
                    score_mean=0.8,
                    score_std=0.0,
                    steps_total_mean=3.0,
                    steps_to_success_mean=2.0,
                    tip_speed_max_mean_mm_s=15.0,
                    wall_force_max_mean=0.2,
                    wall_force_max_mean_newton=0.1,
                    force_available_rate=1.0,
                ),
            ),
            trials=(),
        )

        anatomy_discovery = _AnatomyDiscoveryStub((scenario.anatomy,))
        policy_discovery = _PolicyDiscoveryStub(
            execution_wires=(execution_wire,),
            startable_wires=(execution_wire,),
            registry_policies=(policy,),
            explicit_policies=(),
        )
        explicit_policy_discovery = _ExplicitPolicyDiscoveryStub(())
        target_discovery = _TargetDiscoveryStub()
        runner = _RunnerStub(report)

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            job = EvaluationJob(
                name="archive_job",
                scenarios=(scenario,),
                candidates=(candidate,),
                output_root=output_root,
            )
            service = DefaultEvaluationService(
                anatomy_discovery=anatomy_discovery,
                policy_discovery=policy_discovery,
                explicit_policy_discovery=explicit_policy_discovery,
                target_discovery=target_discovery,
                evaluation_runner=runner,
            )

            service.run_evaluation_job(job)
            summaries = service.list_historical_reports(output_root=output_root)

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary.job_name, "archive_job")
        self.assertEqual(summary.generated_at, "2026-04-21T10:00:00+00:00")
        self.assertIn("scenario_archive", summary.anatomy)
        self.assertEqual(summary.tested_wires, ("steve_default/standard_j",))
        self.assertTrue(summary.manifest_json_path.name == "manifest.json")

    def test_load_manifest_from_disk_reconstructs_full_evaluation_report(self) -> None:
        execution_wire = _wire("steve_default", "standard_j")
        policy = _policy("policy_a", execution_wire)
        candidate = _candidate("candidate_a", execution_wire, policy)
        scenario = _scenario("scenario_load")
        report = EvaluationReport(
            job_name="load_job",
            generated_at="2026-04-21T11:00:00+00:00",
            summaries=(
                CandidateSummary(
                    scenario_name="scenario_load",
                    candidate_name="candidate_a",
                    execution_wire=execution_wire,
                    trained_on_wire=execution_wire,
                    trial_count=1,
                    success_rate=1.0,
                    valid_rate=1.0,
                    soft_score_mean_valid=0.8,
                    soft_score_std_valid=0.0,
                    candidate_score_final=0.8,
                    score_mean=0.8,
                    score_std=0.0,
                    steps_total_mean=3.0,
                    steps_to_success_mean=2.0,
                    tip_speed_max_mean_mm_s=15.0,
                    wall_force_max_mean=0.2,
                    wall_force_max_mean_newton=0.1,
                    force_available_rate=1.0,
                ),
            ),
            trials=(
                TrialResult(
                    scenario_name="scenario_load",
                    candidate_name="candidate_a",
                    execution_wire=execution_wire,
                    policy=policy,
                    trial_index=0,
                    seed=123,
                    policy_seed=None,
                    score=ScoreBreakdown(
                        total=0.8,
                        success=1.0,
                        efficiency=1.0,
                        safety=1.0,
                        smoothness=1.0,
                    ),
                    telemetry=TrialTelemetrySummary(
                        success=True,
                        steps_total=3,
                        steps_to_success=2,
                        episode_reward=0.8,
                        tip_speed_max_mm_s=15.0,
                        forces=ForceTelemetrySummary(
                            available_for_score=True,
                            validation_status="stub",
                            tip_force_available=True,
                            tip_force_validation_status="ok",
                            tip_force_records=(
                                {"wire_collision_dof": 2, "is_tip": True},
                            ),
                            tip_force_total_vector_N=(0.1, 0.2, 0.3),
                            tip_force_total_norm_N=0.3741657387,
                            tip_force_peak_normal_N=0.2,
                            tip_force_total_mean_N=0.1,
                        ),
                    ),
                    valid_for_ranking=True,
                    force_within_safety_threshold=True,
                ),
            ),
        )

        anatomy_discovery = _AnatomyDiscoveryStub((scenario.anatomy,))
        policy_discovery = _PolicyDiscoveryStub(
            execution_wires=(execution_wire,),
            startable_wires=(execution_wire,),
            registry_policies=(policy,),
            explicit_policies=(),
        )
        explicit_policy_discovery = _ExplicitPolicyDiscoveryStub(())
        target_discovery = _TargetDiscoveryStub()
        runner = _RunnerStub(report)

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            job = EvaluationJob(
                name="load_job",
                scenarios=(scenario,),
                candidates=(candidate,),
                output_root=output_root,
            )
            service = DefaultEvaluationService(
                anatomy_discovery=anatomy_discovery,
                policy_discovery=policy_discovery,
                explicit_policy_discovery=explicit_policy_discovery,
                target_discovery=target_discovery,
                evaluation_runner=runner,
            )

            persisted = service.run_evaluation_job(job)
            assert persisted.artifacts is not None
            loaded = service.load_manifest_from_disk(
                persisted.artifacts.manifest_json_path
            )

        self.assertEqual(loaded.job_name, "load_job")
        self.assertEqual(loaded.generated_at, "2026-04-21T11:00:00+00:00")
        self.assertEqual(len(loaded.summaries), 1)
        self.assertEqual(loaded.summaries[0].candidate_name, "candidate_a")
        assert loaded.trials[0].telemetry.forces is not None
        self.assertTrue(loaded.trials[0].telemetry.forces.available_for_score)
        self.assertAlmostEqual(
            loaded.trials[0].telemetry.forces.tip_force_peak_normal_N, 0.2
        )

    def test_save_clinical_feedback_writes_feedback_json_next_to_report(self) -> None:
        execution_wire = _wire("steve_default", "standard_j")
        policy = _policy("policy_a", execution_wire)
        candidate = _candidate("candidate_a", execution_wire, policy)
        scenario = _scenario("scenario_feedback")
        report = EvaluationReport(
            job_name="feedback_job",
            generated_at="2026-04-21T12:00:00+00:00",
            summaries=(),
            trials=(),
        )

        anatomy_discovery = _AnatomyDiscoveryStub((scenario.anatomy,))
        policy_discovery = _PolicyDiscoveryStub(
            execution_wires=(execution_wire,),
            startable_wires=(execution_wire,),
            registry_policies=(policy,),
            explicit_policies=(),
        )
        explicit_policy_discovery = _ExplicitPolicyDiscoveryStub(())
        target_discovery = _TargetDiscoveryStub()
        runner = _RunnerStub(report)

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            job = EvaluationJob(
                name="feedback_job",
                scenarios=(scenario,),
                candidates=(candidate,),
                output_root=output_root,
            )
            service = DefaultEvaluationService(
                anatomy_discovery=anatomy_discovery,
                policy_discovery=policy_discovery,
                explicit_policy_discovery=explicit_policy_discovery,
                target_discovery=target_discovery,
                evaluation_runner=runner,
            )

            persisted = service.run_evaluation_job(job)
            assert persisted.artifacts is not None
            feedback_path = service.save_clinical_feedback(
                report_id=str(persisted.artifacts.output_dir),
                feedback_data={
                    "wire_actually_used": "steve_default/standard_j",
                    "clinical_outcome": "Success",
                    "clinician_notes": "Stable vessel access",
                },
            )

            payload = json.loads(feedback_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["wire_actually_used"], "steve_default/standard_j")
        self.assertEqual(payload["clinical_outcome"], "Success")
        self.assertEqual(payload["clinician_notes"], "Stable vessel access")


if __name__ == "__main__":
    unittest.main()
