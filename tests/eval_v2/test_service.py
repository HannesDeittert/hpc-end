from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path

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
    PolicySpec,
    ScoreBreakdown,
    ScoringSpec,
    TargetModeDescriptor,
    TrialResult,
    TrialTelemetrySummary,
    WireRef,
)
from steve_recommender.eval_v2.runtime import PreparedEvaluationRuntime
from steve_recommender.eval_v2.service import DefaultEvaluationService, LocalEvaluationRunner


def _wire(model: str, wire: str) -> WireRef:
    return WireRef(model=model, wire=wire)


def _policy(name: str, trained_on_wire: WireRef | None) -> PolicySpec:
    return PolicySpec(
        name=name,
        checkpoint_path=Path(f"/tmp/{name}.everl"),
        trained_on_wire=trained_on_wire,
    )


def _candidate(name: str, execution_wire: WireRef, policy: PolicySpec) -> EvaluationCandidate:
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


def _trial(
    *,
    scenario_name: str,
    candidate_name: str,
    execution_wire: WireRef,
    policy: PolicySpec,
    trial_index: int,
    seed: int,
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

    def run_evaluation_job(self, job: EvaluationJob) -> EvaluationReport:
        self.jobs.append(job)
        return self.report


class LocalEvaluationRunnerTests(unittest.TestCase):
    def test_run_evaluation_job_aggregates_trials_into_candidate_summaries(self) -> None:
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

        runtime_factory_calls: list[tuple[EvaluationCandidate, EvaluationScenario, str]] = []
        trial_runner_calls: list[tuple[int, int]] = []
        play_policy = _Closable()
        intervention = _Closable()

        def runtime_factory(*, candidate, scenario, simulation=None, registry_path=None, policy_device="cpu"):
            _ = simulation, registry_path
            runtime_factory_calls.append((candidate, scenario, policy_device))
            return _RuntimeStub(
                candidate=candidate,
                scenario=scenario,
                play_policy=play_policy,
                intervention=intervention,
            )

        def trial_runner(*, runtime, trial_index, seed, execution, scoring):
            _ = runtime, execution, scoring
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
        self.assertEqual(report.artifacts.output_dir, Path("/tmp/eval_outputs") / "job_a")
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
        self.assertAlmostEqual(summary.score_std, 0.25)
        self.assertAlmostEqual(summary.steps_total_mean, 5.5)
        self.assertAlmostEqual(summary.steps_to_success_mean, 4.0)
        self.assertAlmostEqual(summary.tip_speed_max_mean_mm_s, 30.0)
        self.assertAlmostEqual(summary.wall_force_max_mean, 0.4)
        self.assertAlmostEqual(summary.wall_force_max_mean_newton, 0.2)
        self.assertAlmostEqual(summary.force_available_rate, 0.5)

    def test_run_evaluation_job_closes_runtime_resources_on_trial_failure(self) -> None:
        execution_wire = _wire("steve_default", "standard_j")
        policy = _policy("policy_a", execution_wire)
        candidate = _candidate("candidate_a", execution_wire, policy)
        scenario = _scenario("scenario_a")
        job = EvaluationJob(
            name="job_fail",
            scenarios=(scenario,),
            candidates=(candidate,),
            execution=ExecutionPlan(trials_per_candidate=1, base_seed=200, policy_device="cpu"),
        )
        play_policy = _Closable()
        intervention = _Closable()

        def runtime_factory(*, candidate, scenario, simulation=None, registry_path=None, policy_device="cpu"):
            _ = candidate, scenario, simulation, registry_path, policy_device
            return _RuntimeStub(
                candidate=candidate,
                scenario=scenario,
                play_policy=play_policy,
                intervention=intervention,
            )

        def failing_trial_runner(*, runtime, trial_index, seed, execution, scoring):
            _ = runtime, trial_index, seed, execution, scoring
            raise RuntimeError("boom")

        runner = LocalEvaluationRunner(
            runtime_factory=runtime_factory,
            trial_runner=failing_trial_runner,
        )

        with self.assertRaises(RuntimeError):
            runner.run_evaluation_job(job)

        self.assertTrue(play_policy.closed)
        self.assertTrue(intervention.closed)


class DefaultEvaluationServiceTests(unittest.TestCase):
    def test_service_delegates_discovery_target_selection_candidate_building_and_execution(self) -> None:
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
        target_discovery = _TargetDiscoveryStub()
        runner = _RunnerStub(report)
        service = DefaultEvaluationService(
            anatomy_discovery=anatomy_discovery,
            policy_discovery=policy_discovery,
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
        self.assertEqual(service.list_branches(anatomy), target_discovery.list_branches(anatomy))
        self.assertEqual(service.list_target_modes(), target_discovery.list_target_modes())

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
        self.assertTrue(any(candidate.policy == cross_wire_policy for candidate in all_candidates))
        self.assertFalse(any(candidate.policy == cross_wire_policy for candidate in no_cross_wire))

        job = EvaluationJob(
            name="job_x",
            scenarios=(_scenario("scenario_x"),),
            candidates=(built,),
        )
        returned_report = service.run_evaluation_job(job)

        self.assertIs(returned_report, report)
        self.assertEqual(runner.jobs, [job])


if __name__ == "__main__":
    unittest.main()
