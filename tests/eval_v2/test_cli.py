from __future__ import annotations

import io
import unittest
from pathlib import Path

from steve_recommender.eval_v2.cli import run_cli
from steve_recommender.eval_v2.models import (
    AgentRef,
    AnatomyBranch,
    AorticArchAnatomy,
    BranchIndexTarget,
    CandidateSummary,
    EvaluationArtifacts,
    EvaluationCandidate,
    EvaluationJob,
    EvaluationReport,
    EvaluationScenario,
    ManualTarget,
    PolicySpec,
    TargetModeDescriptor,
    VisualizationSpec,
    WireRef,
)


class _ServiceStub:
    def __init__(self) -> None:
        self.anatomy = AorticArchAnatomy(
            arch_type="II",
            seed=42,
            record_id="Tree_00",
        )
        self.branch = AnatomyBranch(
            name="lcca",
            centerline_points_vessel_cs=((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
            length_mm=12.5,
        )
        self.mode = TargetModeDescriptor(
            kind="branch_end",
            label="Branch End",
            description="Select one or more named branches and target a terminal endpoint.",
            requires_branch_selection=True,
            requires_index_selection=False,
            allows_multi_branch_selection=True,
            requires_manual_points=False,
        )
        self.execution_wire = WireRef(model="steve_default", wire="standard_j")
        self.cross_wire = WireRef(model="universal_ii", wire="standard_j")
        self.registry_policy = PolicySpec(
            name="policy_a",
            checkpoint_path=Path("/tmp/policy_a.everl"),
            trained_on_wire=self.execution_wire,
            source="registry",
        )
        self.report = EvaluationReport(
            job_name="job_x",
            generated_at="2026-04-20T12:00:00+00:00",
            summaries=(
                CandidateSummary(
                    scenario_name="scenario_x",
                    candidate_name="candidate_x",
                    execution_wire=self.execution_wire,
                    trained_on_wire=self.execution_wire,
                    trial_count=2,
                    success_rate=0.5,
                    score_mean=0.75,
                    score_std=0.1,
                    steps_total_mean=5.0,
                    steps_to_success_mean=4.0,
                    tip_speed_max_mean_mm_s=20.0,
                    wall_force_max_mean=None,
                    wall_force_max_mean_newton=None,
                    force_available_rate=0.0,
                ),
            ),
            trials=(),
            artifacts=EvaluationArtifacts(output_dir=Path("/tmp/eval_outputs/job_x")),
        )
        self.built_candidates: list[tuple[str, WireRef, PolicySpec]] = []
        self.jobs: list[EvaluationJob] = []

    def list_anatomies(self, *, registry_path=None):
        _ = registry_path
        return (self.anatomy,)

    def get_anatomy(self, *, record_id: str, registry_path=None):
        _ = registry_path
        if record_id != self.anatomy.record_id:
            raise KeyError(record_id)
        return self.anatomy

    def list_branches(self, anatomy: AorticArchAnatomy):
        assert anatomy == self.anatomy
        return (self.branch,)

    def get_branch(self, anatomy: AorticArchAnatomy, *, branch_name: str):
        assert anatomy == self.anatomy
        if branch_name != self.branch.name:
            raise KeyError(branch_name)
        return self.branch

    def list_target_modes(self):
        return (
            self.mode,
            TargetModeDescriptor(
                kind="branch_index",
                label="Branch Index",
                description="Select one branch and one exact centerline index for a fixed target.",
                requires_branch_selection=True,
                requires_index_selection=True,
                allows_multi_branch_selection=False,
                requires_manual_points=False,
            ),
            TargetModeDescriptor(
                kind="manual",
                label="Manual Coordinates",
                description="Provide explicit vessel-space coordinates instead of a named branch target.",
                requires_branch_selection=False,
                requires_index_selection=False,
                allows_multi_branch_selection=False,
                requires_manual_points=True,
            ),
        )

    def list_execution_wires(self):
        return (self.execution_wire, self.cross_wire)

    def list_startable_wires(self):
        return (self.execution_wire,)

    def list_registry_policies(self, *, execution_wire=None):
        _ = execution_wire
        return (self.registry_policy,)

    def list_explicit_policies(self, *, execution_wire=None):
        _ = execution_wire
        return ()

    def resolve_policy_from_agent_ref(self, agent_ref):
        raise KeyError(agent_ref)

    def build_candidate(
        self,
        *,
        name: str,
        execution_wire: WireRef,
        policy: PolicySpec,
    ) -> EvaluationCandidate:
        self.built_candidates.append((name, execution_wire, policy))
        return EvaluationCandidate(
            name=name,
            execution_wire=execution_wire,
            policy=policy,
        )

    def list_candidates(self, *, execution_wire: WireRef, include_cross_wire: bool = True):
        _ = include_cross_wire
        return (
            EvaluationCandidate(
                name="candidate_from_registry",
                execution_wire=execution_wire,
                policy=self.registry_policy,
            ),
        )

    def run_evaluation_job(self, job: EvaluationJob) -> EvaluationReport:
        self.jobs.append(job)
        return self.report


class CliAdapterTests(unittest.TestCase):
    def test_list_anatomies_command_prints_available_anatomies(self) -> None:
        service = _ServiceStub()
        stdout = io.StringIO()

        rc = run_cli(["list-anatomies"], service=service, stdout=stdout)

        self.assertEqual(rc, 0)
        output = stdout.getvalue()
        self.assertIn("Tree_00", output)
        self.assertIn("arch_type=II", output)

    def test_list_branches_command_prints_branch_summary(self) -> None:
        service = _ServiceStub()
        stdout = io.StringIO()

        rc = run_cli(
            ["list-branches", "--anatomy", "Tree_00"],
            service=service,
            stdout=stdout,
        )

        self.assertEqual(rc, 0)
        output = stdout.getvalue()
        self.assertIn("lcca", output)
        self.assertIn("length_mm=12.5", output)
        self.assertIn("terminal_index=1", output)

    def test_list_target_modes_command_prints_supported_modes(self) -> None:
        service = _ServiceStub()
        stdout = io.StringIO()

        rc = run_cli(["list-target-modes"], service=service, stdout=stdout)

        self.assertEqual(rc, 0)
        output = stdout.getvalue()
        self.assertIn("branch_end", output)
        self.assertIn("branch_index", output)
        self.assertIn("manual", output)

    def test_run_command_builds_branch_index_job_and_calls_service(self) -> None:
        service = _ServiceStub()
        stdout = io.StringIO()

        rc = run_cli(
            [
                "run",
                "--job-name",
                "job_x",
                "--scenario-name",
                "scenario_x",
                "--anatomy",
                "Tree_00",
                "--execution-wire",
                "steve_default/standard_j",
                "--policy-name",
                "policy_a",
                "--candidate-label",
                "candidate_x",
                "--target-mode",
                "branch_index",
                "--target-branch",
                "lcca",
                "--target-index",
                "7",
                "--threshold-mm",
                "4.5",
                "--trial-count",
                "3",
                "--base-seed",
                "500",
                "--max-episode-steps",
                "75",
                "--policy-device",
                "cpu",
                "--policy-mode",
                "deterministic",
                "--friction",
                "0.002",
                "--image-frequency-hz",
                "10.0",
                "--image-rot-z-deg",
                "15.0",
                "--image-rot-x-deg",
                "-5.0",
                "--output-root",
                "/tmp/eval_runs",
            ],
            service=service,
            stdout=stdout,
        )

        self.assertEqual(rc, 0)
        self.assertEqual(len(service.jobs), 1)
        job = service.jobs[0]
        self.assertEqual(job.name, "job_x")
        self.assertEqual(job.output_root, Path("/tmp/eval_runs"))
        self.assertEqual(job.execution.trials_per_candidate, 3)
        self.assertEqual(job.execution.base_seed, 500)
        self.assertEqual(job.execution.max_episode_steps, 75)
        self.assertEqual(job.execution.policy_device, "cpu")
        self.assertEqual(job.execution.policy_mode, "deterministic")
        self.assertEqual(job.execution.environment_seeds, (500, 501, 502))
        self.assertEqual(job.execution.policy_seeds, (None, None, None))

        scenario = job.scenarios[0]
        self.assertEqual(scenario.name, "scenario_x")
        self.assertEqual(scenario.anatomy, service.anatomy)
        self.assertEqual(scenario.friction, 0.002)
        self.assertEqual(scenario.fluoroscopy.image_frequency_hz, 10.0)
        self.assertEqual(scenario.fluoroscopy.image_rot_zx_deg, (15.0, -5.0))
        self.assertIsInstance(scenario.target, BranchIndexTarget)
        self.assertEqual(scenario.target.branch, "lcca")
        self.assertEqual(scenario.target.index, 7)
        self.assertEqual(scenario.target.threshold_mm, 4.5)

        candidate = job.candidates[0]
        self.assertEqual(candidate.name, "candidate_x")
        self.assertEqual(candidate.execution_wire, service.execution_wire)
        self.assertEqual(candidate.policy, service.registry_policy)
        self.assertIn("summaries=1", stdout.getvalue())
        self.assertIn("/tmp/eval_outputs/job_x", stdout.getvalue())

    def test_run_command_supports_explicit_checkpoint_and_manual_target(self) -> None:
        service = _ServiceStub()
        stdout = io.StringIO()

        rc = run_cli(
            [
                "run",
                "--anatomy",
                "Tree_00",
                "--execution-wire",
                "steve_default/standard_j",
                "--policy-checkpoint",
                "/tmp/manual_policy.everl",
                "--policy-label",
                "manual_policy",
                "--policy-trained-on-wire",
                "universal_ii/standard_j",
                "--target-mode",
                "manual",
                "--manual-target",
                "1.0,2.0,3.0",
                "--manual-target",
                "4.0,5.0,6.0",
            ],
            service=service,
            stdout=stdout,
        )

        self.assertEqual(rc, 0)
        self.assertEqual(len(service.jobs), 1)
        job = service.jobs[0]
        scenario = job.scenarios[0]
        candidate = job.candidates[0]
        self.assertIsInstance(scenario.target, ManualTarget)
        self.assertEqual(
            scenario.target.targets_vessel_cs,
            ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        )
        self.assertEqual(candidate.policy.name, "manual_policy")
        self.assertEqual(candidate.policy.checkpoint_path, Path("/tmp/manual_policy.everl"))
        self.assertEqual(candidate.policy.source, "explicit")
        self.assertEqual(candidate.policy.trained_on_wire, service.cross_wire)

    def test_run_command_accepts_explicit_environment_and_policy_seed_lists(self) -> None:
        service = _ServiceStub()
        stdout = io.StringIO()

        rc = run_cli(
            [
                "run",
                "--anatomy",
                "Tree_00",
                "--execution-wire",
                "steve_default/standard_j",
                "--policy-name",
                "policy_a",
                "--target-mode",
                "branch_end",
                "--target-branches",
                "lcca",
                "--trial-count",
                "3",
                "--env-seeds",
                "123,999,42",
                "--policy-mode",
                "stochastic",
                "--policy-seeds",
                "1000,1001,1002",
            ],
            service=service,
            stdout=stdout,
        )

        self.assertEqual(rc, 0)
        job = service.jobs[0]
        self.assertEqual(job.execution.environment_seeds, (123, 999, 42))
        self.assertEqual(job.execution.policy_seeds, (1000, 1001, 1002))

    def test_run_command_accepts_fixed_start_stochastic_mode(self) -> None:
        service = _ServiceStub()
        stdout = io.StringIO()

        rc = run_cli(
            [
                "run",
                "--anatomy",
                "Tree_00",
                "--execution-wire",
                "steve_default/standard_j",
                "--policy-name",
                "policy_a",
                "--target-mode",
                "branch_end",
                "--target-branches",
                "lcca",
                "--trial-count",
                "3",
                "--base-seed",
                "123",
                "--policy-mode",
                "stochastic",
                "--policy-base-seed",
                "1000",
                "--stochastic-env-mode",
                "fixed_start",
            ],
            service=service,
            stdout=stdout,
        )

        self.assertEqual(rc, 0)
        job = service.jobs[0]
        self.assertEqual(job.execution.environment_seeds, (123, 123, 123))
        self.assertEqual(job.execution.policy_seeds, (1000, 1001, 1002))

    def test_run_command_rejects_environment_seed_list_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "env-seeds"):
            run_cli(
                [
                    "run",
                    "--anatomy",
                    "Tree_00",
                    "--execution-wire",
                    "steve_default/standard_j",
                    "--policy-name",
                    "policy_a",
                    "--target-mode",
                    "branch_end",
                    "--target-branches",
                    "lcca",
                    "--trial-count",
                    "3",
                    "--env-seeds",
                    "123,999",
                ],
                service=_ServiceStub(),
                stdout=io.StringIO(),
            )

    def test_run_command_rejects_policy_seed_list_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "policy-seeds"):
            run_cli(
                [
                    "run",
                    "--anatomy",
                    "Tree_00",
                    "--execution-wire",
                    "steve_default/standard_j",
                    "--policy-name",
                    "policy_a",
                    "--target-mode",
                    "branch_end",
                    "--target-branches",
                    "lcca",
                    "--trial-count",
                    "3",
                    "--policy-mode",
                    "stochastic",
                    "--policy-seeds",
                    "1000,1001",
                ],
                service=_ServiceStub(),
                stdout=io.StringIO(),
            )

    def test_run_command_can_select_policy_by_agent_ref_when_names_collide(self) -> None:
        class _AmbiguousServiceStub(_ServiceStub):
            def __init__(self) -> None:
                super().__init__()
                self.same_name_same_wire = PolicySpec(
                    name="shared_policy",
                    checkpoint_path=Path("/tmp/shared_policy_registry.everl"),
                    trained_on_wire=self.execution_wire,
                    source="registry",
                    registry_agent=AgentRef(wire=self.execution_wire, agent="registry_agent"),
                )
                self.same_name_explicit = PolicySpec(
                    name="shared_policy",
                    checkpoint_path=Path("/tmp/shared_policy_explicit.everl"),
                    trained_on_wire=self.execution_wire,
                    source="explicit",
                    registry_agent=AgentRef(wire=self.execution_wire, agent="explicit_agent"),
                )

            def list_registry_policies(self, *, execution_wire=None):
                _ = execution_wire
                return (self.same_name_same_wire,)

            def list_explicit_policies(self, *, execution_wire=None):
                _ = execution_wire
                return (self.same_name_explicit,)

            def resolve_policy_from_agent_ref(self, agent_ref):
                if agent_ref == self.same_name_explicit.registry_agent:
                    return self.same_name_explicit
                if agent_ref == self.same_name_same_wire.registry_agent:
                    return self.same_name_same_wire
                raise KeyError(agent_ref)

        service = _AmbiguousServiceStub()
        stdout = io.StringIO()

        rc = run_cli(
            [
                "run",
                "--anatomy",
                "Tree_00",
                "--execution-wire",
                "steve_default/standard_j",
                "--policy-agent-ref",
                "steve_default/standard_j:explicit_agent",
                "--target-mode",
                "manual",
                "--manual-target",
                "1.0,2.0,3.0",
            ],
            service=service,
            stdout=stdout,
        )

        self.assertEqual(rc, 0)
        self.assertEqual(len(service.jobs), 1)
        self.assertEqual(service.jobs[0].candidates[0].policy, service.same_name_explicit)
        self.assertIn("summaries=1", stdout.getvalue())

    def test_run_command_builds_visualization_execution_settings(self) -> None:
        service = _ServiceStub()
        stdout = io.StringIO()

        rc = run_cli(
            [
                "run",
                "--anatomy",
                "Tree_00",
                "--execution-wire",
                "steve_default/standard_j",
                "--policy-name",
                "policy_a",
                "--target-mode",
                "branch_end",
                "--target-branches",
                "lcca",
                "--visualize",
                "--visualize-trials-per-candidate",
                "2",
            ],
            service=service,
            stdout=stdout,
        )

        self.assertEqual(rc, 0)
        self.assertEqual(len(service.jobs), 1)
        self.assertEqual(
            service.jobs[0].execution.visualization,
            VisualizationSpec(enabled=True, rendered_trials_per_candidate=2),
        )

    def test_run_command_stores_worker_count(self) -> None:
        service = _ServiceStub()
        stdout = io.StringIO()

        rc = run_cli(
            [
                "run",
                "--anatomy",
                "Tree_00",
                "--execution-wire",
                "steve_default/standard_j",
                "--policy-name",
                "policy_a",
                "--target-mode",
                "branch_end",
                "--target-branches",
                "lcca",
                "--workers",
                "4",
            ],
            service=service,
            stdout=stdout,
        )

        self.assertEqual(rc, 0)
        self.assertEqual(service.jobs[0].execution.worker_count, 4)

    def test_run_command_rejects_parallel_visualization(self) -> None:
        service = _ServiceStub()
        stdout = io.StringIO()

        with self.assertRaises(ValueError):
            run_cli(
                [
                    "run",
                    "--anatomy",
                    "Tree_00",
                    "--execution-wire",
                    "steve_default/standard_j",
                    "--policy-name",
                    "policy_a",
                    "--target-mode",
                    "branch_end",
                    "--target-branches",
                    "lcca",
                    "--visualize",
                    "--workers",
                    "2",
                ],
                service=service,
                stdout=stdout,
            )


if __name__ == "__main__":
    unittest.main()
