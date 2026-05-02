"""
Integration test: train → persist → eval scoring.

Verifies that:
  1. A trial with a known wire_force_normal_trial_max_N value is persisted and
     reloaded with that value intact.
  2. score_safety() and force_within_safety_threshold() operate on
     wire_force_normal_trial_max_N and produce the expected values.
  3. Every persisted artifact (HDF5 trials, JSON manifest) carries
     metric_version = "v2_normal_component" and does NOT contain any legacy
     field names (total_force_norm_max_newton, tip_force_peak_normal_N, etc.).
  4. The candidate summary aggregates wire_force_normal_trial_max_mean_N
     correctly.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import h5py

from steve_recommender.eval_v2.models import (
    AorticArchAnatomy,
    BranchEndTarget,
    CandidateSummary,
    EvaluationArtifacts,
    EvaluationCandidate,
    EvaluationJob,
    EvaluationReport,
    EvaluationScenario,
    ExecutionPlan,
    ForceScoringSpec,
    ForceTelemetrySummary,
    PolicySpec,
    SafetyScoreSpec,
    ScoreBreakdown,
    ScoringSpec,
    TrialArtifactPaths,
    TrialResult,
    TrialTelemetrySummary,
    WireRef,
)
from steve_recommender.eval_v2.scoring import (
    force_within_safety_threshold,
    score_safety,
    score_trial,
)
from steve_recommender.eval_v2.service import DefaultEvaluationService, summarize_trials
from tests.eval_v2.test_service import (
    _AnatomyDiscoveryStub,
    _ExplicitPolicyDiscoveryStub,
    _PolicyDiscoveryStub,
    _TargetDiscoveryStub,
    _scenario,
)


def _wire() -> WireRef:
    return WireRef(model="steve_default", wire="standard_j")


def _policy() -> PolicySpec:
    return PolicySpec(
        name="policy_a",
        checkpoint_path=Path("/tmp/policy_a.everl"),
        trained_on_wire=_wire(),
    )


def _candidate() -> EvaluationCandidate:
    return EvaluationCandidate(name="candidate_a", execution_wire=_wire(), policy=_policy())


def _force_summary(*, normal_max_N: float, available: bool) -> ForceTelemetrySummary:
    return ForceTelemetrySummary(
        available_for_score=available,
        validation_status="ok" if available else "unavailable",
        wire_force_normal_instant_N=normal_max_N,
        wire_force_normal_trial_max_N=normal_max_N,
        wire_force_normal_trial_mean_N=normal_max_N * 0.7,
        tip_force_normal_instant_N=normal_max_N * 0.4,
        tip_force_normal_trial_max_N=normal_max_N * 0.4,
        tip_force_normal_trial_mean_N=normal_max_N * 0.3,
    )


def _trial(
    *,
    index: int,
    normal_max_N: float,
    steps_to_success: int,
    scoring: ScoringSpec,
) -> TrialResult:
    forces = _force_summary(normal_max_N=normal_max_N, available=True)
    telemetry = TrialTelemetrySummary(
        success=True,
        steps_total=steps_to_success,
        steps_to_success=steps_to_success,
        episode_reward=1.0,
        forces=forces,
    )
    breakdown = score_trial(telemetry=telemetry, max_episode_steps=20, scoring=scoring)
    return TrialResult(
        scenario_name="scenario_a",
        candidate_name="candidate_a",
        execution_wire=_wire(),
        policy=_policy(),
        trial_index=index,
        seed=100 + index,
        policy_seed=None,
        score=breakdown,
        telemetry=telemetry,
        valid_for_ranking=True,
        force_within_safety_threshold=force_within_safety_threshold(
            telemetry=telemetry, scoring=scoring
        ),
        artifacts=TrialArtifactPaths(trace_h5_path=Path(f"/tmp/trace_{index}.h5")),
    )


class _RunnerStub:
    def __init__(self, report: EvaluationReport) -> None:
        self.report = report

    def run_evaluation_job(self, job, *, frame_callback=None, progress_callback=None):
        _ = job, frame_callback, progress_callback
        return self.report


class ForcePipelineIntegrationTests(unittest.TestCase):
    """End-to-end: known force value → persist → reload → score correctly."""

    def setUp(self) -> None:
        self.scoring = ScoringSpec(
            force=ForceScoringSpec(force_max_N=2.0),
            safety_score=SafetyScoreSpec(c=0.30, p=2.0, k=10.0, F50_N=1.55, F_max_N=2.0),
        )

    def test_score_safety_uses_wire_force_normal_trial_max_not_any_other_field(self) -> None:
        forces_correct = ForceTelemetrySummary(
            available_for_score=True,
            validation_status="ok",
            wire_force_normal_trial_max_N=1.0,
        )
        forces_decoy = ForceTelemetrySummary(
            available_for_score=True,
            validation_status="ok",
            wire_force_normal_trial_max_N=1.0,
            total_force_norm_max=99.0,
        )
        telemetry = TrialTelemetrySummary(
            success=True, steps_total=5, steps_to_success=5,
            episode_reward=1.0, forces=forces_correct,
        )
        telemetry_decoy = TrialTelemetrySummary(
            success=True, steps_total=5, steps_to_success=5,
            episode_reward=1.0, forces=forces_decoy,
        )
        expected = score_safety(force_N=1.0, scoring=self.scoring)
        self.assertAlmostEqual(
            score_trial(telemetry=telemetry, max_episode_steps=10, scoring=self.scoring).safety,
            expected,
        )
        self.assertAlmostEqual(
            score_trial(telemetry=telemetry_decoy, max_episode_steps=10, scoring=self.scoring).safety,
            expected,
        )

    def test_force_within_safety_threshold_gates_on_wire_force_normal_trial_max(self) -> None:
        below = TrialTelemetrySummary(
            success=True, steps_total=5, steps_to_success=5, episode_reward=1.0,
            forces=ForceTelemetrySummary(
                available_for_score=True, validation_status="ok",
                wire_force_normal_trial_max_N=1.9,
                total_force_norm_max=9.9,
            ),
        )
        above = TrialTelemetrySummary(
            success=True, steps_total=5, steps_to_success=5, episode_reward=1.0,
            forces=ForceTelemetrySummary(
                available_for_score=True, validation_status="ok",
                wire_force_normal_trial_max_N=2.1,
                total_force_norm_max=0.1,
            ),
        )
        self.assertTrue(force_within_safety_threshold(telemetry=below, scoring=self.scoring))
        self.assertFalse(force_within_safety_threshold(telemetry=above, scoring=self.scoring))

    def test_persist_and_reload_round_trips_wire_force_normal_trial_max(self) -> None:
        scenario = _scenario("scenario_a")
        candidate = _candidate()
        trial_a = _trial(index=0, normal_max_N=0.8, steps_to_success=5, scoring=self.scoring)
        trial_b = _trial(index=1, normal_max_N=1.4, steps_to_success=8, scoring=self.scoring)

        summary = summarize_trials((trial_a, trial_b), scoring=self.scoring)
        self.assertAlmostEqual(summary.wire_force_normal_trial_max_mean_N, 1.1)

        report = EvaluationReport(
            job_name="integration_test",
            generated_at="2026-05-01T00:00:00+00:00",
            summaries=(summary,),
            trials=(trial_a, trial_b),
            artifacts=EvaluationArtifacts(output_dir=Path("/tmp/integration_test")),
        )
        service = DefaultEvaluationService(
            anatomy_discovery=_AnatomyDiscoveryStub((scenario.anatomy,)),
            policy_discovery=_PolicyDiscoveryStub(
                execution_wires=(candidate.execution_wire,),
                startable_wires=(candidate.execution_wire,),
                registry_policies=(candidate.policy,),
                explicit_policies=(),
            ),
            explicit_policy_discovery=_ExplicitPolicyDiscoveryStub(()),
            target_discovery=_TargetDiscoveryStub(),
            evaluation_runner=_RunnerStub(report),
        )

        with tempfile.TemporaryDirectory() as tmp:
            job = EvaluationJob(
                name="integration_test",
                scenarios=(scenario,),
                candidates=(candidate,),
                execution=ExecutionPlan(trials_per_candidate=2, base_seed=100, policy_device="cpu"),
                scoring=self.scoring,
                output_root=Path(tmp),
            )
            persisted = service.run_evaluation_job(job)
            artifacts = persisted.artifacts
            assert artifacts is not None

            # Verify JSON manifest
            manifest = json.loads(artifacts.manifest_json_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["metric_version"], "v2_normal_component")
            self.assertAlmostEqual(
                manifest["summaries"][0]["wire_force_normal_trial_max_mean_N"], 1.1
            )
            # Old field names must not appear in persisted JSON
            manifest_text = artifacts.manifest_json_path.read_text(encoding="utf-8")
            for banned in (
                "total_force_norm_max_newton",
                "total_force_norm_mean_newton",
                "tip_force_total_norm_N",
                "tip_force_peak_normal_N",
                "wall_force_max_mean_newton",
                "force_total_norm_max_N",
                "force_total_norm_mean_N",
                "tip_force_total_mean_N",
            ):
                self.assertNotIn(banned, manifest_text, msg=f"banned field '{banned}' found in manifest JSON")

            # Verify HDF5 trials
            with h5py.File(artifacts.trials_h5_path, "r") as handle:
                self.assertEqual(handle.attrs["metric_version"], "v2_normal_component")
                group = handle["trials"]
                # New fields must be present
                self.assertIn("wire_force_normal_trial_max_N", group)
                self.assertIn("wire_force_normal_instant_N", group)
                self.assertIn("metric_version", group)
                # Old fields must not be present
                for banned_col in (
                    "force_total_norm_max_N",
                    "force_total_norm_mean_N",
                    "tip_force_peak_normal_N",
                    "tip_force_total_mean_N",
                ):
                    self.assertNotIn(banned_col, group, msg=f"banned column '{banned_col}' found in HDF5")
                # Values round-trip correctly
                self.assertAlmostEqual(float(group["wire_force_normal_trial_max_N"][0]), 0.8)
                self.assertAlmostEqual(float(group["wire_force_normal_trial_max_N"][1]), 1.4)
                self.assertEqual(
                    group["metric_version"][0].decode("utf-8"), "v2_normal_component"
                )

            # Reload via manifest and verify scoring is consistent
            loaded = service.load_manifest_from_disk(artifacts.manifest_json_path)
            loaded_trial = loaded.trials[0]
            assert loaded_trial.telemetry.forces is not None
            self.assertAlmostEqual(
                loaded_trial.telemetry.forces.wire_force_normal_trial_max_N, 0.8
            )
            recomputed_safety = score_safety(
                force_N=loaded_trial.telemetry.forces.wire_force_normal_trial_max_N,
                scoring=self.scoring,
            )
            self.assertGreater(recomputed_safety, 0.0)
            self.assertLessEqual(recomputed_safety, 1.0)


if __name__ == "__main__":
    unittest.main()
