from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import h5py

from steve_recommender.eval_v2.models import (
    CandidateScoreSpec,
    EvaluationArtifacts,
    EvaluationCandidate,
    EvaluationJob,
    EvaluationReport,
    ExecutionPlan,
    ForceScoringSpec,
    ForceTelemetrySummary,
    PolicySpec,
    SafetyScoreSpec,
    ScoreBreakdown,
    ScoringSpec,
    SmoothnessScoreSpec,
    TrialArtifactPaths,
    TrialResult,
    TrialTelemetrySummary,
    WireRef,
)
from steve_recommender.eval_v2.scoring import (
    force_within_safety_threshold,
    score_efficiency,
    score_safety,
    valid_for_ranking,
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
    wire = _wire()
    return PolicySpec(
        name="policy_a",
        checkpoint_path=Path("/tmp/policy_a.everl"),
        trained_on_wire=wire,
    )


def _candidate() -> EvaluationCandidate:
    wire = _wire()
    return EvaluationCandidate(name="candidate_a", execution_wire=wire, policy=_policy())


def _trial(
    *,
    success: bool,
    steps_to_success: int | None,
    force_max_N: float | None,
    force_available: bool,
    total_score: float,
    valid: bool,
    force_within_threshold: bool,
    smoothness: float | None = None,
) -> TrialResult:
    wire = _wire()
    policy = _policy()
    forces = ForceTelemetrySummary(
        available_for_score=force_available,
        validation_status="ok" if force_available else "missing",
        wire_force_magnitude_instant_N=force_max_N,
        wire_force_magnitude_trial_max_N=force_max_N,
        wire_force_magnitude_trial_mean_N=force_max_N,
        wire_force_normal_instant_N=force_max_N,
        wire_force_normal_trial_max_N=force_max_N,
        wire_force_normal_trial_mean_N=force_max_N,
        tip_force_magnitude_instant_N=0.4 if force_available else 0.0,
        tip_force_magnitude_trial_max_N=0.4 if force_available else 0.0,
        tip_force_magnitude_trial_mean_N=0.2 if force_available else 0.0,
        tip_force_normal_instant_N=0.4 if force_available else 0.0,
        tip_force_normal_trial_max_N=0.4 if force_available else 0.0,
        tip_force_normal_trial_mean_N=0.2 if force_available else 0.0,
        total_force_norm_max=force_max_N,
        total_force_norm_mean=force_max_N,
    )
    return TrialResult(
        scenario_name="scenario_a",
        candidate_name="candidate_a",
        execution_wire=wire,
        policy=policy,
        trial_index=0,
        seed=123,
        policy_seed=1000,
        score=ScoreBreakdown(
            total=total_score,
            success=1.0 if success else 0.0,
            efficiency=0.75 if success else 0.0,
            safety=0.5 if force_available else 0.0,
            smoothness=smoothness,
        ),
        telemetry=TrialTelemetrySummary(
            success=success,
            steps_total=10,
            steps_to_success=steps_to_success,
            episode_reward=1.0,
            sim_time_s=1.0,
            wall_time_s=2.0,
            tip_speed_max_mm_s=10.0,
            tip_speed_mean_mm_s=5.0,
            tip_total_distance_mm=12.0,
            tip_acc_p95=1.0,
            tip_acc_max=2.0,
            tip_jerk_p95=3.0,
            tip_jerk_max=4.0,
            forces=forces,
        ),
        valid_for_ranking=valid,
        force_within_safety_threshold=force_within_threshold,
        artifacts=TrialArtifactPaths(trace_h5_path=Path("/tmp/trace.h5")),
    )


class _RunnerStub:
    def __init__(self, report: EvaluationReport) -> None:
        self.report = report

    def run_evaluation_job(self, job, *, frame_callback=None, progress_callback=None):
        _ = job, frame_callback, progress_callback
        return self.report


class ScoringFormulaTests(unittest.TestCase):
    def test_score_efficiency_matches_requested_formula(self) -> None:
        self.assertEqual(score_efficiency(success=False, steps_to_success=None, max_episode_steps=10), 0.0)
        self.assertAlmostEqual(
            score_efficiency(success=True, steps_to_success=1, max_episode_steps=10),
            1.0,
        )
        self.assertAlmostEqual(
            score_efficiency(success=True, steps_to_success=10, max_episode_steps=10),
            0.0,
        )

    def test_score_safety_matches_nonlinear_formula(self) -> None:
        scoring = ScoringSpec(
            force=ForceScoringSpec(force_max_N=2.0),
            safety_score=SafetyScoreSpec(c=0.30, p=2.0, k=10.0, F50_N=1.55, F_max_N=2.0),
        )
        force_value = 1.0
        def g(force: float) -> float:
            return 1.0 / (1.0 + math.exp(10.0 * (force - 1.55)))
        expected = max(
            0.0,
            min(
                1.0,
                (1.0 - 0.30 * force_value**2)
                * ((g(force_value) - g(2.0)) / (g(0.0) - g(2.0))),
            ),
        )
        self.assertAlmostEqual(
            score_safety(force_N=force_value, scoring=scoring),
            expected,
        )

    def test_force_threshold_and_trial_indicator_follow_hard_constraints(self) -> None:
        scoring = ScoringSpec(force=ForceScoringSpec(force_max_N=2.0))
        telemetry = TrialTelemetrySummary(
            success=True,
            steps_total=10,
            steps_to_success=8,
            episode_reward=1.0,
            forces=ForceTelemetrySummary(
                available_for_score=True,
                validation_status="ok",
                wire_force_normal_trial_max_N=1.9,
            ),
        )
        self.assertTrue(force_within_safety_threshold(telemetry=telemetry, scoring=scoring))
        self.assertTrue(valid_for_ranking(telemetry=telemetry, max_episode_steps=10, scoring=scoring))

        telemetry_fail = TrialTelemetrySummary(
            success=True,
            steps_total=10,
            steps_to_success=8,
            episode_reward=1.0,
            forces=ForceTelemetrySummary(
                available_for_score=True,
                validation_status="ok",
                wire_force_normal_trial_max_N=2.1,
            ),
        )
        self.assertFalse(force_within_safety_threshold(telemetry=telemetry_fail, scoring=scoring))
        self.assertFalse(valid_for_ranking(telemetry=telemetry_fail, max_episode_steps=10, scoring=scoring))

    def test_scoring_uses_wire_force_normal_trial_max_not_legacy_total_force_field(self) -> None:
        scoring = ScoringSpec(force=ForceScoringSpec(force_max_N=2.0))
        telemetry = TrialTelemetrySummary(
            success=True,
            steps_total=10,
            steps_to_success=8,
            episode_reward=1.0,
            forces=ForceTelemetrySummary(
                available_for_score=True,
                validation_status="ok",
                wire_force_normal_trial_max_N=1.9,
                total_force_norm_max=9.9,
            ),
        )
        self.assertTrue(force_within_safety_threshold(telemetry=telemetry, scoring=scoring))
        breakdown = valid_for_ranking(
            telemetry=telemetry, max_episode_steps=10, scoring=scoring
        )
        self.assertTrue(breakdown)

    def test_candidate_score_aggregation_uses_valid_rate_and_valid_trial_soft_scores(self) -> None:
        scoring = ScoringSpec(
            candidate_score=CandidateScoreSpec(
                lambda_=1.0,
                beta=0.0,
                default_weights={"score_safety": 0.5, "score_efficiency": 0.5},
                active_components=("score_safety", "score_efficiency"),
            )
        )
        trials = (
            _trial(success=True, steps_to_success=5, force_max_N=1.0, force_available=True, total_score=0.8, valid=True, force_within_threshold=True),
            _trial(success=True, steps_to_success=6, force_max_N=1.0, force_available=True, total_score=0.6, valid=True, force_within_threshold=True),
            _trial(success=False, steps_to_success=None, force_max_N=None, force_available=False, total_score=0.0, valid=False, force_within_threshold=False),
        )
        summary = summarize_trials(trials, scoring=scoring)
        self.assertAlmostEqual(summary.valid_rate, 2.0 / 3.0)
        self.assertAlmostEqual(summary.soft_score_mean_valid, 0.625)
        self.assertAlmostEqual(summary.candidate_score_final, (2.0 / 3.0) * 0.625)

    def test_candidate_summary_aggregates_wire_force_normal_trial_max_mean(self) -> None:
        scoring = ScoringSpec()
        first = _trial(
            success=True,
            steps_to_success=5,
            force_max_N=0.8,
            force_available=True,
            total_score=0.8,
            valid=True,
            force_within_threshold=True,
        )
        second = _trial(
            success=True,
            steps_to_success=6,
            force_max_N=1.4,
            force_available=True,
            total_score=0.6,
            valid=True,
            force_within_threshold=True,
        )
        summary = summarize_trials((first, second), scoring=scoring)
        self.assertAlmostEqual(summary.wire_force_normal_trial_max_mean_N, 1.1)


class ManifestAndTrialsPersistenceTests(unittest.TestCase):
    def test_manifest_and_trials_h5_persist_scoring_defaults_and_tip_length(self) -> None:
        scenario = _scenario("scenario_manifest")
        candidate = _candidate()
        scoring = ScoringSpec(
            force=ForceScoringSpec(force_max_N=1.7, tip_length_mm=7.5),
            safety_score=SafetyScoreSpec(c=0.30, p=2.0, k=10.0, F50_N=1.55, F_max_N=1.7),
            candidate_score=CandidateScoreSpec(
                lambda_=1.0,
                beta=0.0,
                default_weights={"score_safety": 0.5, "score_efficiency": 0.5},
                active_components=("score_safety", "score_efficiency"),
            ),
            smoothness_score=SmoothnessScoreSpec(jerk_scale_mm_s3=None),
        )
        trial = _trial(
            success=True,
            steps_to_success=8,
            force_max_N=1.2,
            force_available=True,
            total_score=0.7,
            valid=True,
            force_within_threshold=True,
            smoothness=None,
        )
        summary = summarize_trials((trial,), scoring=scoring)
        report = EvaluationReport(
            job_name="manifest_job",
            generated_at="2026-04-30T20:00:00+00:00",
            summaries=(summary,),
            trials=(trial,),
            artifacts=EvaluationArtifacts(output_dir=Path("/tmp/manifest_job")),
        )
        anatomy_discovery = _AnatomyDiscoveryStub((scenario.anatomy,))
        policy_discovery = _PolicyDiscoveryStub(
            execution_wires=(candidate.execution_wire,),
            startable_wires=(candidate.execution_wire,),
            registry_policies=(candidate.policy,),
            explicit_policies=(),
        )
        service = DefaultEvaluationService(
            anatomy_discovery=anatomy_discovery,
            policy_discovery=policy_discovery,
            explicit_policy_discovery=_ExplicitPolicyDiscoveryStub(()),
            target_discovery=_TargetDiscoveryStub(),
            evaluation_runner=_RunnerStub(report),
        )

        with tempfile.TemporaryDirectory() as tmp:
            job = EvaluationJob(
                name="manifest_job",
                scenarios=(scenario,),
                candidates=(candidate,),
                execution=ExecutionPlan(trials_per_candidate=1, base_seed=123, policy_device="cpu"),
                scoring=scoring,
                output_root=Path(tmp),
            )
            persisted = service.run_evaluation_job(job)
            artifacts = persisted.artifacts
            assert artifacts is not None

            manifest = json.loads(artifacts.manifest_json_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["metric_version"], "v2_normal_component")
            self.assertEqual(manifest["scoring_spec"]["force"]["tip_length_mm"], 7.5)
            self.assertEqual(manifest["scoring_spec"]["force"]["force_max_N"], 1.7)
            self.assertEqual(manifest["scoring_spec"]["candidate_score"]["default_weights"]["score_safety"], 0.5)
            self.assertEqual(manifest["counts"]["n_trials_total"], 1)
            self.assertEqual(
                manifest["summaries"][0]["wire_force_normal_trial_max_mean_N"], 1.2
            )

            with h5py.File(artifacts.trials_h5_path, "r") as handle:
                self.assertEqual(handle.attrs["metric_version"], "v2_normal_component")
                trials_group = handle["trials"]
                required_columns = {
                    "metric_version",
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
                    "end_reason",
                    "max_episode_steps",
                    "episode_reward",
                    "sim_time_s",
                    "wall_time_s",
                    "tip_speed_max_mm_s",
                    "tip_speed_mean_mm_s",
                    "tip_total_distance_mm",
                    "force_available_for_score",
                    "wire_force_magnitude_instant_N",
                    "wire_force_magnitude_trial_max_N",
                    "wire_force_magnitude_trial_mean_N",
                    "wire_force_normal_instant_N",
                    "wire_force_normal_trial_max_N",
                    "wire_force_normal_trial_mean_N",
                    "tip_force_magnitude_instant_N",
                    "tip_force_magnitude_trial_max_N",
                    "tip_force_magnitude_trial_mean_N",
                    "tip_force_normal_instant_N",
                    "tip_force_normal_trial_max_N",
                    "tip_force_normal_trial_mean_N",
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
                }
                self.assertTrue(required_columns.issubset(set(trials_group.keys())))
                self.assertEqual(
                    trials_group["metric_version"][0].decode("utf-8"),
                    "v2_normal_component",
                )
                self.assertAlmostEqual(float(trials_group["tip_length_mm"][0]), 7.5)
                self.assertAlmostEqual(
                    float(trials_group["wire_force_normal_instant_N"][0]), 1.2
                )
                self.assertAlmostEqual(
                    float(trials_group["wire_force_normal_trial_max_N"][0]), 1.2
                )
                self.assertTrue(bool(trials_group["force_within_safety_threshold"][0]))
                self.assertTrue(bool(trials_group["valid_for_ranking"][0]))


if __name__ == "__main__":
    unittest.main()
