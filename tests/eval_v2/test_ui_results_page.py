from __future__ import annotations

import os
import unittest
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from steve_recommender.eval_v2.models import (
    CandidateScoreSpec,
    EvaluationReport,
    ExecutionPlan,
    ForceTelemetrySummary,
    PolicySpec,
    ScoreBreakdown,
    ScoringSpec,
    TrialArtifactPaths,
    TrialResult,
    TrialTelemetrySummary,
    WireRef,
)
from steve_recommender.eval_v2.ui_controller import ClinicalUIController
from steve_recommender.eval_v2.ui_wizard_pages import ResultsPage


class _NoopService:
    def list_execution_wires(self):
        return ()


def _trial(
    *,
    wire: WireRef,
    candidate_name: str,
    trial_index: int,
    steps_to_success: int,
    force_max_N: float,
) -> TrialResult:
    return TrialResult(
        scenario_name="scenario",
        candidate_name=candidate_name,
        execution_wire=wire,
        policy=PolicySpec(
            name=f"{candidate_name}_policy",
            checkpoint_path=Path("/tmp/policy.everl"),
            trained_on_wire=wire,
        ),
        trial_index=trial_index,
        seed=123 + trial_index,
        policy_seed=None,
        score=ScoreBreakdown(total=0.0, success=1.0, efficiency=0.0, safety=0.0, smoothness=None),
        telemetry=TrialTelemetrySummary(
            success=True,
            steps_total=steps_to_success,
            steps_to_success=steps_to_success,
            episode_reward=1.0,
            tip_speed_max_mm_s=10.0,
            tip_speed_mean_mm_s=5.0,
            forces=ForceTelemetrySummary(
                available_for_score=True,
                validation_status="ok",
                total_force_norm_max_newton=force_max_N,
                total_force_norm_mean_newton=force_max_N,
                total_force_norm_max=force_max_N,
                total_force_norm_mean=force_max_N,
            ),
        ),
        valid_for_ranking=True,
        force_within_safety_threshold=True,
        artifacts=TrialArtifactPaths(trace_h5_path=Path(f"/tmp/{candidate_name}_{trial_index}.h5")),
    )


def _empty_summary_for_wire(wire: WireRef, candidate_name: str):
    from steve_recommender.eval_v2.models import CandidateSummary

    return CandidateSummary(
        scenario_name="scenario",
        candidate_name=candidate_name,
        execution_wire=wire,
        trained_on_wire=wire,
        trial_count=1,
        success_rate=1.0,
        valid_rate=1.0,
        soft_score_mean_valid=0.0,
        soft_score_std_valid=0.0,
        candidate_score_final=0.0,
        score_mean=0.0,
        score_std=0.0,
        steps_total_mean=1.0,
        steps_to_success_mean=1.0,
        tip_speed_max_mean_mm_s=0.0,
        wall_force_max_mean=0.0,
        wall_force_max_mean_newton=0.0,
        force_available_rate=1.0,
    )


class ResultsPageScoringControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication(["-platform", "offscreen"])

    def _make_page(self) -> ResultsPage:
        controller = ClinicalUIController(service=_NoopService(), runtime_preflight=lambda: None)
        page = ResultsPage(controller=controller)
        page.show()
        return page

    def test_weight_sliders_reorder_leaderboard(self) -> None:
        wire_safe = WireRef(model="steve_default", wire="safe_wire")
        wire_fast = WireRef(model="steve_default", wire="fast_wire")
        report = EvaluationReport(
            job_name="job",
            generated_at="2026-04-30T12:00:00+00:00",
            summaries=(
                _empty_summary_for_wire(wire_safe, "safe_candidate"),
                _empty_summary_for_wire(wire_fast, "fast_candidate"),
            ),
            trials=(
                _trial(
                    wire=wire_safe,
                    candidate_name="safe_candidate",
                    trial_index=0,
                    steps_to_success=90,
                    force_max_N=0.1,
                ),
                _trial(
                    wire=wire_fast,
                    candidate_name="fast_candidate",
                    trial_index=0,
                    steps_to_success=10,
                    force_max_N=1.6,
                ),
            ),
            execution_plan=ExecutionPlan(max_episode_steps=100, worker_count=1, policy_device="cpu"),
            scoring_spec=ScoringSpec(
                candidate_score=CandidateScoreSpec(
                    lambda_=1.0,
                    beta=0.0,
                    default_weights={"score_safety": 1.0, "score_efficiency": 0.0},
                    active_components=("score_safety", "score_efficiency"),
                )
            ),
        )

        page = self._make_page()
        page.set_report(report)

        self.assertEqual(page.leaderboard_table.item(0, 1).text(), wire_safe.tool_ref)

        page._control_sliders["score_weight_safety"].setValue(0)
        page._control_sliders["score_weight_efficiency"].setValue(100)
        page._on_weight_changed()

        self.assertEqual(page.leaderboard_table.item(0, 1).text(), wire_fast.tool_ref)

    def test_force_threshold_slider_changes_validity_and_score(self) -> None:
        wire_safe = WireRef(model="steve_default", wire="safe_wire")
        wire_risky = WireRef(model="steve_default", wire="risky_wire")
        report = EvaluationReport(
            job_name="job",
            generated_at="2026-04-30T12:00:00+00:00",
            summaries=(
                _empty_summary_for_wire(wire_safe, "safe_candidate"),
                _empty_summary_for_wire(wire_risky, "risky_candidate"),
            ),
            trials=(
                _trial(
                    wire=wire_safe,
                    candidate_name="safe_candidate",
                    trial_index=0,
                    steps_to_success=60,
                    force_max_N=0.4,
                ),
                _trial(
                    wire=wire_risky,
                    candidate_name="risky_candidate",
                    trial_index=0,
                    steps_to_success=10,
                    force_max_N=2.5,
                ),
            ),
            execution_plan=ExecutionPlan(max_episode_steps=100, worker_count=1, policy_device="cpu"),
            scoring_spec=ScoringSpec(),
        )

        page = self._make_page()
        page.set_report(report)

        risky_row_before = None
        for row_index in range(page.leaderboard_table.rowCount()):
            if page.leaderboard_table.item(row_index, 1).text() == wire_risky.tool_ref:
                risky_row_before = row_index
                break
        self.assertIsNotNone(risky_row_before)
        self.assertEqual(page.leaderboard_table.item(risky_row_before, 3).text(), "0.0000")
        score_before = float(page.leaderboard_table.item(risky_row_before, 2).text())

        page._control_sliders["force_max_N"].setValue(59)
        page._on_weight_changed()

        risky_row_after = None
        for row_index in range(page.leaderboard_table.rowCount()):
            if page.leaderboard_table.item(row_index, 1).text() == wire_risky.tool_ref:
                risky_row_after = row_index
                break
        self.assertIsNotNone(risky_row_after)
        self.assertGreater(float(page.leaderboard_table.item(risky_row_after, 3).text()), 0.0)
        self.assertGreater(float(page.leaderboard_table.item(risky_row_after, 2).text()), score_before)


if __name__ == "__main__":
    unittest.main()
