from __future__ import annotations

import unittest

from steve_recommender.eval_v2.models import CandidateSummary, WireRef
from steve_recommender.eval_v2.scoring import (
    calculate_overall_score,
    normalize_safety_score,
    normalize_success_score,
    normalize_time_score,
)


def _summary(*, success_rate: float, time_value: float, force_value: float) -> CandidateSummary:
    wire = WireRef(model="steve_default", wire="standard_j")
    return CandidateSummary(
        scenario_name="scenario",
        candidate_name="candidate",
        execution_wire=wire,
        trained_on_wire=wire,
        trial_count=1,
        success_rate=success_rate,
        valid_rate=success_rate,
        soft_score_mean_valid=0.0,
        soft_score_std_valid=0.0,
        candidate_score_final=0.0,
        score_mean=0.0,
        score_std=0.0,
        steps_total_mean=time_value,
        steps_to_success_mean=time_value,
        tip_speed_max_mean_mm_s=0.0,
        wall_force_max_mean=force_value,
        wall_force_max_mean_newton=force_value,
        force_available_rate=1.0,
    )


class ScoringDashboardMathTests(unittest.TestCase):
    def test_normalized_axis_scores_are_clamped_between_zero_and_one(self) -> None:
        self.assertGreaterEqual(normalize_success_score(success_rate=-10.0), 0.0)
        self.assertLessEqual(normalize_success_score(success_rate=10.0), 1.0)

        self.assertGreaterEqual(
            normalize_time_score(insertion_time=10_000.0, max_expected_time=100.0),
            0.0,
        )
        self.assertLessEqual(
            normalize_time_score(insertion_time=-10.0, max_expected_time=100.0),
            1.0,
        )

        self.assertGreaterEqual(
            normalize_safety_score(max_force=100.0, safe_force_threshold=0.1),
            0.0,
        )
        self.assertLessEqual(
            normalize_safety_score(max_force=-1.0, safe_force_threshold=0.1),
            1.0,
        )

    def test_zero_weight_fully_removes_axis_contribution(self) -> None:
        summary = _summary(success_rate=1.0, time_value=95.0, force_value=0.9)
        weights = {"success": 1.0, "speed": 0.0, "safety": 0.0}

        overall, axes = calculate_overall_score(
            summary,
            axes_weights=weights,
            max_expected_time=100.0,
            safe_force_threshold=1.0,
        )

        self.assertAlmostEqual(overall, axes["success"], places=6)

    def test_overall_score_is_weighted_average_and_zero_when_all_weights_zero(self) -> None:
        summary = _summary(success_rate=0.5, time_value=50.0, force_value=0.5)

        overall, axes = calculate_overall_score(
            summary,
            axes_weights={"success": 0.0, "speed": 0.0, "safety": 0.0},
            max_expected_time=100.0,
            safe_force_threshold=1.0,
        )
        self.assertEqual(overall, 0.0)

        overall2, axes2 = calculate_overall_score(
            summary,
            axes_weights={"success": 1.0, "speed": 1.0, "safety": 1.0},
            max_expected_time=100.0,
            safe_force_threshold=1.0,
        )
        expected = (axes2["success"] + axes2["speed"] + axes2["safety"]) / 3.0
        self.assertAlmostEqual(overall2, expected, places=6)


if __name__ == "__main__":
    unittest.main()
