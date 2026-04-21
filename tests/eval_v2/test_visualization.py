from __future__ import annotations

import unittest
from types import SimpleNamespace

from steve_recommender.eval_v2.models import ExecutionPlan, VisualizationSpec
from steve_recommender.eval_v2.visualization import (
    build_trial_visualisation,
    should_visualize_trial,
)


class TrialVisualizationAdapterTests(unittest.TestCase):
    def test_should_visualize_trial_false_when_disabled(self) -> None:
        self.assertFalse(
            should_visualize_trial(
                VisualizationSpec(enabled=False, rendered_trials_per_candidate=3),
                trial_index=0,
            )
        )

    def test_should_visualize_trial_false_after_render_limit(self) -> None:
        self.assertFalse(
            should_visualize_trial(
                VisualizationSpec(enabled=True, rendered_trials_per_candidate=2),
                trial_index=2,
            )
        )

    def test_build_trial_visualisation_uses_factory_for_enabled_trial(self) -> None:
        runtime = SimpleNamespace(intervention=object())
        created = []

        def factory(intervention):
            created.append(intervention)
            return "viewer"

        visualisation = build_trial_visualisation(
            runtime,
            execution=ExecutionPlan(
                policy_device="cpu",
                visualization=VisualizationSpec(
                    enabled=True,
                    rendered_trials_per_candidate=2,
                ),
            ),
            trial_index=1,
            visualisation_factory=factory,
        )

        self.assertEqual(visualisation, "viewer")
        self.assertEqual(created, [runtime.intervention])

    def test_build_trial_visualisation_returns_none_for_non_visualised_trials(self) -> None:
        runtime = SimpleNamespace(intervention=object())

        visualisation = build_trial_visualisation(
            runtime,
            execution=ExecutionPlan(
                policy_device="cpu",
                visualization=VisualizationSpec(
                    enabled=True,
                    rendered_trials_per_candidate=1,
                ),
            ),
            trial_index=1,
            visualisation_factory=lambda intervention: intervention,
        )

        self.assertIsNone(visualisation)

    def test_build_trial_visualisation_rejects_force_debug_overlay(self) -> None:
        runtime = SimpleNamespace(intervention=object())

        with self.assertRaises(NotImplementedError):
            build_trial_visualisation(
                runtime,
                execution=ExecutionPlan(
                    policy_device="cpu",
                    visualization=VisualizationSpec(
                        enabled=True,
                        rendered_trials_per_candidate=1,
                        force_debug_overlay=True,
                    ),
                ),
                trial_index=0,
                visualisation_factory=lambda intervention: intervention,
            )


if __name__ == "__main__":
    unittest.main()
