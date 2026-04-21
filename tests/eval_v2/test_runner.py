from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from third_party.stEVE.eve.env import Env
from third_party.stEVE.eve.intervention.simulation import SimulationDummy

from steve_recommender.eval_v2.models import (
    AorticArchAnatomy,
    BranchEndTarget,
    EvaluationCandidate,
    EvaluationScenario,
    ExecutionPlan,
    ManualTarget,
    PolicySpec,
    ScoringSpec,
    TrialResult,
    VisualizationSpec,
    WireRef,
)
from steve_recommender.eval_v2.runner import build_single_trial_env, run_single_trial
from steve_recommender.eval_v2.runtime import PreparedEvaluationRuntime, build_intervention


class StubPlayPolicy:
    def __init__(
        self,
        *,
        eval_action: tuple[float, float] = (0.0, 0.0),
        exploration_action: tuple[float, float] | None = None,
    ) -> None:
        self.device = "cpu"
        self._eval_action = np.asarray(eval_action, dtype=np.float32)
        self._exploration_action = (
            None
            if exploration_action is None
            else np.asarray(exploration_action, dtype=np.float32)
        )
        self.reset_calls = 0
        self.eval_calls = 0
        self.exploration_calls = 0
        self.flat_states: list[np.ndarray] = []

    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        self.eval_calls += 1
        self.flat_states.append(np.array(flat_state, copy=True))
        return np.array(self._eval_action, copy=True)

    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        if self._exploration_action is None:
            raise NotImplementedError("No exploration action configured for this stub")
        self.exploration_calls += 1
        self.flat_states.append(np.array(flat_state, copy=True))
        return np.array(self._exploration_action, copy=True)

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        return None

    def to(self, device: object) -> None:
        self.device = str(device)


class StubVisualisation:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.render_calls = 0
        self.close_calls = 0

    def reset(self, episode_nr: int = 0) -> None:
        _ = episode_nr
        self.reset_calls += 1

    def render(self):
        self.render_calls += 1
        return None

    def close(self) -> None:
        self.close_calls += 1


class SingleTrialRunnerIntegrationTests(unittest.TestCase):
    def _write_wire_registry(self, root: Path) -> Path:
        registry_root = root / "wire_registry"
        tool_dir = registry_root / "steve_default" / "wire_versions" / "standard_j"
        tool_dir.mkdir(parents=True)

        tool_definition = {
            "type": "procedural",
            "spec": {
                "name": "guidewire",
                "velocity_limit": [35.0, 3.14],
                "length": 450.0,
                "tip_radius": 12.1,
                "tip_angle": 1.2566370614359172,
                "tip_outer_diameter": 0.7,
                "tip_inner_diameter": 0.0,
                "straight_outer_diameter": 0.89,
                "straight_inner_diameter": 0.0,
                "poisson_ratio": 0.49,
                "young_modulus_tip": 17000.0,
                "young_modulus_straight": 80000.0,
                "mass_density_tip": 2.1e-05,
                "mass_density_straight": 2.1e-05,
                "visu_edges_per_mm": 0.5,
                "collis_edges_per_mm_tip": 2.0,
                "collis_edges_per_mm_straight": 0.1,
                "beams_per_mm_tip": 1.4,
                "beams_per_mm_straight": 0.5,
                "color": [0.0, 0.0, 0.0],
            },
        }
        tool_definition_path = tool_dir / "tool_definition.json"
        tool_definition_path.write_text(
            json.dumps(tool_definition, indent=2),
            encoding="utf-8",
        )

        index_payload = {
            "schema_version": 1,
            "wires": {
                "steve_default/standard_j": {
                    "model": "steve_default",
                    "name": "standard_j",
                    "tool_ref": "steve_default/standard_j",
                    "tool_definition_path": "steve_default/wire_versions/standard_j/tool_definition.json",
                    "wire_dir": "steve_default/wire_versions/standard_j",
                    "agents_dir": None,
                }
            },
            "agents": {},
        }
        registry_path = registry_root / "index.json"
        registry_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
        return registry_path

    def _candidate(self) -> EvaluationCandidate:
        return EvaluationCandidate(
            name="candidate_a",
            execution_wire=WireRef(model="steve_default", wire="standard_j"),
            policy=PolicySpec(name="policy_stub", checkpoint_path=Path("/tmp/policy_stub.everl")),
        )

    def _anatomy(self) -> AorticArchAnatomy:
        return AorticArchAnatomy(
            arch_type="II",
            seed=42,
            rotation_yzx_deg=(1.0, 2.0, 3.0),
            scaling_xyzd=(1.0, 1.1, 1.2, 1.3),
            omit_axis="z",
        )

    def _prepare_runtime(
        self,
        *,
        registry_path: Path,
        scenario: EvaluationScenario,
        play_policy: StubPlayPolicy,
    ) -> PreparedEvaluationRuntime:
        candidate = self._candidate()
        intervention, device = build_intervention(
            candidate=candidate,
            scenario=scenario,
            simulation=SimulationDummy(),
            registry_path=registry_path,
        )
        return PreparedEvaluationRuntime(
            candidate=candidate,
            scenario=scenario,
            device=device,
            intervention=intervention,
            play_policy=play_policy,
        )

    def test_build_single_trial_env_returns_real_steve_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = self._write_wire_registry(Path(tmp))
            runtime = self._prepare_runtime(
                registry_path=registry_path,
                scenario=EvaluationScenario(
                    name="scenario_a",
                    anatomy=self._anatomy(),
                    target=BranchEndTarget(threshold_mm=5.0, branches=("lcca",)),
                ),
                play_policy=StubPlayPolicy(),
            )

            env = build_single_trial_env(runtime, max_episode_steps=3)

        self.assertIsInstance(env, Env)
        self.assertEqual(tuple(env.observation.observations.keys()), ("tracking", "target", "last_action"))
        self.assertEqual(env.action_space.shape, (1, 2))

    def test_run_single_trial_returns_successful_trial_result_for_manual_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = self._write_wire_registry(Path(tmp))
            play_policy = StubPlayPolicy(eval_action=(0.0, 0.0))
            runtime = self._prepare_runtime(
                registry_path=registry_path,
                scenario=EvaluationScenario(
                    name="manual_success",
                    anatomy=self._anatomy(),
                    target=ManualTarget(
                        threshold_mm=5.0,
                        targets_vessel_cs=((0.0, 0.0, 0.0),),
                    ),
                ),
                play_policy=play_policy,
            )

            result = run_single_trial(
                runtime=runtime,
                trial_index=0,
                seed=123,
                execution=ExecutionPlan(
                    trials_per_candidate=1,
                    base_seed=123,
                    max_episode_steps=3,
                    policy_device="cpu",
                ),
                scoring=ScoringSpec(),
            )

        self.assertIsInstance(result, TrialResult)
        self.assertEqual(result.scenario_name, "manual_success")
        self.assertEqual(result.candidate_name, "candidate_a")
        self.assertEqual(result.seed, 123)
        self.assertTrue(result.telemetry.success)
        self.assertEqual(result.telemetry.steps_total, 1)
        self.assertEqual(result.telemetry.steps_to_success, 1)
        self.assertGreater(result.telemetry.episode_reward, 0.0)
        self.assertEqual(result.telemetry.tip_speed_max_mm_s, 0.0)
        self.assertEqual(result.telemetry.tip_speed_mean_mm_s, 0.0)
        self.assertEqual(result.score.total, 1.0)
        self.assertEqual(result.score.success, 1.0)
        self.assertEqual(result.score.efficiency, 1.0)
        self.assertEqual(result.score.smoothness, 1.0)
        self.assertIsNone(result.score.safety)
        self.assertEqual(play_policy.reset_calls, 1)
        self.assertEqual(play_policy.eval_calls, 1)
        self.assertEqual(play_policy.exploration_calls, 0)
        self.assertEqual(len(play_policy.flat_states), 1)
        self.assertEqual(play_policy.flat_states[0].ndim, 1)
        self.assertGreater(play_policy.flat_states[0].size, 0)

    def test_run_single_trial_returns_truncated_result_when_target_is_not_reached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = self._write_wire_registry(Path(tmp))
            play_policy = StubPlayPolicy(eval_action=(0.0, 0.0))
            runtime = self._prepare_runtime(
                registry_path=registry_path,
                scenario=EvaluationScenario(
                    name="branch_failure",
                    anatomy=self._anatomy(),
                    target=BranchEndTarget(
                        threshold_mm=5.0,
                        branches=("lcca",),
                    ),
                ),
                play_policy=play_policy,
            )

            result = run_single_trial(
                runtime=runtime,
                trial_index=2,
                seed=321,
                execution=ExecutionPlan(
                    trials_per_candidate=1,
                    base_seed=321,
                    max_episode_steps=2,
                    policy_device="cpu",
                ),
                scoring=ScoringSpec(),
            )

        self.assertFalse(result.telemetry.success)
        self.assertEqual(result.telemetry.steps_total, 1)
        self.assertIsNone(result.telemetry.steps_to_success)
        self.assertLess(result.telemetry.episode_reward, 0.0)
        self.assertAlmostEqual(result.score.success, 0.0)
        self.assertAlmostEqual(result.score.efficiency, 0.0)
        self.assertAlmostEqual(result.score.smoothness, 1.0)
        self.assertIsNone(result.score.safety)
        self.assertAlmostEqual(result.score.total, 0.25 / 3.25)
        self.assertEqual(play_policy.reset_calls, 1)
        self.assertEqual(play_policy.eval_calls, 1)

    def test_run_single_trial_renders_and_closes_visualisation_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = self._write_wire_registry(Path(tmp))
            play_policy = StubPlayPolicy(eval_action=(0.0, 0.0))
            visualisation = StubVisualisation()
            runtime = self._prepare_runtime(
                registry_path=registry_path,
                scenario=EvaluationScenario(
                    name="manual_visualised",
                    anatomy=self._anatomy(),
                    target=ManualTarget(
                        threshold_mm=5.0,
                        targets_vessel_cs=((0.0, 0.0, 0.0),),
                    ),
                ),
                play_policy=play_policy,
            )

            with patch(
                "steve_recommender.eval_v2.runner.build_trial_visualisation",
                return_value=visualisation,
            ):
                result = run_single_trial(
                    runtime=runtime,
                    trial_index=0,
                    seed=123,
                    execution=ExecutionPlan(
                        trials_per_candidate=1,
                        base_seed=123,
                        max_episode_steps=3,
                        policy_device="cpu",
                        visualization=VisualizationSpec(
                            enabled=True,
                            rendered_trials_per_candidate=1,
                        ),
                    ),
                    scoring=ScoringSpec(),
                )

        self.assertTrue(result.telemetry.success)
        self.assertEqual(visualisation.reset_calls, 1)
        self.assertEqual(visualisation.render_calls, 1)
        self.assertEqual(visualisation.close_calls, 1)


class RunnerActionMappingTests(unittest.TestCase):
    def _runtime(self, *, normalize_action: bool, play_policy: StubPlayPolicy) -> PreparedEvaluationRuntime:
        candidate = EvaluationCandidate(
            name="candidate_a",
            execution_wire=WireRef(model="steve_default", wire="standard_j"),
            policy=PolicySpec(
                name="policy_stub",
                checkpoint_path=Path("/tmp/policy_stub.everl"),
            ),
        )
        scenario = EvaluationScenario(
            name="scenario_a",
            anatomy=AorticArchAnatomy(
                arch_type="II",
                seed=42,
                rotation_yzx_deg=(1.0, 2.0, 3.0),
                scaling_xyzd=(1.0, 1.1, 1.2, 1.3),
                omit_axis="z",
            ),
            target=ManualTarget(
                threshold_mm=5.0,
                targets_vessel_cs=((0.0, 0.0, 0.0),),
            ),
            normalize_action=normalize_action,
        )
        intervention = SimpleNamespace(
            velocity_limits=np.asarray([[35.0, 3.14]], dtype=np.float32),
        )
        return PreparedEvaluationRuntime(
            candidate=candidate,
            scenario=scenario,
            device=SimpleNamespace(),
            intervention=intervention,
            play_policy=play_policy,
        )

    def test_run_single_trial_maps_normalized_policy_action_to_env_action_space(self) -> None:
        class StubEnv:
            def __init__(self) -> None:
                self.action_space = SimpleNamespace(
                    shape=(1, 2),
                    low=np.asarray([[-35.0, -3.14]], dtype=np.float32),
                    high=np.asarray([[35.0, 3.14]], dtype=np.float32),
                )
                self.received_actions: list[np.ndarray] = []

            def step(self, action: np.ndarray):
                self.received_actions.append(np.asarray(action, dtype=np.float32))
                observation = {"tracking": np.zeros((1, 2), dtype=np.float32)}
                info = {
                    "success": True,
                    "steps": 1,
                    "path_ratio": 0.0,
                    "trajectory_length": 0.0,
                    "average_translation_speed": 0.0,
                }
                return observation, 0.0, True, False, info

            def render(self):
                return None

        env = StubEnv()
        runtime = self._runtime(
            normalize_action=True,
            play_policy=StubPlayPolicy(eval_action=(1.0, -1.0)),
        )

        with patch(
            "steve_recommender.eval_v2.runner.build_single_trial_env",
            return_value=env,
        ), patch(
            "steve_recommender.eval_v2.runner._reset_single_trial_env",
            return_value=({"tracking": np.zeros((1, 2), dtype=np.float32)}, {}),
        ):
            result = run_single_trial(
                runtime=runtime,
                trial_index=0,
                seed=123,
                execution=ExecutionPlan(
                    trials_per_candidate=1,
                    base_seed=123,
                    max_episode_steps=2,
                    policy_device="cpu",
                ),
                scoring=ScoringSpec(),
            )

        self.assertEqual(len(env.received_actions), 1)
        np.testing.assert_allclose(
            env.received_actions[0],
            np.asarray([[35.0, -3.14]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertAlmostEqual(result.telemetry.tip_speed_max_mm_s, 35.0)

    def test_run_single_trial_keeps_physical_policy_action_when_normalization_is_disabled(self) -> None:
        class StubEnv:
            def __init__(self) -> None:
                self.action_space = SimpleNamespace(
                    shape=(1, 2),
                    low=np.asarray([[-35.0, -3.14]], dtype=np.float32),
                    high=np.asarray([[35.0, 3.14]], dtype=np.float32),
                )
                self.received_actions: list[np.ndarray] = []

            def step(self, action: np.ndarray):
                self.received_actions.append(np.asarray(action, dtype=np.float32))
                observation = {"tracking": np.zeros((1, 2), dtype=np.float32)}
                info = {
                    "success": True,
                    "steps": 1,
                    "path_ratio": 0.0,
                    "trajectory_length": 0.0,
                    "average_translation_speed": 0.0,
                }
                return observation, 0.0, True, False, info

            def render(self):
                return None

        env = StubEnv()
        runtime = self._runtime(
            normalize_action=False,
            play_policy=StubPlayPolicy(eval_action=(1.0, -1.0)),
        )

        with patch(
            "steve_recommender.eval_v2.runner.build_single_trial_env",
            return_value=env,
        ), patch(
            "steve_recommender.eval_v2.runner._reset_single_trial_env",
            return_value=({"tracking": np.zeros((1, 2), dtype=np.float32)}, {}),
        ):
            run_single_trial(
                runtime=runtime,
                trial_index=0,
                seed=123,
                execution=ExecutionPlan(
                    trials_per_candidate=1,
                    base_seed=123,
                    max_episode_steps=2,
                    policy_device="cpu",
                ),
                scoring=ScoringSpec(),
            )

        self.assertEqual(len(env.received_actions), 1)
        np.testing.assert_allclose(
            env.received_actions[0],
            np.asarray([[1.0, -1.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
