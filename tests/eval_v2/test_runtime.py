from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from third_party.stEVE.eve.intervention import MonoPlaneStatic
from third_party.stEVE.eve.intervention.device import JShaped
from third_party.stEVE.eve.intervention.simulation import SimulationDummy

from steve_recommender.eval_v2.discovery import FileBasedWireRegistryDiscovery
from steve_recommender.eval_v2.models import (
    AorticArchAnatomy,
    BranchEndTarget,
    EvaluationCandidate,
    EvaluationScenario,
    WireRef,
)
from steve_recommender.eval_v2.runtime import (
    PreparedEvaluationRuntime,
    build_device,
    load_play_policy,
    prepare_evaluation_runtime,
    safe_reset_intervention,
)


class RuntimeDeviceBuilderTests(unittest.TestCase):
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

    def test_build_device_reads_tool_definition_and_returns_real_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = self._write_wire_registry(Path(tmp))

            device = build_device(
                WireRef(model="steve_default", wire="standard_j"),
                registry_path=registry_path,
            )

        self.assertIsInstance(device, JShaped)
        self.assertEqual(device.name, "guidewire")
        self.assertEqual(device.velocity_limit, (35.0, 3.14))
        self.assertEqual(device.length, 450.0)
        self.assertEqual(device.tip_radius, 12.1)
        self.assertEqual(device.tip_angle, 1.2566370614359172)
        self.assertEqual(device.sofa_device.radius, 0.89 / 2.0)
        self.assertEqual(device.sofa_device.radius_extremity, 0.7 / 2.0)


class RuntimeAssemblyIntegrationTests(unittest.TestCase):
    def _real_policy(self):
        policies = FileBasedWireRegistryDiscovery().list_registry_policies()
        if not policies:
            self.skipTest("No registry-backed policies with checkpoints available")
        return policies[0]

    def test_load_play_policy_loads_real_checkpoint(self) -> None:
        policy = self._real_policy()

        play_policy = load_play_policy(policy, device="cpu")

        self.assertEqual(type(play_policy).__name__, "SACPlayOnly")
        self.assertTrue(callable(play_policy.get_eval_action))
        self.assertEqual(str(play_policy.device), "cpu")

    def test_prepare_evaluation_runtime_builds_real_intervention_and_policy(self) -> None:
        policy = self._real_policy()
        candidate = EvaluationCandidate(
            name="registry_candidate",
            execution_wire=policy.trained_on_wire or WireRef(model="steve_default", wire="standard_j"),
            policy=policy,
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
            target=BranchEndTarget(threshold_mm=5.0, branches=("lcca",)),
            friction=0.001,
        )

        runtime = prepare_evaluation_runtime(
            candidate=candidate,
            scenario=scenario,
            simulation=SimulationDummy(),
            policy_device="cpu",
        )
        runtime.intervention.reset()

        self.assertIsInstance(runtime, PreparedEvaluationRuntime)
        self.assertIsInstance(runtime.intervention, MonoPlaneStatic)
        self.assertEqual(len(runtime.intervention.devices), 1)
        self.assertIs(runtime.device, runtime.intervention.devices[0])
        self.assertEqual(runtime.candidate, candidate)
        self.assertEqual(runtime.scenario, scenario)
        self.assertEqual(type(runtime.play_policy).__name__, "SACPlayOnly")
        self.assertEqual(runtime.intervention.target.__class__.__name__, "BranchEnd")
        self.assertGreater(len(runtime.intervention.vessel_tree.branches), 0)
        self.assertEqual(runtime.intervention.device_lengths_inserted, [0.0])


class SafeResetIntegrationTests(unittest.TestCase):
    def _write_wire_registry(self, root: Path) -> Path:
        return RuntimeDeviceBuilderTests()._write_wire_registry(root)

    def test_safe_reset_intervention_accepts_seed_for_branch_random_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = self._write_wire_registry(Path(tmp))
            candidate = EvaluationCandidate(
                name="candidate_a",
                execution_wire=WireRef(model="steve_default", wire="standard_j"),
                policy=self._policy_stub(),
            )
            scenario = EvaluationScenario(
                name="scenario_seeded_reset",
                anatomy=AorticArchAnatomy(
                    arch_type="II",
                    seed=42,
                    rotation_yzx_deg=(1.0, 2.0, 3.0),
                    scaling_xyzd=(1.0, 1.1, 1.2, 1.3),
                    omit_axis="z",
                ),
                target=BranchEndTarget(
                    threshold_mm=5.0,
                    branches=("lcca", "rcca"),
                ),
            )

            intervention, _ = prepare_intervention_for_reset_test(
                candidate=candidate,
                scenario=scenario,
                registry_path=registry_path,
            )

            safe_reset_intervention(intervention, seed=123)
            first_coordinates = intervention.target.coordinates3d.copy()

            safe_reset_intervention(intervention, seed=123)
            second_coordinates = intervention.target.coordinates3d.copy()

        self.assertEqual(intervention.device_lengths_inserted, [0.0])
        self.assertEqual(first_coordinates.shape, (3,))
        self.assertTrue((first_coordinates == second_coordinates).all())

    @staticmethod
    def _policy_stub():
        from steve_recommender.eval_v2.models import PolicySpec

        return PolicySpec(
            name="policy_stub",
            checkpoint_path=Path("/tmp/policy_stub.everl"),
        )


def prepare_intervention_for_reset_test(
    *,
    candidate: EvaluationCandidate,
    scenario: EvaluationScenario,
    registry_path: Path,
) -> tuple[MonoPlaneStatic, JShaped]:
    from steve_recommender.eval_v2.runtime import build_intervention

    intervention, device = build_intervention(
        candidate=candidate,
        scenario=scenario,
        simulation=SimulationDummy(),
        registry_path=registry_path,
    )
    return intervention, device


if __name__ == "__main__":
    unittest.main()
