from __future__ import annotations

import unittest

import numpy as np
import pytest

from steve_recommender.eval_v2.force_telemetry import _project_constraint_forces
from steve_recommender.eval_v2.tests.scenes.validation_sphere_on_plane import createScene


SCENE_FORCE_TO_NEWTON = 1e-3
SPHERE_MASS_KG = 1.0
GRAVITY_MM_PER_S2 = 9810.0


class _Node:
    def __init__(self) -> None:
        self.children: dict[str, _Node] = {}
        self.objects: list[tuple[str, dict]] = []
        self.gravity = None
        self.dt = None

    def addObject(self, object_type: str, **kwargs):
        self.objects.append((object_type, dict(kwargs)))
        return type("_Object", (), kwargs)()

    def addChild(self, name: str):
        child = _Node()
        self.children[name] = child
        return child


class ValidationSphereOnPlaneTests(unittest.TestCase):
    def test_scene_builds_expected_graph(self) -> None:
        root = _Node()
        createScene(root)

        object_names = [name for name, _ in root.objects]
        self.assertIn("FreeMotionAnimationLoop", object_names)
        self.assertIn("DefaultPipeline", object_names)
        self.assertIn("LCPConstraintSolver", object_names)
        self.assertIn("DefaultContactManager", object_names)
        self.assertIn("Floor", root.children)
        self.assertIn("Sphere", root.children)
        self.assertEqual(root.gravity, [0.0, 0.0, -9810.0])

    def test_scene_file_is_importable_without_sofa(self) -> None:
        self.assertTrue(callable(createScene))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_scene_boots_and_produces_constraint_rows(self) -> None:
        """Verify the scaffold actually boots SOFA and produces constraint rows.

        This is the foundational test for the validation scene.
        If it fails, every later test in Phase B will fail for the wrong reason.
        We animate enough steps for contact handling, then assert separately that
        geometry exists, contact rows are emitted, and the solver returns impulses.
        """
        try:
            import Sofa
        except ImportError:
            self.skipTest("SOFA not available; run with: source scripts/sofa_env.sh && pytest")

        # Create SOFA root and populate the scene
        root = Sofa.Core.Node("root")
        createScene(root)

        # Initialize the simulation
        try:
            Sofa.Simulation.init(root)
        except Exception as e:
            self.fail(f"SOFA init failed: {e}")

        floor_positions = np.asarray(root.Floor.dofs.position.value)
        sphere_collision_positions = np.asarray(root.Sphere.Collision.dofs.position.value)
        self.assertGreater(
            floor_positions.size,
            0,
            "Floor MechanicalObject has no positions; the triangle topology is empty.",
        )
        self.assertGreater(
            sphere_collision_positions.size,
            0,
            "Sphere collision MechanicalObject has no positions; the collision model is empty.",
        )

        # Animate several steps to let collision detection and constraint solving run.
        dt_s = getattr(root.dt, "value", 0.01)
        for step in range(20):
            try:
                Sofa.Simulation.animate(root, dt_s)
            except Exception as e:
                self.fail(f"SOFA animate failed at step {step}: {e}")

        sphere_constraint = str(getattr(root.Sphere.Collision.dofs.constraint, "value", ""))
        self.assertNotEqual(
            sphere_constraint.strip(),
            "",
            "Sphere collision DOFs have no constraint rows after animation; "
            "collision detection/contact response did not emit rows.",
        )

        # Check that LCP.constraintForces is non-empty
        lcp = getattr(root, "LCP", None)
        self.assertIsNotNone(lcp, "No LCP solver found (root.LCP is None)")

        constraint_forces_raw = getattr(lcp, "constraintForces", None)
        if constraint_forces_raw is None:
            # Try findData as fallback
            try:
                constraint_forces_raw = lcp.findData("constraintForces")
            except Exception:
                constraint_forces_raw = None

        if hasattr(constraint_forces_raw, "value"):
            constraint_forces_raw = constraint_forces_raw.value

        self.assertIsNotNone(
            constraint_forces_raw,
            "LCP.constraintForces not found; check that LCPConstraintSolver "
            "has computeConstraintForces=True",
        )

        # Convert to array and verify non-empty
        try:
            arr = np.asarray(constraint_forces_raw, dtype=np.float64).reshape(-1)
        except Exception as e:
            self.fail(f"Failed to convert constraintForces to array: {e}")

        self.assertGreater(
            arr.size,
            0,
            "No constraint rows produced (LCP.constraintForces is empty). "
            "Check collision geometry and contact parameters.",
        )

    def _project_validation_scene_force(
        self,
        *,
        friction_mu: float = 0.0,
        dt_s: float = 0.01,
        steps: int = 20,
    ) -> np.ndarray:
        try:
            import Sofa
        except ImportError:
            self.skipTest("SOFA not available; run with: source scripts/sofa_env.sh && pytest")

        root = Sofa.Core.Node("root")
        createScene(root, friction_mu=friction_mu, dt_s=dt_s)
        try:
            Sofa.Simulation.init(root)
        except Exception as e:
            self.fail(f"SOFA init failed: {e}")

        step_dt_s = getattr(root.dt, "value", dt_s)
        for step in range(steps):
            try:
                Sofa.Simulation.animate(root, step_dt_s)
            except Exception as e:
                self.fail(f"SOFA animate failed at step {step}: {e}")

        constraint_raw = getattr(root.Sphere.Collision.dofs.constraint, "value", "")
        collision_positions = np.asarray(root.Sphere.Collision.dofs.position.value)
        lcp_forces = np.asarray(root.LCP.constraintForces.value, dtype=np.float64).reshape(-1)
        projected, _rows = _project_constraint_forces(
            lcp_forces,
            constraint_raw,
            int(collision_positions.reshape(-1, 3).shape[0]),
            float(step_dt_s),
        )
        return np.asarray(projected, dtype=np.float64).reshape(-1, 3).sum(axis=0)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_validation_scene_recovers_mg_within_1pct(self) -> None:
        projected_scene_force = self._project_validation_scene_force()

        expected_scene_force = SPHERE_MASS_KG * GRAVITY_MM_PER_S2
        self.assertAlmostEqual(projected_scene_force[0], 0.0, delta=1e-6)
        self.assertAlmostEqual(projected_scene_force[1], 0.0, delta=1e-6)
        self.assertAlmostEqual(
            projected_scene_force[2],
            expected_scene_force,
            delta=0.01 * expected_scene_force,
        )

        projected_si_force = projected_scene_force * SCENE_FORCE_TO_NEWTON
        self.assertAlmostEqual(
            projected_si_force[2],
            expected_scene_force * SCENE_FORCE_TO_NEWTON,
            delta=0.01 * expected_scene_force * SCENE_FORCE_TO_NEWTON,
        )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_validation_scene_with_friction_normal_force_unchanged(self) -> None:
        frictionless_force = self._project_validation_scene_force(friction_mu=0.0)
        frictional_force = self._project_validation_scene_force(friction_mu=0.5)

        expected_scene_force = SPHERE_MASS_KG * GRAVITY_MM_PER_S2
        self.assertAlmostEqual(
            frictional_force[2],
            expected_scene_force,
            delta=0.01 * expected_scene_force,
        )
        self.assertAlmostEqual(
            frictional_force[2],
            frictionless_force[2],
            delta=0.01 * expected_scene_force,
        )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_validation_scene_force_invariant_across_dt(self) -> None:
        expected_scene_force = SPHERE_MASS_KG * GRAVITY_MM_PER_S2

        for dt_s in (0.005, 0.01, 0.02):
            with self.subTest(dt_s=dt_s):
                projected_scene_force = self._project_validation_scene_force(dt_s=dt_s)
                self.assertAlmostEqual(
                    projected_scene_force[2],
                    expected_scene_force,
                    delta=0.01 * expected_scene_force,
                )


if __name__ == "__main__":
    unittest.main()
