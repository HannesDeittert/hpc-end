from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from steve_recommender.eval_v2.force_telemetry import (
    DEFAULT_TIP_THRESHOLD_MM,
    EvalV2ForceTelemetryCollector,
    _parse_constraint_rows,
    _project_constraint_forces,
    _unit_scale_to_newton,
)
from steve_recommender.eval_v2.models import ForceTelemetrySpec, ForceUnits
from steve_recommender.eval_v2.tests.helpers.fake_contact_export import FakeContactExport


class _Data:
    def __init__(self, value):
        self.value = value


class _CollisionDOFs:
    def __init__(self, constraint: str, positions: list[list[float]]) -> None:
        self.constraint = _Data(constraint)
        self.position = _Data(positions)


class _Root:
    def __init__(self, *, lcp: list[float], dt: float, collision: _CollisionDOFs, export: FakeContactExport) -> None:
        self.LCP = type("_LCP", (), {"constraintForces": _Data(lcp)})()
        self.dt = _Data(dt)
        self.InstrumentCombined = type(
            "_InstrumentCombined",
            (),
            {"CollisionModel": type("_CollisionModel", (), {"CollisionDOFs": collision})()},
        )()
        self.wire_wall_contact_export = export
        self.wire_wall_force_monitor = None


class _Simulation:
    def __init__(self, root: _Root) -> None:
        self.root = root


class _Intervention:
    def __init__(self, root: _Root) -> None:
        self.simulation = _Simulation(root)


class MappingHarnessTests(unittest.TestCase):
    def _collector(
        self,
        *,
        units: ForceUnits | None = None,
        tip_threshold_mm: float = DEFAULT_TIP_THRESHOLD_MM,
    ) -> EvalV2ForceTelemetryCollector:
        spec = ForceTelemetrySpec(
            mode="constraint_projected_si_validated",
            required=False,
            contact_epsilon=1e-7,
            tip_threshold_mm=tip_threshold_mm,
            units=units or ForceUnits(length_unit="mm", mass_unit="kg", time_unit="s"),
        )
        return EvalV2ForceTelemetryCollector(spec=spec, action_dt_s=0.1)

    def test_unit_scale_mm_kg_s_is_exactly_1e_minus_3(self) -> None:
        units = ForceUnits(length_unit="mm", mass_unit="kg", time_unit="s")
        self.assertAlmostEqual(_unit_scale_to_newton(units), 1e-3, places=15)

    def test_unit_scaling_applied_exactly_once_to_triangle_and_wire_records(self) -> None:
        collision = _CollisionDOFs("0 1 7 1.0 0.0 0.0", [[float(i), 0.0, 0.0] for i in range(8)])
        export = FakeContactExport(
            constraintRowIndices=[0],
            wallTriangleIds=[12],
            collisionDofIndices=[7],
            contactCount=1,
            explicitCoverage=1.0,
        )
        root = _Root(lcp=[4.0], dt=0.5, collision=collision, export=export)
        collector = self._collector()
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)

        self.assertEqual(len(collector._triangle_force_records), 1)
        self.assertEqual(len(collector._wire_force_records), 1)
        self.assertAlmostEqual(collector._triangle_force_records[0]["norm_N"], 4.0 * 1e-3 / 0.5, places=9)
        self.assertAlmostEqual(collector._wire_force_records[0]["norm_N"], 4.0 * 1e-3 / 0.5, places=9)

    def test_single_row_maps_to_single_triangle(self) -> None:
        collision = _CollisionDOFs("0 1 0 0.0 0.0 1.0", [[0.0, 0.0, 0.0]])
        export = FakeContactExport(constraintRowIndices=[0], wallTriangleIds=[12], collisionDofIndices=[0])
        root = _Root(lcp=[2.0], dt=1.0, collision=collision, export=export)
        collector = self._collector()
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)

        triangle_ids = [record["triangle_id"] for record in collector._triangle_force_records]
        self.assertEqual(triangle_ids, [12])
        self.assertEqual(collector._triangle_force_records[0]["contributing_rows"], 1)

    def test_multiple_rows_same_triangle_accumulate(self) -> None:
        collision = _CollisionDOFs(
            "\n".join(["0 1 0 1.0 0.0 0.0", "1 1 0 2.0 0.0 0.0"]),
            [[0.0, 0.0, 0.0]],
        )
        export = FakeContactExport(constraintRowIndices=[0, 1], wallTriangleIds=[12, 12], collisionDofIndices=[0, 0])
        root = _Root(lcp=[1.0, 1.0], dt=1.0, collision=collision, export=export)
        collector = self._collector()
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)

        self.assertEqual(len(collector._triangle_force_records), 1)
        self.assertEqual(collector._triangle_force_records[0]["triangle_id"], 12)
        self.assertEqual(collector._triangle_force_records[0]["contributing_rows"], 2)
        self.assertAlmostEqual(collector._triangle_force_records[0]["norm_N"], 3.0 * 1e-3, places=9)

    def test_rows_with_invalid_flag_excluded_from_aggregation(self) -> None:
        collision = _CollisionDOFs("0 1 0 1.0 0.0 0.0", [[0.0, 0.0, 0.0]])
        export = FakeContactExport(
            constraintRowIndices=[0],
            constraintRowValidFlags=[0],
            wallTriangleIds=[12],
            collisionDofIndices=[0],
            contactCount=1,
            explicitCoverage=1.0,
        )
        root = _Root(lcp=[1.0], dt=1.0, collision=collision, export=export)
        collector = self._collector()
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)
        summary = collector.build_summary()

        self.assertFalse(summary.available_for_score)
        self.assertEqual(summary.validation_status, "lcp_only_unmapped")
        self.assertEqual(collector._triangle_force_records, [])

    def test_explicit_coverage_false_marks_summary_unscoreable(self) -> None:
        collision = _CollisionDOFs("0 1 0 0.0 0.0 1.0", [[0.0, 0.0, 0.0]])
        export = FakeContactExport(
            constraintRowIndices=[0],
            wallTriangleIds=[12],
            collisionDofIndices=[0],
            contactCount=1,
            explicitCoverage=0.0,
        )
        root = _Root(lcp=[1.0], dt=1.0, collision=collision, export=export)
        collector = self._collector()
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)
        summary = collector.build_summary()

        self.assertFalse(summary.available_for_score)
        self.assertEqual(summary.validation_status, "partial")

    def test_row_maps_to_correct_wire_collision_dof(self) -> None:
        positions = [[float(i), 2.0, 3.0] for i in range(8)]
        positions[7] = [1.0, 2.0, 3.0]
        collision = _CollisionDOFs("0 1 7 0.0 1.0 0.0", positions)
        export = FakeContactExport(constraintRowIndices=[0], wallTriangleIds=[12], collisionDofIndices=[7])
        root = _Root(lcp=[2.0], dt=1.0, collision=collision, export=export)
        collector = self._collector()
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)

        self.assertEqual(len(collector._wire_force_records), 1)
        self.assertEqual(collector._wire_force_records[0]["wire_collision_dof"], 7)
        self.assertEqual(collector._wire_force_records[0]["world_pos"], [1.0, 2.0, 3.0])

    def test_wire_dof_record_force_matches_projection(self) -> None:
        collision = _CollisionDOFs("0 1 7 0.0 0.0 1.0", [[float(i), 2.0, 3.0] for i in range(8)])
        export = FakeContactExport(constraintRowIndices=[0], wallTriangleIds=[12], collisionDofIndices=[7])
        root = _Root(lcp=[3.0], dt=0.5, collision=collision, export=export)
        collector = self._collector()
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)

        expected_force = 3.0 / 0.5 * 1e-3
        self.assertAlmostEqual(collector._wire_force_records[0]["norm_N"], expected_force, places=9)

    def test_one_row_to_many_dofs_distributes_correctly(self) -> None:
        collision = _CollisionDOFs("0 2 7 1.0 0.0 0.0 8 0.5 0.0 0.0", [[float(i), 0.0, 0.0] for i in range(9)])
        export = FakeContactExport(constraintRowIndices=[0, 0], wallTriangleIds=[12, 12], collisionDofIndices=[7, 8])
        root = _Root(lcp=[2.0], dt=1.0, collision=collision, export=export)
        collector = self._collector()
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)

        self.assertEqual(len(collector._wire_force_records), 4)
        self.assertAlmostEqual(
            sum(record["norm_N"] for record in collector._wire_force_records),
            0.006,
            places=9,
        )

    def test_many_rows_to_one_dof_accumulate(self) -> None:
        collision = _CollisionDOFs("\n".join(["0 1 7 1.0 0.0 0.0", "1 1 7 2.0 0.0 0.0"]), [[float(i), 0.0, 0.0] for i in range(8)])
        export = FakeContactExport(constraintRowIndices=[0, 1], wallTriangleIds=[12, 12], collisionDofIndices=[7, 7])
        root = _Root(lcp=[1.0, 1.0], dt=1.0, collision=collision, export=export)
        collector = self._collector()
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)

        self.assertEqual(len(collector._wire_force_records), 2)
        self.assertTrue(all(record["wire_collision_dof"] == 7 for record in collector._wire_force_records))

    def test_record_count_matches_expected_sparse_size(self) -> None:
        collision = _CollisionDOFs("\n".join(["0 1 0 1.0 0.0 0.0", "1 1 1 0.0 1.0 0.0"]), [[float(i), 0.0, 0.0] for i in range(2)])
        export = FakeContactExport(constraintRowIndices=[0, 1], wallTriangleIds=[12, 13], collisionDofIndices=[0, 1])
        root = _Root(lcp=[1.0, 2.0], dt=1.0, collision=collision, export=export)
        collector = self._collector()
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)

        self.assertEqual(len(collector._triangle_force_records), 2)
        self.assertEqual(len(collector._wire_force_records), 2)

    def test_wire_record_has_is_tip_flag(self) -> None:
        collision = _CollisionDOFs("0 1 2 1.0 0.0 0.0", [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        export = FakeContactExport(constraintRowIndices=[0], wallTriangleIds=[12], collisionDofIndices=[2])
        root = _Root(lcp=[1.0], dt=1.0, collision=collision, export=export)
        collector = self._collector(tip_threshold_mm=10.0)
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)

        record = collector._wire_force_records[0]
        self.assertIn("is_tip", record)
        self.assertIn("arc_length_from_distal_mm", record)
        self.assertTrue(record["is_tip"])
        self.assertAlmostEqual(record["arc_length_from_distal_mm"], 0.0)

    def test_tip_threshold_is_inclusive(self) -> None:
        collision = _CollisionDOFs("0 1 1 1.0 0.0 0.0", [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        export = FakeContactExport(constraintRowIndices=[0], wallTriangleIds=[12], collisionDofIndices=[1])
        root = _Root(lcp=[1.0], dt=1.0, collision=collision, export=export)
        collector = self._collector(tip_threshold_mm=5.0)
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)

        self.assertAlmostEqual(collector._wire_force_records[0]["arc_length_from_distal_mm"], 5.0)
        self.assertTrue(collector._wire_force_records[0]["is_tip"])

    def test_tip_threshold_default_documented(self) -> None:
        collector = self._collector()

        self.assertAlmostEqual(collector._tip_threshold_mm, DEFAULT_TIP_THRESHOLD_MM)

    def test_tip_threshold_configurable(self) -> None:
        collision = _CollisionDOFs("0 1 1 1.0 0.0 0.0", [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        export = FakeContactExport(constraintRowIndices=[0], wallTriangleIds=[12], collisionDofIndices=[1])
        root = _Root(lcp=[1.0], dt=1.0, collision=collision, export=export)
        collector = self._collector(tip_threshold_mm=4.0)
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)

        self.assertAlmostEqual(collector._wire_force_records[0]["arc_length_from_distal_mm"], 5.0)
        self.assertFalse(collector._wire_force_records[0]["is_tip"])

    def test_distal_dof_filter_isolates_tip_records(self) -> None:
        collision = _CollisionDOFs(
            "\n".join(["0 1 0 1.0 0.0 0.0", "1 1 2 0.0 2.0 0.0"]),
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        )
        export = FakeContactExport(
            constraintRowIndices=[0, 1],
            wallTriangleIds=[12, 13],
            collisionDofIndices=[0, 2],
        )
        root = _Root(lcp=[1.0, 1.0], dt=1.0, collision=collision, export=export)
        collector = self._collector(tip_threshold_mm=1.0)
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)

        summary = collector.build_summary()

        self.assertEqual(len(summary.tip_force_records), 1)
        self.assertEqual(summary.tip_force_records[0]["wire_collision_dof"], 2)
        self.assertTrue(summary.tip_force_records[0]["is_tip"])

    def test_tip_force_total_equals_sum_of_tip_records(self) -> None:
        collision = _CollisionDOFs(
            "\n".join(["0 1 1 1.0 0.0 0.0", "1 1 2 0.0 2.0 0.0"]),
            [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        )
        export = FakeContactExport(
            constraintRowIndices=[0, 1],
            wallTriangleIds=[12, 13],
            collisionDofIndices=[1, 2],
        )
        root = _Root(lcp=[1.0, 1.0], dt=1.0, collision=collision, export=export)
        collector = self._collector(tip_threshold_mm=2.0)
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)

        summary = collector.build_summary()
        expected = np.asarray(
            [
                sum(record["fx_N"] for record in summary.tip_force_records),
                sum(record["fy_N"] for record in summary.tip_force_records),
                sum(record["fz_N"] for record in summary.tip_force_records),
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(
            np.asarray(summary.tip_force_total_vector_N, dtype=np.float64),
            expected,
            rtol=0.0,
            atol=1e-12,
        )

    def test_tip_records_unit_invariant(self) -> None:
        collision = _CollisionDOFs("0 1 2 0.0 0.0 2.0", [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        export = FakeContactExport(constraintRowIndices=[0], wallTriangleIds=[12], collisionDofIndices=[2])
        root = _Root(lcp=[3.0], dt=0.5, collision=collision, export=export)
        collector = self._collector(tip_threshold_mm=1.0)
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)

        summary = collector.build_summary()
        record = summary.tip_force_records[0]
        self.assertAlmostEqual(record["norm_N"], record["norm_scene"] * 1e-3, places=9)

    def test_tip_filter_runs_on_real_v23_06_fixture(self) -> None:
        constraint_raw = (Path(__file__).resolve().parents[2] / "steve_recommender" / "eval_v2" / "tests" / "fixtures" / "constraint_string_v23_06_real.txt").read_text(encoding="utf-8")
        lcp = np.load(Path(__file__).resolve().parents[2] / "steve_recommender" / "eval_v2" / "tests" / "fixtures" / "lcp_constraintForces_real.npy")
        rows = _parse_constraint_rows(constraint_raw)
        row_ids = [row_idx for row_idx, _dof_idx, _coeff in rows]
        dof_ids = [dof_idx for _row_idx, dof_idx, _coeff in rows]
        positions = [[float(i), 0.0, 0.0] for i in range(max(dof_ids) + 1)]
        collision = _CollisionDOFs(constraint_raw, positions)
        export = FakeContactExport(
            constraintRowIndices=row_ids,
            wallTriangleIds=[100 + idx for idx in range(len(row_ids))],
            collisionDofIndices=dof_ids,
            contactCount=len(row_ids),
            explicitCoverage=1.0,
        )
        root = _Root(lcp=np.asarray(lcp, dtype=np.float64).reshape(-1).tolist(), dt=0.01, collision=collision, export=export)
        collector = self._collector(tip_threshold_mm=DEFAULT_TIP_THRESHOLD_MM)
        collector.ensure_runtime(intervention=_Intervention(root))
        collector.capture_step(intervention=_Intervention(root), step_index=1)

        summary = collector.build_summary()

        self.assertGreater(len(collector._wire_force_records), 0)
        self.assertGreater(len(summary.tip_force_records), 0)
        self.assertTrue(all("is_tip" in record for record in collector._wire_force_records))


class ProjectionHelperTests(unittest.TestCase):
    def test_project_constraint_rows_matches_expected_projection(self) -> None:
        proj, rows = _project_constraint_forces(
            lcp_forces=np.asarray([2.0], dtype=np.float32),
            constraint_raw="0 1 7 1.0 0.0 0.0",
            n_points=8,
            dt_s=0.5,
        )
        self.assertAlmostEqual(float(proj[7, 0]), 4.0, places=6)
        self.assertEqual(rows[0]["row_idx"], 0)
        self.assertEqual(rows[0]["dof_idx"], 7)


if __name__ == "__main__":
    unittest.main()
