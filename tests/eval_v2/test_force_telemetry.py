from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from steve_recommender.eval_v2.force_telemetry import (
    DEFAULT_TIP_THRESHOLD_MM,
    EvalV2ForceTelemetryCollector,
    collision_dof_arc_lengths_from_distal_mm,
    _parse_constraint_rows,
    _project_constraint_forces,
    _unit_scale_to_newton,
)
import numpy as np
from steve_recommender.eval_v2.models import ForceTelemetrySpec, ForceUnits


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "steve_recommender" / "eval_v2" / "tests" / "fixtures"


def _assert_rows_equal(
    testcase: unittest.TestCase,
    parsed: list[tuple[int, int, np.ndarray]],
    expected: list[tuple[int, int, tuple[float, float, float]]],
) -> None:
    testcase.assertEqual(len(parsed), len(expected))
    for (row_idx, dof_idx, coeff), (exp_row, exp_dof, exp_coeff) in zip(parsed, expected):
        testcase.assertEqual(int(row_idx), int(exp_row))
        testcase.assertEqual(int(dof_idx), int(exp_dof))
        np.testing.assert_allclose(np.asarray(coeff, dtype=np.float64), np.asarray(exp_coeff, dtype=np.float64), rtol=0.0, atol=1e-6)


class _Data:
    def __init__(self, value):
        self.value = value


class _Monitor:
    def __init__(self) -> None:
        self.available = _Data(True)
        self.source = _Data("passive_monitor_wall_triangles")
        self.status = _Data("ok:wire.force:nearest_triangle_centroid")
        self.totalForceNorm = _Data(0.2)
        self.contactCount = _Data(3)
        self.wallSegmentCount = _Data(10)
        self.segmentForceVectors = _Data(
            [
                [0.1, 0.0, 0.0],
                [0.0, 0.3, 0.0],
                [0.0, 0.0, 0.2],
            ]
        )


class _LCP:
    def __init__(self) -> None:
        self.constraintForces = _Data([1.0, -2.0, 0.5])


class _Export:
    def __init__(self, *, contact_count: int = 1, valid_rows: list[int] | None = None) -> None:
        rows = list(valid_rows or [0])
        self.available = _Data(True)
        self.source = _Data("native_contact_export")
        self.status = _Data("ok:records=1:explicit=1")
        self.contactCount = _Data(int(contact_count))
        self.explicitCoverage = _Data(1.0)
        self.orderingStable = _Data(True)
        self.constraintRowIndices = _Data(rows)
        self.constraintRowValidFlags = _Data([1 for _ in rows])
        self.wallTriangleIds = _Data([0 for _ in rows])
        self.triangleIdValidFlags = _Data([1 for _ in rows])


class _Root:
    def __init__(self, monitor: _Monitor | None, dt: float = 0.1, export: _Export | None = None) -> None:
        self.wire_wall_force_monitor = monitor
        self.wire_wall_contact_export = export
        self.LCP = _LCP()
        self.dt = _Data(dt)


class _Simulation:
    def __init__(self, monitor: _Monitor | None, dt: float = 0.1, export: _Export | None = None) -> None:
        self.root = _Root(monitor, dt=dt, export=export)


class _Intervention:
    def __init__(self, monitor: _Monitor | None, dt: float = 0.1, export: _Export | None = None) -> None:
        self.simulation = _Simulation(monitor, dt=dt, export=export)


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

class UnitConversionTests(unittest.TestCase):
    def test_default_tip_threshold_constant_is_3mm(self) -> None:
        self.assertAlmostEqual(DEFAULT_TIP_THRESHOLD_MM, 3.0)

    def test_mm_kg_s_gives_1e_minus_3(self) -> None:
        units = ForceUnits(length_unit="mm", mass_unit="kg", time_unit="s")
        scale = _unit_scale_to_newton(units)
        self.assertAlmostEqual(scale, 1e-3, places=15)


class TipThresholdConfigTests(unittest.TestCase):
    def test_collector_uses_default_when_unspecified(self) -> None:
        collector = EvalV2ForceTelemetryCollector(
            spec=ForceTelemetrySpec(mode="passive", required=False),
            action_dt_s=0.1,
        )

        self.assertAlmostEqual(collector._tip_threshold_mm, DEFAULT_TIP_THRESHOLD_MM)

    def test_collector_accepts_explicit_threshold(self) -> None:
        collector = EvalV2ForceTelemetryCollector(
            spec=ForceTelemetrySpec(mode="passive", required=False, tip_threshold_mm=5.0),
            action_dt_s=0.1,
        )

        self.assertAlmostEqual(collector._tip_threshold_mm, 5.0)

    def test_collector_rejects_negative_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "tip_threshold_mm.*> 0.*-1.0"):
            ForceTelemetrySpec(mode="passive", required=False, tip_threshold_mm=-1.0)

    def test_collector_rejects_zero_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "tip_threshold_mm.*> 0.*0.0"):
            ForceTelemetrySpec(mode="passive", required=False, tip_threshold_mm=0.0)


class TipArcLengthMapTests(unittest.TestCase):
    def test_arc_length_map_for_straight_wire_matches_segment_lengths(self) -> None:
        positions_mm = np.asarray(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
            dtype=np.float64,
        )

        arc_lengths = collision_dof_arc_lengths_from_distal_mm(positions_mm)

        self.assertEqual(set(arc_lengths), {0, 1, 2, 3})
        self.assertAlmostEqual(arc_lengths[0], 9.0)
        self.assertAlmostEqual(arc_lengths[1], 7.0)
        self.assertAlmostEqual(arc_lengths[2], 4.0)
        self.assertAlmostEqual(arc_lengths[3], 0.0)

    def test_arc_length_map_collision_dof_subset(self) -> None:
        wire_positions_mm = np.asarray([[float(i), 0.0, 0.0] for i in range(10)], dtype=np.float64)

        arc_lengths = collision_dof_arc_lengths_from_distal_mm(
            wire_positions_mm,
            collision_dof_wire_indices=[0, 2, 4, 6, 9],
        )

        self.assertEqual(set(arc_lengths), {0, 1, 2, 3, 4})
        self.assertAlmostEqual(arc_lengths[0], 9.0)
        self.assertAlmostEqual(arc_lengths[1], 7.0)
        self.assertAlmostEqual(arc_lengths[2], 5.0)
        self.assertAlmostEqual(arc_lengths[3], 3.0)
        self.assertAlmostEqual(arc_lengths[4], 0.0)

    def test_arc_length_map_distal_direction_documented(self) -> None:
        positions_mm = np.asarray([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float64)

        arc_lengths = collision_dof_arc_lengths_from_distal_mm(positions_mm)

        self.assertAlmostEqual(arc_lengths[0], 3.0)
        self.assertAlmostEqual(arc_lengths[1], 0.0)

    def test_m_kg_s_gives_1(self) -> None:
        units = ForceUnits(length_unit="m", mass_unit="kg", time_unit="s")
        scale = _unit_scale_to_newton(units)
        self.assertAlmostEqual(scale, 1.0, places=15)


# ---------------------------------------------------------------------------
# dt conversion
# ---------------------------------------------------------------------------

class DtConversionTests(unittest.TestCase):
    def test_lcp_force_uses_root_dt_not_action_dt(self) -> None:
        # root.dt = 0.006 (SOFA simulation step), action_dt_s = 1/7.5 (image frequency)
        # The lcp force-like conversion must divide by root.dt, not action_dt.
        root_dt = 0.006
        action_dt = 1.0 / 7.5
        monitor = _Monitor()
        monitor.available = _Data(False)
        monitor.totalForceNorm = _Data(0.0)
        monitor.contactCount = _Data(0)
        monitor.segmentForceVectors = _Data([])
        intervention = _Intervention(monitor, dt=root_dt)

        collector = EvalV2ForceTelemetryCollector(
            spec=ForceTelemetrySpec(mode="passive", required=False, contact_epsilon=1e-7),
            action_dt_s=action_dt,
        )
        collector.ensure_runtime(intervention=intervention)
        collector.capture_step(intervention=intervention, step_index=1)

        summary = collector.build_summary()
        # LCP has [1.0, -2.0, 0.5]; max(|lambda|) = 2.0
        # Using root_dt=0.006: force_like = 2.0 / 0.006 ≈ 333.3
        # Using action_dt=1/7.5≈0.133: force_like = 2.0 / 0.133 ≈ 15.0
        # The collector must use root_dt (333.3), not action_dt (15.0).
        lcp_max = summary.lcp_max_abs_max
        self.assertIsNotNone(lcp_max)
        assert lcp_max is not None
        # raw lcp_max_abs_max is max(|lambda|) = 2.0 (before dt division)
        self.assertAlmostEqual(lcp_max, 2.0, places=6)
        # total_force_norm_max comes from lcp_force_like: 2.0 / root_dt
        total = summary.total_force_norm_max
        self.assertIsNotNone(total)
        assert total is not None
        expected_force_like = 2.0 / root_dt
        self.assertAlmostEqual(total, expected_force_like, places=2,
                               msg=f"Force-like should use root.dt={root_dt}, got {total}")
        # Ensure it's not accidentally equal to the action_dt version
        wrong_force_like = 2.0 / action_dt
        self.assertNotAlmostEqual(total, wrong_force_like, places=1,
                                  msg="Force-like must NOT use action_dt for dt division")


# ---------------------------------------------------------------------------
# Parser robustness
# ---------------------------------------------------------------------------

class ConstraintParserTests(unittest.TestCase):
    def test_parse_old_format_single_dof_per_row(self) -> None:
        raw = "\n".join(
            [
                "0 1 146 0.00447023 -0.0720868 -0.00441247",
                "1 1 158 0.000850771 -0.0137195 -0.000839777",
                "2 1 170 0.0228333 -0.36821 -0.0225383",
            ]
        )

        parsed = _parse_constraint_rows(raw)

        _assert_rows_equal(
            self,
            parsed,
            [
                (0, 146, (0.00447023, -0.0720868, -0.00441247)),
                (1, 158, (0.000850771, -0.0137195, -0.000839777)),
                (2, 170, (0.0228333, -0.36821, -0.0225383)),
            ],
        )

    def test_parse_new_format_single_dof_per_row(self) -> None:
        raw = "\n".join(
            [
                "Constraint ID : 0 / dof ID : 146 / value : 0.00447023 -0.0720868 -0.00441247",
                "Constraint ID : 1 / dof ID : 158 / value : 0.000850771 -0.0137195 -0.000839777",
                "Constraint ID : 2 / dof ID : 170 / value : 0.0228333 -0.36821 -0.0225383",
            ]
        )

        parsed = _parse_constraint_rows(raw)

        _assert_rows_equal(
            self,
            parsed,
            [
                (0, 146, (0.00447023, -0.0720868, -0.00441247)),
                (1, 158, (0.000850771, -0.0137195, -0.000839777)),
                (2, 170, (0.0228333, -0.36821, -0.0225383)),
            ],
        )

    def test_parse_old_and_new_yield_identical_rows(self) -> None:
        old_raw = "\n".join(
            [
                "0 1 146 0.00447023 -0.0720868 -0.00441247",
                "1 1 158 0.000850771 -0.0137195 -0.000839777",
            ]
        )
        new_raw = "\n".join(
            [
                "Constraint ID : 0 / dof ID : 146 / value : 0.00447023 -0.0720868 -0.00441247",
                "Constraint ID : 1 / dof ID : 158 / value : 0.000850771 -0.0137195 -0.000839777",
            ]
        )

        old_rows = _parse_constraint_rows(old_raw)
        new_rows = _parse_constraint_rows(new_raw)
        self.assertEqual(len(old_rows), len(new_rows))
        for (old_row, old_dof, old_coeff), (new_row, new_dof, new_coeff) in zip(old_rows, new_rows):
            self.assertEqual(int(old_row), int(new_row))
            self.assertEqual(int(old_dof), int(new_dof))
            np.testing.assert_array_equal(np.asarray(old_coeff), np.asarray(new_coeff))

    def test_parse_multidof_row_keeps_all_dofs(self) -> None:
        raw = "\n".join(
            [
                "Constraint ID : 9 / dof ID : 146 / value : 0.7 0.0 0.0 Constraint ID : 9 / dof ID : 158 / value : 0.3 0.0 0.0",
            ]
        )

        parsed = _parse_constraint_rows(raw)

        _assert_rows_equal(
            self,
            parsed,
            [
                (9, 146, (0.7, 0.0, 0.0)),
                (9, 158, (0.3, 0.0, 0.0)),
            ],
        )

    def test_parse_empty_string_returns_empty_rows(self) -> None:
        self.assertEqual(_parse_constraint_rows(""), [])
        self.assertEqual(_parse_constraint_rows("   \n\t  "), [])

    def test_parse_real_v23_06_fixture(self) -> None:
        fixture_path = FIXTURE_DIR / "constraint_string_v23_06_real.txt"
        raw = fixture_path.read_text(encoding="utf-8")

        parsed = _parse_constraint_rows(raw)

        self.assertEqual(len(parsed), 12)
        self.assertEqual(sorted({row_idx for row_idx, _, _ in parsed}), [0, 1, 2])
        self.assertEqual(sorted({dof_idx for _, dof_idx, _ in parsed}), [146, 158, 170, 171])
        for _, _, coeff in parsed:
            self.assertEqual(np.asarray(coeff).shape, (3,))
            self.assertTrue(np.all(np.isfinite(coeff)))


# ---------------------------------------------------------------------------
# Projection mathematics
# ---------------------------------------------------------------------------

class ConstraintProjectionMathTests(unittest.TestCase):
    def test_project_unit_lambda_recovers_h_row(self) -> None:
        constraint_raw = "0 1 7 -1.5 2.25 0.5"
        proj, row_contribs = _project_constraint_forces(
            lcp_forces=np.asarray([1.0], dtype=np.float32),
            constraint_raw=constraint_raw,
            n_points=8,
            dt_s=1.0,
        )

        expected = np.zeros((8, 3), dtype=np.float32)
        expected[7] = np.asarray([-1.5, 2.25, 0.5], dtype=np.float32)
        np.testing.assert_allclose(proj, expected, rtol=0.0, atol=1e-7)
        self.assertEqual(len(row_contribs), 1)
        self.assertEqual(row_contribs[0]["row_idx"], 0)
        self.assertEqual(row_contribs[0]["dof_idx"], 7)
        np.testing.assert_allclose(row_contribs[0]["force_vec"], expected[7], rtol=0.0, atol=1e-7)

    def test_project_zero_lambda_yields_zero_forces(self) -> None:
        constraint_raw = "0 1 0 1.0 0.0 0.0"
        proj, row_contribs = _project_constraint_forces(
            lcp_forces=np.asarray([0.0], dtype=np.float32),
            constraint_raw=constraint_raw,
            n_points=1,
            dt_s=0.25,
        )

        np.testing.assert_array_equal(proj, np.zeros((1, 3), dtype=np.float32))
        self.assertEqual(len(row_contribs), 1)
        self.assertEqual(row_contribs[0]["row_idx"], 0)
        self.assertEqual(row_contribs[0]["dof_idx"], 0)
        np.testing.assert_array_equal(row_contribs[0]["force_vec"], np.zeros((3,), dtype=np.float32))
        self.assertEqual(float(row_contribs[0]["force_norm"]), 0.0)

    def test_project_three_friction_rows_recompose_3d_vector(self) -> None:
        constraint_raw = "\n".join(
            [
                "0 1 0 1.0 0.0 0.0",
                "1 1 0 0.0 1.0 0.0",
                "2 1 0 0.0 0.0 1.0",
            ]
        )
        proj, _ = _project_constraint_forces(
            lcp_forces=np.asarray([3.0, -2.0, 0.5], dtype=np.float32),
            constraint_raw=constraint_raw,
            n_points=1,
            dt_s=1.0,
        )

        np.testing.assert_allclose(proj[0], np.asarray([3.0, -2.0, 0.5], dtype=np.float32), rtol=0.0, atol=1e-7)

    def test_project_multidof_row_distributes_with_barycentric_weights(self) -> None:
        constraint_raw = "0 2 0 0.7 0.0 0.0 1 0.3 0.0 0.0"
        proj, row_contribs = _project_constraint_forces(
            lcp_forces=np.asarray([2.0], dtype=np.float32),
            constraint_raw=constraint_raw,
            n_points=2,
            dt_s=1.0,
        )

        np.testing.assert_allclose(proj[0], np.asarray([1.4, 0.0, 0.0], dtype=np.float32), rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(proj[1], np.asarray([0.6, 0.0, 0.0], dtype=np.float32), rtol=0.0, atol=1e-7)
        self.assertAlmostEqual(float(np.sum(proj[:, 0])), 2.0, places=7)
        self.assertEqual(len(row_contribs), 2)

    def test_project_dt_scales_inversely(self) -> None:
        constraint_raw = "0 1 0 0.0 0.0 1.0"
        lcp = np.asarray([2.5], dtype=np.float32)

        proj_dt1, _ = _project_constraint_forces(
            lcp_forces=lcp,
            constraint_raw=constraint_raw,
            n_points=1,
            dt_s=1.0,
        )
        proj_dt_half, _ = _project_constraint_forces(
            lcp_forces=lcp,
            constraint_raw=constraint_raw,
            n_points=1,
            dt_s=0.5,
        )

        np.testing.assert_allclose(proj_dt_half, proj_dt1 * 2.0, rtol=0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Monitor path (valid)
# ---------------------------------------------------------------------------

class MonitorPathTests(unittest.TestCase):
    def test_collects_summary_from_runtime_monitor(self) -> None:
        monitor = _Monitor()
        intervention = _Intervention(monitor)
        collector = EvalV2ForceTelemetryCollector(
            spec=ForceTelemetrySpec(mode="passive", required=False, contact_epsilon=1e-7),
            action_dt_s=0.1,
        )

        status = collector.ensure_runtime(intervention=intervention)
        self.assertTrue(status.configured)

        collector.capture_step(intervention=intervention, step_index=1)

        monitor.totalForceNorm.value = 0.6
        monitor.segmentForceVectors.value = [[0.0, 0.0, 0.9]]
        collector.capture_step(intervention=intervention, step_index=2)

        summary = collector.build_summary()
        # New design: monitor is diagnostic only. Without mapped LCP rows the
        # primary validated source is LCP (unmapped -> diagnostic only).
        self.assertFalse(summary.available_for_score)
        self.assertEqual(summary.validation_status, "lcp_only_unmapped")
        self.assertEqual(summary.channel, "lcp.constraintForces/dt")
        self.assertEqual(summary.association_method, "force_points_nearest_triangle")
        # Monitor-derived diagnostics still populated
        self.assertAlmostEqual(float(summary.total_force_norm_max or 0.0), 0.6, places=6)
        self.assertAlmostEqual(float(summary.total_force_norm_mean or 0.0), 0.4, places=6)
        self.assertAlmostEqual(float(summary.lcp_max_abs_max or 0.0), 2.0, places=6)
        self.assertAlmostEqual(float(summary.lcp_sum_abs_mean or 0.0), 3.5, places=6)
        self.assertAlmostEqual(float(summary.peak_segment_force_norm or 0.0), 0.9, places=6)
        self.assertEqual(summary.peak_segment_force_step, 2)
        self.assertAlmostEqual(float(summary.peak_segment_force_time_s or 0.0), 0.2, places=6)

    def test_tip_force_unavailable_on_monitor_path(self) -> None:
        monitor = _Monitor()
        intervention = _Intervention(monitor)
        collector = EvalV2ForceTelemetryCollector(
            spec=ForceTelemetrySpec(mode="passive", required=False),
            action_dt_s=0.1,
        )
        collector.ensure_runtime(intervention=intervention)
        collector.capture_step(intervention=intervention, step_index=1)
        summary = collector.build_summary()

        self.assertFalse(summary.tip_force_available,
                         "Tip force must be explicitly unavailable until spatial mapping is implemented")
        self.assertEqual(summary.tip_force_validation_status, "unmapped")


# ---------------------------------------------------------------------------
# LCP fallback semantics (the critical fix)
# ---------------------------------------------------------------------------

class LcpFallbackSemanticsTests(unittest.TestCase):
    """
    LCP fallback fires when the C++ monitor is zero but LCP has nonzero rows.
    In this case, constraint rows are NOT mapped to wire-wall contacts — the LCP
    vector includes friction, beam corrections, and all contacts.  The fallback
    must NOT be treated as validated wall force.
    """

    def _make_collector_with_zero_monitor(
        self,
        *,
        mode: str = "passive",
        units: ForceUnits | None = None,
        action_dt: float = 0.1,
    ) -> tuple[EvalV2ForceTelemetryCollector, _Intervention]:
        monitor = _Monitor()
        monitor.totalForceNorm.value = 0.0
        monitor.contactCount.value = 0
        monitor.segmentForceVectors.value = [[0.0, 0.0, 0.0]]
        intervention = _Intervention(monitor)
        spec = ForceTelemetrySpec(
            mode=mode,
            required=False,
            contact_epsilon=1e-7,
            units=units,
        )
        collector = EvalV2ForceTelemetryCollector(spec=spec, action_dt_s=action_dt)
        return collector, intervention

    def test_lcp_fallback_not_available_for_score(self) -> None:
        collector, intervention = self._make_collector_with_zero_monitor()
        collector.ensure_runtime(intervention=intervention)
        collector.capture_step(intervention=intervention, step_index=1)

        summary = collector.build_summary()

        self.assertFalse(
            summary.available_for_score,
            "LCP-only fallback without contact-row mapping must NOT be available_for_score",
        )

    def test_lcp_fallback_validation_status_is_lcp_only_unmapped(self) -> None:
        collector, intervention = self._make_collector_with_zero_monitor()
        collector.ensure_runtime(intervention=intervention)
        collector.capture_step(intervention=intervention, step_index=1)

        summary = collector.build_summary()

        self.assertEqual(
            summary.validation_status,
            "lcp_only_unmapped",
            "LCP fallback without mapping must carry validation_status='lcp_only_unmapped'",
        )

    def test_lcp_fallback_channel_is_set_for_diagnostics(self) -> None:
        collector, intervention = self._make_collector_with_zero_monitor()
        collector.ensure_runtime(intervention=intervention)
        collector.capture_step(intervention=intervention, step_index=1)

        summary = collector.build_summary()

        self.assertEqual(summary.channel, "lcp.constraintForces/dt",
                         "Channel must name the LCP path even when not available_for_score")
        self.assertIn("fallback_lcp_dt", summary.source)

    def test_lcp_fallback_lcp_stats_populated_for_diagnostics(self) -> None:
        collector, intervention = self._make_collector_with_zero_monitor()
        collector.ensure_runtime(intervention=intervention)
        collector.capture_step(intervention=intervention, step_index=1)

        summary = collector.build_summary()

        # Raw LCP values still present for downstream diagnostics
        self.assertIsNotNone(summary.lcp_max_abs_max, "lcp_max_abs_max should be populated")
        self.assertIsNotNone(summary.total_force_norm_max,
                             "total_force_norm_max should contain LCP force-like value for diagnostics")
        assert summary.total_force_norm_max is not None
        self.assertGreater(summary.total_force_norm_max, 0.0)

    def test_mapped_lcp_becomes_available_for_score(self) -> None:
        units = ForceUnits(length_unit="mm", mass_unit="kg", time_unit="s")
        monitor = _Monitor()
        monitor.totalForceNorm.value = 0.0
        monitor.contactCount.value = 0
        monitor.segmentForceVectors.value = []
        export = _Export(contact_count=1, valid_rows=[0])
        intervention = _Intervention(monitor, dt=0.5, export=export)
        collector = EvalV2ForceTelemetryCollector(
            spec=ForceTelemetrySpec(
                mode="constraint_projected_si_validated",
                required=False,
                contact_epsilon=1e-7,
                units=units,
            ),
            action_dt_s=0.1,
        )
        collector.ensure_runtime(intervention=intervention)
        collector.capture_step(intervention=intervention, step_index=1)

        summary = collector.build_summary()
        self.assertTrue(summary.available_for_score)
        self.assertEqual(summary.validation_status, "ok")
        self.assertEqual(summary.channel, "lcp.constraintForces/dt")
        self.assertGreater(summary.lcp_mapped_wall_row_count_max, 0)
        self.assertIsNotNone(summary.lcp_contact_export_coverage)
        self.assertGreater(float(summary.total_force_norm_max or 0.0), 0.0)

    def test_lcp_fallback_quality_tier_is_degraded_not_unavailable(self) -> None:
        # "degraded" = data present but unmapped; "unavailable" = no data at all.
        collector, intervention = self._make_collector_with_zero_monitor()
        collector.ensure_runtime(intervention=intervention)
        collector.capture_step(intervention=intervention, step_index=1)

        summary = collector.build_summary()

        self.assertEqual(summary.quality_tier, "degraded",
                         "LCP fallback has data, so quality_tier should be 'degraded' not 'unavailable'")

    def test_lcp_fallback_si_scaling_applied_for_validated_mode(self) -> None:
        # Even though not available_for_score, the SI-scaled value is in total_force_norm_max
        # for diagnostic display.  max(|lambda|)=2.0, dt=0.1 → 20 scene units;
        # mm,kg,s → ×1e-3 → 0.02 N.
        units = ForceUnits(length_unit="mm", mass_unit="kg", time_unit="s")
        collector, intervention = self._make_collector_with_zero_monitor(
            mode="constraint_projected_si_validated", units=units, action_dt=0.1
        )
        collector.ensure_runtime(intervention=intervention)
        collector.capture_step(intervention=intervention, step_index=1)

        summary = collector.build_summary()

        self.assertFalse(summary.available_for_score)
        self.assertEqual(summary.validation_status, "lcp_only_unmapped")
        self.assertAlmostEqual(float(summary.total_force_norm_max or 0.0), 0.02, places=9)


# ---------------------------------------------------------------------------
# GUI regression: monitor=0, contact=0, LCP nonzero → not valid wall force
# ---------------------------------------------------------------------------

class GuiRegressionTests(unittest.TestCase):
    """
    Regression for the scenario seen in eval_v2 GUI comparison reports:
      channel = "lcp.constraintForces/dt"
      contact_count_max = 0
      contact_detected_any = false
      quality_tier = "degraded"
    The system must not claim C++ monitor produced valid wall-contact forces.
    """

    def test_monitor_zero_lcp_nonzero_is_not_valid_wall_force(self) -> None:
        monitor = _Monitor()
        monitor.available = _Data(True)   # monitor initialized, links valid
        monitor.totalForceNorm.value = 0.0
        monitor.contactCount.value = 0
        monitor.segmentForceVectors.value = []
        intervention = _Intervention(monitor)

        collector = EvalV2ForceTelemetryCollector(
            spec=ForceTelemetrySpec(mode="passive", required=False),
            action_dt_s=0.1,
        )
        collector.ensure_runtime(intervention=intervention)
        # Simulate a run with nonzero LCP but monitor consistently reporting zero
        for step in range(1, 51):
            collector.capture_step(intervention=intervention, step_index=step)

        summary = collector.build_summary()

        # Despite LCP having nonzero values, this is NOT valid wall force
        self.assertFalse(
            summary.available_for_score,
            "monitor=0 + LCP nonzero must NOT be available_for_score=True",
        )
        self.assertEqual(
            summary.validation_status,
            "lcp_only_unmapped",
            "Must explicitly name unmapped LCP fallback",
        )
        self.assertFalse(
            summary.contact_detected_any,
            "contact_detected_any must be False when monitor reports zero contacts",
        )
        self.assertEqual(summary.contact_count_max, 0)

    def test_contact_detected_any_false_when_monitor_always_zero(self) -> None:
        monitor = _Monitor()
        monitor.available = _Data(True)
        monitor.totalForceNorm.value = 0.0
        monitor.contactCount.value = 0
        monitor.segmentForceVectors.value = []
        intervention = _Intervention(monitor)

        collector = EvalV2ForceTelemetryCollector(
            spec=ForceTelemetrySpec(mode="passive", required=False),
            action_dt_s=0.1,
        )
        collector.ensure_runtime(intervention=intervention)
        collector.capture_step(intervention=intervention, step_index=1)

        summary = collector.build_summary()
        self.assertFalse(summary.contact_detected_any)
        self.assertEqual(summary.contact_count_max, 0)


# ---------------------------------------------------------------------------
# required + no data
# ---------------------------------------------------------------------------

class RequiredMissingTests(unittest.TestCase):
    def test_required_without_runtime_data_marks_missing(self) -> None:
        class _NoRootIntervention:
            simulation = object()

        spec = replace(ForceTelemetrySpec(mode="passive", required=False), required=True)
        collector = EvalV2ForceTelemetryCollector(spec=spec, action_dt_s=0.1)
        status = collector.ensure_runtime(intervention=_NoRootIntervention())
        self.assertFalse(status.configured)

        summary = collector.build_summary()
        self.assertFalse(summary.available_for_score)
        self.assertEqual(summary.validation_status, "required_missing")


# ---------------------------------------------------------------------------
# Tip force
# ---------------------------------------------------------------------------

class TipForceTests(unittest.TestCase):
    def test_tip_force_always_unavailable_until_spatial_mapping_implemented(self) -> None:
        # Tip force requires mapping contact/force data to the distal wire region.
        # Until that is implemented, it must be explicitly unavailable — never silently
        # fabricated from total LCP max.
        monitor = _Monitor()
        intervention = _Intervention(monitor)
        collector = EvalV2ForceTelemetryCollector(
            spec=ForceTelemetrySpec(mode="passive", required=False),
            action_dt_s=0.1,
        )
        collector.ensure_runtime(intervention=intervention)
        collector.capture_step(intervention=intervention, step_index=1)
        summary = collector.build_summary()

        self.assertFalse(summary.tip_force_available)
        self.assertEqual(summary.tip_force_validation_status, "unmapped")


# ---------------------------------------------------------------------------
# Constraint projection unit tests
# ---------------------------------------------------------------------------


class ConstraintProjectionTests(unittest.TestCase):
    def test_project_constraint_forces_basic(self) -> None:
        # Single row mapping to dof 0 with unit x-direction
        constraint_raw = "0 1 0 1.0 0.0 0.0"
        lcp = [2.0]
        proj, row_contribs = _project_constraint_forces(
            lcp_forces=np.asarray(lcp, dtype=np.float32),
            constraint_raw=constraint_raw,
            n_points=1,
            dt_s=0.5,
        )
        # contribution: lambda=2.0, dt=0.5 -> 4.0 in scene units on dof 0 x
        self.assertEqual(proj.shape, (1, 3))
        self.assertAlmostEqual(float(proj[0, 0]), 4.0, places=6)
        self.assertAlmostEqual(float(proj[0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(proj[0, 2]), 0.0, places=6)
        self.assertEqual(len(row_contribs), 1)
        self.assertEqual(row_contribs[0]["row_idx"], 0)
        self.assertEqual(row_contribs[0]["dof_idx"], 0)

    def test_projection_and_mapping_records(self) -> None:
        # Create fake runtime objects: root with LCP, collision state and export
        class _RootObj:
            pass

        class _LCP:
            def __init__(self):
                self.constraintForces = [2.0]

        class _Collision:
            def __init__(self):
                # constraint: row 0 -> dof 0, x-direction
                self.constraint = _Data("0 1 0 1.0 0.0 0.0")
                self.position = _Data([[0.0, 0.0, 0.0]])

        class _ExportObj:
            def __init__(self):
                self.available = _Data(True)
                self.constraintRowIndices = _Data([0])
                self.constraintRowValidFlags = _Data([1])
                self.wallTriangleIds = _Data([5])
                self.collisionDofIndices = _Data([0])
                self.contactCount = _Data(1)
                self.explicitCoverage = _Data(1.0)

        root = _RootObj()
        root.LCP = _LCP()
        root.dt = _Data(0.5)
        root.InstrumentCombined = _Data(None)
        # monkeypatch attribute path used in collector
        # place CollisionDOFs at root.InstrumentCombined.CollisionModel.CollisionDOFs
        class _CM:
            pass

        cm = _CM()
        cm.CollisionDOFs = _Collision()
        instr = type("I", (), {"CollisionModel": cm})
        root.InstrumentCombined = instr
        root.wire_wall_contact_export = _ExportObj()

        class _Sim:
            def __init__(self, root):
                self.root = root

        class _Interv:
            def __init__(self, sim):
                self.simulation = sim

        intervention = _Interv(_Sim(root))

        collector = EvalV2ForceTelemetryCollector(
            spec=ForceTelemetrySpec(mode="constraint_projected_si_validated", required=False, contact_epsilon=1e-7, units=ForceUnits(length_unit="mm", mass_unit="kg", time_unit="s")),
            action_dt_s=0.1,
        )
        collector.ensure_runtime(intervention=intervention)
        collector.capture_step(intervention=intervention, step_index=1)

        # triangle records should contain an entry for triangle 5
        tri_recs = [r for r in collector._triangle_force_records if r.get("triangle_id") == 5]
        self.assertTrue(len(tri_recs) > 0)
        rec = tri_recs[0]
        # scene force = 4.0, unit scale mm/kg/s -> *1e-3 -> 0.004 N
        self.assertAlmostEqual(rec.get("norm_N", 0.0), 4.0 * 1e-3, places=9)

    def test_tip_force_unavailable_on_lcp_fallback(self) -> None:
        monitor = _Monitor()
        monitor.totalForceNorm.value = 0.0
        monitor.contactCount.value = 0
        monitor.segmentForceVectors.value = []
        intervention = _Intervention(monitor)
        collector = EvalV2ForceTelemetryCollector(
            spec=ForceTelemetrySpec(mode="passive", required=False),
            action_dt_s=0.1,
        )
        collector.ensure_runtime(intervention=intervention)
        collector.capture_step(intervention=intervention, step_index=1)
        summary = collector.build_summary()

        self.assertFalse(summary.tip_force_available)
        self.assertEqual(summary.tip_force_validation_status, "unmapped")


if __name__ == "__main__":
    unittest.main()
