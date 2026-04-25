from __future__ import annotations

import unittest
from dataclasses import replace

from steve_recommender.eval_v2.force_telemetry import EvalV2ForceTelemetryCollector
from steve_recommender.eval_v2.models import ForceTelemetrySpec, ForceUnits


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


class _Root:
    def __init__(self, monitor: _Monitor | None) -> None:
        self.wire_wall_force_monitor = monitor
        self.LCP = _LCP()
        self.dt = _Data(0.1)


class _Simulation:
    def __init__(self, monitor: _Monitor | None) -> None:
        self.root = _Root(monitor)


class _Intervention:
    def __init__(self, monitor: _Monitor | None) -> None:
        self.simulation = _Simulation(monitor)


class ForceTelemetryCollectorTests(unittest.TestCase):
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
        self.assertTrue(summary.available_for_score)
        self.assertEqual(summary.validation_status, "ok")
        self.assertEqual(summary.channel, "wire_wall_force_monitor")
        self.assertEqual(summary.association_method, "force_points_nearest_triangle")
        self.assertAlmostEqual(float(summary.total_force_norm_max or 0.0), 0.6, places=6)
        self.assertAlmostEqual(float(summary.total_force_norm_mean or 0.0), 0.4, places=6)
        self.assertAlmostEqual(float(summary.lcp_max_abs_max or 0.0), 2.0, places=6)
        self.assertAlmostEqual(float(summary.lcp_sum_abs_mean or 0.0), 3.5, places=6)
        self.assertAlmostEqual(float(summary.peak_segment_force_norm or 0.0), 0.9, places=6)
        self.assertEqual(summary.peak_segment_force_step, 2)
        self.assertAlmostEqual(float(summary.peak_segment_force_time_s or 0.0), 0.2, places=6)

    def test_uses_lcp_fallback_when_monitor_reports_zero_norms(self) -> None:
        monitor = _Monitor()
        monitor.totalForceNorm.value = 0.0
        monitor.contactCount.value = 0
        monitor.segmentForceVectors.value = [[0.0, 0.0, 0.0]]
        intervention = _Intervention(monitor)
        collector = EvalV2ForceTelemetryCollector(
            spec=ForceTelemetrySpec(mode="passive", required=False, contact_epsilon=1e-7),
            action_dt_s=0.1,
        )

        status = collector.ensure_runtime(intervention=intervention)
        self.assertTrue(status.configured)
        collector.capture_step(intervention=intervention, step_index=1)

        summary = collector.build_summary()
        self.assertTrue(summary.available_for_score)
        self.assertEqual(summary.channel, "lcp.constraintForces/dt")
        self.assertIn("fallback_lcp_dt", summary.source)
        self.assertAlmostEqual(float(summary.total_force_norm_max or 0.0), 20.0, places=6)

    def test_validated_mode_applies_si_scaling_for_fallback(self) -> None:
        monitor = _Monitor()
        monitor.totalForceNorm.value = 0.0
        monitor.contactCount.value = 0
        monitor.segmentForceVectors.value = [[0.0, 0.0, 0.0]]
        intervention = _Intervention(monitor)
        collector = EvalV2ForceTelemetryCollector(
            spec=ForceTelemetrySpec(
                mode="constraint_projected_si_validated",
                required=False,
                contact_epsilon=1e-7,
                units=ForceUnits(length_unit="mm", mass_unit="kg", time_unit="s"),
            ),
            action_dt_s=0.1,
        )

        status = collector.ensure_runtime(intervention=intervention)
        self.assertTrue(status.configured)
        collector.capture_step(intervention=intervention, step_index=1)

        summary = collector.build_summary()
        self.assertTrue(summary.available_for_score)
        self.assertEqual(summary.channel, "lcp.constraintForces/dt")
        # max(|lambda|)=2.0, dt=0.1 -> 20 in scene force units; mm,kg,s to N => *1e-3.
        self.assertAlmostEqual(float(summary.total_force_norm_max or 0.0), 0.02, places=9)

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


if __name__ == "__main__":
    unittest.main()
