from __future__ import annotations

import json
import unittest
from pathlib import Path


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "steve_recommender"
    / "eval_v2"
    / "tests"
    / "fixtures"
    / "trace_smoke_seed123.json"
)


class ForceTraceFixtureTests(unittest.TestCase):
    def test_smoke_fixture_has_expected_shape(self) -> None:
        payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["trial_seed"], 123)
        self.assertEqual(payload["metadata"]["source"], "live_eval_v2_runner_force_trace")
        self.assertGreaterEqual(len(payload["triangle_records"]), 1)
        self.assertGreaterEqual(len(payload["wire_records"]), 1)
        self.assertIn("summary", payload)
        self.assertIn("links", payload)
        self.assertIn("record_counts", payload)

    def test_force_trace_smoke_matches_golden(self) -> None:
        payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["metadata"]["force_channel"], "lcp.constraintForces/dt")
        self.assertEqual(payload["metadata"]["force_validation_status"], "ok")
        self.assertEqual(payload["metadata"]["force_quality_tier"], "validated")
        self.assertTrue(payload["links"]["passive_monitor_attached"])
        self.assertTrue(payload["links"]["contact_export_attached"])
        self.assertTrue(payload["links"]["lcp_computeConstraintForces"])
        self.assertEqual(payload["links"]["vessel_triangle_count"], 1890)

        summary = payload["summary"]
        self.assertEqual(summary["steps_total"], 20)
        self.assertEqual(summary["lcp_nonzero_steps"], 9)
        self.assertEqual(summary["mapping"]["total_rows"], 9)
        self.assertEqual(summary["mapping"]["mapped_rows"], 9)
        self.assertEqual(summary["mapping"]["unmapped_rows"], 0)
        self.assertAlmostEqual(summary["mapping"]["mapping_coverage"], 1.0)
        self.assertAlmostEqual(summary["total_force_norm_max_n"], 0.7865626057786536)
        self.assertAlmostEqual(summary["lcp_max_abs_max_raw"], 0.004719375634671921)

        self.assertEqual(payload["record_counts"]["triangle_records_total"], 10)
        self.assertEqual(payload["record_counts"]["wire_records_total"], 36)

        first_triangle = payload["triangle_records"][0]
        self.assertEqual(first_triangle["timestep"], 12)
        self.assertEqual(first_triangle["triangle_id"], 870)
        self.assertEqual(first_triangle["contributing_rows"], 3)
        self.assertTrue(first_triangle["mapped"])
        self.assertAlmostEqual(first_triangle["norm_N"], 0.5953568816184998)

        first_wire = payload["wire_records"][0]
        self.assertEqual(first_wire["timestep"], 12)
        self.assertEqual(first_wire["wire_collision_dof"], 75)
        self.assertEqual(first_wire["row_idx"], 0)
        self.assertTrue(first_wire["mapped"])
        self.assertAlmostEqual(first_wire["norm_N"], 0.5953565835952759)

    def test_real_scene_has_nonzero_scoreable_coverage(self) -> None:
        payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

        summary = payload["summary"]
        mapping = summary["mapping"]
        self.assertEqual(payload["metadata"]["force_validation_status"], "ok")
        self.assertEqual(payload["metadata"]["force_quality_tier"], "validated")
        self.assertGreater(mapping["total_rows"], 0)
        self.assertGreater(mapping["mapped_rows"], 0)
        self.assertEqual(mapping["unmapped_rows"], 0)
        self.assertGreater(mapping["mapping_coverage"], 0.0)
        self.assertGreater(summary["total_force_norm_max_n"], 0.0)


if __name__ == "__main__":
    unittest.main()
