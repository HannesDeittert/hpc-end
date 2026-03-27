from __future__ import annotations

import unittest

import numpy as np

from steve_recommender.evaluation.config import config_from_dict
from steve_recommender.evaluation.pipeline import (
    _reconstruct_dense_wall_segment_forces,
    _validate_force_signal,
    _validate_force_vector_consistency,
)
from steve_recommender.evaluation.scoring import score_trial
from steve_recommender.evaluation.force_calibration import (
    _build_calibration_probe_config,
    _series_reproducible,
    _tolerance_profile,
    calibration_fingerprint,
    load_force_calibration,
)


class ForceExtractionConfigTests(unittest.TestCase):
    def test_config_parses_force_extraction_block(self) -> None:
        cfg = config_from_dict(
            {
                "name": "x",
                "agents": [
                    {
                        "name": "a",
                        "tool": "M/W",
                        "checkpoint": "/tmp/a.everl",
                    }
                ],
                "force_extraction": {
                    "mode": "intrusive_lcp",
                    "required": True,
                    "contact_epsilon": 1e-5,
                    "plugin_path": "/tmp/libSofaWireForceMonitor.so",
                    "calibration": {
                        "required": True,
                        "cache_path": "/tmp/force_cache.json",
                        "tolerance_profile": "default_v1",
                    },
                },
            }
        )
        self.assertEqual(cfg.force_extraction.mode, "intrusive_lcp")
        self.assertTrue(cfg.force_extraction.required)
        self.assertAlmostEqual(cfg.force_extraction.contact_epsilon, 1e-5)
        self.assertEqual(
            cfg.force_extraction.plugin_path,
            "/tmp/libSofaWireForceMonitor.so",
        )
        self.assertEqual(cfg.force_extraction.calibration.cache_path, "/tmp/force_cache.json")

    def test_config_rejects_invalid_force_mode(self) -> None:
        with self.assertRaises(ValueError):
            config_from_dict(
                {
                    "name": "x",
                    "agents": [
                        {
                            "name": "a",
                            "tool": "M/W",
                            "checkpoint": "/tmp/a.everl",
                        }
                    ],
                    "force_extraction": {"mode": "broken_mode"},
                }
            )

    def test_validated_mode_requires_units(self) -> None:
        with self.assertRaises(ValueError):
            config_from_dict(
                {
                    "name": "x",
                    "agents": [
                        {
                            "name": "a",
                            "tool": "M/W",
                            "checkpoint": "/tmp/a.everl",
                        }
                    ],
                    "force_extraction": {"mode": "constraint_projected_si_validated"},
                }
            )

    def test_validated_mode_parses_units(self) -> None:
        cfg = config_from_dict(
            {
                "name": "x",
                "agents": [
                    {
                        "name": "a",
                        "tool": "M/W",
                        "checkpoint": "/tmp/a.everl",
                    }
                ],
                "force_extraction": {
                    "mode": "constraint_projected_si_validated",
                    "units": {
                        "length_unit": "mm",
                        "mass_unit": "kg",
                        "time_unit": "s",
                    },
                },
            }
        )
        self.assertEqual(cfg.force_extraction.mode, "constraint_projected_si_validated")
        self.assertIsNotNone(cfg.force_extraction.units)
        self.assertEqual(cfg.force_extraction.units.length_unit, "mm")

    def test_validated_mode_rejects_partial_units_block(self) -> None:
        with self.assertRaises(ValueError):
            config_from_dict(
                {
                    "name": "x",
                    "agents": [
                        {
                            "name": "a",
                            "tool": "M/W",
                            "checkpoint": "/tmp/a.everl",
                        }
                    ],
                    "force_extraction": {
                        "mode": "constraint_projected_si_validated",
                        "units": {
                            "length_unit": "mm",
                            "mass_unit": "kg",
                        },
                    },
                }
            )

    def test_config_parses_visual_force_debug_settings(self) -> None:
        cfg = config_from_dict(
            {
                "name": "x",
                "agents": [
                    {
                        "name": "a",
                        "tool": "M/W",
                        "checkpoint": "/tmp/a.everl",
                    }
                ],
                "visualize_force_debug": True,
                "visualize_force_debug_top_k": 7,
            }
        )
        self.assertTrue(cfg.visualize_force_debug)
        self.assertEqual(cfg.visualize_force_debug_top_k, 7)


class ForceSignalValidationTests(unittest.TestCase):
    def test_validate_force_signal_detects_contact_without_force(self) -> None:
        ok, reason = _validate_force_signal(
            force_available_series=np.asarray([True, True, True], dtype=np.bool_),
            contact_count_series=np.asarray([0, 2, 1], dtype=np.int32),
            contact_detected_series=np.asarray([False, True, True], dtype=np.bool_),
            lcp_active_count_series=np.asarray([0, 1, 1], dtype=np.int32),
            total_force_norm_series=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            contact_epsilon=1e-7,
        )
        self.assertFalse(ok)
        self.assertIn("active contact constraints", reason)

    def test_validate_force_signal_accepts_no_contact_episode(self) -> None:
        ok, reason = _validate_force_signal(
            force_available_series=np.asarray([True, True], dtype=np.bool_),
            contact_count_series=np.asarray([0, 0], dtype=np.int32),
            contact_detected_series=np.asarray([False, False], dtype=np.bool_),
            lcp_active_count_series=np.asarray([0, 0], dtype=np.int32),
            total_force_norm_series=np.asarray([0.0, 0.0], dtype=np.float32),
            contact_epsilon=1e-7,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_validate_force_signal_ignores_geometry_only_contact_flag(self) -> None:
        ok, reason = _validate_force_signal(
            force_available_series=np.asarray([True, True], dtype=np.bool_),
            contact_count_series=np.asarray([0, 0], dtype=np.int32),
            contact_detected_series=np.asarray([False, True], dtype=np.bool_),
            lcp_active_count_series=np.asarray([0, 0], dtype=np.int32),
            total_force_norm_series=np.asarray([0.0, 0.0], dtype=np.float32),
            contact_epsilon=1e-7,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_validate_force_signal_uses_lcp_active_rows(self) -> None:
        ok, reason = _validate_force_signal(
            force_available_series=np.asarray([True, True], dtype=np.bool_),
            contact_count_series=np.asarray([0, 0], dtype=np.int32),
            contact_detected_series=np.asarray([False, False], dtype=np.bool_),
            lcp_active_count_series=np.asarray([0, 3], dtype=np.int32),
            total_force_norm_series=np.asarray([0.0, 0.0], dtype=np.float32),
            contact_epsilon=1e-7,
        )
        self.assertFalse(ok)
        self.assertIn("active contact constraints", reason)

    def test_validate_force_signal_prefers_active_constraint_step_series(self) -> None:
        ok, reason = _validate_force_signal(
            force_available_series=np.asarray([True, True], dtype=np.bool_),
            contact_count_series=np.asarray([0, 0], dtype=np.int32),
            contact_detected_series=np.asarray([False, False], dtype=np.bool_),
            lcp_active_count_series=np.asarray([0, 0], dtype=np.int32),
            total_force_norm_series=np.asarray([0.0, 0.0], dtype=np.float32),
            contact_epsilon=1e-7,
            active_constraint_step_series=np.asarray([False, True], dtype=np.bool_),
        )
        self.assertFalse(ok)
        self.assertIn("active contact constraints", reason)

    def test_validate_force_signal_active_constraint_step_with_force_passes(self) -> None:
        ok, reason = _validate_force_signal(
            force_available_series=np.asarray([True, True], dtype=np.bool_),
            contact_count_series=np.asarray([0, 0], dtype=np.int32),
            contact_detected_series=np.asarray([False, False], dtype=np.bool_),
            lcp_active_count_series=np.asarray([0, 0], dtype=np.int32),
            total_force_norm_series=np.asarray([0.0, 0.5], dtype=np.float32),
            contact_epsilon=1e-7,
            active_constraint_step_series=np.asarray([False, True], dtype=np.bool_),
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_validate_force_vector_consistency_detects_mismatch(self) -> None:
        ok, reason = _validate_force_vector_consistency(
            segment_force_series=np.asarray(
                [
                    np.asarray([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32),
                ],
                dtype=object,
            ),
            total_force_vector_series=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            atol=1e-7,
        )
        self.assertFalse(ok)
        self.assertIn("segment-force sum mismatch", reason)

    def test_reconstruct_dense_wall_segment_forces(self) -> None:
        dense = _reconstruct_dense_wall_segment_forces(
            wall_segment_count_series=np.asarray([4, 4], dtype=np.int32),
            wall_active_segment_ids_series=np.asarray(
                [
                    np.asarray([1, 3], dtype=np.int32),
                    np.asarray([0], dtype=np.int32),
                ],
                dtype=object,
            ),
            wall_active_segment_force_vectors_series=np.asarray(
                [
                    np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
                    np.asarray([[0.0, 0.0, 3.0]], dtype=np.float32),
                ],
                dtype=object,
            ),
        )
        self.assertEqual(dense.shape, (2, 4, 3))
        self.assertTrue(np.allclose(dense[0, 1], np.asarray([1.0, 0.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(dense[0, 3], np.asarray([0.0, 2.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(dense[1, 0], np.asarray([0.0, 0.0, 3.0], dtype=np.float32)))


class ForceScoringTests(unittest.TestCase):
    def test_force_unavailable_excludes_safety_weight(self) -> None:
        from steve_recommender.evaluation.config import ScoringConfig

        scoring = ScoringConfig(
            w_success=2.0,
            w_efficiency=1.0,
            w_safety=1.0,
            w_smoothness=0.25,
            normalize_weights=True,
        )

        ts = score_trial(
            scoring=scoring,
            success=True,
            steps_to_success=None,
            max_episode_steps=1000,
            tip_speed_max_mm_s=0.0,
            wall_wire_force_norm_max=np.nan,
            wall_collision_force_norm_max=np.nan,
            wall_total_force_norm_max=np.nan,
            wall_lcp_max_abs_max=np.nan,
            force_available=False,
        )

        expected = (2.0 * 1.0 + 1.0 * 0.0 + 0.25 * 1.0) / (2.0 + 1.0 + 0.25)
        self.assertAlmostEqual(ts.score, expected)
        self.assertTrue(np.isnan(ts.safety))

    def test_scoring_prefers_wall_total_force_metric(self) -> None:
        from steve_recommender.evaluation.config import ScoringConfig

        scoring = ScoringConfig(force_scale=1.0, lcp_scale=1.0)
        ts = score_trial(
            scoring=scoring,
            success=True,
            steps_to_success=10,
            max_episode_steps=1000,
            tip_speed_max_mm_s=0.0,
            wall_wire_force_norm_max=0.0,
            wall_collision_force_norm_max=0.0,
            wall_total_force_norm_max=5.0,
            wall_lcp_max_abs_max=0.0,
            force_available=True,
        )
        # safety should be clearly penalized by exp(-5), not remain near 1.
        self.assertLess(ts.safety, 0.1)


class ForceCalibrationTests(unittest.TestCase):
    def test_load_force_calibration_not_found(self) -> None:
        state = load_force_calibration(
            cache_path="/tmp/non_existing_force_cache.json",
            cache_key="abc",
        )
        self.assertFalse(state["found"])
        self.assertFalse(state["passed"])

    def test_calibration_fingerprint_contains_units(self) -> None:
        cfg = config_from_dict(
            {
                "name": "x",
                "agents": [
                    {
                        "name": "a",
                        "tool": "M/W",
                        "checkpoint": "/tmp/a.everl",
                    }
                ],
                "force_extraction": {
                    "mode": "constraint_projected_si_validated",
                    "units": {
                        "length_unit": "mm",
                        "mass_unit": "kg",
                        "time_unit": "s",
                    },
                },
            }
        )
        fp = calibration_fingerprint(cfg, cfg.agents[0])
        self.assertEqual(fp["units"]["length_unit"], "mm")

    def test_build_probe_config_disables_calibration_gate(self) -> None:
        cfg = config_from_dict(
            {
                "name": "x",
                "agents": [
                    {
                        "name": "a",
                        "tool": "M/W",
                        "checkpoint": "/tmp/a.everl",
                    }
                ],
                "force_extraction": {
                    "mode": "constraint_projected_si_validated",
                    "required": True,
                    "units": {
                        "length_unit": "mm",
                        "mass_unit": "kg",
                        "time_unit": "s",
                    },
                    "calibration": {
                        "required": True,
                        "cache_path": "/tmp/cache.json",
                        "tolerance_profile": "default_v1",
                    },
                },
            }
        )
        probe_cfg = _build_calibration_probe_config(cfg, cfg.agents[0])
        self.assertFalse(probe_cfg.force_extraction.required)
        self.assertFalse(probe_cfg.force_extraction.calibration.required)
        self.assertTrue(probe_cfg.name.endswith("_probe"))

    def test_series_reproducibility_checker(self) -> None:
        ok, err = _series_reproducible(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([1.0, 2.0005, 3.0]),
            atol=1e-3,
            rtol=1e-3,
        )
        self.assertTrue(ok)
        self.assertEqual(err, "")

        ok2, err2 = _series_reproducible(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([1.0, 2.2, 3.0]),
            atol=1e-3,
            rtol=1e-3,
        )
        self.assertFalse(ok2)
        self.assertIn("trace_mismatch", err2)

    def test_tolerance_profile_defaults(self) -> None:
        self.assertEqual(_tolerance_profile("default_v1")["trace_atol_N"], 1e-3)
        self.assertEqual(_tolerance_profile("strict_v1")["trace_atol_N"], 5e-4)


if __name__ == "__main__":
    unittest.main()
