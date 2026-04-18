from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from debug.force_playground.config import ForcePlaygroundConfig, parse_units
from debug.force_playground.controllers import build_controller
from debug.force_playground.io import PlaygroundRunIO
from debug.force_playground.oracle import NormalForceBalanceOracle
from debug.force_playground.plotting import LivePlotter
from debug.force_playground.scene_factory import select_camera_pose
from debug.force_playground.telemetry import TelemetryCollector


class ForcePlaygroundCoreTests(unittest.TestCase):
    def test_config_rejects_open_loop_force_for_guidewire(self) -> None:
        with self.assertRaises(ValueError):
            ForcePlaygroundConfig(
                probe="guidewire",
                mode="open_loop_force",
            )

    def test_parse_units(self) -> None:
        units = parse_units("mm,kg,s")
        self.assertEqual(units.length_unit, "mm")
        self.assertEqual(units.mass_unit, "kg")
        self.assertEqual(units.time_unit, "s")

    def test_camera_preset_plane_is_not_edge_on(self) -> None:
        cfg = ForcePlaygroundConfig(scene="plane_wall", camera_preset="plane_oblique")
        vertices = np.asarray(
            [
                [-110.0, -60.0, 0.0],
                [110.0, -60.0, 0.0],
                [110.0, 60.0, 0.0],
                [-110.0, 60.0, 0.0],
            ],
            dtype=np.float32,
        )
        preset, position, look_at = select_camera_pose(
            cfg,
            vertices,
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        )
        self.assertEqual(preset, "plane_oblique")
        view_dir = look_at - position
        normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        cos_to_normal = abs(float(np.dot(view_dir, normal))) / float(np.linalg.norm(view_dir))
        # non-edge-on: camera view direction must have strong normal component.
        self.assertGreater(cos_to_normal, 0.4)

    def test_normal_tangent_decomposition(self) -> None:
        fn_scalar, fn_vec, ft_vec, fn_abs, ft_abs = TelemetryCollector._normal_tangent(  # type: ignore[attr-defined]
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        )
        self.assertAlmostEqual(fn_scalar, 3.0, places=6)
        self.assertTrue(np.allclose(fn_vec, np.asarray([0.0, 0.0, 3.0], dtype=np.float32)))
        self.assertTrue(np.allclose(ft_vec, np.asarray([1.0, 2.0, 0.0], dtype=np.float32)))
        self.assertAlmostEqual(fn_abs, 3.0, places=6)
        self.assertAlmostEqual(ft_abs, np.sqrt(5.0), places=6)

    def test_oracle_pass_and_fail(self) -> None:
        cfg = ForcePlaygroundConfig(
            scene="plane_wall",
            probe="rigid_probe",
            mode="open_loop_force",
            steps=5,
        )
        oracle = NormalForceBalanceOracle(cfg, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))

        base = {
            "wall_contact_detected": True,
            "lambda_active_rows_count": 1,
            "total_force_norm": 1.0,
            "commanded_force_vector_n": [0.0, 0.0, -1.0],
            "total_force_vector": [0.0, 0.0, -1.0],
        }
        # warmup record
        rec1 = dict(base)
        rec1["step"] = 1
        out1 = oracle.evaluate_step(rec1)
        self.assertIsNone(out1["oracle_physical_pass"])
        self.assertEqual(out1["oracle_reason"], "warmup")

        # force a post-warmup step that passes
        cfg2 = ForcePlaygroundConfig(
            scene="plane_wall",
            probe="rigid_probe",
            mode="open_loop_force",
            steps=5,
            oracle=cfg.oracle.__class__(
                oracle_type="normal_force_balance",
                enabled=True,
                rel_tol=0.10,
                abs_tol_n=0.01,
                near_zero_ref_n=0.02,
                warmup_steps=0,
                window_steps=5,
            ),
        )
        oracle2 = NormalForceBalanceOracle(cfg2, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        ok = oracle2.evaluate_step({**base, "step": 1})
        self.assertTrue(ok["oracle_physical_pass"])

        bad = oracle2.evaluate_step(
            {
                "step": 2,
                "wall_contact_detected": True,
                "lambda_active_rows_count": 1,
                "total_force_norm": 0.5,
                "commanded_force_vector_n": [0.0, 0.0, -1.0],
                "total_force_vector": [0.0, 0.0, -0.5],
            }
        )
        self.assertFalse(bad["oracle_physical_pass"])

        rep = oracle2.finalize()
        self.assertFalse(rep["passed"])
        self.assertEqual(rep["window"]["fails"], 1)

    def test_oracle_applicability_flags(self) -> None:
        cfg_displacement = ForcePlaygroundConfig(
            scene="plane_wall",
            probe="rigid_probe",
            mode="displacement",
            require_oracle_applicable=True,
        )
        oracle_displacement = NormalForceBalanceOracle(
            cfg_displacement,
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        )
        self.assertFalse(oracle_displacement.applicable)

        cfg_force = ForcePlaygroundConfig(
            scene="plane_wall",
            probe="rigid_probe",
            mode="open_loop_force",
            require_oracle_applicable=True,
        )
        oracle_force = NormalForceBalanceOracle(
            cfg_force,
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        )
        self.assertTrue(oracle_force.applicable)

    def test_wait_loop_calls_idle_callback_when_paused(self) -> None:
        class _DummyPlt:
            def pause(self, _: float) -> None:
                return

        plotter = LivePlotter.__new__(LivePlotter)
        plotter.enabled = True
        plotter.interactive = True
        plotter.should_quit = False
        plotter.run_continuous = False
        plotter._pending_steps = 0
        plotter._status_note = ""
        plotter._plt = _DummyPlt()
        plotter._set_title = lambda: None  # type: ignore[method-assign]

        calls = {"n": 0}

        def _idle() -> None:
            calls["n"] += 1
            if calls["n"] >= 3:
                plotter._pending_steps = 1

        allowed = LivePlotter.wait_until_step_allowed(plotter, idle_callback=_idle)
        self.assertTrue(allowed)
        self.assertGreaterEqual(calls["n"], 3)

    def test_run_io_writes_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            io = PlaygroundRunIO(run_dir)
            io.write_config({"k": 1})
            io.append_step(
                {
                    "step": 1,
                    "scene": "plane_wall",
                    "probe": "rigid_probe",
                    "mode": "displacement",
                    "norm_sum_vector": 0.0,
                    "sum_norm": 0.0,
                    "peak_triangle_force": 0.0,
                    "total_force_vector": [0.0, 0.0, 0.0],
                    "sum_abs_fn": 0.0,
                    "sum_abs_ft": 0.0,
                    "lambda_abs_sum": 0.0,
                    "lambda_dt_abs_sum": 0.0,
                    "lambda_active_rows_count": 0,
                }
            )
            io.append_triangles(
                [
                    {
                        "step": 1,
                        "triangle_id": 0,
                        "active": 0,
                        "force_x": 0.0,
                        "force_y": 0.0,
                        "force_z": 0.0,
                        "force_norm": 0.0,
                        "normal_x": 0.0,
                        "normal_y": 0.0,
                        "normal_z": 1.0,
                        "fn_scalar": 0.0,
                        "fn_x": 0.0,
                        "fn_y": 0.0,
                        "fn_z": 0.0,
                        "fn_abs": 0.0,
                        "ft_x": 0.0,
                        "ft_y": 0.0,
                        "ft_z": 0.0,
                        "ft_abs": 0.0,
                        "ft_over_fn": 0.0,
                    }
                ]
            )
            io.write_oracle_report({"passed": True})
            io.write_summary("ok")
            io.close()

            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "steps.csv").exists())
            self.assertTrue((run_dir / "steps.jsonl").exists())
            self.assertTrue((run_dir / "triangle_forces.csv").exists())
            self.assertTrue((run_dir / "oracle_report.json").exists())
            self.assertTrue((run_dir / "summary.md").exists())

    def test_open_loop_controller_uses_runtime_insert_action(self) -> None:
        cfg = ForcePlaygroundConfig(
            scene="plane_wall",
            probe="rigid_probe",
            mode="open_loop_force",
        )
        ctl = build_controller(cfg, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        # Default comes from open_loop_insert_action, not generic insert_action.
        cmd0 = ctl.command(1)
        self.assertAlmostEqual(float(cmd0.action[0, 0]), float(cfg.control.open_loop_insert_action))
        # Runtime updates (used by constraint_audit preload/hold/ramp) must apply.
        ctl.insert_action = 0.75
        cmd1 = ctl.command(2)
        self.assertAlmostEqual(float(cmd1.action[0, 0]), 0.75, places=6)


if __name__ == "__main__":
    unittest.main()
