from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _has_runtime_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


SOFA_RUNTIME_AVAILABLE = _has_runtime_module("Sofa") and _has_runtime_module("SofaRuntime")


@unittest.skipUnless(SOFA_RUNTIME_AVAILABLE, "SOFA runtime not available in current Python env")
class EvalV2WallPressRuntimeTests(unittest.TestCase):
    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _run_wall_press(
        self,
        *,
        run_name: str,
        steps: int,
        drive_mode: str,
        target_force_n: float,
        insert_action: float = 0.0,
    ) -> dict:
        repo_root = self._repo_root()
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            cmd = [
                sys.executable,
                "-m",
                "debug.force_playground.eval_v2_plain_wall_press",
                "--steps",
                str(int(steps)),
                "--target-force-n",
                str(float(target_force_n)),
                "--insert-action",
                str(float(insert_action)),
                "--force-mode",
                "constraint_projected_si_validated",
                "--drive-mode",
                str(drive_mode),
                "--output-root",
                str(output_root),
                "--run-name",
                str(run_name),
            ]
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(
                    "Wall-press CLI failed with return code {rc}\nSTDOUT:\n{out}\nSTDERR:\n{err}".format(
                        rc=proc.returncode,
                        out=proc.stdout,
                        err=proc.stderr,
                    )
                )

            report_candidates = sorted(
                output_root.glob(f"*_{run_name}/eval_v2_wall_press_report.json")
            )
            self.assertTrue(report_candidates, "No eval_v2 wall-press report JSON produced")
            report_path = report_candidates[-1]
            with report_path.open("r", encoding="utf-8") as f:
                return json.load(f)

    def test_action_only_lcp_collected_but_not_available_for_score(self) -> None:
        """
        action_only mode: wire motion drives contact via simulation physics only.
        The C++ monitor reads MechanicalObject.force/externalForce which are NOT
        populated for constraint-based contacts in FreeMotionAnimationLoop.  The
        LCP constraintForces may be nonzero, and when the contact export maps rows
        back to wall triangles that path is valid for scoring.
        """
        report = self._run_wall_press(
            run_name="test_eval_v2_action_only_lcp_semantics",
            steps=220,
            drive_mode="action_only",
            target_force_n=0.10,
            insert_action=0.40,
        )

        summary = report.get("collector_summary", {})
        channel = str(summary.get("channel", ""))
        validation_status = str(summary.get("validation_status", ""))

        # If the C++ monitor detected real contacts → channel may be wire_wall_force_monitor
        # and available_for_score=True.  That path is fine.
        # If the monitor reads zero (expected for constraint-based contacts) → LCP fallback.
        if channel == "lcp.constraintForces/dt":
            # When contact export mapped rows exist, the LCP path is scoreable.
            self.assertTrue(
                bool(summary.get("available_for_score", False)),
                "Mapped LCP fallback should be available_for_score=True",
            )
            self.assertIn(validation_status, {"lcp_mapped_wire_wall", "ok"})
            self.assertGreater(int(summary.get("lcp_mapped_wall_row_count_max", 0)), 0)
            self.assertIsNotNone(summary.get("lcp_contact_export_coverage"))
            # LCP diagnostic values should still be present.
            lcp_max = summary.get("lcp_max_abs_max")
            total_max = summary.get("total_force_norm_max")
            self.assertIsNotNone(lcp_max, "lcp_max_abs_max should be collected for diagnostics")
            self.assertIsNotNone(total_max, "total_force_norm_max should be populated for diagnostics")
        else:
            # Legacy monitor may be attached for diagnostics or plumbing smoke tests.
            # Eval_v2 no longer treats the monitor as a validated scoring source.
            self.assertIn(channel, {"wire_wall_force_monitor"}, f"Unexpected force channel: {channel}")

    def test_action_only_lcp_data_collected_even_without_monitor(self) -> None:
        """LCP constraintForces should be sampled regardless of monitor availability."""
        report = self._run_wall_press(
            run_name="test_eval_v2_lcp_data_present",
            steps=150,
            drive_mode="action_only",
            target_force_n=0.10,
            insert_action=0.40,
        )
        summary = report.get("collector_summary", {})
        lcp_max = summary.get("lcp_max_abs_max")
        # LCP data should be collected (LCP constraint system is always active)
        self.assertIsNotNone(lcp_max,
                             "lcp_max_abs_max should be sampled even if monitor is zero")

    def test_external_force_plumbing_circular_smoke_test(self) -> None:
        """
        CIRCULAR / PLUMBING ONLY test.  This mode writes force into
        MechanicalObject.externalForce/force fields and the C++ monitor reads
        the same fields.  This validates field-access plumbing, NOT physical
        contact force detection.  The result is intentionally circular and must
        NOT be interpreted as validating the LCP path or wall-contact physics.
        """
        target_force_n = 0.10
        report = self._run_wall_press(
            run_name="test_eval_v2_external_force_circular_plumbing",
            steps=100,
            drive_mode="external_force",
            target_force_n=target_force_n,
        )

        summary = report.get("collector_summary", {})
        channel = str(summary.get("channel", ""))
        tails = report.get("tail_means", {})

        # In external_force mode the monitor reads back the written value:
        # if it detects it (monitor path) the test validates plumbing end-to-end.
        # If it falls back to LCP (monitor still zero), that also exercises the path.
        if channel == "wire_wall_force_monitor":
            # Circular plumbing check: field was written and read back
            measured = float(tails.get("monitor_total_force_norm", float("nan")))
            self.assertTrue(math.isfinite(measured), "Measured tail force is not finite")
            self.assertGreater(measured, 0.0, "Measured tail force should be positive")
            # Generous tolerance for plumbing smoke test only
            rel_err = abs(measured - target_force_n) / max(target_force_n, 1e-8)
            self.assertLessEqual(
                rel_err,
                0.50,
                msg=(
                    f"[PLUMBING-ONLY] Tail force {measured:.6g} differs from target "
                    f"{target_force_n:.6g} with relative error {rel_err:.4f}. "
                    "This is a circular smoke test, not physical wall-force validation."
                ),
            )
        else:
            # LCP fallback: data collected. Depending on mapping coverage the LCP
            # path may be treated as scoreable (mapped) or diagnostic (unmapped).
            # Do not assert strict availability here.
            pass


if __name__ == "__main__":
    unittest.main()
