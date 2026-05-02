from __future__ import annotations

import importlib.util
import json
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
class EvalV2MonitorDiagnosticTests(unittest.TestCase):
    """Diagnostic tests to understand why the passive monitor is not detecting contacts."""

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _run_wall_press_with_contact(self) -> dict:
        """Run wall-press with motion to force contact with wall."""
        repo_root = self._repo_root()
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            cmd = [
                sys.executable,
                "-m",
                "debug.force_playground.eval_v2_plain_wall_press",
                "--steps",
                "120",
                "--target-force-n",
                "0.10",
                "--insert-action",
                "0.50",
                "--force-mode",
                "constraint_projected_si_validated",
                "--drive-mode",
                "external_force",
                "--output-root",
                str(output_root),
                "--run-name",
                "diagnostic_monitor_check",
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
                output_root.glob("*_diagnostic_monitor_check/eval_v2_wall_press_report.json")
            )
            self.assertTrue(report_candidates, "No eval_v2 wall-press report JSON produced")
            report_path = report_candidates[-1]
            with report_path.open("r", encoding="utf-8") as f:
                return json.load(f)

    def test_monitor_contact_detection_status(self) -> None:
        """
        Diagnostic: check whether the passive monitor detected any contacts.

        Expected behavior in SOFA FreeMotionAnimationLoop + LCPConstraintSolver:
        - The C++ monitor reads MechanicalObject.force/externalForce which are NOT
          populated for constraint-based contacts.
        - If drive_mode=external_force writes to externalForce, the monitor may read it
          back (circular plumbing), reporting nonzero force on the monitor path.
        - If monitor reads zero, the system falls back to LCP (unmapped, not scored).

        Either outcome is valid here — this is a diagnostic test, not a correctness test.
        """
        report = self._run_wall_press_with_contact()

        summary = report.get("collector_summary", {})
        channel = str(summary.get("channel", ""))
        source = str(summary.get("source", ""))
        validation_status = str(summary.get("validation_status", ""))
        available_for_score = bool(summary.get("available_for_score", False))

        print(f"\n[Diagnostic] Active channel: {channel}")
        print(f"[Diagnostic] Source: {source}")
        print(f"[Diagnostic] Validation status: {validation_status}")
        print(f"[Diagnostic] Available for score: {available_for_score}")

        # The test accepts either outcome explicitly — documenting what the system produces.
        if channel == "lcp.constraintForces/dt":
            self.assertEqual(
                validation_status,
                "lcp_only_unmapped",
                "LCP fallback must carry validation_status='lcp_only_unmapped'",
            )
            self.assertFalse(
                available_for_score,
                "LCP-only unmapped fallback must NOT be available_for_score=True.\n"
                "Root cause: C++ monitor reads MechanicalObject.force/externalForce which\n"
                "are not populated for constraint contacts in FreeMotionAnimationLoop.\n"
                "Fix: read LCP constraintForces with explicit contact-row-to-wall mapping.",
            )
            print(
                "\n[KNOWN ISSUE] Passive monitor zero → LCP fallback."
                "\n  C++ WireWallForceMonitor reads MechanicalObject.force/externalForce."
                "\n  In FreeMotionAnimationLoop, contact forces are constraint impulses (LCP),"
                "\n  not stored in force/externalForce vectors."
                "\n  Fix: use WireWallContactExport to map LCP rows to wall-triangle contacts."
            )
        elif channel == "wire_wall_force_monitor":
            # Monitor path active — legacy diagnostic. Eval_v2 no longer treats the
            # native monitor as a validated scoring source; accept diagnostic outcome.
            print(f"\n[DIAG] Passive monitor active on channel: {channel}")
        else:
            # No data collected at all
            self.assertFalse(available_for_score)
            print(f"\n[WARN] No force channel active: {channel}")

    def test_lcp_data_always_collected(self) -> None:
        """LCP raw values are collected regardless of monitor state, for diagnostics."""
        report = self._run_wall_press_with_contact()

        summary = report.get("collector_summary", {})
        lcp_max = summary.get("lcp_max_abs_max")

        print(f"\n[Diagnostic] lcp_max_abs_max = {lcp_max}")
        print(f"[Diagnostic] lcp_sum_abs_mean = {summary.get('lcp_sum_abs_mean')}")

        # LCP solver is always active in these scenes; constraintForces should be nonzero
        # during contact phases.
        self.assertIsNotNone(
            lcp_max,
            "lcp_max_abs_max should always be collected; if None the LCP data pipeline is broken",
        )

    def test_force_values_presence_for_diagnostics(self) -> None:
        """
        Verify force diagnostic values are present regardless of validity path.
        total_force_norm_max is populated from monitor (valid path) or LCP fallback
        (diagnostic only), so callers always have something to display.
        """
        report = self._run_wall_press_with_contact()

        summary = report.get("collector_summary", {})

        total_max = summary.get("total_force_norm_max")
        total_mean = summary.get("total_force_norm_mean")

        print(f"\n[Force Summary]")
        print(f"  total_force_norm_max: {total_max}")
        print(f"  total_force_norm_mean: {total_mean}")
        print(f"  LCP max (raw):  {summary.get('lcp_max_abs_max')}")
        print(f"  LCP mean (raw): {summary.get('lcp_sum_abs_mean')}")
        print(f"  available_for_score: {summary.get('available_for_score')}")
        print(f"  validation_status:   {summary.get('validation_status')}")

        # Values may be present even when not available_for_score (diagnostic)
        self.assertIsNotNone(total_max, "total_force_norm_max should be populated for display")
        if total_max is not None:
            self.assertGreater(float(total_max), 0.0, "Should have some force signal for diagnostics")


if __name__ == "__main__":
    unittest.main()
