from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from steve_recommender.eval_v2.viewer.cli import main
from tests.eval_v2.test_force_trace_persistence import _TrialTraceTestHelpers


class ViewerCliTests(_TrialTraceTestHelpers, unittest.TestCase):
    def test_cli_opens_named_trace_file_off_screen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=2)
            stdout = io.StringIO()

            code = main([str(trace_path), "--off-screen"], stdout=stdout)

        self.assertEqual(code, 0)
        self.assertIn("Opened trace viewer", stdout.getvalue())

    def test_cli_resolves_job_directory_to_first_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "job"
            traces_dir = job_dir / "traces"
            traces_dir.mkdir(parents=True)
            first = traces_dir / "a_trace.h5"
            second = traces_dir / "b_trace.h5"
            self._write_trace(second, n_steps=2)
            self._write_trace(first, n_steps=2)
            stdout = io.StringIO()

            code = main([str(job_dir), "--off-screen"], stdout=stdout)

        self.assertEqual(code, 0)
        self.assertIn("a_trace.h5", stdout.getvalue())

    def test_cli_invalid_path_exits_with_clear_message(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = main(
            ["/definitely/missing/trace.h5", "--off-screen"],
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(code, 2)
        self.assertIn("Trace path does not exist", stderr.getvalue())
