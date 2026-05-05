from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "scripts" / "monitor_e1_progress.py"


def load_module():
    spec = importlib.util.spec_from_file_location("monitor_e1_progress", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MonitorE1ProgressTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_parse_progress_text_running(self):
        text = "\n".join(
            [
                "[eval_v2] parallel_start workers=29 total=15000",
                "[eval_v2] parallel_trial_done completed=412 total=15000 scenario=X candidate=Y trial_index=0",
            ]
        )
        completed, total, progress, last_line, status = self.mod.parse_progress_text(text)
        self.assertEqual(completed, 412)
        self.assertEqual(total, 15000)
        self.assertAlmostEqual(progress, 412 / 15000)
        self.assertIn("parallel_trial_done completed=412 total=15000", last_line)
        self.assertEqual(status, "running")

    def test_parse_progress_text_completed(self):
        text = "\n".join(
            [
                "[eval_v2] parallel_trial_done completed=15000 total=15000 scenario=X candidate=Y trial_index=0",
                "[eval_v2] parallel_end completed=15000 total=15000",
                "=== JOB_STATISTICS ===",
                "[E1] summary scenario=X candidate=Y success_rate=1.0 score_mean=0.9 trials=1000",
            ]
        )
        completed, total, progress, last_line, status = self.mod.parse_progress_text(text)
        self.assertEqual(completed, 15000)
        self.assertEqual(total, 15000)
        self.assertEqual(progress, 1.0)
        self.assertEqual(status, "completed")
        self.assertIn("parallel_end completed=15000 total=15000", last_line)

    def test_summarize_group_uses_log_progress(self):
        jobs = [
            self.mod.ManifestJob(partition="a100", array_index=0, config_id=1, job_name="job-a", output_dir="out-a", trial_count=1, write_full_trace=False),
            self.mod.ManifestJob(partition="a100", array_index=1, config_id=1, job_name="job-b", output_dir="out-b", trial_count=1, write_full_trace=False),
        ]
        logs = {
            "job-a": self.mod.LogProgress(
                log_path=Path("/tmp/job-a.out"),
                job_name="job-a",
                output_dir="out-a",
                completed=412,
                total=1000,
                progress=0.412,
                status="running",
                last_line="[eval_v2] parallel_trial_done completed=412 total=1000 scenario=X candidate=Y trial_index=0",
            ),
            "job-b": self.mod.LogProgress(
                log_path=Path("/tmp/job-b.out"),
                job_name="job-b",
                output_dir="out-b",
                completed=1000,
                total=1000,
                progress=1.0,
                status="completed",
                last_line="[eval_v2] parallel_end completed=1000 total=1000",
            ),
        }
        group = self.mod._summarize_group(jobs, {}, logs)
        self.assertEqual(group["total_jobs"], 2)
        self.assertEqual(group["states"]["RUNNING"], 1)
        self.assertEqual(group["states"]["COMPLETED"], 1)
        self.assertAlmostEqual(group["progress_mean"], (0.412 + 1.0) / 2)
        self.assertAlmostEqual(group["progress_min"], 0.412)
        self.assertAlmostEqual(group["progress_max"], 1.0)

    def test_summarize_group_log_only_mode_skips_traces(self):
        jobs = [
            self.mod.ManifestJob(partition="work", array_index=0, config_id=2, job_name="job-a", output_dir="out-a", trial_count=1000, write_full_trace=False),
        ]
        logs = {
            "job-a": self.mod.LogProgress(
                log_path=Path("/tmp/job-a.out"),
                job_name="job-a",
                output_dir="out-a",
                completed=12,
                total=15000,
                progress=12 / 15000,
                status="running",
                last_line="[eval_v2] parallel_trial_done completed=12 total=15000 scenario=X candidate=Y trial_index=0",
            ),
        }
        group = self.mod._summarize_group(jobs, {}, logs, progress_source="log")
        self.assertEqual(group["trial_row_count"], 0)
        self.assertEqual(group["trial_row_expected"], 0)
        self.assertEqual(group["trace_count"], 0)
        self.assertEqual(group["trace_expected"], 0)
        self.assertAlmostEqual(group["progress_mean"], 12 / 15000)
        self.assertEqual(group["states"]["RUNNING"], 1)

    @unittest.skipUnless(importlib.util.find_spec("h5py") is not None, "h5py not available")
    def test_summarize_group_uses_trial_rows_when_available(self):
        mod = self.mod
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "results" / "master_thesis" / "e1_tinyfat" / "job-a"
            output_dir.mkdir(parents=True)
            with mod.h5py.File(output_dir / "trials.h5", "w") as handle:  # type: ignore[attr-defined]
                group = handle.create_group("trials")
                group.create_dataset("trial_index", data=[0, 1, 2, 3])

            jobs = [
                mod.ManifestJob(
                    partition="work",
                    array_index=0,
                    config_id=2,
                    job_name="job-a",
                    output_dir=str(output_dir),
                    trial_count=1000,
                    write_full_trace=False,
                )
            ]
            group = mod._summarize_group(jobs, {}, {}, progress_source="auto")
            self.assertEqual(group["trial_row_count"], 4)
            self.assertEqual(group["trial_row_expected"], 15000)
            self.assertAlmostEqual(group["trial_row_progress"], 4 / 15000)
            self.assertEqual(group["states"]["RUNNING"], 1)

    def test_load_trace_progress_counts_trace_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            traces = root / "results" / "master_thesis" / "e1" / "job-a" / "traces"
            traces.mkdir(parents=True)
            for index in range(7):
                (traces / f"trial_{index}.h5").write_bytes(b"test")

            progress = self.mod.load_trace_progress(
                root / "results" / "master_thesis" / "e1" / "job-a",
                trial_count=1000,
            )

            self.assertEqual(progress.count, 7)
            self.assertEqual(progress.expected, 15000)
            self.assertAlmostEqual(progress.progress, 7 / 15000)


if __name__ == "__main__":
    unittest.main()
