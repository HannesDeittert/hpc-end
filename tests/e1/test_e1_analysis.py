from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

HELPER_DIR = Path(__file__).resolve().parents[2] / "experiments" / "master-thesis" / "notebook_helpers"
import sys

if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

from e1_analysis import compute_k_convergence_table, compute_kendall_matrix, compute_snr_table, iqm, load_e1_result_rows, summarize_trials_by_config  # noqa: E402


class E1AnalysisTests(unittest.TestCase):
    def test_loader_reads_manifest_candidate_and_trial_tables(self) -> None:
        try:
            import h5py  # type: ignore
            import numpy as np  # type: ignore
        except ModuleNotFoundError:
            self.skipTest("h5py is not installed in this test environment")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "config_1" / "Tree_625__target_0__seedbase_123"
            run_dir.mkdir(parents=True, exist_ok=True)

            manifest = {
                "job_name": "cfg1__Tree_625__target_0__seed123",
                "config_id": 1,
                "config_name": "config_1",
                "anatomy_id": "Tree_625",
                "target_index": 0,
                "target_seed": 9000,
                "trial_count": 1,
                "max_episode_steps": 1000,
                "generated_time": "2026-05-02T12:00:00+00:00",
                "candidate_names": ["cand_a"],
                "counts": {"n_trials_total": 1},
                "artifact_paths": {
                    "candidate_summaries_csv": "candidate_summaries.csv",
                    "trials_h5": "trials.h5",
                },
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with (run_dir / "candidate_summaries.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["candidate_name", "success_rate", "score_mean", "trial_count"])
                writer.writeheader()
                writer.writerow(
                    {
                        "candidate_name": "cand_a",
                        "success_rate": "1.0",
                        "score_mean": "0.5",
                        "trial_count": "1",
                    }
                )

            with h5py.File(run_dir / "trials.h5", "w") as handle:
                group = handle.create_group("trials")
                group.create_dataset("config_id", data=np.array([1], dtype=np.int32))
                group.create_dataset("anatomy_id", data=np.array([b"Tree_625"], dtype="S8"))
                group.create_dataset("target_index", data=np.array([0], dtype=np.int32))
                group.create_dataset("score_total", data=np.array([0.5], dtype=np.float64))
                group.create_dataset("success", data=np.array([1], dtype=np.int8))

            manifest_rows, summary_rows, trial_rows = load_e1_result_rows(root)

            self.assertEqual(len(manifest_rows), 1)
            self.assertEqual(len(summary_rows), 1)
            self.assertEqual(len(trial_rows), 1)
            self.assertEqual(manifest_rows[0]["max_episode_steps"], 1000)
            self.assertEqual(summary_rows[0]["candidate_name"], "cand_a")
            self.assertAlmostEqual(trial_rows[0]["score_total"], 0.5)
            self.assertEqual(iqm([0.1, 0.2, 0.3, 1.0]), 0.25)

    def test_summarize_trials_by_config(self) -> None:
        rows = [
            {"config_id": 1, "score_total": 0.2},
            {"config_id": 1, "score_total": 0.4},
            {"config_id": 2, "score_total": 0.8},
        ]
        summaries = summarize_trials_by_config(rows)
        self.assertEqual([row["config_id"] for row in summaries], [1, 2])
        self.assertEqual(summaries[0]["n_trials"], 2)

    def test_compute_snr_table_groups_by_config_and_anatomy(self) -> None:
        rows = [
            {"config_id": 1, "anatomy_id": "Tree_625", "target_index": 0, "candidate_name": "wire_a", "score_total": 0.2},
            {"config_id": 1, "anatomy_id": "Tree_625", "target_index": 0, "candidate_name": "wire_a", "score_total": 0.4},
            {"config_id": 1, "anatomy_id": "Tree_625", "target_index": 0, "candidate_name": "wire_b", "score_total": 0.8},
            {"config_id": 1, "anatomy_id": "Tree_625", "target_index": 0, "candidate_name": "wire_b", "score_total": 1.0},
        ]
        snr_rows = compute_snr_table(rows, n_boot=10, seed=1)
        self.assertEqual(len(snr_rows), 1)
        self.assertEqual(snr_rows[0]["config_id"], 1)
        self.assertEqual(snr_rows[0]["anatomy_id"], "Tree_625")
        self.assertGreaterEqual(snr_rows[0]["n_wires"], 2)
        self.assertIn("snr", snr_rows[0])

    def test_compute_kendall_matrix_detects_perfect_agreement(self) -> None:
        rows = [
            {"config_id": 1, "anatomy_id": "Tree_625", "target_index": 0, "candidate_name": "wire_a", "score_total": 0.1},
            {"config_id": 1, "anatomy_id": "Tree_625", "target_index": 0, "candidate_name": "wire_b", "score_total": 0.9},
            {"config_id": 2, "anatomy_id": "Tree_625", "target_index": 0, "candidate_name": "wire_a", "score_total": 0.2},
            {"config_id": 2, "anatomy_id": "Tree_625", "target_index": 0, "candidate_name": "wire_b", "score_total": 1.0},
        ]
        pair_rows, candidate_rows = compute_kendall_matrix(rows)
        self.assertTrue(candidate_rows)
        self.assertEqual(len(pair_rows), 4)
        self.assertTrue(any(row["config_left"] == 1 and row["config_right"] == 2 for row in pair_rows))
        forward = next(row for row in pair_rows if row["config_left"] == 1 and row["config_right"] == 2)
        self.assertAlmostEqual(forward["mean_tau"], 1.0)

    def test_compute_k_convergence_table_returns_requested_grid(self) -> None:
        rows = []
        for trial_index, score in enumerate([0.1, 0.2, 0.3, 0.4], start=1):
            rows.append({"config_id": 1, "anatomy_id": "Tree_625", "target_index": 0, "candidate_name": "wire_a", "score_total": score})
            rows.append({"config_id": 1, "anatomy_id": "Tree_625", "target_index": 0, "candidate_name": "wire_b", "score_total": score + 0.5})
        k_rows = compute_k_convergence_table(rows, k_values=(1, 2, 4), n_boot=5, seed=1)
        self.assertEqual([row["k"] for row in k_rows], [1, 2, 4])
        self.assertEqual({row["config_id"] for row in k_rows}, {1})


if __name__ == "__main__":
    unittest.main()
