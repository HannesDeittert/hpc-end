from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

HELPER_DIR = Path(__file__).resolve().parents[2] / "experiments" / "master-thesis" / "notebook_helpers"
import sys

if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

from e0 import load_e0_results  # noqa: E402


class E0LoaderTests(unittest.TestCase):
    def test_chunk_filter_limits_loaded_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for chunk_name, score in [("chunk_1", 0.1), ("chunk_2", 0.2)]:
                run_dir = root / chunk_name / "run_0"
                run_dir.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "job_name": f"job_{chunk_name}",
                    "artifact_paths": {},
                    "counts": {"n_trials_total": 1},
                    "anatomy_metadata": [1, 2],
                }
                (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                (run_dir / "candidate_summaries.csv").write_text(
                    "candidate_name,score_mean\nwire_a,{score}\n".format(score=score),
                    encoding="utf-8",
                )

            manifest_rows, summary_rows, trial_rows = load_e0_results(root, chunk_dir="chunk_1")
            self.assertEqual(len(manifest_rows), 1)
            self.assertEqual(len(summary_rows), 1)
            self.assertEqual(len(trial_rows), 0)
            self.assertEqual(manifest_rows[0]["chunk_dir"], "chunk_1")


if __name__ == "__main__":
    unittest.main()
