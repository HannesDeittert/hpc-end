from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


class ComparisonCliTests(unittest.TestCase):
    def test_list_agent_refs(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "steve_recommender.comparison.run_cli",
                "--list-agent-refs",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("[compare] available agent refs", proc.stdout)

    def test_dry_run_resolves_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt = tmp_path / "checkpoint123.everl"
            ckpt.write_text("dummy", encoding="utf-8")

            cfg = {
                "name": "dry_run_demo",
                "candidates": [
                    {
                        "name": "agent_1",
                        "tool": "TestModel_StandardJ035/StandardJ035_PTFE",
                        "checkpoint": str(ckpt),
                    }
                ],
                "n_trials": 2,
                "base_seed": 123,
            }
            cfg_path = tmp_path / "compare.yml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "steve_recommender.comparison.run_cli",
                    "--config",
                    str(cfg_path),
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("[compare] resolved candidates", proc.stdout)
            self.assertIn(str(ckpt.resolve()), proc.stdout)

    def test_dry_run_accepts_visualize_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt = tmp_path / "checkpoint999.everl"
            ckpt.write_text("dummy", encoding="utf-8")

            cfg = {
                "name": "dry_run_vis_demo",
                "candidates": [
                    {
                        "name": "agent_1",
                        "tool": "TestModel_StandardJ035/StandardJ035_PTFE",
                        "checkpoint": str(ckpt),
                    }
                ],
                "n_trials": 1,
                "base_seed": 1,
            }
            cfg_path = tmp_path / "compare_vis.yml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "steve_recommender.comparison.run_cli",
                    "--config",
                    str(cfg_path),
                    "--dry-run",
                    "--visualize",
                    "--visualize-trials-per-agent",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("[compare] resolved candidates", proc.stdout)


if __name__ == "__main__":
    unittest.main()
