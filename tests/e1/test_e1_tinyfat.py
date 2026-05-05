from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "experiments" / "master-thesis" / "notebook_helpers" / "e1_tinyfat.py"


def load_module():
    spec = importlib.util.spec_from_file_location("e1_tinyfat", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class E1TinyFatTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_probe_cluster_partitions_parses_tinyfat_output(self):
        text = "\n".join(
            [
                "work*|4|idle|gpu:none|128",
                "broadwell256|3|alloc|N/A|28",
                "broadwell512|1|mix|N/A|56",
            ]
        )
        rows = self.mod.probe_cluster_partitions(sinfo_output=text)
        self.assertEqual([row.partition for row in rows], ["broadwell256", "broadwell512", "work"])

    def test_build_sbatch_script_is_cpu_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "metadata" / "job_manifest_work.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "partition": "work",
                        "jobs": [
                            {
                                "cpus_per_task": 64,
                                "worker_count": 61,
                                "hint": "nomultithread",
                            }
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            script = self.mod.build_sbatch_script(
                project_root=root,
                partition="work",
                jobs_manifest_path=manifest_path,
                logs_root=root / "logs",
                walltime="24:00:00",
            )
            self.assertIn("#SBATCH --partition=work", script)
            self.assertIn("#SBATCH --hint=nomultithread", script)
            self.assertIn("--worker-count \"61\"", script)
            self.assertNotIn("#SBATCH --gres=", script)


if __name__ == "__main__":
    unittest.main()
