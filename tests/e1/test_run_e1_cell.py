from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

HELPER_DIR = Path(__file__).resolve().parents[2] / "experiments" / "master-thesis" / "notebook_helpers"
import sys

if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

from e1 import config_spec  # noqa: E402
from steve_recommender.eval_v2.models import AorticArchAnatomy, EvaluationCandidate, PolicySpec, WireRef


RUN_E1_CELL_PATH = Path(__file__).resolve().parents[2] / "experiments" / "master-thesis" / "run_e1_cell.py"
spec = importlib.util.spec_from_file_location("run_e1_cell", RUN_E1_CELL_PATH)
assert spec is not None and spec.loader is not None
run_e1_cell = importlib.util.module_from_spec(spec)
sys.modules["run_e1_cell"] = run_e1_cell
spec.loader.exec_module(run_e1_cell)


class _StubService:
    def __init__(self) -> None:
        self.jobs: list[object] = []
        self.anatomy = AorticArchAnatomy(arch_type="II", seed=42, record_id="Tree_625")
        self.wire = WireRef(model="steve_default", wire="standard_j")
        self.candidate = EvaluationCandidate(
            name="candidate_from_registry",
            execution_wire=self.wire,
            policy=PolicySpec(name="policy", checkpoint_path=Path("/tmp/policy.everl")),
        )

    def get_anatomy(self, *, record_id: str, registry_path=None):
        _ = registry_path
        assert record_id == self.anatomy.record_id
        return self.anatomy

    def list_startable_wires(self):
        return (self.wire,)

    def list_candidates(self, *, execution_wire: WireRef, include_cross_wire: bool = False):
        _ = include_cross_wire
        assert execution_wire == self.wire
        return (self.candidate,)

    def run_evaluation_job(self, job, *, progress_callback=None):
        _ = progress_callback
        self.jobs.append(job)
        return SimpleNamespace(
            job_name=job.name,
            generated_at="2026-05-02T12:00:00+00:00",
            summaries=(SimpleNamespace(
                scenario_name=job.scenarios[0].name,
                candidate_name=job.candidates[0].name,
                success_rate=1.0,
                score_mean=0.5,
                trial_count=job.execution.trials_per_candidate,
            ),),
            artifacts=SimpleNamespace(output_dir=job.resume_output_dir),
        )


class E1RunnerTests(unittest.TestCase):
    def test_runner_builds_job_from_direct_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            target_json = tmp / "target.json"
            target_json.write_text(
                json.dumps(
                    {
                        "kind": "centerline_random",
                        "target_seed": 9000,
                        "target_index": 0,
                        "threshold_mm": 5.0,
                        "branches": ["bct", "lcca", "lsa"],
                    }
                ),
                encoding="utf-8",
            )
            wires_json = tmp / "wires.json"
            wires_json.write_text(
                json.dumps(
                    {"wires": [{"tool_ref": "steve_default/standard_j"}]},
                    indent=2,
                ),
                encoding="utf-8",
            )
            output_path = tmp / "runs" / "config_1" / "Tree_625__target_0__seedbase_123"
            stub = _StubService()

            with patch.object(run_e1_cell, "make_default_service", return_value=stub):
                rc = run_e1_cell.main(
                    [
                        "--anatomy-id",
                        "Tree_625",
                        "--target-spec",
                        str(target_json),
                        "--config-id",
                        "1",
                        "--num-trials",
                        "1",
                        "--seed-base",
                        "123",
                        "--output-path",
                        str(output_path),
                        "--step-budget",
                        "50",
                        "--wires-json",
                        str(wires_json),
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertEqual(len(stub.jobs), 1)
            job = stub.jobs[0]
            self.assertEqual(job.execution.policy_mode, "deterministic")
            self.assertEqual(job.execution.stochastic_environment_mode, "fixed_start")
            self.assertEqual(job.execution.trials_per_candidate, 1)
            self.assertEqual(job.resume_output_dir, output_path)
            self.assertEqual(job.scenarios[0].name, "Tree_625__target_0")

    def test_runner_manifest_mode_uses_manifest_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / "manifest.json"
            target_json = tmp / "target.json"
            target_json.write_text(
                json.dumps(
                    {
                        "kind": "centerline_random",
                        "target_seed": 9001,
                        "target_index": 1,
                        "threshold_mm": 5.0,
                        "branches": ["bct", "lcca", "lsa"],
                    }
                ),
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "jobs": [
                            {
                                "anatomy_id": "Tree_625",
                                "config_id": 4,
                                "config_spec": config_spec(4),
                                "target_spec": json.loads(target_json.read_text(encoding="utf-8")),
                                "seed_base": 123,
                                "output_dir": str(tmp / "runs" / "config_4" / "Tree_625__target_1__seedbase_123"),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            stub = _StubService()

            with patch.object(run_e1_cell, "make_default_service", return_value=stub):
                rc = run_e1_cell.main(["--manifest", str(manifest), "--array-index", "0"])

            self.assertEqual(rc, 0)
            self.assertEqual(len(stub.jobs), 1)
            self.assertEqual(stub.jobs[0].execution.policy_mode, "stochastic")
            self.assertEqual(stub.jobs[0].execution.stochastic_environment_mode, "random_start")


if __name__ == "__main__":
    unittest.main()
