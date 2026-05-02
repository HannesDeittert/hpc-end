from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HELPER_DIR = Path(__file__).resolve().parents[2] / "experiments" / "master-thesis" / "notebook_helpers"
import sys

if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

from e1 import (  # noqa: E402
    CONFIGS,
    PARTITION_PROFILES,
    build_execution_plan,
    build_job_manifest,
    build_sbatch_script,
    choose_partition_assignment,
    derive_partition_weights_from_probe,
    parse_partition_weights,
    probe_cluster_partitions,
    sample_target_specs,
    target_equivalence_report,
    write_targets_json,
)


class E1HelperTests(unittest.TestCase):
    def _sample_json(self, tmp: Path) -> Path:
        path = tmp / "sample_12.json"
        payload = {
            "selected_anatomies": [
                {"record_id": "Tree_625", "seed": 123, "arch_type": "II"},
                {"record_id": "Tree_857", "seed": 223, "arch_type": "II"},
                {"record_id": "Tree_1428", "seed": 323, "arch_type": "II"},
            ]
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_sample_targets_are_stable_across_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sample_json = self._sample_json(tmp)
            payload_a = sample_target_specs(sample_json=sample_json, target_seed_start=9000)
            payload_b = sample_target_specs(sample_json=sample_json, target_seed_start=9000)
            self.assertEqual(payload_a, payload_b)
            seeds = [
                target["target_seed"]
                for anatomy in payload_a["selected_anatomies"]
                for target in anatomy["targets"]
            ]
            self.assertEqual(seeds, [9000, 9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008])

    def test_build_job_manifest_cartesian_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sample_json = self._sample_json(tmp)
            targets_json = tmp / "targets.json"
            write_targets_json(sample_json=sample_json, output_path=targets_json)
            manifest = build_job_manifest(
                sample_json=sample_json,
                targets_json=targets_json,
                output_root=tmp / "e1",
                max_episode_steps=777,
            )
            self.assertEqual(len(manifest["jobs"]), 36)
            self.assertEqual({row["config_id"] for row in manifest["jobs"]}, {1, 2, 3, 4})
            self.assertEqual(manifest["max_episode_steps"], 777)
            self.assertEqual(manifest["jobs"][0]["config_spec"]["max_episode_steps"], 777)

    def test_target_equivalence_report_detects_same_targets_across_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sample_json = self._sample_json(tmp)
            targets_json = tmp / "targets.json"
            write_targets_json(sample_json=sample_json, output_path=targets_json)
            manifest = build_job_manifest(
                sample_json=sample_json,
                targets_json=targets_json,
                output_root=tmp / "e1",
            )
            report = target_equivalence_report(manifest)
            self.assertTrue(report["same_targets_across_configs"])
            self.assertEqual(report["n_mismatches"], 0)

    def test_partition_assignment_defaults_to_work(self) -> None:
        self.assertEqual(choose_partition_assignment(job_count=4), ("work",) * 4)

    def test_partition_weight_parsing(self) -> None:
        weights = parse_partition_weights("work:0.7,a100:0.3")
        self.assertEqual(weights, (("work", 0.7), ("a100", 0.3)))

    def test_partition_weight_parsing_accepts_plain_names(self) -> None:
        self.assertEqual(parse_partition_weights("work"), (("work", 1.0),))
        self.assertEqual(parse_partition_weights("work,a100"), (("work", 1.0), ("a100", 1.0)))

    def test_derived_partition_weights_follow_idle_capacity(self) -> None:
        rows = probe_cluster_partitions(
            sinfo_output=(
                "work|4|mix|gpu:4|32\n"
                "work|2|idle|gpu:4|32\n"
                "a100|1|idle|gpu:a100:1|32\n"
                "v100|1|alloc|gpu:v100:1|32\n"
            )
        )
        weights = derive_partition_weights_from_probe(rows)
        self.assertEqual([name for name, _weight in weights], ["a100", "work"])
        self.assertAlmostEqual(sum(weight for _name, weight in weights), 1.0)

    def test_execution_plan_mapping(self) -> None:
        plan = build_execution_plan(config_id=3, seed_base=123, trial_count=1000, max_episode_steps=50)
        self.assertEqual(plan["policy_mode"], "stochastic")
        self.assertEqual(plan["stochastic_environment_mode"], "fixed_start")
        self.assertEqual(plan["trials_per_candidate"], 1000)
        self.assertEqual(plan["worker_count"], 29)

    def test_probe_parser_handles_sinfo_lines(self) -> None:
        rows = probe_cluster_partitions(
            sinfo_output=(
                "work|4|mix|gpu:4|32\n"
                "work|2|idle|gpu:4|32\n"
                "a100|1|idle|gpu:a100:1|32\n"
            )
        )
        self.assertEqual([row.partition for row in rows], ["a100", "work"])
        work = next(row for row in rows if row.partition == "work")
        self.assertEqual(work.nodes_total, 6)
        self.assertEqual(work.states["mix"], 4)
        self.assertEqual(work.states["idle"], 2)

    def test_probe_prefers_tinygpu_suffix_then_fallbacks(self) -> None:
        outputs = iter(
            [
                subprocess.CalledProcessError(1, ["sinfo.tinygpu"]),
                subprocess.CompletedProcess(
                    args=["sinfo", "--clusters=tinygpu", "--noheader", "--format=%P|%D|%t|%G|%c"],
                    returncode=0,
                    stdout="work|1|idle|gpu:4|32\n",
                ),
            ]
        )

        def fake_run(command, *args, **kwargs):
            result = next(outputs)
            if isinstance(result, subprocess.CalledProcessError):
                raise result
            return result

        with patch("e1.subprocess.run", side_effect=fake_run):
            rows = probe_cluster_partitions()

        self.assertEqual([row.partition for row in rows], ["work"])

    def test_sbatch_script_includes_partition_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / "job_manifest_work.json"
            manifest.write_text(
                json.dumps({"jobs": [{"gres": "gpu:4", "cpus_per_task": 32}]}, indent=2),
                encoding="utf-8",
            )
            script = build_sbatch_script(
                project_root=Path("/repo"),
                partition="work",
                jobs_manifest_path=manifest,
                logs_root=tmp / "logs",
                walltime="24:00:00",
                gres="gpu:4",
                cpus_per_task=32,
            )
            self.assertIn("#SBATCH --partition=work", script)
            self.assertIn("#SBATCH --gres=gpu:4", script)
            self.assertIn("#SBATCH --cpus-per-task=32", script)

    def test_partition_profiles_match_tinygpu_documentation(self) -> None:
        self.assertEqual(PARTITION_PROFILES["work"]["gres"], "gpu:4")
        self.assertEqual(PARTITION_PROFILES["rtx3080"]["gres"], "gpu:rtx3080:4")
        self.assertEqual(PARTITION_PROFILES["v100"]["gres"], "gpu:v100:4")
        self.assertEqual(PARTITION_PROFILES["a100"]["gres"], "gpu:a100:1")


if __name__ == "__main__":
    unittest.main()
