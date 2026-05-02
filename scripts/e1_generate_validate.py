#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
HELPER_DIR = PROJECT_ROOT / "experiments" / "master-thesis" / "notebook_helpers"
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

from e1 import (  # noqa: E402
    derive_partition_weights_from_probe,
    parse_partition_weights,
    probe_cluster_partitions,
    target_equivalence_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe TinyGPU, generate E1 jobs, and validate them locally")
    parser.add_argument("--sample-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, default=PROJECT_ROOT / "scripts")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--worker-count", type=int, default=29)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--seed-base-start", type=int, default=123)
    parser.add_argument("--target-seed-start", type=int, default=9000)
    parser.add_argument("--target-count-per-anatomy", type=int, default=3)
    parser.add_argument("--trial-count", type=int, default=None)
    parser.add_argument("--target-branches", default="bct,lcca,lsa")
    parser.add_argument("--probe-cluster", action="store_true", help="Probe Slurm and derive partition weights from it")
    parser.add_argument("--partitions", default=None, help="Explicit partition weights, e.g. work:0.7,rtx3080:0.3")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> int:
    args = _parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    probe_rows = ()
    partition_weights = ()
    if args.probe_cluster:
        probe_rows = probe_cluster_partitions()
        partition_weights = derive_partition_weights_from_probe(probe_rows)
    elif args.partitions:
        partition_weights = parse_partition_weights(args.partitions)

    partition_weights_arg = args.partitions
    if args.probe_cluster:
        partition_weights_arg = ",".join(f"{name}:{weight:.6f}" for name, weight in partition_weights)
    elif not partition_weights_arg:
        partition_weights_arg = "work"

    generator = PROJECT_ROOT / "experiments" / "master-thesis" / "generate_e1_jobs.py"
    generator_cmd = [
        args.python_bin,
        str(generator),
        "--sample-json",
        str(Path(args.sample_json).resolve()),
        "--output-root",
        str(output_root),
        "--scripts-root",
        str(Path(args.scripts_root).resolve()),
        "--seed-base-start",
        str(args.seed_base_start),
        "--target-seed-start",
        str(args.target_seed_start),
        "--target-count-per-anatomy",
        str(args.target_count_per_anatomy),
        "--target-branches",
        str(args.target_branches),
        "--partitions",
        partition_weights_arg,
        "--walltime",
        str(args.walltime),
        "--worker-count",
        str(args.worker_count),
        "--max-episode-steps",
        str(args.max_episode_steps),
    ]
    if args.trial_count is not None:
        generator_cmd.extend(["--trial-count", str(args.trial_count)])
    if args.probe_cluster:
        generator_cmd.append("--probe-cluster")
    _run(generator_cmd)

    manifest_path = output_root / "metadata" / "job_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = target_equivalence_report(manifest)

    script_paths = []
    missing = []
    for partition in sorted({str(job["partition"]) for job in manifest.get("jobs", [])}):
        script_path = Path(args.scripts_root).resolve() / f"e1_{partition}.sbatch"
        script_paths.append(script_path)
        if not script_path.exists():
            missing.append(str(script_path))
    payload = {
        "output_root": str(output_root),
        "n_jobs": len(manifest.get("jobs", [])),
        "n_scripts": len(script_paths),
        "missing_scripts": missing,
        "target_report": report,
        "partition_weights": list(partition_weights),
        "probe_rows": [
            {
                "partition": row.partition,
                "nodes_total": row.nodes_total,
                "states": row.states,
                "gres": list(row.gres),
                "cpus_per_node": row.cpus_per_node,
            }
            for row in probe_rows
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report["same_targets_across_configs"] and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
