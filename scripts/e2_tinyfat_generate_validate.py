#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
HELPER_DIR = PROJECT_ROOT / "experiments" / "master-thesis" / "notebook_helpers"
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

from e2_tinyfat import (  # noqa: E402
    DEFAULT_E2_ROOT,
    DEFAULT_FRICTION,
    DEFAULT_SAMPLE_JSON,
    DEFAULT_TARGET_POSITIONS,
    DEFAULT_TARGET_SEED_START,
    DEFAULT_TARGETS_PER_BRANCH,
    DEFAULT_TRIAL_COUNT,
    build_job_manifest,
    build_sbatch_script,
    load_sample_anatomies,
    parse_partition_weights,
    probe_cluster_partitions,
    write_partition_bucket_manifests,
    write_targets_json,
    write_wires_json,
)


def make_default_service():
    from steve_recommender.eval_v2.service import DefaultEvaluationService
    return DefaultEvaluationService()


def _wire_to_dict(wire: object) -> dict[str, str]:
    model = str(getattr(wire, "model", "")) if not isinstance(wire, dict) else str(wire.get("model", ""))
    wire_name = str(getattr(wire, "wire", "")) if not isinstance(wire, dict) else str(wire.get("wire", ""))
    tool_ref = str(getattr(wire, "tool_ref", "")) if not isinstance(wire, dict) else str(wire.get("tool_ref", ""))
    return {"model": model, "wire": wire_name, "tool_ref": tool_ref or f"{model}/{wire_name}"}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TinyFat E2 job manifests and Slurm scripts")
    parser.add_argument("--sample-json", type=Path, default=DEFAULT_SAMPLE_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_E2_ROOT)
    parser.add_argument("--scripts-root", type=Path, default=PROJECT_ROOT / "scripts")
    parser.add_argument("--config-id", type=int, required=True, help="One E1-style config id to materialize: 1..4")
    parser.add_argument("--num-batches", "--trial-count", dest="trial_count", type=int, default=DEFAULT_TRIAL_COUNT)
    parser.add_argument("--seed-base-start", type=int, default=123)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--partitions", default=None)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--worker-count", type=int, default=None)
    parser.add_argument("--friction", type=float, default=DEFAULT_FRICTION)
    parser.add_argument(
        "--targets-per-branch",
        type=int,
        default=DEFAULT_TARGETS_PER_BRANCH,
        help="Number of targets to create for each non-aorta branch.",
    )
    parser.add_argument(
        "--target-positions",
        default=",".join(DEFAULT_TARGET_POSITIONS),
        help='Comma list matching --targets-per-branch, e.g. "start,end" or "end,random,random".',
    )
    parser.add_argument(
        "--target-seed-start",
        type=int,
        default=DEFAULT_TARGET_SEED_START,
        help="First deterministic seed used to resolve random target positions.",
    )
    parser.add_argument("--probe-cluster", action="store_true")
    trace = parser.add_mutually_exclusive_group()
    trace.add_argument("--write-trace", dest="write_full_trace", action="store_true")
    trace.add_argument("--no-write-trace", dest="write_full_trace", action="store_false")
    parser.set_defaults(write_full_trace=False)
    parser.add_argument("--write-diagnostics", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    service = make_default_service()
    output_root = Path(args.output_root).resolve()
    metadata_root = output_root / "metadata"
    logs_root = output_root / "logs"
    scripts_root = Path(args.scripts_root).resolve()
    metadata_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    scripts_root.mkdir(parents=True, exist_ok=True)

    targets_json = metadata_root / "targets.json"
    targets_payload = write_targets_json(
        sample_json=args.sample_json,
        output_path=targets_json,
        service=service,
        targets_per_branch=int(args.targets_per_branch),
        target_positions=str(args.target_positions),
        target_seed_start=int(args.target_seed_start),
    )
    wires = [_wire_to_dict(wire) for wire in service.list_startable_wires()]
    write_wires_json(output_path=metadata_root / "wires.json", wires=wires)

    probe_rows = ()
    if args.probe_cluster:
        try:
            probe_rows = probe_cluster_partitions()
        except Exception:
            probe_rows = ()
    partition_weights = parse_partition_weights(args.partitions) if args.partitions else (("work", 1.0),)

    manifest = build_job_manifest(
        sample_json=args.sample_json,
        targets_json=targets_json,
        output_root=output_root,
        config_id=args.config_id,
        trial_count=args.trial_count,
        seed_base_start=args.seed_base_start,
        max_episode_steps=args.max_episode_steps,
        partitions=args.partitions,
        partition_weights=partition_weights,
        probe_rows=probe_rows,
        worker_count=args.worker_count,
        walltime=args.walltime,
        friction=float(args.friction),
        write_full_trace=bool(args.write_full_trace),
        write_diagnostics=bool(args.write_diagnostics),
    )
    (metadata_root / "job_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (metadata_root / "anatomies.json").write_text(
        json.dumps({"schema_version": 1, "source_sample_json": str(Path(args.sample_json).resolve()), "selected_anatomies": list(load_sample_anatomies(args.sample_json))}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (metadata_root / "experiment_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "e2",
                "config_id": int(args.config_id),
                "num_batches": int(args.trial_count),
                "max_episode_steps": int(args.max_episode_steps),
                "friction": float(args.friction),
                "target_rule": targets_payload.get("target_rule"),
                "targets_per_branch": int(targets_payload.get("targets_per_branch", args.targets_per_branch)),
                "target_positions": list(targets_payload.get("target_positions", [])),
                "target_seed_start": int(targets_payload.get("target_seed_start", args.target_seed_start)),
                "n_anatomies": len(targets_payload.get("selected_anatomies", [])),
                "n_jobs": len(manifest.get("jobs", [])),
                "partition_weights": list(partition_weights),
                "default_walltime": args.walltime,
                "default_worker_count": args.worker_count,
                "probe_rows": [row.__dict__ for row in probe_rows],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    bucket_paths = write_partition_bucket_manifests(output_root=output_root, manifest=manifest)
    script_paths = {}
    for partition, bucket_path in bucket_paths.items():
        script_path = scripts_root / f"e2_tinyfat_{partition}.sbatch"
        script_path.write_text(
            build_sbatch_script(project_root=PROJECT_ROOT, partition=partition, jobs_manifest_path=bucket_path, logs_root=logs_root, walltime=args.walltime),
            encoding="utf-8",
        )
        script_path.chmod(0o755)
        script_paths[partition] = str(script_path)

    print(f"[E2-TF] output_root={output_root}")
    print(f"[E2-TF] metadata_root={metadata_root}")
    print(f"[E2-TF] n_anatomies={len(targets_payload.get('selected_anatomies', []))}")
    print(f"[E2-TF] n_jobs={len(manifest.get('jobs', []))}")
    print(f"[E2-TF] config_id={args.config_id}")
    print(f"[E2-TF] num_batches={args.trial_count}")
    print(f"[E2-TF] friction={float(args.friction)}")
    print(f"[E2-TF] targets_per_branch={targets_payload.get('targets_per_branch')}")
    print(f"[E2-TF] target_positions={targets_payload.get('target_positions')}")
    print(f"[E2-TF] scripts={script_paths}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
