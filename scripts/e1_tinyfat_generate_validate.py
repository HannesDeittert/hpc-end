#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HELPER_DIR = SCRIPT_DIR.parent / "experiments" / "master-thesis" / "notebook_helpers"
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

from e1_tinyfat import (  # noqa: E402
    DEFAULT_E1_ROOT,
    DEFAULT_SAMPLE_JSON,
    DEFAULT_TARGET_BRANCHES,
    build_job_manifest,
    derive_partition_weights_from_probe,
    load_sample_anatomies,
    parse_partition_weights,
    probe_cluster_partitions,
    target_equivalence_report,
    write_partition_bucket_manifests,
    write_targets_json,
    write_wires_json,
    build_sbatch_script,
)


def make_default_service():
    try:
        from steve_recommender.eval_v2.service import DefaultEvaluationService
    except Exception:
        return None
    return DefaultEvaluationService()


def _wire_to_dict(wire: object) -> dict[str, str]:
    if isinstance(wire, dict):
        model = str(wire.get("model", ""))
        wire_name = str(wire.get("wire", ""))
        tool_ref = str(wire.get("tool_ref", "")) or f"{model}/{wire_name}"
    else:
        model = str(getattr(wire, "model", ""))
        wire_name = str(getattr(wire, "wire", ""))
        tool_ref = str(getattr(wire, "tool_ref", "")) or f"{model}/{wire_name}"
    return {"model": model, "wire": wire_name, "tool_ref": tool_ref}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TinyFat E1 job manifests and Slurm scripts")
    parser.add_argument("--sample-json", type=Path, default=DEFAULT_SAMPLE_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_E1_ROOT)
    parser.add_argument("--scripts-root", type=Path, default=PROJECT_ROOT / "scripts")
    parser.add_argument("--seed-base-start", type=int, default=123)
    parser.add_argument("--target-seed-start", type=int, default=9000)
    parser.add_argument("--target-count-per-anatomy", type=int, default=3)
    parser.add_argument("--target-branches", default=",".join(DEFAULT_TARGET_BRANCHES))
    parser.add_argument("--partitions", default=None)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--worker-count", type=int, default=None)
    parser.add_argument("--friction", type=float, default=0.1)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--trial-count", type=int, default=None)
    parser.add_argument("--config-id", type=int, default=None, help="Optional single config to materialize")
    parser.add_argument("--probe-cluster", action="store_true")
    write_trace_group = parser.add_mutually_exclusive_group()
    write_trace_group.add_argument(
        "--write-trace",
        dest="write_full_trace",
        action="store_true",
        help="Enable per-trial HDF5 trace files for generated jobs.",
    )
    write_trace_group.add_argument(
        "--no-write-trace",
        dest="write_full_trace",
        action="store_false",
        help="Disable per-trial HDF5 trace files for generated jobs.",
    )
    parser.add_argument(
        "--write-diagnostics",
        action="store_true",
        default=False,
        help="Write optional diagnostic datasets into trial trace files.",
    )
    parser.set_defaults(write_full_trace=True)
    return parser.parse_args(argv)


def _load_wires(service=None) -> list[dict[str, str]]:
    if service is not None:
        return [_wire_to_dict(wire) for wire in service.list_startable_wires()]
    registry_path = PROJECT_ROOT / "data" / "wire_registry" / "index.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    wires: list[dict[str, str]] = []
    for tool_ref in payload.get("agents", {}).keys():
        if "/" not in tool_ref:
            continue
        model, wire = tool_ref.split("/", maxsplit=1)
        wires.append({"model": model, "wire": wire, "tool_ref": tool_ref})
    if not wires:
        raise RuntimeError(f"No wires found in {registry_path}")
    return wires


def _materialize_scripts(
    *,
    project_root: Path,
    output_root: Path,
    jobs_by_partition: dict[str, list[dict[str, object]]],
    scripts_root: Path,
    walltime: str,
) -> dict[str, Path]:
    script_paths: dict[str, Path] = {}
    metadata_root = output_root / "metadata"
    logs_root = output_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    scripts_root.mkdir(parents=True, exist_ok=True)
    for partition, rows in jobs_by_partition.items():
        bucket_manifest = metadata_root / f"job_manifest_{partition}.json"
        if not bucket_manifest.exists():
            continue
        script_path = scripts_root / f"e1_tinyfat_{partition}.sbatch"
        script_text = build_sbatch_script(
            project_root=project_root,
            partition=partition,
            jobs_manifest_path=bucket_manifest,
            logs_root=logs_root,
            walltime=walltime,
        )
        script_path.write_text(script_text, encoding="utf-8")
        script_path.chmod(0o755)
        script_paths[partition] = script_path
    return script_paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    service = make_default_service()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_root = output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    scripts_root = Path(args.scripts_root).resolve()

    target_branches = tuple(str(item).strip() for item in args.target_branches.split(",") if item.strip())
    partition_weights = (("work", 1.0),)
    if args.partitions:
        partition_weights = parse_partition_weights(args.partitions)
    targets_json = metadata_root / "targets.json"
    targets_payload = write_targets_json(
        sample_json=args.sample_json,
        output_path=targets_json,
        target_seed_start=args.target_seed_start,
        target_count_per_anatomy=args.target_count_per_anatomy,
        target_branches=target_branches,
    )

    wires = _load_wires(service)
    write_wires_json(output_path=metadata_root / "wires.json", wires=wires)

    probe_rows = ()
    if args.probe_cluster:
        try:
            probe_rows = probe_cluster_partitions()
        except Exception:
            probe_rows = ()
    if args.partitions is None:
        partition_weights = (("work", 1.0),)

    manifest = build_job_manifest(
        sample_json=args.sample_json,
        targets_json=targets_json,
        output_root=output_root,
        seed_base_start=args.seed_base_start,
        target_seed_start=args.target_seed_start,
        target_count_per_anatomy=args.target_count_per_anatomy,
        max_episode_steps=args.max_episode_steps,
        partitions=args.partitions,
        partition_weights=partition_weights if partition_weights else None,
        probe_rows=probe_rows,
        worker_count=args.worker_count,
        walltime=args.walltime,
        friction=float(args.friction),
        write_full_trace=bool(args.write_full_trace),
        write_diagnostics=bool(args.write_diagnostics),
    )
    if args.trial_count is not None:
        trial_count = int(args.trial_count)
        manifest["configs"] = {
            str(config_id): {**spec, "trial_count": trial_count}
            for config_id, spec in manifest["configs"].items()
        }
        for job in manifest["jobs"]:
            job["config_spec"] = {**job["config_spec"], "trial_count": trial_count}
        for job in manifest["jobs"]:
            if "trial_count" in job.get("execution", {}):
                job["execution"]["trials_per_candidate"] = trial_count
        manifest["max_episode_steps"] = int(args.max_episode_steps)
        (metadata_root / "job_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        (metadata_root / "job_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (metadata_root / "anatomies.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sample_json": str(Path(args.sample_json).resolve()),
                "selected_anatomies": list(load_sample_anatomies(args.sample_json)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (metadata_root / "experiment_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "configs": {str(key): value for key, value in manifest["configs"].items()},
                "partition_weights": list(partition_weights),
                "target_branches": list(target_branches),
                "target_seed_start": args.target_seed_start,
                "seed_base_start": args.seed_base_start,
                "target_count_per_anatomy": args.target_count_per_anatomy,
                "trial_count_override": args.trial_count,
                "max_episode_steps": args.max_episode_steps,
                "default_walltime": args.walltime,
                "default_worker_count": args.worker_count,
                "default_friction": float(args.friction),
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
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    jobs_by_partition: dict[str, list[dict[str, object]]] = {}
    for job in manifest["jobs"]:
        jobs_by_partition.setdefault(str(job["partition"]), []).append(job)

    bucket_manifests = write_partition_bucket_manifests(
        output_root=output_root,
        manifest=manifest,
    )
    script_paths = _materialize_scripts(
        project_root=PROJECT_ROOT,
        output_root=output_root,
        jobs_by_partition=jobs_by_partition,
        scripts_root=scripts_root,
        walltime=args.walltime,
    )

    print(f"[E1-TF] output_root={output_root}")
    print(f"[E1-TF] metadata_root={metadata_root}")
    print(f"[E1-TF] n_jobs={len(manifest['jobs'])}")
    print(f"[E1-TF] partitions={sorted(bucket_manifests)}")
    print(f"[E1-TF] bucket_manifests={ {k: str(v) for k, v in bucket_manifests.items()} }")
    print(f"[E1-TF] script_paths={ {k: str(v) for k, v in script_paths.items()} }")
    print(f"[E1-TF] wires={len(wires)}")
    print(f"[E1-TF] targets={len(targets_payload.get('selected_anatomies', []))}")
    report = target_equivalence_report(manifest)
    print(
        json.dumps(
            {
                "missing_scripts": [str(path) for path in script_paths.values() if not path.exists()],
                "n_jobs": len(manifest["jobs"]),
                "n_scripts": len(script_paths),
                "output_root": str(output_root),
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
                "target_report": report,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["same_targets_across_configs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
