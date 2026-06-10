#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
HELPER_DIR = PROJECT_ROOT / "experiments" / "master-thesis" / "notebook_helpers"
for path in (PROJECT_ROOT, HELPER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from e1 import load_sample_anatomies, write_wires_json  # type: ignore  # noqa: E402
from e1_tinyfat import (  # type: ignore  # noqa: E402
    build_job_manifest,
    build_sbatch_script,
    target_equivalence_report,
    write_partition_bucket_manifests,
)


DEFAULT_BRANCHES = ("bct", "rcca", "rsa", "lcca", "lsa")


def _parse_csv(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("branch list must not be empty")
    return values


def _load_registry_wires(project_root: Path) -> list[dict[str, str]]:
    registry_path = project_root / "data" / "wire_registry" / "index.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    wires: list[dict[str, str]] = []
    for tool_ref in payload.get("agents", {}).keys():
        if "/" not in str(tool_ref):
            continue
        model, wire = str(tool_ref).split("/", maxsplit=1)
        wires.append({"model": model, "wire": wire, "tool_ref": str(tool_ref)})
    if not wires:
        raise RuntimeError(f"No wires found in {registry_path}")
    return wires


def _write_branch_targets(
    *,
    sample_json: Path,
    targets_json: Path,
    branches: Sequence[str],
    target_seed_start: int,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    target_offset = 0
    for anatomy in load_sample_anatomies(sample_json):
        anatomy_id = str(anatomy["record_id"])
        targets: list[dict[str, object]] = []
        for target_index, branch in enumerate(branches):
            target_seed = int(target_seed_start) + target_offset
            targets.append(
                {
                    "target_index": int(target_index),
                    "kind": "centerline_random",
                    "target_seed": int(target_seed),
                    "threshold_mm": 5.0,
                    "branches": [str(branch)],
                    "selected_branch_hint": str(branch),
                    "selected_anatomy_record_id": anatomy_id,
                    "selected_anatomy_seed": int(anatomy.get("seed", 0)),
                }
            )
            target_offset += 1
        records.append(
            {
                "record_id": anatomy_id,
                "arch_type": anatomy.get("arch_type"),
                "anatomy_seed": int(anatomy.get("seed", 0)),
                "targets": targets,
            }
        )

    payload: dict[str, object] = {
        "schema_version": 1,
        "sample_json": str(sample_json.resolve()),
        "target_rule": "one_centerline_random_target_per_non_aorta_branch",
        "target_branches": list(branches),
        "target_seed_start": int(target_seed_start),
        "target_count_per_anatomy": len(tuple(branches)),
        "selected_anatomies": records,
    }
    targets_json.parent.mkdir(parents=True, exist_ok=True)
    targets_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _patch_sbatch_text(*, text: str, partition: str) -> str:
    return text.replace(
        f"#SBATCH --job-name=e1_tinyfat_{partition}",
        f"#SBATCH --job-name=e1_01_all5_{partition}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare TinyFat E1 friction 0.1 jobs for all five target branches.")
    parser.add_argument("--sample-json", type=Path, default=PROJECT_ROOT / "results" / "experimental_prep" / "sample_12_e1.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "master_thesis" / "e1_tinyfat_friction_01_all5branches")
    parser.add_argument("--scripts-root", type=Path, default=PROJECT_ROOT / "scripts")
    parser.add_argument("--branches", default=",".join(DEFAULT_BRANCHES))
    parser.add_argument("--seed-base-start", type=int, default=123)
    parser.add_argument("--target-seed-start", type=int, default=9000)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--friction", type=float, default=0.1)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output root exists and is not empty: {output_root}")

    metadata_root = output_root / "metadata"
    logs_root = output_root / "logs"
    metadata_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    branches = _parse_csv(args.branches)
    targets_payload = _write_branch_targets(
        sample_json=Path(args.sample_json).resolve(),
        targets_json=metadata_root / "targets.json",
        branches=branches,
        target_seed_start=int(args.target_seed_start),
    )
    wires = _load_registry_wires(PROJECT_ROOT)
    wires_json = metadata_root / "wires.json"
    write_wires_json(output_path=wires_json, wires=wires)

    manifest = build_job_manifest(
        sample_json=Path(args.sample_json).resolve(),
        targets_json=metadata_root / "targets.json",
        output_root=output_root,
        seed_base_start=int(args.seed_base_start),
        target_seed_start=int(args.target_seed_start),
        target_count_per_anatomy=len(branches),
        max_episode_steps=int(args.max_episode_steps),
        partitions=None,
        partition_weights=(("work", 1.0),),
        probe_rows=(),
        worker_count=None,
        walltime=str(args.walltime),
        friction=float(args.friction),
        write_full_trace=False,
        write_diagnostics=False,
    )
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
                "partition_weights": [("work", 1.0)],
                "target_rule": targets_payload["target_rule"],
                "target_branches": list(branches),
                "target_seed_start": int(args.target_seed_start),
                "seed_base_start": int(args.seed_base_start),
                "target_count_per_anatomy": len(branches),
                "trial_count_override": None,
                "max_episode_steps": int(args.max_episode_steps),
                "default_walltime": str(args.walltime),
                "default_worker_count": None,
                "default_friction": float(args.friction),
                "write_full_trace": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    bucket_paths = write_partition_bucket_manifests(output_root=output_root, manifest=manifest)
    script_paths: dict[str, Path] = {}
    Path(args.scripts_root).mkdir(parents=True, exist_ok=True)
    for partition, bucket_path in bucket_paths.items():
        script_path = Path(args.scripts_root).resolve() / f"e1_tinyfat_friction_01_all5branches_{partition}.sbatch"
        script_text = build_sbatch_script(
            project_root=PROJECT_ROOT,
            partition=partition,
            jobs_manifest_path=bucket_path,
            logs_root=logs_root,
            walltime=str(args.walltime),
        )
        script_text = _patch_sbatch_text(text=script_text, partition=partition)
        script_path.write_text(script_text, encoding="utf-8")
        script_path.chmod(0o755)
        (metadata_root / script_path.name).write_text(script_text, encoding="utf-8")
        script_paths[partition] = script_path

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "n_jobs": len(manifest["jobs"]),
                "branches": list(branches),
                "targets_per_anatomy": len(branches),
                "wires": len(wires),
                "bucket_paths": {key: str(value) for key, value in bucket_paths.items()},
                "script_paths": {key: str(value) for key, value in script_paths.items()},
                "target_report": target_equivalence_report(manifest),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
