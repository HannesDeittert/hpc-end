from __future__ import annotations

import csv
import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE_JSON = PROJECT_ROOT / "results" / "experimental_prep" / "sample_12.json"
DEFAULT_E1_ROOT = PROJECT_ROOT / "results" / "master_thesis" / "e1"
DEFAULT_E1_METADATA_ROOT = DEFAULT_E1_ROOT / "metadata"
DEFAULT_E1_RUNS_ROOT = DEFAULT_E1_ROOT / "runs"
DEFAULT_E1_LOGS_ROOT = DEFAULT_E1_ROOT / "logs"

CONFIGS = {
    1: {
        "policy_mode": "deterministic",
        "stochastic_environment_mode": "fixed_start",
        "trial_count": 1,
    },
    2: {
        "policy_mode": "deterministic",
        "stochastic_environment_mode": "random_start",
        "trial_count": 1000,
    },
    3: {
        "policy_mode": "stochastic",
        "stochastic_environment_mode": "fixed_start",
        "trial_count": 1000,
    },
    4: {
        "policy_mode": "stochastic",
        "stochastic_environment_mode": "random_start",
        "trial_count": 1000,
    },
}

PARTITION_PROFILES = {
    "work": {"partition": "work", "gres": "gpu:4", "cpus_per_task": 32},
    "rtx3080": {"partition": "rtx3080", "gres": "gpu:rtx3080:4", "cpus_per_task": 32},
    "v100": {"partition": "v100", "gres": "gpu:v100:4", "cpus_per_task": 32},
    "a100": {"partition": "a100", "gres": "gpu:a100:1", "cpus_per_task": 32},
}

DEFAULT_TARGET_BRANCHES = ("bct", "lcca", "lsa")
DEFAULT_WORKER_COUNT = 29
DEFAULT_WALLTIME = "24:00:00"


@dataclass(frozen=True)
class PartitionProbeRow:
    partition: str
    nodes_total: int
    states: dict[str, int]
    gres: tuple[str, ...]
    cpus_per_node: int | None = None

    @property
    def idle_nodes(self) -> int:
        return self.states.get("idle", 0)

    @property
    def mix_nodes(self) -> int:
        return self.states.get("mix", 0)


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_csv_list(text: str) -> tuple[str, ...]:
    parts = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not parts:
        raise ValueError("partition list must not be empty")
    return parts


def load_sample_anatomies(sample_json: Path) -> tuple[dict[str, Any], ...]:
    payload = _load_json(Path(sample_json))
    selected = payload.get("selected_anatomies", [])
    if not selected:
        raise ValueError(f"No selected_anatomies found in {sample_json}")
    return tuple(dict(item) for item in selected)


def list_anatomy_record_ids(sample_json: Path) -> tuple[str, ...]:
    return tuple(item["record_id"] for item in load_sample_anatomies(sample_json))


def sample_target_specs(
    *,
    sample_json: Path,
    target_seed_start: int = 9000,
    target_count_per_anatomy: int = 3,
    target_branches: Sequence[str] = DEFAULT_TARGET_BRANCHES,
) -> dict[str, Any]:
    sample_json = Path(sample_json).resolve()
    anatomies = load_sample_anatomies(sample_json)
    records: list[dict[str, Any]] = []
    seed_offset = 0
    for anatomy_index, anatomy in enumerate(anatomies):
        anatomy_record_id = str(anatomy["record_id"])
        branch_pool = tuple(str(branch) for branch in target_branches)
        target_entries: list[dict[str, Any]] = []
        for target_index in range(int(target_count_per_anatomy)):
            target_seed = int(target_seed_start + seed_offset)
            rng = random.Random(target_seed)
            branch = rng.choice(branch_pool)
            # Deterministic but anatomy-independent slot seed. The exact centerline
            # index is sampled later by eval_v2 from this stable seed and branch set.
            target_entries.append(
                {
                    "target_index": target_index,
                    "kind": "centerline_random",
                    "target_seed": target_seed,
                    "threshold_mm": 5.0,
                    "branches": list(branch_pool),
                    "selected_branch_hint": branch,
                    "selected_anatomy_record_id": anatomy_record_id,
                    "selected_anatomy_seed": int(anatomy.get("seed", 0)),
                }
            )
            seed_offset += 1
        records.append(
            {
                "record_id": anatomy_record_id,
                "arch_type": anatomy.get("arch_type"),
                "anatomy_seed": int(anatomy.get("seed", 0)),
                "targets": target_entries,
            }
        )
    return {
        "schema_version": 1,
        "sample_json": str(sample_json),
        "target_seed_start": int(target_seed_start),
        "target_count_per_anatomy": int(target_count_per_anatomy),
        "target_branches": list(target_branches),
        "selected_anatomies": records,
    }


def write_targets_json(
    *,
    sample_json: Path,
    output_path: Path,
    target_seed_start: int = 9000,
    target_count_per_anatomy: int = 3,
    target_branches: Sequence[str] = DEFAULT_TARGET_BRANCHES,
) -> dict[str, Any]:
    payload = sample_target_specs(
        sample_json=sample_json,
        target_seed_start=target_seed_start,
        target_count_per_anatomy=target_count_per_anatomy,
        target_branches=target_branches,
    )
    _dump_json(Path(output_path), payload)
    return payload


def load_targets_json(targets_json: Path) -> dict[str, Any]:
    return dict(_load_json(Path(targets_json)))


def partition_profile(partition: str) -> dict[str, Any]:
    return dict(PARTITION_PROFILES.get(partition, PARTITION_PROFILES["work"]))


def parse_partition_weights(text: str | None) -> tuple[tuple[str, float], ...]:
    if text is None:
        return ()
    items = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not items:
        return ()
    parsed: list[tuple[str, float]] = []
    for item in items:
        if ":" not in item:
            parsed.append((item, 1.0))
            continue
        name, weight = item.split(":", maxsplit=1)
        parsed.append((name.strip(), float(weight)))
    if sum(weight for _name, weight in parsed) <= 0:
        raise ValueError("partition weights must sum to a positive value")
    return tuple(parsed)


def probe_cluster_partitions(
    *,
    sinfo_output: str | None = None,
) -> tuple[PartitionProbeRow, ...]:
    if sinfo_output is None:
        commands = (
            ["sinfo.tinygpu", "--noheader", "--format=%P|%D|%t|%G|%c"],
            ["sinfo", "--clusters=tinygpu", "--noheader", "--format=%P|%D|%t|%G|%c"],
            ["sinfo", "--noheader", "--format=%P|%D|%t|%G|%c"],
        )
        last_error: Exception | None = None
        for command in commands:
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                sinfo_output = result.stdout
                break
            except Exception as exc:  # pragma: no cover - fallback path is environment-specific
                last_error = exc
        else:
            raise RuntimeError("Unable to probe Slurm partitions") from last_error

    aggregated: dict[str, dict[str, Any]] = {}
    for raw_line in sinfo_output.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        partition_raw, nodes_raw, state_raw, gres_raw, cpus_raw = line.split("|", maxsplit=4)
        partition = partition_raw.rstrip("*")
        data = aggregated.setdefault(
            partition,
            {"nodes_total": 0, "states": {}, "gres": set(), "cpus_per_node": None},
        )
        nodes = int(nodes_raw)
        data["nodes_total"] += nodes
        data["states"][state_raw.lower()] = data["states"].get(state_raw.lower(), 0) + nodes
        if gres_raw and gres_raw != "N/A":
            data["gres"].add(gres_raw)
        if cpus_raw and cpus_raw.isdigit():
            cpus = int(cpus_raw)
            data["cpus_per_node"] = cpus if data["cpus_per_node"] is None else max(data["cpus_per_node"], cpus)

    rows = []
    for partition, data in sorted(aggregated.items()):
        rows.append(
            PartitionProbeRow(
                partition=partition,
                nodes_total=int(data["nodes_total"]),
                states=dict(data["states"]),
                gres=tuple(sorted(data["gres"])),
                cpus_per_node=data["cpus_per_node"],
            )
        )
    return tuple(rows)


def derive_partition_weights_from_probe(
    probe_rows: Sequence[PartitionProbeRow],
) -> tuple[tuple[str, float], ...]:
    if not probe_rows:
        return (("work", 1.0),)

    weighted: list[tuple[str, float]] = []
    for row in probe_rows:
        score = float(row.idle_nodes) + 0.5 * float(row.mix_nodes)
        if score > 0:
            weighted.append((row.partition, score))
    if not weighted:
        return (("work", 1.0),)
    total = sum(weight for _partition, weight in weighted)
    return tuple((partition, weight / total) for partition, weight in weighted)


def choose_partition_assignment(
    *,
    job_count: int,
    weights: tuple[tuple[str, float], ...] | None = None,
    probe_rows: Sequence[PartitionProbeRow] = (),
) -> tuple[str, ...]:
    if job_count < 1:
        raise ValueError("job_count must be positive")
    if not weights:
        return tuple("work" for _ in range(job_count))

    known = {row.partition for row in probe_rows} if probe_rows else set(PARTITION_PROFILES)
    filtered = [(name, weight) for name, weight in weights if name in known]
    if not filtered:
        filtered = [("work", 1.0)]

    total_weight = sum(weight for _name, weight in filtered)
    raw = [job_count * weight / total_weight for _name, weight in filtered]
    counts = [int(math.floor(value)) for value in raw]
    remainder = job_count - sum(counts)
    order = sorted(
        range(len(filtered)),
        key=lambda idx: (raw[idx] - counts[idx], filtered[idx][0]),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1

    assignments: list[str] = []
    for (name, _weight), count in zip(filtered, counts):
        assignments.extend([name] * count)
    while len(assignments) < job_count:
        assignments.append(filtered[0][0])
    return tuple(assignments[:job_count])


def config_spec(config_id: int) -> dict[str, Any]:
    try:
        return dict(CONFIGS[int(config_id)])
    except KeyError as exc:
        raise KeyError(f"Unknown config id: {config_id}") from exc


def build_execution_plan(
    *,
    config_id: int,
    seed_base: int,
    trial_count: int,
    max_episode_steps: int,
    worker_count: int = DEFAULT_WORKER_COUNT,
) -> dict[str, Any]:
    spec = config_spec(config_id)
    policy_mode = spec["policy_mode"]
    stochastic_environment_mode = spec["stochastic_environment_mode"]
    policy_base_seed = int(seed_base) + 100_000
    if policy_mode == "deterministic":
        policy_base_seed = int(seed_base) + 1_000_000
    return {
        "trials_per_candidate": int(trial_count),
        "base_seed": int(seed_base),
        "policy_base_seed": int(policy_base_seed),
        "max_episode_steps": int(max_episode_steps),
        "policy_mode": policy_mode,
        "stochastic_environment_mode": stochastic_environment_mode,
        "worker_count": int(worker_count),
    }


def build_job_manifest(
    *,
    sample_json: Path,
    targets_json: Path,
    output_root: Path,
    seed_base_start: int = 123,
    target_seed_start: int = 9000,
    target_count_per_anatomy: int = 3,
    max_episode_steps: int = 1000,
    partitions: str | None = None,
    partition_weights: Sequence[tuple[str, float]] | None = None,
    probe_rows: Sequence[PartitionProbeRow] = (),
    worker_count: int = DEFAULT_WORKER_COUNT,
    walltime: str = DEFAULT_WALLTIME,
) -> dict[str, Any]:
    sample_json = Path(sample_json).resolve()
    targets_json = Path(targets_json).resolve()
    output_root = Path(output_root).resolve()

    anatomies = load_sample_anatomies(sample_json)
    targets = load_targets_json(targets_json)
    target_rows = {
        str(row["record_id"]): tuple(row["targets"])
        for row in targets.get("selected_anatomies", [])
    }
    if partition_weights is None:
        partition_weights = parse_partition_weights(partitions)
    job_rows: list[dict[str, Any]] = []
    for config_id in sorted(CONFIGS):
        for anatomy_index, anatomy in enumerate(anatomies):
            anatomy_id = str(anatomy["record_id"])
            for target_index in range(int(target_count_per_anatomy)):
                seed_base = int(seed_base_start + anatomy_index * 100 + target_index)
                target_seed = int(target_seed_start + anatomy_index * target_count_per_anatomy + target_index)
                target_spec = target_rows[anatomy_id][target_index]
                job_rows.append(
                    {
                        "config_id": int(config_id),
                        "config_spec": {
                            **config_spec(config_id),
                            "max_episode_steps": int(max_episode_steps),
                        },
                        "anatomy_id": anatomy_id,
                        "anatomy_index": anatomy_index,
                        "target_index": target_index,
                        "seed_base": seed_base,
                        "target_seed": target_seed,
                        "target_spec": target_spec,
                        "job_name": f"cfg{config_id}__{anatomy_id}__target{target_index}__seed{seed_base}",
                        "output_dir": str(
                            output_root
                            / "runs"
                            / f"config_{config_id}"
                            / f"{anatomy_id}__target_{target_index}__seedbase_{seed_base}"
                        ),
                    }
                )

    assignments = choose_partition_assignment(
        job_count=len(job_rows),
        weights=partition_weights,
        probe_rows=probe_rows,
    )
    for job_row, partition in zip(job_rows, assignments):
        profile = partition_profile(partition)
        job_row["partition"] = partition
        job_row["gres"] = profile["gres"]
        job_row["cpus_per_task"] = int(profile["cpus_per_task"])
        job_row["worker_count"] = int(worker_count)
        job_row["walltime"] = str(walltime)

    manifest = {
        "schema_version": 1,
        "sample_json": str(sample_json),
        "targets_json": str(targets_json),
        "output_root": str(output_root),
        "seed_base_start": int(seed_base_start),
        "target_seed_start": int(target_seed_start),
        "target_count_per_anatomy": int(target_count_per_anatomy),
        "configs": {str(key): value for key, value in CONFIGS.items()},
        "partition_weights": list(partition_weights if partition_weights is not None else parse_partition_weights(partitions)),
        "worker_count": int(worker_count),
        "walltime": str(walltime),
        "max_episode_steps": int(max_episode_steps),
        "jobs": job_rows,
    }
    return manifest


def write_manifest_bundle(
    *,
    sample_json: Path,
    output_root: Path,
    target_seed_start: int = 9000,
    target_count_per_anatomy: int = 3,
    seed_base_start: int = 123,
    partitions: str | None = None,
    target_branches: Sequence[str] = DEFAULT_TARGET_BRANCHES,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    metadata_root = output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    targets_json = metadata_root / "targets.json"
    targets_payload = write_targets_json(
        sample_json=sample_json,
        output_path=targets_json,
        target_seed_start=target_seed_start,
        target_count_per_anatomy=target_count_per_anatomy,
        target_branches=target_branches,
    )

    manifest = build_job_manifest(
        sample_json=sample_json,
        targets_json=targets_json,
        output_root=output_root,
        seed_base_start=seed_base_start,
        target_seed_start=target_seed_start,
        target_count_per_anatomy=target_count_per_anatomy,
        partitions=partitions,
    )
    _dump_json(metadata_root / "job_manifest.json", manifest)
    _dump_json(
        metadata_root / "experiment_config.json",
        {
            "schema_version": 1,
            "configs": CONFIGS,
            "default_worker_count": DEFAULT_WORKER_COUNT,
            "default_walltime": DEFAULT_WALLTIME,
            "partition_profiles": PARTITION_PROFILES,
            "target_branches": list(target_branches),
            "target_seed_start": int(target_seed_start),
            "seed_base_start": int(seed_base_start),
            "target_count_per_anatomy": int(target_count_per_anatomy),
        },
    )
    _dump_json(metadata_root / "anatomies.json", {"selected_anatomies": load_sample_anatomies(sample_json)})
    return {
        "targets_json": targets_json,
        "manifest": manifest,
        "metadata_root": metadata_root,
        "targets_payload": targets_payload,
    }


def write_partition_bucket_manifests(
    *,
    output_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    metadata_root = Path(output_root).resolve() / "metadata"
    jobs = list(manifest.get("jobs", ()))
    bucket_paths: dict[str, Path] = {}
    buckets: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        buckets.setdefault(str(job.get("partition", "work")), []).append(job)
    for partition, rows in buckets.items():
        bucket_path = metadata_root / f"job_manifest_{partition}.json"
        _dump_json(
            bucket_path,
            {
                "schema_version": 1,
                "partition": partition,
                "jobs": rows,
            },
        )
        bucket_paths[partition] = bucket_path
    return bucket_paths


def write_wires_json(*, output_path: Path, wires: Sequence[Any]) -> Path:
    payload = {
        "schema_version": 1,
        "wires": [
            {
                "model": wire.get("model") if isinstance(wire, dict) else getattr(wire, "model", None),
                "wire": wire.get("wire") if isinstance(wire, dict) else getattr(wire, "wire", None),
                "tool_ref": (
                    wire.get("tool_ref")
                    if isinstance(wire, dict)
                    else getattr(wire, "tool_ref", None)
                )
                or (
                    f"{wire.get('model', '')}/{wire.get('wire', '')}"
                    if isinstance(wire, dict)
                    else f"{getattr(wire, 'model', '')}/{getattr(wire, 'wire', '')}"
                ),
            }
            for wire in wires
        ],
    }
    _dump_json(output_path, payload)
    return Path(output_path)


def target_equivalence_report(manifest: dict[str, Any]) -> dict[str, Any]:
    jobs = list(manifest.get("jobs", ()))
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for job in jobs:
        key = (str(job.get("anatomy_id", "")), int(job.get("target_index", 0)))
        grouped.setdefault(key, []).append(job)

    mismatches: list[dict[str, Any]] = []
    for (anatomy_id, target_index), rows in sorted(grouped.items()):
        canonical = json.dumps(rows[0].get("target_spec"), sort_keys=True)
        configs_seen = []
        for row in rows:
            target_spec = json.dumps(row.get("target_spec"), sort_keys=True)
            configs_seen.append(int(row.get("config_id", 0)))
            if target_spec != canonical:
                mismatches.append(
                    {
                        "anatomy_id": anatomy_id,
                        "target_index": target_index,
                        "config_id": int(row.get("config_id", 0)),
                    }
                )
    return {
        "n_jobs": len(jobs),
        "n_groups": len(grouped),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "same_targets_across_configs": len(mismatches) == 0,
    }


def build_sbatch_script(
    *,
    project_root: Path,
    partition: str,
    jobs_manifest_path: Path,
    logs_root: Path,
    walltime: str,
    gres: str,
    cpus_per_task: int,
) -> str:
    project_root = Path(project_root).resolve()
    jobs_manifest_path = Path(jobs_manifest_path).resolve()
    logs_root = Path(logs_root).resolve()
    job_count = len(_load_json(jobs_manifest_path).get("jobs", []))
    return f"""#!/bin/bash -l
#SBATCH --job-name=e1_{partition}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={walltime}
#SBATCH --array=0-{max(job_count - 1, 0)}
#SBATCH --output={logs_root}/slurm-%A_%a.out
#SBATCH --error={logs_root}/slurm-%A_%a.err
#SBATCH --export=NONE

set -euo pipefail
unset SLURM_EXPORT_ENV

cd {project_root}
module load python
eval "$(conda shell.bash hook)"
conda activate "${{E1_CONDA_ENV:-/home/woody/iwhr/iwhr106h/conda/envs/master-project}}"
export SOFA_ROOT="${{SOFA_ROOT:-/home/woody/iwhr/iwhr106h/opt/SOFA_v23.06.00_Linux}}"
source scripts/sofa_env.sh
export STEVE_WALL_FORCE_MONITOR_PLUGIN="${{STEVE_WALL_FORCE_MONITOR_PLUGIN:-$PWD/native/sofa_wire_force_monitor/build/libSofaWireForceMonitor.so}}"
export PYTHON_BIN="${{PYTHON_BIN:-python3}}"
export E1_JOB_MANIFEST="{jobs_manifest_path}"

exec "$PYTHON_BIN" experiments/master-thesis/run_e1_cell.py --manifest "$E1_JOB_MANIFEST" --array-index "${{SLURM_ARRAY_TASK_ID}}"
"""
