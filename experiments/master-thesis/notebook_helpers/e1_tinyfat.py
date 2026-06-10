from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from e1 import (  # type: ignore
    CONFIGS,
    DEFAULT_TARGET_BRANCHES,
    build_execution_plan,
    config_spec,
    load_sample_anatomies,
    load_targets_json,
    parse_partition_weights,
    target_equivalence_report,
    write_targets_json,
    write_wires_json,
)


DEFAULT_SAMPLE_JSON = PROJECT_ROOT / "results" / "experimental_prep" / "sample_12_e1.json"
DEFAULT_E1_ROOT = PROJECT_ROOT / "results" / "master_thesis" / "e1_tinyfat"
DEFAULT_E1_METADATA_ROOT = DEFAULT_E1_ROOT / "metadata"
DEFAULT_E1_RUNS_ROOT = DEFAULT_E1_ROOT / "runs"
DEFAULT_E1_LOGS_ROOT = DEFAULT_E1_ROOT / "logs"

PARTITION_PROFILES = {
    "work": {
        "partition": "work",
        "gres": "",
        "cpus_per_task": 32,
        "worker_count": 60,
        "hint": "multithread",
    },
    "broadwell256": {
        "partition": "broadwell256",
        "gres": "",
        "cpus_per_task": 12,
        "worker_count": 9,
        "hint": "nomultithread",
    },
    "broadwell512": {
        "partition": "broadwell512",
        "gres": "",
        "cpus_per_task": 28,
        "worker_count": 25,
        "hint": "nomultithread",
    },
    "long256": {
        "partition": "long256",
        "gres": "",
        "cpus_per_task": 12,
        "worker_count": 9,
        "hint": "nomultithread",
    },
}

DEFAULT_WORKER_COUNT = 60
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


def partition_profile(partition: str) -> dict[str, Any]:
    return dict(PARTITION_PROFILES.get(partition, PARTITION_PROFILES["work"]))


def probe_cluster_partitions(
    *,
    sinfo_output: str | None = None,
) -> tuple[PartitionProbeRow, ...]:
    if sinfo_output is None:
        commands = (
            ["sinfo.tinyfat", "--noheader", "--format=%P|%D|%t|%G|%c"],
            ["sinfo", "--clusters=tinyfat", "--noheader", "--format=%P|%D|%t|%G|%c"],
            ["sinfo", "--noheader", "--format=%P|%D|%t|%G|%c"],
        )
        last_error: Exception | None = None
        for command in commands:
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                sinfo_output = result.stdout
                break
            except Exception as exc:  # pragma: no cover - environment-specific fallback
                last_error = exc
        else:
            raise RuntimeError("Unable to probe TinyFat Slurm partitions") from last_error

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
    worker_count: int | None = None,
    walltime: str = DEFAULT_WALLTIME,
    friction: float = 0.1,
    write_full_trace: bool = True,
    write_diagnostics: bool = False,
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
                        "write_full_trace": bool(write_full_trace),
                        "write_diagnostics": bool(write_diagnostics),
                        "anatomy_id": anatomy_id,
                        "anatomy_index": anatomy_index,
                        "target_index": target_index,
                        "seed_base": seed_base,
                        "target_seed": target_seed,
                        "target_spec": target_spec,
                        "friction": float(friction),
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
        job_row["gres"] = profile.get("gres", "")
        job_row["cpus_per_task"] = int(profile["cpus_per_task"])
        job_row["worker_count"] = int(worker_count if worker_count is not None else profile["worker_count"])
        job_row["walltime"] = str(walltime)
        job_row["hint"] = str(profile.get("hint", "nomultithread"))

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
        "worker_count": int(worker_count if worker_count is not None else DEFAULT_WORKER_COUNT),
        "walltime": str(walltime),
        "friction": float(friction),
        "write_full_trace": bool(write_full_trace),
        "write_diagnostics": bool(write_diagnostics),
        "max_episode_steps": int(max_episode_steps),
        "jobs": job_rows,
    }
    return manifest


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


def build_sbatch_script(
    *,
    project_root: Path,
    partition: str,
    jobs_manifest_path: Path,
    logs_root: Path,
    walltime: str,
) -> str:
    project_root = Path(project_root).resolve()
    jobs_manifest_path = Path(jobs_manifest_path).resolve()
    logs_root = Path(logs_root).resolve()
    payload = _load_json(jobs_manifest_path)
    jobs = payload.get("jobs", [])
    job_count = len(jobs)
    first_row = dict(jobs[0]) if jobs else {}
    cpus_per_task = int(first_row.get("cpus_per_task", partition_profile(partition)["cpus_per_task"]))
    worker_count = int(first_row.get("worker_count", partition_profile(partition)["worker_count"]))
    hint = str(first_row.get("hint", "nomultithread"))
    friction = float(first_row.get("friction", payload.get("friction", 0.1)))
    trace_flag = "--write-trace" if bool(first_row.get("write_full_trace", True)) else "--no-write-trace"
    diagnostics_flag = " --write-diagnostics" if bool(first_row.get("write_diagnostics", False)) else ""
    job_name = f"e1_tinyfat_{partition}"
    return f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --hint={hint}
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export E1_JOB_MANIFEST="{jobs_manifest_path}"
export E1_WORKER_COUNT="{worker_count}"
export E1_FRICTION="{friction}"

exec "$PYTHON_BIN" experiments/master-thesis/run_e1_cell.py \
  --manifest "$E1_JOB_MANIFEST" \
  --array-index "$SLURM_ARRAY_TASK_ID" \
  --worker-count "$E1_WORKER_COUNT" \
  --friction "$E1_FRICTION" \
  {trace_flag}{diagnostics_flag}
"""
