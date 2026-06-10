#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "results" / "master_thesis" / "e2_tinyfat_finished_readable_20260515"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "master_thesis" / "e2_force_aware_20260515"
DEFAULT_REMOTE_PROJECT_ROOT = Path("/home/woody/iwhr/iwhr106h/master-project")

WIRE_MODELS = ("amplatz_super_stiff", "steve_default", "universal_ii")
ALPHAS = ("a001", "a005", "a05")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_rel_run_dir(output_dir: str) -> Path:
    marker = "results/master_thesis/e2_tinyfat/"
    text = str(output_dir)
    if marker not in text:
        raise ValueError(f"source output_dir does not contain {marker!r}: {text}")
    return Path(text.split(marker, maxsplit=1)[1])


def _force_aware_candidates(*, project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in WIRE_MODELS:
        for alpha in ALPHAS:
            name = f"{model}_standard_j_{alpha}_force_aware"
            checkpoint = Path("data") / "archive" / "agents" / "force_aware" / f"{model}_standard_j_relu_{alpha}" / "checkpoints" / "best_checkpoint.everl"
            checkpoint_path = project_root / checkpoint
            rows.append(
                {
                    "name": name,
                    "model": model,
                    "wire": "standard_j",
                    "tool_ref": f"{model}/standard_j",
                    "alpha": alpha,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_project_relative": checkpoint.as_posix(),
                }
            )
    return rows


def _build_sbatch(*, project_root: Path, manifest_path: Path, logs_root: Path, job_count: int, worker_count: int, friction: float, walltime: str, partition: str, cpus_per_task: int, hint: str) -> str:
    return f"""#!/bin/bash -l
#SBATCH --job-name=e2_force_aware
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --hint={hint}
#SBATCH --time={walltime}
#SBATCH --array=0-{max(int(job_count) - 1, 0)}
#SBATCH --output={logs_root}/slurm-%A_%a.out
#SBATCH --error={logs_root}/slurm-%A_%a.err
#SBATCH --export=NONE

set -euo pipefail
unset SLURM_EXPORT_ENV

cd {project_root}
module load python
eval "$(conda shell.bash hook)"
conda activate "${{E2_CONDA_ENV:-/home/woody/iwhr/iwhr106h/conda/envs/master-project}}"
export SOFA_ROOT="${{SOFA_ROOT:-/home/woody/iwhr/iwhr106h/opt/SOFA_v23.06.00_Linux}}"
source scripts/sofa_env.sh
export STEVE_WALL_FORCE_MONITOR_PLUGIN="${{STEVE_WALL_FORCE_MONITOR_PLUGIN:-$PWD/native/sofa_wire_force_monitor/build/libSofaWireForceMonitor.so}}"
export PYTHON_BIN="${{PYTHON_BIN:-python3}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export E2_JOB_MANIFEST="{manifest_path}"
export E2_WORKER_COUNT="{worker_count}"
export E2_FRICTION="{friction}"

exec "$PYTHON_BIN" experiments/master-thesis/run_e2_force_aware_cell.py \\
  --manifest "$E2_JOB_MANIFEST" \\
  --array-index "$SLURM_ARRAY_TASK_ID" \\
  --worker-count "$E2_WORKER_COUNT"
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare an E2 force-aware rerun manifest for the completed E2 job prefix.")
    ap.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--source-manifest", type=Path, default=None)
    ap.add_argument("--job-count", type=int, default=1700)
    ap.add_argument("--cluster-project-root", type=Path, default=DEFAULT_REMOTE_PROJECT_ROOT)
    ap.add_argument("--local-project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--partition", default="work")
    ap.add_argument("--cpus-per-task", type=int, default=32)
    ap.add_argument("--worker-count", type=int, default=60)
    ap.add_argument("--walltime", default="24:00:00")
    ap.add_argument("--hint", default="multithread")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root_local = args.output_root.resolve()
    source_manifest_path = args.source_manifest or source_root / "metadata" / "job_manifest_work.json"
    source_payload = _read_json(source_manifest_path)
    source_jobs = list(source_payload.get("jobs", []))
    if int(args.job_count) <= 0 or int(args.job_count) > len(source_jobs):
        raise ValueError(f"--job-count must be in 1..{len(source_jobs)}, got {args.job_count}")

    cluster_project_root = Path(args.cluster_project_root)
    remote_output_root = cluster_project_root / "results" / "master_thesis" / output_root_local.name
    metadata_root = output_root_local / "metadata"
    logs_root_remote = remote_output_root / "logs"
    manifest_path_remote = remote_output_root / "metadata" / "job_manifest_force_aware.json"

    candidates = _force_aware_candidates(project_root=cluster_project_root)
    missing_local = [
        row["checkpoint_project_relative"]
        for row in _force_aware_candidates(project_root=Path(args.local_project_root))
        if not Path(row["checkpoint"]).exists()
    ]
    if missing_local:
        raise FileNotFoundError("missing local force-aware checkpoints:\n" + "\n".join(missing_local))

    jobs: list[dict[str, Any]] = []
    for idx, source_job in enumerate(source_jobs[: int(args.job_count)]):
        rel_run_dir = _source_rel_run_dir(str(source_job["output_dir"]))
        job = dict(source_job)
        job["source_e2_array_index"] = idx
        job["source_e2_output_dir"] = source_job["output_dir"]
        job["output_dir"] = str(remote_output_root / rel_run_dir)
        job["force_aware_candidates"] = candidates
        job["candidate_count"] = len(candidates)
        job["job_name"] = str(source_job["job_name"]).replace("cfg4__", "cfg4_force_aware__", 1)
        job["partition"] = args.partition
        job["cpus_per_task"] = int(args.cpus_per_task)
        job["worker_count"] = int(args.worker_count)
        job["walltime"] = str(args.walltime)
        job["hint"] = str(args.hint)
        jobs.append(job)

    payload = {
        "schema_version": 1,
        "experiment": "e2_force_aware",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest_path),
        "source_result_root": str(source_root),
        "source_job_count": len(source_jobs),
        "selected_source_job_count": len(jobs),
        "selection_rule": "source array indices 0..1699, the contiguous completed E2 prefix",
        "output_root": str(remote_output_root),
        "force_aware_candidates": candidates,
        "jobs": jobs,
    }
    _write_json(metadata_root / "job_manifest_force_aware.json", payload)
    _write_json(metadata_root / "force_aware_candidates.json", candidates)

    sbatch = _build_sbatch(
        project_root=cluster_project_root,
        manifest_path=manifest_path_remote,
        logs_root=logs_root_remote,
        job_count=len(jobs),
        worker_count=int(args.worker_count),
        friction=float(jobs[0].get("friction", 0.1)),
        walltime=str(args.walltime),
        partition=str(args.partition),
        cpus_per_task=int(args.cpus_per_task),
        hint=str(args.hint),
    )
    script_path = PROJECT_ROOT / "scripts" / "e2_force_aware_work.sbatch"
    script_path.write_text(sbatch, encoding="utf-8")
    print(f"wrote {metadata_root / 'job_manifest_force_aware.json'}")
    print(f"wrote {metadata_root / 'force_aware_candidates.json'}")
    print(f"wrote {script_path}")
    print(f"jobs={len(jobs)} candidates_per_job={len(candidates)} total_candidate_runs={len(jobs) * len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
