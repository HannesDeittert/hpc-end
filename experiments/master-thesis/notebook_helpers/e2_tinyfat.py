from __future__ import annotations

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

from e1 import CONFIGS, config_spec, load_sample_anatomies, parse_partition_weights, write_wires_json  # type: ignore
from e1_tinyfat import (  # type: ignore
    DEFAULT_WALLTIME,
    PARTITION_PROFILES,
    PartitionProbeRow,
    choose_partition_assignment,
    partition_profile,
    probe_cluster_partitions,
)

DEFAULT_SAMPLE_JSON = PROJECT_ROOT / "results" / "experimental_prep" / "sample_1000_e2.json"
DEFAULT_E2_ROOT = PROJECT_ROOT / "results" / "master_thesis" / "e2_tinyfat"
DEFAULT_FRICTION = 0.1
DEFAULT_TRIAL_COUNT = 50
DEFAULT_WORKER_COUNT = 60
DEFAULT_TARGETS_PER_BRANCH = 2
DEFAULT_TARGET_POSITIONS = ("start", "end")
DEFAULT_TARGET_SEED_START = 9000


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_targets_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_target_positions(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        raw_positions = [part.strip().strip("'\"") for part in text.split(",")]
    else:
        raw_positions = [str(part).strip().strip("'\"") for part in value]
    positions = tuple(position.lower() for position in raw_positions if position)
    allowed = {"start", "mid", "end", "random"}
    unsupported = sorted(set(positions) - allowed)
    if unsupported:
        raise ValueError(f"unsupported target positions {unsupported}; expected only {sorted(allowed)}")
    if not positions:
        raise ValueError("at least one target position must be provided")
    return positions


def _branches_for_anatomy(service: Any, anatomy_id: str) -> list[Any]:
    anatomy = service.get_anatomy(record_id=anatomy_id)
    return [branch for branch in service.list_branches(anatomy) if str(branch.name).lower() != "aorta"]


def _target_for_position(
    *,
    branch: Any,
    position: str,
    target_index: int,
    target_seed: int,
    threshold_mm: float,
) -> dict[str, Any]:
    branch_name = str(branch.name)
    terminal_index = int(branch.terminal_index)
    base = {
        "target_index": int(target_index),
        "target_position": str(position),
        "target_seed": int(target_seed),
        "branch": branch_name,
        "branch_point_count": int(branch.point_count),
        "branch_terminal_index": terminal_index,
        "threshold_mm": float(threshold_mm),
    }
    if position == "end":
        return {
            **base,
            "kind": "branch_end",
            "branches": [branch_name],
            "index": terminal_index,
        }
    if position == "start":
        index = 0
    elif position == "mid":
        index = terminal_index // 2
    elif position == "random":
        lower = 1 if terminal_index > 1 else 0
        upper = terminal_index - 1 if terminal_index > 1 else terminal_index
        index = random.Random(target_seed).randint(lower, upper)
    else:
        raise ValueError(f"unsupported target position: {position!r}")
    return {
        **base,
        "kind": "branch_index",
        "index": int(index),
    }


def write_targets_json(
    *,
    sample_json: Path,
    output_path: Path,
    service: Any,
    threshold_mm: float = 5.0,
    targets_per_branch: int = DEFAULT_TARGETS_PER_BRANCH,
    target_positions: str | Sequence[str] = DEFAULT_TARGET_POSITIONS,
    target_seed_start: int = DEFAULT_TARGET_SEED_START,
) -> dict[str, Any]:
    positions = parse_target_positions(target_positions)
    if int(targets_per_branch) < 1:
        raise ValueError("--targets-per-branch must be >= 1")
    if len(positions) != int(targets_per_branch):
        raise ValueError(
            f"--target-positions must contain exactly --targets-per-branch entries "
            f"({len(positions)} != {int(targets_per_branch)})"
        )
    records: list[dict[str, Any]] = []
    global_target_offset = 0
    for anatomy in load_sample_anatomies(sample_json):
        anatomy_id = str(anatomy["record_id"])
        targets: list[dict[str, Any]] = []
        for branch in _branches_for_anatomy(service, anatomy_id):
            for position in positions:
                target_seed = int(target_seed_start) + global_target_offset
                targets.append(
                    _target_for_position(
                        branch=branch,
                        position=position,
                        target_index=len(targets),
                        target_seed=target_seed,
                        threshold_mm=threshold_mm,
                    )
                )
                global_target_offset += 1
        records.append(
            {
                "record_id": anatomy_id,
                "arch_type": anatomy.get("arch_type"),
                "anatomy_seed": int(anatomy.get("seed", 0)),
                "targets": targets,
            }
        )
    payload = {
        "schema_version": 1,
        "source_sample_json": str(Path(sample_json).resolve()),
        "target_rule": "all_non_aorta_branches_configurable_positions",
        "targets_per_branch": int(targets_per_branch),
        "target_positions": list(positions),
        "target_seed_start": int(target_seed_start),
        "selected_anatomies": records,
    }
    _dump_json(output_path, payload)
    return payload


def build_job_manifest(
    *,
    sample_json: Path,
    targets_json: Path,
    output_root: Path,
    config_id: int,
    trial_count: int = DEFAULT_TRIAL_COUNT,
    seed_base_start: int = 123,
    max_episode_steps: int = 1000,
    partitions: str | None = None,
    partition_weights: Sequence[tuple[str, float]] | None = None,
    probe_rows: Sequence[PartitionProbeRow] = (),
    worker_count: int | None = None,
    walltime: str = DEFAULT_WALLTIME,
    friction: float = DEFAULT_FRICTION,
    write_full_trace: bool = False,
    write_diagnostics: bool = False,
) -> dict[str, Any]:
    if int(config_id) not in CONFIGS:
        raise ValueError(f"unknown config_id={config_id}; expected one of {sorted(CONFIGS)}")
    sample_json = Path(sample_json).resolve()
    targets_json = Path(targets_json).resolve()
    output_root = Path(output_root).resolve()
    anatomies = load_sample_anatomies(sample_json)
    targets = load_targets_json(targets_json)
    target_rows = {str(row["record_id"]): tuple(row["targets"]) for row in targets.get("selected_anatomies", [])}
    if partition_weights is None:
        partition_weights = parse_partition_weights(partitions)

    spec = {**config_spec(int(config_id)), "trial_count": int(trial_count), "max_episode_steps": int(max_episode_steps)}
    job_rows: list[dict[str, Any]] = []
    for anatomy_index, anatomy in enumerate(anatomies):
        anatomy_id = str(anatomy["record_id"])
        for target_index, target_spec in enumerate(target_rows.get(anatomy_id, ())) :
            seed_base = int(seed_base_start + anatomy_index * 1000 + target_index)
            branch = str(target_spec.get("branch", "target"))
            position = str(target_spec.get("target_position", target_spec.get("kind", "target")))
            job_rows.append(
                {
                    "config_id": int(config_id),
                    "config_spec": spec,
                    "write_full_trace": bool(write_full_trace),
                    "write_diagnostics": bool(write_diagnostics),
                    "anatomy_id": anatomy_id,
                    "anatomy_index": anatomy_index,
                    "target_index": int(target_index),
                    "seed_base": seed_base,
                    "target_spec": target_spec,
                    "friction": float(friction),
                    "job_name": f"cfg{config_id}__{anatomy_id}__{branch}__{position}__seed{seed_base}",
                    "output_dir": str(output_root / "runs" / f"config_{config_id}" / f"{anatomy_id}__target_{target_index:03d}__{branch}__{position}__seedbase_{seed_base}"),
                }
            )
    assignments = choose_partition_assignment(job_count=len(job_rows), weights=partition_weights, probe_rows=probe_rows)
    for job_row, partition in zip(job_rows, assignments):
        profile = partition_profile(partition)
        job_row["partition"] = partition
        job_row["gres"] = profile.get("gres", "")
        job_row["cpus_per_task"] = int(profile["cpus_per_task"])
        job_row["worker_count"] = int(worker_count if worker_count is not None else profile["worker_count"])
        job_row["walltime"] = str(walltime)
        job_row["hint"] = str(profile.get("hint", "nomultithread"))
    return {
        "schema_version": 1,
        "experiment": "e2",
        "sample_json": str(sample_json),
        "targets_json": str(targets_json),
        "output_root": str(output_root),
        "seed_base_start": int(seed_base_start),
        "config_id": int(config_id),
        "configs": {str(config_id): spec},
        "partition_weights": list(partition_weights if partition_weights is not None else parse_partition_weights(partitions)),
        "worker_count": int(worker_count if worker_count is not None else DEFAULT_WORKER_COUNT),
        "walltime": str(walltime),
        "friction": float(friction),
        "write_full_trace": bool(write_full_trace),
        "write_diagnostics": bool(write_diagnostics),
        "max_episode_steps": int(max_episode_steps),
        "trial_count": int(trial_count),
        "jobs": job_rows,
    }


def write_partition_bucket_manifests(*, output_root: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    metadata_root = Path(output_root).resolve() / "metadata"
    buckets: dict[str, list[dict[str, Any]]] = {}
    for job in manifest.get("jobs", []):
        buckets.setdefault(str(job.get("partition", "work")), []).append(job)
    paths: dict[str, Path] = {}
    for partition, rows in buckets.items():
        path = metadata_root / f"job_manifest_{partition}.json"
        _dump_json(path, {"schema_version": 1, "partition": partition, "jobs": rows})
        paths[partition] = path
    return paths


def build_sbatch_script(*, project_root: Path, partition: str, jobs_manifest_path: Path, logs_root: Path, walltime: str) -> str:
    project_root = Path(project_root).resolve()
    jobs_manifest_path = Path(jobs_manifest_path).resolve()
    logs_root = Path(logs_root).resolve()
    payload = json.loads(jobs_manifest_path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs", [])
    first = dict(jobs[0]) if jobs else {}
    cpus_per_task = int(first.get("cpus_per_task", partition_profile(partition)["cpus_per_task"]))
    worker_count = int(first.get("worker_count", partition_profile(partition)["worker_count"]))
    hint = str(first.get("hint", "nomultithread"))
    friction = float(first.get("friction", DEFAULT_FRICTION))
    return f'''#!/bin/bash -l
#SBATCH --job-name=e2_tinyfat_{partition}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --hint={hint}
#SBATCH --time={walltime}
#SBATCH --array=0-{max(len(jobs) - 1, 0)}
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
export E2_JOB_MANIFEST="{jobs_manifest_path}"
export E2_WORKER_COUNT="{worker_count}"
export E2_FRICTION="{friction}"

exec "$PYTHON_BIN" experiments/master-thesis/run_e2_cell.py \
  --manifest "$E2_JOB_MANIFEST" \
  --array-index "$SLURM_ARRAY_TASK_ID" \
  --worker-count "$E2_WORKER_COUNT" \
  --wires-json "{jobs_manifest_path.parent / 'wires.json'}"
'''


def _decimate_points(points: Sequence[Sequence[float]], max_points: int) -> list[list[float]]:
    rows = [[float(point[0]), float(point[1]), float(point[2])] for point in points]
    if len(rows) <= int(max_points):
        return rows
    if int(max_points) < 2:
        return [rows[0], rows[-1]]
    step = (len(rows) - 1) / float(int(max_points) - 1)
    indices = sorted({min(round(i * step), len(rows) - 1) for i in range(int(max_points))})
    return [rows[int(index)] for index in indices]


def _target_point_from_spec(branches_by_name: dict[str, Any], target_spec: dict[str, Any]) -> list[float] | None:
    branch_name = str(target_spec.get("branch", ""))
    branch = branches_by_name.get(branch_name)
    if branch is None:
        return None
    points = branch.centerline_points_vessel_cs
    if not points:
        return None
    index = int(target_spec.get("index", len(points) - 1))
    index = max(0, min(index, len(points) - 1))
    point = points[index]
    return [float(point[0]), float(point[1]), float(point[2])]


def _target_point_from_point_map(branch_points_by_name: dict[str, Any], target_spec: dict[str, Any]) -> list[float] | None:
    branch_name = str(target_spec.get("branch", ""))
    points = branch_points_by_name.get(branch_name)
    if points is None or len(points) == 0:
        return None
    index = int(target_spec.get("index", len(points) - 1))
    index = max(0, min(index, len(points) - 1))
    point = points[index]
    return [float(point[0]), float(point[1]), float(point[2])]


def _sample_records_by_id_from_targets_payload(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sample_json = payload.get("source_sample_json")
    if not sample_json:
        return {}
    sample_path = Path(str(sample_json))
    if not sample_path.exists():
        return {}
    sample_payload = json.loads(sample_path.read_text(encoding="utf-8"))
    return {
        str(item.get("record_id")): dict(item)
        for item in sample_payload.get("selected_anatomies", [])
        if item.get("record_id")
    }


def _branch_rows_from_sample_record(
    sample_record: dict[str, Any],
    *,
    max_points_per_branch: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    centerline_path = Path(str(sample_record.get("centerline_bundle_path", "")))
    if not centerline_path.exists():
        return [], {}
    branch_rows: list[dict[str, Any]] = []
    branch_points_by_name: dict[str, Any] = {}
    with np.load(centerline_path, allow_pickle=True) as data:
        for key in sorted(data.files):
            if not (key.startswith("branch_") and key.endswith("_coords")):
                continue
            branch_name = key[len("branch_") : -len("_coords")]
            points = data[key]
            if len(points) == 0:
                continue
            branch_points_by_name[branch_name] = points
            branch_rows.append(
                {
                    "name": branch_name,
                    "point_count": int(len(points)),
                    "terminal_index": int(len(points) - 1),
                    "is_aorta": branch_name.lower() == "aorta",
                    "points": _decimate_points(points, max_points=max_points_per_branch),
                }
            )
    return branch_rows, branch_points_by_name


def build_e2_target_review_records(
    *,
    targets_json: Path,
    service: Any,
    registry_path: Path | None = None,
    max_points_per_branch: int = 220,
) -> list[dict[str, Any]]:
    payload = load_targets_json(Path(targets_json))
    sample_records_by_id = _sample_records_by_id_from_targets_payload(payload)
    records: list[dict[str, Any]] = []
    for item in payload.get("selected_anatomies", []):
        anatomy_id = str(item["record_id"])
        sample_record = sample_records_by_id.get(anatomy_id)
        branch_points_by_name: dict[str, Any] = {}
        if sample_record is not None:
            branch_rows, branch_points_by_name = _branch_rows_from_sample_record(
                sample_record,
                max_points_per_branch=max_points_per_branch,
            )
        else:
            branch_rows = []
        branches_by_name = {}
        if not branch_rows:
            anatomy = service.get_anatomy(record_id=anatomy_id, registry_path=registry_path)
            branches = list(service.list_branches(anatomy))
            branches_by_name = {str(branch.name): branch for branch in branches}
            for branch in branches:
                branch_rows.append(
                    {
                        "name": str(branch.name),
                        "point_count": int(branch.point_count),
                        "terminal_index": int(branch.terminal_index),
                        "is_aorta": str(branch.name).lower() == "aorta",
                        "points": _decimate_points(branch.centerline_points_vessel_cs, max_points=max_points_per_branch),
                    }
                )
        target_rows = []
        for target in item.get("targets", []):
            target_spec = dict(target)
            point = (
                _target_point_from_point_map(branch_points_by_name, target_spec)
                if branch_points_by_name
                else _target_point_from_spec(branches_by_name, target_spec)
            )
            target_rows.append(
                {
                    **target_spec,
                    "point": point,
                }
            )
        records.append(
            {
                "record_id": anatomy_id,
                "arch_type": item.get("arch_type"),
                "anatomy_seed": item.get("anatomy_seed"),
                "branches": branch_rows,
                "targets": target_rows,
            }
        )
    return records


def write_e2_target_review_html(
    *,
    targets_json: Path,
    output_html: Path,
    service: Any,
    registry_path: Path | None = None,
    title: str = "E2 target review",
    max_points_per_branch: int = 220,
) -> Path:
    records = build_e2_target_review_records(
        targets_json=targets_json,
        service=service,
        registry_path=registry_path,
        max_points_per_branch=max_points_per_branch,
    )
    data_json = json.dumps(records, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    title_json = json.dumps(str(title))
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --bg: #101418; --panel: #171d23; --ink: #e8edf2; --muted: #9aa7b5; --accent: #f2ac32; --bad: #ef476f; --good: #7bd88f; }}
  body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.35 ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }}
  .wrap {{ display: grid; grid-template-columns: 1fr 320px; gap: 12px; height: 100vh; padding: 12px; box-sizing: border-box; }}
  .main, .side {{ background: var(--panel); border: 1px solid #2a333d; border-radius: 12px; overflow: hidden; }}
  .bar {{ display: flex; align-items: center; gap: 8px; padding: 10px; border-bottom: 1px solid #2a333d; flex-wrap: wrap; }}
  button, select, input, textarea {{ background: #0f151b; color: var(--ink); border: 1px solid #34404c; border-radius: 8px; padding: 7px 10px; }}
  button {{ cursor: pointer; }}
  button:hover {{ border-color: var(--accent); }}
  input[type=range] {{ width: 220px; }}
  .plots {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; padding: 10px; }}
  svg {{ width: 100%; height: calc(100vh - 110px); min-height: 420px; background: #0b0f13; border: 1px solid #27313b; border-radius: 10px; }}
  .side {{ padding: 12px; overflow: auto; }}
  .meta {{ color: var(--muted); margin-bottom: 10px; }}
  .target {{ border-left: 3px solid var(--accent); padding: 5px 0 5px 8px; margin: 4px 0; color: #dbe4ed; }}
  .branch {{ display: inline-block; margin: 2px 5px 2px 0; color: var(--muted); }}
  .status-valid {{ color: var(--good); }}
  .status-invalid {{ color: var(--bad); }}
  .hint {{ color: var(--muted); font-size: 12px; }}
  @media (max-width: 1000px) {{ .wrap {{ grid-template-columns: 1fr; height: auto; }} .plots {{ grid-template-columns: 1fr; }} svg {{ height: 360px; }} }}
</style>
</head>
<body>
<div class="wrap">
  <section class="main">
    <div class="bar">
      <button id="prev">Prev</button>
      <button id="next">Next</button>
      <label>Index <input id="idx" type="number" min="0" value="0" style="width:80px"></label>
      <input id="slider" type="range" min="0" value="0">
      <strong id="counter"></strong>
      <select id="status">
        <option value="">unreviewed</option>
        <option value="valid">valid</option>
        <option value="invalid">invalid</option>
        <option value="unsure">unsure</option>
      </select>
      <button id="download">Download review JSON</button>
      <span class="hint">Arrow keys or swipe left/right. Review state is kept in browser localStorage.</span>
    </div>
    <div class="plots">
      <svg id="xy" data-proj="xy"></svg>
      <svg id="xz" data-proj="xz"></svg>
      <svg id="yz" data-proj="yz"></svg>
    </div>
  </section>
  <aside class="side">
    <h2 id="title"></h2>
    <div id="meta" class="meta"></div>
    <label>Notes</label>
    <textarea id="notes" rows="4" style="width:100%; box-sizing:border-box; margin:6px 0 12px"></textarea>
    <h3>Branches</h3>
    <div id="branches"></div>
    <h3>Targets</h3>
    <div id="targets"></div>
  </aside>
</div>
<script id="review-data" type="application/json">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById('review-data').textContent);
const TITLE = {title_json};
const KEY = 'e2-target-review:' + TITLE + ':' + DATA.length;
let review = JSON.parse(localStorage.getItem(KEY) || '{{}}');
let current = 0;
const colors = ['#f2ac32','#60a5fa','#7bd88f','#f472b6','#c084fc','#f97316','#22d3ee','#fde047','#a3e635','#fb7185'];
function saveReview() {{ localStorage.setItem(KEY, JSON.stringify(review)); }}
function branchColor(name, idx, isAorta) {{ return isAorta ? '#5d6670' : colors[idx % colors.length]; }}
function project(point, proj) {{
  if (proj === 'xy') return [point[0], point[1]];
  if (proj === 'xz') return [point[0], point[2]];
  return [point[1], point[2]];
}}
function bounds(record, proj) {{
  const pts = [];
  record.branches.forEach(b => b.points.forEach(p => pts.push(project(p, proj))));
  record.targets.forEach(t => {{ if (t.point) pts.push(project(t.point, proj)); }});
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  let minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  if (!isFinite(minX)) {{ minX=0; maxX=1; minY=0; maxY=1; }}
  const padX = (maxX-minX || 1) * 0.08, padY = (maxY-minY || 1) * 0.08;
  return [minX-padX, maxX+padX, minY-padY, maxY+padY];
}}
function scaleFn(svg, record, proj) {{
  const w = svg.clientWidth || 500, h = svg.clientHeight || 500;
  const [minX,maxX,minY,maxY] = bounds(record, proj);
  return p => {{
    const [x,y] = project(p, proj);
    return [40 + (x-minX)/(maxX-minX || 1)*(w-70), h-35 - (y-minY)/(maxY-minY || 1)*(h-75)];
  }};
}}
function pathFor(points, scale) {{
  return points.map((p,i) => {{
    const [x,y] = scale(p);
    return (i ? 'L' : 'M') + x.toFixed(1) + ',' + y.toFixed(1);
  }}).join(' ');
}}
function renderSvg(svg, record, proj) {{
  const scale = scaleFn(svg, record, proj);
  const branchParts = record.branches.map((b, i) => `<path d="${{pathFor(b.points, scale)}}" fill="none" stroke="${{branchColor(b.name, i, b.is_aorta)}}" stroke-width="${{b.is_aorta ? 2 : 3}}" opacity="${{b.is_aorta ? 0.35 : 0.9}}"/>`).join('');
  const targetParts = record.targets.map((t, i) => {{
    if (!t.point) return '';
    const [x,y] = scale(t.point);
    const label = `${{t.branch}}:${{t.target_position}}`;
    return `<g><circle cx="${{x.toFixed(1)}}" cy="${{y.toFixed(1)}}" r="5.5" fill="#ef476f" stroke="#fff" stroke-width="1.2"/><text x="${{(x+7).toFixed(1)}}" y="${{(y-7).toFixed(1)}}" fill="#e8edf2" font-size="10">${{label}}</text></g>`;
  }}).join('');
  svg.innerHTML = `<text x="12" y="22" fill="#9aa7b5" font-size="13">${{proj.toUpperCase()}}</text>${{branchParts}}${{targetParts}}`;
}}
function currentReview(record) {{ return review[record.record_id] || {{status:'', notes:''}}; }}
function setReview(record, patch) {{ review[record.record_id] = {{...currentReview(record), ...patch, record_id: record.record_id}}; saveReview(); }}
function render() {{
  current = Math.max(0, Math.min(current, DATA.length - 1));
  const r = DATA[current];
  document.getElementById('counter').textContent = `${{current+1}} / ${{DATA.length}}`;
  document.getElementById('idx').value = current;
  document.getElementById('slider').max = Math.max(DATA.length - 1, 0);
  document.getElementById('slider').value = current;
  document.getElementById('title').textContent = `${{r.record_id}}`;
  document.getElementById('meta').innerHTML = `arch=${{r.arch_type || ''}} seed=${{r.anatomy_seed || ''}}<br>branches=${{r.branches.length}} targets=${{r.targets.length}}`;
  const rv = currentReview(r);
  document.getElementById('status').value = rv.status || '';
  document.getElementById('notes').value = rv.notes || '';
  document.getElementById('branches').innerHTML = r.branches.map((b,i) => `<span class="branch" style="color:${{branchColor(b.name,i,b.is_aorta)}}">${{b.name}}(${{b.point_count}})</span>`).join('');
  document.getElementById('targets').innerHTML = r.targets.map(t => `<div class="target">${{t.target_index}}: <b>${{t.branch}}</b> ${{t.target_position}} index=${{t.index}} seed=${{t.target_seed ?? ''}}</div>`).join('');
  ['xy','xz','yz'].forEach(id => renderSvg(document.getElementById(id), r, id));
}}
document.getElementById('prev').onclick = () => {{ current--; render(); }};
document.getElementById('next').onclick = () => {{ current++; render(); }};
document.getElementById('idx').onchange = e => {{ current = parseInt(e.target.value || '0', 10); render(); }};
document.getElementById('slider').oninput = e => {{ current = parseInt(e.target.value, 10); render(); }};
document.getElementById('status').onchange = e => setReview(DATA[current], {{status:e.target.value}});
document.getElementById('notes').oninput = e => setReview(DATA[current], {{notes:e.target.value}});
document.getElementById('download').onclick = () => {{
  const blob = new Blob([JSON.stringify(review, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'e2_target_review.json';
  a.click();
  URL.revokeObjectURL(a.href);
}};
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') {{ current++; render(); }}
  if (e.key === 'ArrowLeft') {{ current--; render(); }}
}});
let touchX = null;
document.addEventListener('touchstart', e => {{ touchX = e.changedTouches[0].clientX; }}, {{passive:true}});
document.addEventListener('touchend', e => {{
  if (touchX === null) return;
  const dx = e.changedTouches[0].clientX - touchX;
  if (Math.abs(dx) > 50) {{ current += dx < 0 ? 1 : -1; render(); }}
  touchX = null;
}}, {{passive:true}});
window.addEventListener('resize', render);
render();
</script>
</body>
</html>
"""
    output_html = Path(output_html).resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html


def _read_obj_mesh(path: Path, *, max_faces: int = 4000) -> dict[str, Any]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                raw = [part.split("/")[0] for part in line.split()[1:]]
                if len(raw) < 3:
                    continue
                idx = [int(part) - 1 for part in raw]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    if max_faces > 0 and len(faces) > max_faces:
        step = len(faces) / float(max_faces)
        faces = [faces[min(int(i * step), len(faces) - 1)] for i in range(max_faces)]
    return {"vertices": vertices, "faces": faces}


def _write_mesh_review_record(
    *,
    output_path: Path,
    review_record: dict[str, Any],
    sample_record: dict[str, Any],
    max_mesh_faces: int,
    max_points_per_branch: int,
) -> None:
    branch_rows, branch_points_by_name = _branch_rows_from_sample_record(
        sample_record,
        max_points_per_branch=max_points_per_branch,
    )
    target_rows = []
    for target in review_record.get("targets", []):
        target_spec = dict(target)
        target_rows.append(
            {
                **target_spec,
                "point": _target_point_from_point_map(branch_points_by_name, target_spec),
            }
        )
    mesh_path = Path(str(sample_record.get("visualization_mesh_path") or sample_record.get("simulation_mesh_path")))
    mesh = _read_obj_mesh(mesh_path, max_faces=max_mesh_faces) if mesh_path.exists() else {"vertices": [], "faces": []}
    payload = {
        "record_id": review_record.get("record_id"),
        "arch_type": review_record.get("arch_type"),
        "anatomy_seed": review_record.get("anatomy_seed"),
        "mesh_path": str(mesh_path),
        "mesh": mesh,
        "branches": branch_rows,
        "targets": target_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")


def write_e2_mesh_review_viewer(
    *,
    targets_json: Path,
    output_dir: Path,
    title: str = "E2 mesh target review",
    max_mesh_faces: int = 4000,
    max_points_per_branch: int = 180,
) -> Path:
    targets_payload = load_targets_json(Path(targets_json))
    sample_records_by_id = _sample_records_by_id_from_targets_payload(targets_payload)
    selected = list(targets_payload.get("selected_anatomies", []))
    output_dir = Path(output_dir).resolve()
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    for i, item in enumerate(selected):
        anatomy_id = str(item["record_id"])
        sample_record = sample_records_by_id.get(anatomy_id)
        if sample_record is None:
            continue
        rel_path = Path("records") / f"{i:04d}_{anatomy_id}.json"
        _write_mesh_review_record(
            output_path=output_dir / rel_path,
            review_record=dict(item),
            sample_record=sample_record,
            max_mesh_faces=max_mesh_faces,
            max_points_per_branch=max_points_per_branch,
        )
        index_rows.append({"i": i, "record_id": anatomy_id, "path": rel_path.as_posix()})
    index_json = json.dumps(index_rows, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    title_json = json.dumps(str(title))
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body{{margin:0;background:#101418;color:#e8edf2;font:14px system-ui,sans-serif}}.bar{{display:flex;gap:8px;align-items:center;padding:10px;background:#171d23;border-bottom:1px solid #2a333d;flex-wrap:wrap}}button,input,select,textarea{{background:#0f151b;color:#e8edf2;border:1px solid #34404c;border-radius:8px;padding:7px 10px}}button{{cursor:pointer}}#wrap{{display:grid;grid-template-columns:1fr 320px;height:calc(100vh - 56px)}}#canvas{{width:100%;height:100%;display:block;background:#080b0f}}#side{{padding:12px;background:#171d23;border-left:1px solid #2a333d;overflow:auto}}.target{{border-left:3px solid #ef476f;margin:4px 0;padding:5px 0 5px 8px}}.hint,.meta{{color:#9aa7b5}}@media(max-width:900px){{#wrap{{grid-template-columns:1fr;height:auto}}#canvas{{height:70vh}}}}
</style></head><body>
<div class="bar"><button id="prev">Prev</button><button id="next">Next</button><label>Index <input id="idx" type="number" value="0" min="0" style="width:80px"></label><input id="slider" type="range" value="0" min="0"><strong id="counter"></strong><select id="status"><option value="">unreviewed</option><option>valid</option><option>invalid</option><option>unsure</option></select><button id="download">Download review JSON</button><span class="hint">Drag rotate, wheel zoom, Shift+drag pan, arrows/Swipe navigate</span></div>
<div id="wrap"><canvas id="canvas"></canvas><aside id="side"><h2 id="name"></h2><div id="meta" class="meta"></div><label>Notes</label><textarea id="notes" rows="4" style="width:100%;box-sizing:border-box;margin:6px 0 12px"></textarea><h3>Targets</h3><div id="targets"></div></aside></div>
<script>
const INDEX={index_json}; const TITLE={title_json}; const KEY='e2-mesh-review:'+TITLE+':'+INDEX.length;
let review=JSON.parse(localStorage.getItem(KEY)||'{{}}'), current=0, rec=null, yaw=0.7, pitch=0.2, zoom=3.2, panX=0, panY=0, dragging=false, last=null;
const canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
function save(){{localStorage.setItem(KEY,JSON.stringify(review));}} function rv(){{return rec?(review[rec.record_id]||{{status:'',notes:''}}):{{}}}} function setRv(p){{review[rec.record_id]={{...rv(),...p,record_id:rec.record_id}};save();}}
async function load(i){{current=Math.max(0,Math.min(i,INDEX.length-1)); rec=await fetch(INDEX[current].path).then(r=>r.json()); document.getElementById('slider').max=INDEX.length-1; document.getElementById('slider').value=current; document.getElementById('idx').value=current; document.getElementById('counter').textContent=`${{current+1}} / ${{INDEX.length}}`; document.getElementById('name').textContent=rec.record_id; document.getElementById('meta').innerHTML=`arch=${{rec.arch_type||''}} seed=${{rec.anatomy_seed||''}}<br>mesh faces=${{rec.mesh.faces.length}} branches=${{rec.branches.length}} targets=${{rec.targets.length}}`; document.getElementById('targets').innerHTML=rec.targets.map(t=>`<div class="target">${{t.target_index}} <b>${{t.branch}}</b> ${{t.target_position}} index=${{t.index}}</div>`).join(''); document.getElementById('status').value=rv().status||''; document.getElementById('notes').value=rv().notes||''; draw();}}
function centerScale(){{let pts=rec.mesh.vertices.concat(...rec.branches.map(b=>b.points)); let c=[0,0,0]; pts.forEach(p=>{{c[0]+=p[0];c[1]+=p[1];c[2]+=p[2];}}); c=c.map(v=>v/Math.max(pts.length,1)); let r=1; pts.forEach(p=>{{let d=Math.hypot(p[0]-c[0],p[1]-c[1],p[2]-c[2]); if(d>r)r=d;}}); return [c,r];}}
function proj(p,c,r){{let x=(p[0]-c[0])/r,y=(p[1]-c[1])/r,z=(p[2]-c[2])/r; let cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch); let x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=cp*y-sp*z1,z2=sp*y+cp*z1; let s=zoom/(zoom+z2+2.2)*Math.min(canvas.width,canvas.height)*0.42; return [canvas.width/2+panX+x1*s,canvas.height/2+panY-y1*s,z2];}}
function draw(){{if(!rec)return; canvas.width=canvas.clientWidth*devicePixelRatio; canvas.height=canvas.clientHeight*devicePixelRatio; ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,canvas.width,canvas.height); let [c,r]=centerScale(); let tris=rec.mesh.faces.map(f=>{{let ps=f.map(i=>proj(rec.mesh.vertices[i],c,r)); return {{ps,z:(ps[0][2]+ps[1][2]+ps[2][2])/3}};}}).sort((a,b)=>a.z-b.z); ctx.lineWidth=0.6*devicePixelRatio; tris.forEach(t=>{{ctx.beginPath();ctx.moveTo(t.ps[0][0],t.ps[0][1]);ctx.lineTo(t.ps[1][0],t.ps[1][1]);ctx.lineTo(t.ps[2][0],t.ps[2][1]);ctx.closePath();ctx.fillStyle='rgba(120,145,165,0.16)';ctx.strokeStyle='rgba(170,190,210,0.16)';ctx.fill();ctx.stroke();}}); const cols=['#f2ac32','#60a5fa','#7bd88f','#f472b6','#c084fc','#22d3ee']; rec.branches.forEach((b,bi)=>{{ctx.beginPath();b.points.forEach((p,i)=>{{let q=proj(p,c,r); if(i)ctx.lineTo(q[0],q[1]); else ctx.moveTo(q[0],q[1]);}});ctx.strokeStyle=b.is_aorta?'rgba(180,190,200,0.35)':cols[bi%cols.length];ctx.lineWidth=(b.is_aorta?1.5:3)*devicePixelRatio;ctx.stroke();}}); rec.targets.forEach(t=>{{if(!t.point)return;let q=proj(t.point,c,r);ctx.beginPath();ctx.arc(q[0],q[1],6*devicePixelRatio,0,Math.PI*2);ctx.fillStyle='#ef476f';ctx.strokeStyle='#fff';ctx.fill();ctx.stroke();ctx.fillStyle='#fff';ctx.font=`${{11*devicePixelRatio}}px sans-serif`;ctx.fillText(`${{t.branch}}:${{t.target_position}}`,q[0]+8*devicePixelRatio,q[1]-8*devicePixelRatio);}});}}
canvas.onmousedown=e=>{{dragging=true;last=[e.clientX,e.clientY,e.shiftKey];}}; window.onmouseup=()=>dragging=false; window.onmousemove=e=>{{if(!dragging)return;let dx=e.clientX-last[0],dy=e.clientY-last[1]; if(last[2]){{panX+=dx*devicePixelRatio;panY+=dy*devicePixelRatio;}}else{{yaw+=dx*0.01;pitch+=dy*0.01;}} last=[e.clientX,e.clientY,e.shiftKey]; draw();}}; canvas.onwheel=e=>{{e.preventDefault();zoom=Math.max(0.5,Math.min(12,zoom*(e.deltaY>0?0.9:1.1)));draw();}};
document.getElementById('prev').onclick=()=>load(current-1);document.getElementById('next').onclick=()=>load(current+1);document.getElementById('idx').onchange=e=>load(parseInt(e.target.value||0));document.getElementById('slider').oninput=e=>load(parseInt(e.target.value));document.getElementById('status').onchange=e=>setRv({{status:e.target.value}});document.getElementById('notes').oninput=e=>setRv({{notes:e.target.value}});document.getElementById('download').onclick=()=>{{let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(review,null,2)],{{type:'application/json'}}));a.download='e2_mesh_review.json';a.click();}};document.onkeydown=e=>{{if(e.key==='ArrowRight')load(current+1);if(e.key==='ArrowLeft')load(current-1);}};let tx=null;document.ontouchstart=e=>tx=e.changedTouches[0].clientX;document.ontouchend=e=>{{let dx=e.changedTouches[0].clientX-tx;if(Math.abs(dx)>50)load(current+(dx<0?1:-1));}};window.onresize=draw;load(0);
</script></body></html>"""
    html_path = output_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path
