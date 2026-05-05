#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Live reads of partially rewritten HDF5 files on shared filesystems can fail
# with file-locking errors unless locking is disabled explicitly.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

try:  # pragma: no cover - optional dependency in some local environments
    import h5py

    H5PY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency in some local environments
    h5py = None  # type: ignore[assignment]
    H5PY_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_E1_ROOT = PROJECT_ROOT / "results" / "master_thesis" / "e1"
DEFAULT_METADATA_ROOT = DEFAULT_E1_ROOT / "metadata"
DEFAULT_LOGS_ROOT = DEFAULT_E1_ROOT / "logs"


CONFIG_LABELS = {
    1: "deterministic / fixed_start / 1 trial",
    2: "deterministic / random_start / 1000 trials",
    3: "stochastic / fixed_start / 1000 trials",
    4: "stochastic / random_start / 1000 trials",
}

EXPECTED_CANDIDATES_PER_JOB = 15

PROGRESS_PATTERNS = (
    re.compile(r"\[eval_v2\] parallel_trial_done completed=(\d+) total=(\d+)"),
    re.compile(r"\[eval_v2\] parallel_end completed=(\d+) total=(\d+)"),
    re.compile(r"\[eval_v2\] parallel_start workers=(\d+) total=(\d+)"),
)


@dataclass(frozen=True)
class ManifestJob:
    partition: str
    array_index: int
    config_id: int
    job_name: str
    output_dir: str
    trial_count: int
    write_full_trace: bool


@dataclass(frozen=True)
class QueueTask:
    jobid: str
    partition: str
    state: str
    reason: str
    name: str


@dataclass(frozen=True)
class LogProgress:
    log_path: Path
    job_name: str | None
    output_dir: str | None
    completed: int | None
    total: int | None
    progress: float | None
    status: str
    last_line: str | None


@dataclass(frozen=True)
class TrialTableProgress:
    output_dir: Path
    trials_path: Path
    count: int
    expected: int
    progress: float


@dataclass(frozen=True)
class TraceProgress:
    output_dir: Path
    trace_dir: Path
    count: int
    expected: int
    progress: float


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tail_text(path: Path, max_bytes: int = 131072) -> str:
    with Path(path).open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(size - max_bytes, 0), os.SEEK_SET)
        return handle.read().decode("utf-8", errors="replace")


def load_manifest_jobs(metadata_root: Path) -> list[ManifestJob]:
    metadata_root = Path(metadata_root).resolve()
    jobs: list[ManifestJob] = []
    bucket_paths = sorted(metadata_root.glob("job_manifest_*.json"))
    if not bucket_paths and (metadata_root / "job_manifest.json").exists():
        bucket_paths = [metadata_root / "job_manifest.json"]

    for bucket_path in bucket_paths:
        payload = _load_json(bucket_path)
        partition = str(payload.get("partition") or bucket_path.stem.replace("job_manifest_", ""))
        for array_index, job in enumerate(payload.get("jobs", [])):
            config_id = int(job.get("config_id", 0))
            config_spec = dict(job.get("config_spec", {}))
            jobs.append(
                ManifestJob(
                    partition=partition,
                    array_index=array_index,
                    config_id=config_id,
                    job_name=str(job.get("job_name", "")),
                    output_dir=str(job.get("output_dir", "")),
                    trial_count=int(config_spec.get("trial_count", 0)),
                    write_full_trace=bool(job.get("write_full_trace", True)),
                )
            )
    return jobs


def load_queue_tasks(
    user: str | None = None,
    *,
    cluster: str | None = None,
) -> dict[tuple[str, int], QueueTask]:
    cluster = (cluster or "").strip().lower() or None
    if cluster == "tinyfat":
        commands = [
            ["squeue.tinyfat", "-r", "-h", "-o", "%i|%P|%T|%R|%j"],
            ["squeue", "--clusters=tinyfat", "-r", "-h", "-o", "%i|%P|%T|%R|%j"],
            ["squeue", "-r", "-h", "-o", "%i|%P|%T|%R|%j"],
        ]
    elif cluster == "tinygpu":
        commands = [
            ["squeue.tinygpu", "-r", "-h", "-o", "%i|%P|%T|%R|%j"],
            ["squeue", "--clusters=tinygpu", "-r", "-h", "-o", "%i|%P|%T|%R|%j"],
            ["squeue", "-r", "-h", "-o", "%i|%P|%T|%R|%j"],
        ]
    else:
        commands = [
            ["squeue.tinyfat", "-r", "-h", "-o", "%i|%P|%T|%R|%j"],
            ["squeue", "--clusters=tinyfat", "-r", "-h", "-o", "%i|%P|%T|%R|%j"],
            ["squeue.tinygpu", "-r", "-h", "-o", "%i|%P|%T|%R|%j"],
            ["squeue", "--clusters=tinygpu", "-r", "-h", "-o", "%i|%P|%T|%R|%j"],
            ["squeue", "-r", "-h", "-o", "%i|%P|%T|%R|%j"],
        ]
    if user:
        commands = [cmd[:1] + ["-u", user] + cmd[1:] for cmd in commands]

    last_error: Exception | None = None
    output = ""
    for command in commands:
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            output = result.stdout
            break
        except Exception as exc:  # pragma: no cover - environment-specific fallback
            last_error = exc
    else:
        raise RuntimeError("Unable to query Slurm queue") from last_error

    tasks: dict[tuple[str, int], QueueTask] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        jobid, partition, state, reason, name = (part.strip() for part in line.split("|", 4))
        if "_" not in jobid:
            continue
        master_id, task_id = jobid.split("_", 1)
        if not task_id.isdigit():
            continue
        tasks[(partition, int(task_id))] = QueueTask(
            jobid=jobid,
            partition=partition,
            state=state,
            reason=reason,
            name=name,
        )
    return tasks


def parse_progress_text(text: str) -> tuple[int | None, int | None, float | None, str | None, str]:
    completed: int | None = None
    total: int | None = None
    last_line: str | None = None
    status = "unknown"

    for line in text.splitlines():
        for index, pattern in enumerate(PROGRESS_PATTERNS):
            match = pattern.search(line)
            if not match:
                continue
            if index == 2:
                completed = 0
                total = int(match.group(2))
                status = "running"
            else:
                completed = int(match.group(1))
                total = int(match.group(2))
                status = "running"
            last_line = line.strip()

    if "Traceback (most recent call last)" in text:
        status = "failed"
    if "JOB_STATISTICS" in text or "[E1] summary" in text:
        status = "completed"
        if total is None:
            total = completed if completed is not None else 1
        completed = total

    progress = None
    if completed is not None and total not in (None, 0):
        progress = max(0.0, min(1.0, completed / float(total)))
    return completed, total, progress, last_line, status


def load_log_progress(log_path: Path) -> LogProgress:
    text = _tail_text(log_path)
    job_name = None
    output_dir = None
    for line in text.splitlines():
        if line.startswith("[E1] job="):
            match = re.search(r"\[E1\] job=(.+?) generated_at=", line)
            if match:
                job_name = match.group(1).strip()
        if line.startswith("[E1] output_dir="):
            output_dir = line.split("=", 1)[1].strip()

    completed, total, progress, last_line, status = parse_progress_text(text)
    return LogProgress(
        log_path=log_path,
        job_name=job_name,
        output_dir=output_dir,
        completed=completed,
        total=total,
        progress=progress,
        status=status,
        last_line=last_line,
    )


def load_trial_table_progress(output_dir: str | Path, trial_count: int) -> TrialTableProgress | None:
    if not H5PY_AVAILABLE:
        return None
    resolved_output_dir = _resolve_output_dir(output_dir)
    trials_path = resolved_output_dir / "trials.h5"
    if not trials_path.exists():
        return None

    try:
        with h5py.File(trials_path, "r") as handle:  # type: ignore[union-attr]
            group = handle["trials"]
            count = int(group["trial_index"].shape[0]) if "trial_index" in group else 0
    except Exception:
        return None

    expected = max(0, int(trial_count) * EXPECTED_CANDIDATES_PER_JOB)
    progress = 0.0 if expected == 0 else min(1.0, count / float(expected))
    return TrialTableProgress(
        output_dir=resolved_output_dir,
        trials_path=trials_path,
        count=count,
        expected=expected,
        progress=progress,
    )


def _resolve_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_trace_progress(output_dir: str | Path, trial_count: int) -> TraceProgress:
    resolved_output_dir = _resolve_output_dir(output_dir)
    trace_dir = resolved_output_dir / "traces"
    count = 0
    if trace_dir.exists():
        count = sum(1 for path in trace_dir.glob("*.h5") if path.is_file())
    expected = max(0, int(trial_count) * EXPECTED_CANDIDATES_PER_JOB)
    progress = 0.0 if expected == 0 else min(1.0, count / float(expected))
    return TraceProgress(
        output_dir=resolved_output_dir,
        trace_dir=trace_dir,
        count=count,
        expected=expected,
        progress=progress,
    )


def scan_logs(logs_root: Path) -> dict[str, LogProgress]:
    logs_root = Path(logs_root).resolve()
    records: dict[str, LogProgress] = {}
    if not logs_root.exists():
        return records

    for log_path in sorted(logs_root.glob("slurm-*.out")):
        try:
            record = load_log_progress(log_path)
        except Exception:
            continue
        keys = {key for key in [record.job_name, record.output_dir] if key}
        for key in keys:
            prev = records.get(key)
            if prev is None or log_path.stat().st_mtime >= prev.log_path.stat().st_mtime:
                records[key] = record
    return records


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:5.1f}%"


def _summarize_group(
    jobs: list[ManifestJob],
    queue_tasks: dict[tuple[str, int], QueueTask],
    log_records: dict[str, LogProgress],
    *,
    progress_source: str = "auto",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        queue = queue_tasks.get((job.partition, job.array_index))
        log = log_records.get(job.job_name) or log_records.get(job.output_dir)
        trial_table = (
            load_trial_table_progress(job.output_dir, job.trial_count)
            if progress_source in {"auto", "trial-h5"}
            else None
        )
        traces = (
            load_trace_progress(job.output_dir, job.trial_count)
            if progress_source == "trace"
            else None
        )
        if trial_table and trial_table.count > 0:
            state = "COMPLETED" if trial_table.progress >= 1.0 else "RUNNING"
        elif log and log.status == "completed":
            state = "COMPLETED"
        elif log and log.status == "failed":
            state = "FAILED"
        elif traces is not None and traces.count > 0:
            state = "RUNNING" if traces.progress < 1.0 else "COMPLETED"
        elif queue is not None:
            state = queue.state
        elif log and log.progress is not None:
            state = "RUNNING"
        else:
            state = "PENDING"

        if trial_table is not None and trial_table.count > 0:
            progress = trial_table.progress
            metric_label = "trial rows"
            metric_count = trial_table.count
            metric_expected = trial_table.expected
            metric_path = str(trial_table.trials_path)
        elif traces is not None and traces.count > 0:
            progress = traces.progress
            metric_label = "trace files"
            metric_count = traces.count
            metric_expected = traces.expected
            metric_path = str(traces.trace_dir)
        elif log and log.progress is not None:
            progress = log.progress
            metric_label = "log progress"
            metric_count = log.completed if log.completed is not None else 0
            metric_expected = log.total if log.total is not None else 0
            metric_path = str(log.log_path)
        else:
            progress = 1.0 if state == "COMPLETED" else 0.0
            metric_label = "n/a"
            metric_count = 0
            metric_expected = 0
            metric_path = ""
        rows.append(
            {
                "job": job,
                "state": state,
                "progress": progress,
                "metric_label": metric_label,
                "metric_count": metric_count,
                "metric_expected": metric_expected,
                "metric_path": metric_path,
                "trial_row_count": trial_table.count if trial_table is not None else None,
                "trial_row_expected": trial_table.expected if trial_table is not None else None,
                "trial_row_path": str(trial_table.trials_path) if trial_table is not None else "",
                "trace_count": traces.count if traces is not None else None,
                "trace_expected": traces.expected if traces is not None else None,
                "trace_dir": str(traces.trace_dir) if traces is not None else "",
                "queue_reason": queue.reason if queue else "",
                "queue_jobid": queue.jobid if queue else "",
                "last_line": log.last_line if log else None,
                "log_path": str(log.log_path) if log else "",
                "completed": log.completed if log else None,
                "total": log.total if log else None,
                "write_full_trace": job.write_full_trace,
            }
        )

    total_jobs = len(rows)
    states = {}
    for row in rows:
        states[row["state"]] = states.get(row["state"], 0) + 1
    running = [row for row in rows if row["state"] in {"RUNNING", "COMPLETED"} and row["progress"] is not None]
    progress_values = [float(row["progress"]) for row in running]
    total_trial_rows = sum(int(row["trial_row_count"] or 0) for row in rows)
    total_trial_expected = sum(int(row["trial_row_expected"] or 0) for row in rows)
    total_trial_progress = (
        0.0
        if total_trial_expected == 0
        else min(1.0, total_trial_rows / float(total_trial_expected))
    )
    total_trace_count = sum(int(row["trace_count"] or 0) for row in rows)
    total_trace_expected = sum(int(row["trace_expected"] or 0) for row in rows)
    total_trace_progress = (
        0.0
        if total_trace_expected == 0
        else min(1.0, total_trace_count / float(total_trace_expected))
    )
    return {
        "total_jobs": total_jobs,
        "states": states,
        "rows": rows,
        "progress_mean": sum(progress_values) / len(progress_values) if progress_values else None,
        "progress_min": min(progress_values) if progress_values else None,
        "progress_max": max(progress_values) if progress_values else None,
        "trial_row_count": total_trial_rows,
        "trial_row_expected": total_trial_expected,
        "trial_row_progress": total_trial_progress,
        "trace_count": total_trace_count,
        "trace_expected": total_trace_expected,
        "trace_progress": total_trace_progress,
        "trace_enabled_jobs": sum(1 for row in rows if row["write_full_trace"]),
    }


def render_report(
    e1_root: Path,
    *,
    user: str | None = None,
    max_active_jobs: int = 5,
    progress_source: str = "auto",
    cluster: str | None = None,
) -> str:
    e1_root = Path(e1_root).resolve()
    metadata_root = e1_root / "metadata"
    logs_root = e1_root / "logs"

    jobs = load_manifest_jobs(metadata_root)
    queue_tasks = load_queue_tasks(user=user, cluster=cluster)
    log_records = scan_logs(logs_root)

    by_config: dict[int, list[ManifestJob]] = {}
    for job in jobs:
        by_config.setdefault(job.config_id, []).append(job)

    lines: list[str] = []
    lines.append(f"E1 monitor @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"root: {e1_root}")
    lines.append("")

    for config_id in sorted(by_config):
        group = _summarize_group(
            by_config[config_id],
            queue_tasks,
            log_records,
            progress_source=progress_source,
        )
        spec_label = CONFIG_LABELS.get(config_id, "unknown")
        lines.append(f"Config {config_id}: {spec_label}")
        lines.append(
            "  jobs: {total} | pending {pending} | running {running} | completed {completed} | failed {failed}".format(
                total=group["total_jobs"],
                pending=group["states"].get("PENDING", 0),
                running=group["states"].get("RUNNING", 0),
                completed=group["states"].get("COMPLETED", 0),
                failed=group["states"].get("FAILED", 0),
            )
        )
        lines.append(
            (
                "  trial rows: {count}/{expected} ({pct})"
                if group["trial_row_expected"] > 0
                else (
                    "  trace files: {count}/{expected} ({pct})"
                    if group["trace_expected"] > 0
                    else "  progress: n/a"
                )
            ).format(
                count=group["trial_row_count"] if group["trial_row_expected"] > 0 else group["trace_count"],
                expected=group["trial_row_expected"] if group["trial_row_expected"] > 0 else group["trace_expected"],
                pct=_fmt_pct(
                    group["trial_row_progress"]
                    if group["trial_row_expected"] > 0
                    else group["trace_progress"]
                ),
            )
        )
        lines.append(
            "  job progress: mean {mean} | min {minv} | max {maxv}".format(
                mean=_fmt_pct(group["progress_mean"]),
                minv=_fmt_pct(group["progress_min"]),
                maxv=_fmt_pct(group["progress_max"]),
            )
        )

        active_rows = [
            row for row in group["rows"] if row["state"] in {"RUNNING", "COMPLETED"} and row["progress"] is not None
        ]
        active_rows.sort(key=lambda row: (row["state"] != "RUNNING", row["progress"]))
        if active_rows:
            lines.append("  active jobs:")
            for row in active_rows[:max_active_jobs]:
                job: ManifestJob = row["job"]
                if row["trial_row_expected"] is not None and row["trial_row_expected"] > 0:
                    completed = row["trial_row_count"] if row["trial_row_count"] is not None else 0
                    total = row["trial_row_expected"] if row["trial_row_expected"] is not None else 0
                elif row["trace_expected"] is not None and row["trace_expected"] > 0:
                    completed = row["trace_count"] if row["trace_count"] is not None else 0
                    total = row["trace_expected"] if row["trace_expected"] is not None else 0
                else:
                    completed = row["completed"] if row["completed"] is not None else 0
                    total = row["total"] if row["total"] is not None else 0
                lines.append(
                    f"    - [{row['state'][:1]}] idx={job.array_index:03d} {job.job_name} "
                    f"{_fmt_pct(row['progress'])} "
                    f"({row['metric_label']} {completed}/{total})"
                )
                if row["last_line"]:
                    lines.append(f"      {row['last_line']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor E1 cluster progress by config and log output")
    parser.add_argument("--e1-root", type=Path, default=DEFAULT_E1_ROOT)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--user", default=os.environ.get("USER"))
    parser.add_argument(
        "--cluster",
        choices=("auto", "tinyfat", "tinygpu"),
        default="auto",
        help="Which Slurm cluster queue to query for job state.",
    )
    parser.add_argument("--max-active-jobs", type=int, default=5)
    parser.add_argument(
        "--progress-source",
        choices=("auto", "log", "trace", "trial-h5"),
        default="auto",
        help="Where to read per-job progress from. 'auto' prefers live trial rows, then logs, then traces.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cluster = args.cluster
    if cluster == "auto":
        cluster = "tinyfat" if "tinyfat" in args.e1_root.as_posix() else "tinygpu"
    try:
        while True:
            report = render_report(
                args.e1_root,
                user=args.user,
                max_active_jobs=args.max_active_jobs,
                progress_source=args.progress_source,
                cluster=cluster,
            )
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(report, end="")
            if args.once:
                return 0
            time.sleep(max(int(args.interval), 1))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
