"""Utilities to extract and compare training run history.

This module builds a normalized index from run folders that contain:
- ``main.log``
- ``checkpoints/*.everl``
- optional ``<run_name>.csv`` next to the run directory.

The index is intentionally storage-agnostic (JSONL/CSV records) so that
comparison pipelines can be built in plain Python, pandas, SQL, or notebook code.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
import csv
import fnmatch
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TS_FMT = "%Y-%m-%d %H:%M:%S"
CHECKPOINT_NUM_RE = re.compile(r"^checkpoint(?P<steps>\d+)\.everl$")

LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(?P<ms>\d+)"
    r"\s+-\s+(?P<logger>.*?)\s+-\s+(?P<level>\w+)\s+-\s+(?P<msg>.*)$"
)

USING_TOOL_RE = re.compile(r"^Using tool:\s+(?P<tool>.+)$")
LOADING_CKPT_RE = re.compile(r"^Loading checkpoint:\s+(?P<path>.+)$")
LOADED_COUNTERS_RE = re.compile(
    r"^Loaded counters "
    r"heatup=(?P<heatup>\d+) explore=(?P<explore>\d+) "
    r"update=(?P<update>\d+) eval=(?P<eval>\d+)"
    r"\s*\|\s*heatup_steps_run=(?P<heatup_run>\d+) "
    r"training_steps_target=(?P<target>\d+)"
)

UPDATE_RE = re.compile(
    r"^update / exploration:\s*"
    r"(?P<update_seconds>[0-9.]+)/(?P<explore_seconds>[0-9.]+)\s*s\s*\|\s*"
    r"(?P<update_rate>[0-9.]+)/\s*(?P<explore_rate>[0-9.]+)\s*steps/s\s*\|\s*"
    r"(?P<update_steps>\d+)/\s*(?P<explore_steps>\d+)\s*steps,\s*"
    r"(?P<episodes>\d+)\s*episodes\s*\|\s*"
    r"(?P<update_total>\d+)/\s*(?P<explore_total>\d+)\s*steps total,\s*"
    r"(?P<episodes_total>\d+)\s*episodes total$"
)

EVAL_RE = re.compile(
    r"^evaluation\s*:\s*"
    r"(?P<elapsed_seconds>[0-9.]+)s\s*\|\s*"
    r"(?P<step_rate>[0-9.]+)\s*steps/s\s*\|\s*"
    r"(?P<steps>\d+)\s*steps\s*/\s*(?P<episodes>\d+)\s*episodes\s*\|\s*"
    r"Total:\s*(?P<steps_total>\d+)\s*steps\s*/\s*(?P<episodes_total>\d+)\s*episodes$"
)

QUALITY_RE = re.compile(
    r"^Quality:\s*(?P<quality>[-+0-9.eE]+),\s*"
    r"Reward:\s*(?P<reward>[-+0-9.eE]+),\s*"
    r"Exploration steps:\s*(?P<explore_steps>\d+)"
)

RESTART_RE = re.compile(r"^Restaring Agent (?P<worker>\S+) because of (?P<reason>.+)$")

HEARTBEAT_RE = re.compile(
    r"^heartbeat\[(?P<stage>\w+)\]:\s*"
    r"heatup=(?P<heatup>\d+)\s*\(\+(?P<dheatup>\d+)/\s*(?P<dt>\d+)s\)\s*"
    r"explore=(?P<explore>\d+)\s*\(\+(?P<dexplore>\d+)/\s*(?P<dt2>\d+)s\)\s*"
    r"update=(?P<update>\d+)\s*\(\+(?P<dupdate>\d+)/\s*(?P<dt3>\d+)s\)"
)


@dataclass
class RunSummary:
    run_id: str
    run_path: str
    log_path: str
    checkpoints_path: str
    csv_path: Optional[str]

    tool: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None

    loaded_heatup: Optional[int] = None
    loaded_explore: Optional[int] = None
    loaded_update: Optional[int] = None
    loaded_eval: Optional[int] = None
    loaded_heatup_steps_run: Optional[int] = None
    training_steps_target: Optional[int] = None

    started_at: Optional[str] = None
    ended_at: Optional[str] = None

    updates_count: int = 0
    eval_count: int = 0
    quality_count: int = 0

    last_update_explore_total: Optional[int] = None
    last_update_update_total: Optional[int] = None
    last_update_episodes_total: Optional[int] = None

    last_quality: Optional[float] = None
    last_reward: Optional[float] = None
    last_quality_explore_steps: Optional[int] = None
    best_quality: Optional[float] = None
    best_quality_explore_steps: Optional[int] = None

    restart_total: int = 0
    restart_timeout: int = 0
    restart_exception: int = 0

    worker_timeout_kills: int = 0
    worker_tracebacks: int = 0
    worker_sigsegv: int = 0

    checkpoint_count: int = 0
    checkpoint_max_step: Optional[int] = None
    checkpoint_latest_path: Optional[str] = None
    best_checkpoint_path: Optional[str] = None

    csv_last_quality: Optional[float] = None
    csv_best_quality: Optional[float] = None
    csv_last_explore_steps: Optional[int] = None
    csv_training_steps: Optional[int] = None

    parent_run_id: Optional[str] = None
    chain_id: Optional[str] = None

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    runs: List[RunSummary]
    checkpoints: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    chains: List[Dict[str, Any]]


def _to_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: str) -> Optional[float]:
    try:
        v = float(value)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _ts_iso(ts: str) -> str:
    # Keep deterministic, timezone-free timestamps.
    return datetime.strptime(ts, TS_FMT).isoformat(sep=" ")


def discover_run_dirs(runs_root: Path, run_glob: str = "*") -> List[Path]:
    """Discover run directories by finding ``main.log`` files."""
    out: List[Path] = []
    for log_path in runs_root.rglob("main.log"):
        run_dir = log_path.parent
        if not fnmatch.fnmatch(run_dir.name, run_glob):
            continue
        out.append(run_dir)
    out.sort(key=lambda p: p.name)
    return out


def _parse_results_csv(csv_path: Optional[Path]) -> Dict[str, Any]:
    if csv_path is None or not csv_path.exists():
        return {
            "csv_last_quality": None,
            "csv_best_quality": None,
            "csv_last_explore_steps": None,
            "csv_training_steps": None,
        }

    last_quality: Optional[float] = None
    best_quality: Optional[float] = None
    last_steps: Optional[int] = None
    training_steps: Optional[int] = None

    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            if training_steps is None:
                training_steps = _to_int((row.get("TRAINING_STEPS") or "").strip())

            steps = _to_int((row.get("steps explore") or "").strip())
            quality = _to_float((row.get("quality") or "").strip())
            if steps is None or quality is None:
                continue
            last_steps = steps
            last_quality = quality
            if best_quality is None or quality > best_quality:
                best_quality = quality

    return {
        "csv_last_quality": last_quality,
        "csv_best_quality": best_quality,
        "csv_last_explore_steps": last_steps,
        "csv_training_steps": training_steps,
    }


def _scan_worker_logs(logs_dir: Path) -> Dict[str, int]:
    counts = {
        "worker_timeout_kills": 0,
        "worker_tracebacks": 0,
        "worker_sigsegv": 0,
    }
    if not logs_dir.exists():
        return counts

    for path in logs_dir.rglob("*.log"):
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "Killing sofa because of timeout" in line:
                        counts["worker_timeout_kills"] += 1
                    if "Traceback" in line:
                        counts["worker_tracebacks"] += 1
                    if "SIGSEGV" in line:
                        counts["worker_sigsegv"] += 1
        except OSError:
            continue
    return counts


def _parse_checkpoints(run_id: str, run_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    checkpoints_dir = run_dir / "checkpoints"
    out: List[Dict[str, Any]] = []
    summary = {
        "checkpoint_count": 0,
        "checkpoint_max_step": None,
        "checkpoint_latest_path": None,
        "best_checkpoint_path": None,
    }

    if not checkpoints_dir.exists():
        return out, summary

    latest_mtime = -1.0
    max_step: Optional[int] = None

    for ckpt in sorted(checkpoints_dir.glob("*.everl")):
        kind = "other"
        step: Optional[int] = None

        m = CHECKPOINT_NUM_RE.match(ckpt.name)
        if m:
            kind = "checkpoint"
            step = int(m.group("steps"))
            if max_step is None or step > max_step:
                max_step = step
        elif ckpt.name == "best_checkpoint.everl":
            kind = "best"
            summary["best_checkpoint_path"] = str(ckpt)

        stat = ckpt.stat()
        rec = {
            "run_id": run_id,
            "run_path": str(run_dir),
            "checkpoint_path": str(ckpt),
            "checkpoint_name": ckpt.name,
            "checkpoint_kind": kind,
            "checkpoint_step": step,
            "size_bytes": stat.st_size,
            "mtime_epoch": stat.st_mtime,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(sep=" "),
        }
        out.append(rec)

        if stat.st_mtime > latest_mtime:
            latest_mtime = stat.st_mtime
            summary["checkpoint_latest_path"] = str(ckpt)

    summary["checkpoint_count"] = len(out)
    summary["checkpoint_max_step"] = max_step
    return out, summary


def _parse_main_log(run_id: str, log_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summary: Dict[str, Any] = {
        "tool": None,
        "resume_from_checkpoint": None,
        "loaded_heatup": None,
        "loaded_explore": None,
        "loaded_update": None,
        "loaded_eval": None,
        "loaded_heatup_steps_run": None,
        "training_steps_target": None,
        "started_at": None,
        "ended_at": None,
        "updates_count": 0,
        "eval_count": 0,
        "quality_count": 0,
        "last_update_explore_total": None,
        "last_update_update_total": None,
        "last_update_episodes_total": None,
        "last_quality": None,
        "last_reward": None,
        "last_quality_explore_steps": None,
        "best_quality": None,
        "best_quality_explore_steps": None,
        "restart_total": 0,
        "restart_timeout": 0,
        "restart_exception": 0,
    }

    events: List[Dict[str, Any]] = []

    if not log_path.exists():
        return summary, events

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            m = LOG_LINE_RE.match(raw.strip())
            if not m:
                continue

            ts_raw = m.group("ts")
            ts = _ts_iso(ts_raw)
            msg = m.group("msg")
            logger = m.group("logger")
            level = m.group("level")

            if summary["started_at"] is None:
                summary["started_at"] = ts
            summary["ended_at"] = ts

            m_tool = USING_TOOL_RE.match(msg)
            if m_tool and summary["tool"] is None:
                summary["tool"] = m_tool.group("tool").strip()
                events.append(
                    {
                        "run_id": run_id,
                        "ts": ts,
                        "event": "tool",
                        "tool": summary["tool"],
                        "logger": logger,
                        "level": level,
                    }
                )
                continue

            m_load_ckpt = LOADING_CKPT_RE.match(msg)
            if m_load_ckpt:
                summary["resume_from_checkpoint"] = m_load_ckpt.group("path").strip()
                events.append(
                    {
                        "run_id": run_id,
                        "ts": ts,
                        "event": "resume_from",
                        "checkpoint_path": summary["resume_from_checkpoint"],
                        "logger": logger,
                        "level": level,
                    }
                )
                continue

            m_loaded = LOADED_COUNTERS_RE.match(msg)
            if m_loaded:
                summary["loaded_heatup"] = int(m_loaded.group("heatup"))
                summary["loaded_explore"] = int(m_loaded.group("explore"))
                summary["loaded_update"] = int(m_loaded.group("update"))
                summary["loaded_eval"] = int(m_loaded.group("eval"))
                summary["loaded_heatup_steps_run"] = int(m_loaded.group("heatup_run"))
                summary["training_steps_target"] = int(m_loaded.group("target"))
                events.append(
                    {
                        "run_id": run_id,
                        "ts": ts,
                        "event": "loaded_counters",
                        "heatup": summary["loaded_heatup"],
                        "explore": summary["loaded_explore"],
                        "update": summary["loaded_update"],
                        "eval": summary["loaded_eval"],
                        "heatup_steps_run": summary["loaded_heatup_steps_run"],
                        "training_steps_target": summary["training_steps_target"],
                        "logger": logger,
                        "level": level,
                    }
                )
                continue

            m_update = UPDATE_RE.match(msg)
            if m_update:
                summary["updates_count"] += 1
                update_total = int(m_update.group("update_total"))
                explore_total = int(m_update.group("explore_total"))
                episodes_total = int(m_update.group("episodes_total"))
                summary["last_update_update_total"] = update_total
                summary["last_update_explore_total"] = explore_total
                summary["last_update_episodes_total"] = episodes_total
                events.append(
                    {
                        "run_id": run_id,
                        "ts": ts,
                        "event": "update",
                        "update_seconds": float(m_update.group("update_seconds")),
                        "explore_seconds": float(m_update.group("explore_seconds")),
                        "update_rate": float(m_update.group("update_rate")),
                        "explore_rate": float(m_update.group("explore_rate")),
                        "update_steps": int(m_update.group("update_steps")),
                        "explore_steps": int(m_update.group("explore_steps")),
                        "episodes": int(m_update.group("episodes")),
                        "update_total": update_total,
                        "explore_total": explore_total,
                        "episodes_total": episodes_total,
                        "logger": logger,
                        "level": level,
                    }
                )
                continue

            m_eval = EVAL_RE.match(msg)
            if m_eval:
                summary["eval_count"] += 1
                events.append(
                    {
                        "run_id": run_id,
                        "ts": ts,
                        "event": "evaluation",
                        "elapsed_seconds": float(m_eval.group("elapsed_seconds")),
                        "step_rate": float(m_eval.group("step_rate")),
                        "steps": int(m_eval.group("steps")),
                        "episodes": int(m_eval.group("episodes")),
                        "steps_total": int(m_eval.group("steps_total")),
                        "episodes_total": int(m_eval.group("episodes_total")),
                        "logger": logger,
                        "level": level,
                    }
                )
                continue

            m_quality = QUALITY_RE.match(msg)
            if m_quality:
                summary["quality_count"] += 1
                quality = float(m_quality.group("quality"))
                reward = float(m_quality.group("reward"))
                explore_steps = int(m_quality.group("explore_steps"))
                summary["last_quality"] = quality
                summary["last_reward"] = reward
                summary["last_quality_explore_steps"] = explore_steps
                if summary["best_quality"] is None or quality > summary["best_quality"]:
                    summary["best_quality"] = quality
                    summary["best_quality_explore_steps"] = explore_steps
                events.append(
                    {
                        "run_id": run_id,
                        "ts": ts,
                        "event": "quality",
                        "quality": quality,
                        "reward": reward,
                        "explore_steps": explore_steps,
                        "logger": logger,
                        "level": level,
                    }
                )
                continue

            m_restart = RESTART_RE.match(msg)
            if m_restart:
                summary["restart_total"] += 1
                reason = m_restart.group("reason")
                if "timeout" in reason.lower():
                    summary["restart_timeout"] += 1
                if "exception" in reason.lower():
                    summary["restart_exception"] += 1
                events.append(
                    {
                        "run_id": run_id,
                        "ts": ts,
                        "event": "restart",
                        "worker": m_restart.group("worker"),
                        "reason": reason,
                        "logger": logger,
                        "level": level,
                    }
                )
                continue

            m_hb = HEARTBEAT_RE.match(msg)
            if m_hb:
                events.append(
                    {
                        "run_id": run_id,
                        "ts": ts,
                        "event": "heartbeat",
                        "stage": m_hb.group("stage"),
                        "heatup": int(m_hb.group("heatup")),
                        "explore": int(m_hb.group("explore")),
                        "update": int(m_hb.group("update")),
                        "dheatup": int(m_hb.group("dheatup")),
                        "dexplore": int(m_hb.group("dexplore")),
                        "dupdate": int(m_hb.group("dupdate")),
                        "dt": int(m_hb.group("dt")),
                        "logger": logger,
                        "level": level,
                    }
                )

    return summary, events


def extract_run(
    run_dir: Path,
    include_worker_scan: bool = True,
) -> Tuple[RunSummary, List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_dir = run_dir.resolve()
    run_id = run_dir.name
    log_path = run_dir / "main.log"
    checkpoints_path = run_dir / "checkpoints"
    csv_path = run_dir.parent / f"{run_id}.csv"

    parsed_log, events = _parse_main_log(run_id=run_id, log_path=log_path)
    checkpoint_records, checkpoint_summary = _parse_checkpoints(run_id=run_id, run_dir=run_dir)
    csv_summary = _parse_results_csv(csv_path if csv_path.exists() else None)
    worker_counts = _scan_worker_logs(run_dir / "logs_subprocesses") if include_worker_scan else {
        "worker_timeout_kills": 0,
        "worker_tracebacks": 0,
        "worker_sigsegv": 0,
    }

    summary = RunSummary(
        run_id=run_id,
        run_path=str(run_dir),
        log_path=str(log_path),
        checkpoints_path=str(checkpoints_path),
        csv_path=str(csv_path) if csv_path.exists() else None,
        **parsed_log,
        **worker_counts,
        **checkpoint_summary,
        **csv_summary,
    )

    return summary, checkpoint_records, events


def assign_lineage(runs: Sequence[RunSummary], checkpoints: Sequence[Dict[str, Any]]) -> None:
    """Populate ``parent_run_id`` and ``chain_id`` for each run in-place."""
    by_id = {run.run_id: run for run in runs}

    # Map checkpoint paths to producing run.
    checkpoint_owner_exact: Dict[str, str] = {}
    checkpoint_owner_by_run_and_name: Dict[Tuple[str, str], str] = {}
    checkpoint_owner_by_tail3: Dict[Tuple[str, str, str], str] = {}

    for row in checkpoints:
        ckpt_path = Path(str(row["checkpoint_path"]))
        ckpt_name = ckpt_path.name
        run_id = row["run_id"]

        checkpoint_owner_exact[str(ckpt_path.resolve())] = run_id
        checkpoint_owner_by_run_and_name[(run_id, ckpt_name)] = run_id

        parts = ckpt_path.parts
        if len(parts) >= 3:
            checkpoint_owner_by_tail3[(parts[-3], parts[-2], parts[-1])] = run_id

    for run in runs:
        parent = None
        if run.resume_from_checkpoint:
            resume_path = Path(run.resume_from_checkpoint)

            # 1) exact local path match
            key = str(resume_path.resolve())
            parent = checkpoint_owner_exact.get(key)

            # 2) if resume path comes from another machine/root, match by
            #    "<run_id>/checkpoints/<checkpoint_name>" tail
            if parent is None:
                parts = resume_path.parts
                if len(parts) >= 3:
                    parent = checkpoint_owner_by_tail3.get(
                        (parts[-3], parts[-2], parts[-1])
                    )

            # 3) explicit parse around "checkpoints" segment
            if parent is None:
                parts = list(resume_path.parts)
                if "checkpoints" in parts:
                    idx = len(parts) - 1 - parts[::-1].index("checkpoints")
                    if idx >= 1 and idx + 1 < len(parts):
                        run_hint = parts[idx - 1]
                        ckpt_name = parts[idx + 1]
                        parent = checkpoint_owner_by_run_and_name.get((run_hint, ckpt_name))
                        if parent is None and run_hint in by_id:
                            # Fallback if filename changed or only run-level match is possible.
                            parent = run_hint

        run.parent_run_id = parent

    for run in runs:
        cur = run.run_id
        seen = {cur}
        parent = by_id[cur].parent_run_id
        while parent and parent in by_id and parent not in seen:
            cur = parent
            seen.add(cur)
            parent = by_id[cur].parent_run_id
        run.chain_id = cur


def build_chain_summaries(
    runs: Sequence[RunSummary],
    target_steps_default: int = 20_000_000,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[RunSummary]] = {}
    for run in runs:
        key = run.chain_id or run.run_id
        grouped.setdefault(key, []).append(run)

    out: List[Dict[str, Any]] = []
    for chain_id, items in grouped.items():
        items_sorted = sorted(items, key=lambda r: (r.started_at or "", r.run_id))
        latest = max(items_sorted, key=lambda r: (r.ended_at or "", r.started_at or "", r.run_id))

        best_quality: Optional[float] = None
        latest_quality: Optional[float] = None
        max_checkpoint: Optional[int] = None
        max_progress: Optional[int] = None
        target_steps: Optional[int] = None

        tools = []
        restart_total = 0
        timeout_kills = 0
        tracebacks = 0
        sigsegv = 0

        for run in items_sorted:
            if run.tool:
                tools.append(run.tool)
            restart_total += run.restart_total
            timeout_kills += run.worker_timeout_kills
            tracebacks += run.worker_tracebacks
            sigsegv += run.worker_sigsegv

            if run.best_quality is not None:
                best_quality = run.best_quality if best_quality is None else max(best_quality, run.best_quality)
            if run.last_quality is not None:
                latest_quality = run.last_quality
            if run.checkpoint_max_step is not None:
                max_checkpoint = (
                    run.checkpoint_max_step
                    if max_checkpoint is None
                    else max(max_checkpoint, run.checkpoint_max_step)
                )

            run_progress = run.last_update_explore_total
            if run_progress is None:
                run_progress = run.checkpoint_max_step
            if run_progress is not None:
                max_progress = run_progress if max_progress is None else max(max_progress, run_progress)

            if run.training_steps_target:
                target_steps = run.training_steps_target if target_steps is None else max(target_steps, run.training_steps_target)

        if target_steps is None:
            target_steps = target_steps_default

        progress_pct = None
        if max_progress is not None and target_steps and target_steps > 0:
            progress_pct = (100.0 * max_progress) / target_steps

        out.append(
            {
                "chain_id": chain_id,
                "run_count": len(items_sorted),
                "run_ids": [r.run_id for r in items_sorted],
                "tools": sorted(set(tools)),
                "first_started_at": items_sorted[0].started_at,
                "last_ended_at": items_sorted[-1].ended_at,
                "latest_run_id": latest.run_id,
                "latest_quality": latest_quality,
                "best_quality": best_quality,
                "max_checkpoint_step": max_checkpoint,
                "max_progress_step": max_progress,
                "target_steps": target_steps,
                "progress_pct": progress_pct,
                "restart_total": restart_total,
                "worker_timeout_kills": timeout_kills,
                "worker_tracebacks": tracebacks,
                "worker_sigsegv": sigsegv,
            }
        )

    out.sort(key=lambda row: (row.get("progress_pct") or -1.0), reverse=True)
    return out


def enrich_checkpoints_with_run_metadata(
    checkpoints: Sequence[Dict[str, Any]],
    runs: Sequence[RunSummary],
) -> None:
    """Attach run/chain metadata to each checkpoint row in-place."""
    by_run = {run.run_id: run for run in runs}

    for row in checkpoints:
        run = by_run.get(row.get("run_id"))
        if run is None:
            continue

        row["run_tool"] = run.tool
        row["run_chain_id"] = run.chain_id
        row["run_parent_run_id"] = run.parent_run_id
        row["run_training_steps_target"] = run.training_steps_target
        row["run_last_update_explore_total"] = run.last_update_explore_total
        row["run_last_quality"] = run.last_quality
        row["run_best_quality"] = run.best_quality

        ckpt_step = row.get("checkpoint_step")
        target = run.training_steps_target
        if ckpt_step is not None and target and target > 0:
            row["checkpoint_progress_pct"] = (100.0 * float(ckpt_step)) / float(target)
        else:
            row["checkpoint_progress_pct"] = None

        latest_path = run.checkpoint_latest_path
        if latest_path:
            try:
                row["is_run_latest_checkpoint"] = (
                    Path(str(row["checkpoint_path"])).resolve()
                    == Path(str(latest_path)).resolve()
                )
            except OSError:
                row["is_run_latest_checkpoint"] = False
        else:
            row["is_run_latest_checkpoint"] = False

        row["is_run_best_checkpoint_file"] = (
            Path(str(row["checkpoint_path"])).name == "best_checkpoint.everl"
        )


def extract_history(
    runs_root: Path,
    run_glob: str = "*",
    include_worker_scan: bool = True,
    target_steps_default: int = 20_000_000,
) -> ExtractionResult:
    run_dirs = discover_run_dirs(runs_root=runs_root, run_glob=run_glob)

    run_summaries: List[RunSummary] = []
    checkpoints: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        summary, ckpts, evs = extract_run(
            run_dir=run_dir,
            include_worker_scan=include_worker_scan,
        )
        run_summaries.append(summary)
        checkpoints.extend(ckpts)
        events.extend(evs)

    assign_lineage(run_summaries, checkpoints)
    enrich_checkpoints_with_run_metadata(checkpoints, run_summaries)
    chains = build_chain_summaries(run_summaries, target_steps_default=target_steps_default)

    return ExtractionResult(
        runs=run_summaries,
        checkpoints=checkpoints,
        events=events,
        chains=chains,
    )


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_index(result: ExtractionResult, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = [r.to_record() for r in result.runs]
    checkpoint_rows = list(result.checkpoints)
    event_rows = list(result.events)
    chain_rows = list(result.chains)

    n_runs = _write_jsonl(out_dir / "runs.jsonl", run_rows)
    n_ckpt = _write_jsonl(out_dir / "checkpoints.jsonl", checkpoint_rows)
    n_events = _write_jsonl(out_dir / "events.jsonl", event_rows)
    n_chains = _write_jsonl(out_dir / "chains.jsonl", chain_rows)

    _write_csv(out_dir / "runs.csv", run_rows)
    _write_csv(out_dir / "checkpoints.csv", checkpoint_rows)
    _write_csv(out_dir / "events.csv", event_rows)
    _write_csv(out_dir / "chains.csv", chain_rows)

    manifest = {
        "generated_at": datetime.utcnow().isoformat(sep=" ") + "Z",
        "runs_count": n_runs,
        "checkpoints_count": n_ckpt,
        "events_count": n_events,
        "chains_count": n_chains,
        "files": {
            "runs_jsonl": str(out_dir / "runs.jsonl"),
            "checkpoints_jsonl": str(out_dir / "checkpoints.jsonl"),
            "events_jsonl": str(out_dir / "events.jsonl"),
            "chains_jsonl": str(out_dir / "chains.jsonl"),
            "runs_csv": str(out_dir / "runs.csv"),
            "checkpoints_csv": str(out_dir / "checkpoints.csv"),
            "events_csv": str(out_dir / "events.csv"),
            "chains_csv": str(out_dir / "chains.csv"),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
