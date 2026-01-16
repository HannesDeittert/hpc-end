from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from steve_recommender.storage import repo_root


@dataclass(frozen=True)
class TrainedAgentCheckpoint:
    """A discovered (tool, checkpoint) pair that can be evaluated."""

    tool_ref: str
    run_name: str
    checkpoint_path: Path
    run_dir: Optional[Path] = None


_TOOL_RE = re.compile(r"^\[train\]\s+tool=(?P<tool>\S+)")
_PATHS_RE = re.compile(
    r"^\[train\]\s+logs=(?P<log>\S+)\s+results=(?P<results>\S+)\s+checkpoints=(?P<ckpt>\S+)"
)
_CHECKPOINT_NUM_RE = re.compile(r"^checkpoint(?P<steps>\d+)\.everl$")


def _pick_checkpoint(checkpoints_dir: Path) -> Optional[Path]:
    """Pick a sensible checkpoint to evaluate (best_checkpoint > latest checkpointN)."""

    best = checkpoints_dir / "best_checkpoint.everl"
    if best.exists():
        return best

    numbered: List[tuple[int, Path]] = []
    for p in checkpoints_dir.glob("checkpoint*.everl"):
        m = _CHECKPOINT_NUM_RE.match(p.name)
        if not m:
            continue
        numbered.append((int(m.group("steps")), p))
    if numbered:
        return max(numbered, key=lambda t: t[0])[1]

    any_everl = sorted(checkpoints_dir.glob("*.everl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return any_everl[0] if any_everl else None


def _parse_nohup_log(path: Path) -> Optional[TrainedAgentCheckpoint]:
    tool_ref: Optional[str] = None
    checkpoints_dir: Optional[Path] = None
    run_dir: Optional[Path] = None

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None

    for line in lines:
        if tool_ref is None:
            m = _TOOL_RE.match(line)
            if m:
                tool_ref = m.group("tool")
                continue

        m = _PATHS_RE.match(line)
        if m:
            checkpoints_dir = Path(m.group("ckpt"))
            run_dir = Path(m.group("log")).parent
            break

    if not tool_ref or not checkpoints_dir:
        return None

    ckpt = _pick_checkpoint(checkpoints_dir)
    if ckpt is None:
        return None

    run_name = path.stem
    if run_name.startswith("nohup_"):
        run_name = run_name[len("nohup_") :]
    return TrainedAgentCheckpoint(
        tool_ref=tool_ref,
        run_name=run_name,
        checkpoint_path=ckpt,
        run_dir=run_dir,
    )


def discover_trained_agents_from_nohup_logs(
    *,
    roots: Iterable[Path],
) -> List[TrainedAgentCheckpoint]:
    """Discover trained agents from our wrapper-script nohup logs.

    This is a best-effort helper: our paper training wrapper prints a structured header:
      - `[train] tool=<model/wire> ...`
      - `[train] logs=<.../run/main.log> ... checkpoints=<.../run/checkpoints>`
    We parse that to recover the mapping between tool and checkpoint folder.
    """

    agents: List[TrainedAgentCheckpoint] = []
    for root in roots:
        if not root.exists():
            continue
        for log_path in sorted(root.glob("nohup_*.log")):
            parsed = _parse_nohup_log(log_path)
            if parsed is not None:
                agents.append(parsed)
    return agents


def discover_trained_agents() -> List[TrainedAgentCheckpoint]:
    """Discover trained agent checkpoints in this repo.

    Current sources:
    - `results/paper_runs/nohup_*.log` (paper multi-worker)
    - `results/paper_runs_single/nohup_*.log` (if used)
    """

    root = repo_root() / "results"
    return discover_trained_agents_from_nohup_logs(
        roots=[
            root / "paper_runs",
            root / "paper_runs_single",
        ]
    )


def trained_checkpoints_by_tool() -> Dict[str, List[TrainedAgentCheckpoint]]:
    """Return discovered checkpoints grouped by tool, sorted by recency."""

    by_tool: Dict[str, List[TrainedAgentCheckpoint]] = {}
    for agent in discover_trained_agents():
        by_tool.setdefault(agent.tool_ref, []).append(agent)

    for tool, items in by_tool.items():
        items.sort(key=lambda a: a.checkpoint_path.stat().st_mtime if a.checkpoint_path.exists() else 0, reverse=True)
        by_tool[tool] = items
    return by_tool
