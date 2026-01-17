"""Service helpers for browsing training/evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from steve_recommender.domain import EvaluationRunInfo, TrainingRunInfo


def list_training_runs(root: Path | str = "results/paper_runs") -> List[TrainingRunInfo]:
    base = Path(root)
    if not base.exists():
        return []
    runs: List[TrainingRunInfo] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        results_csv = child.with_suffix(".csv")
        log_path = child / "main.log"
        checkpoint_dir = child / "checkpoints"
        runs.append(
            TrainingRunInfo(
                name=child.name,
                path=child,
                log_path=log_path if log_path.exists() else None,
                results_csv=results_csv if results_csv.exists() else None,
                checkpoint_dir=checkpoint_dir if checkpoint_dir.exists() else None,
            )
        )
    return runs


def list_evaluation_runs(root: Path | str = "results/eval_runs") -> List[EvaluationRunInfo]:
    base = Path(root)
    if not base.exists():
        return []
    runs: List[EvaluationRunInfo] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        summary = child / "summary.csv"
        report = child / "report.md"
        runs.append(
            EvaluationRunInfo(
                name=child.name,
                path=child,
                summary_csv=summary if summary.exists() else None,
                report_md=report if report.exists() else None,
            )
        )
    return runs


def tail_text(path: Path, max_lines: int = 200) -> str:
    if max_lines <= 0:
        raise ValueError("max_lines must be > 0")
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def latest_checkpoint(checkpoint_dir: Optional[Path]) -> Optional[Path]:
    if checkpoint_dir is None or not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("*.everl"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None
