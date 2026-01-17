"""Domain models for training/evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TrainingRunInfo:
    name: str
    path: Path
    log_path: Optional[Path]
    results_csv: Optional[Path]
    checkpoint_dir: Optional[Path]


@dataclass(frozen=True)
class EvaluationRunInfo:
    name: str
    path: Path
    summary_csv: Optional[Path]
    report_md: Optional[Path]
