"""Filesystem helpers for train_v2."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple


LATEST_REPLAY_BUFFER_NAME = "latest_replay_buffer.everl"


def mkdir_recursive(path: Path) -> None:
    """Create one path and its parents if needed."""

    path.mkdir(parents=True, exist_ok=True)


def get_result_checkpoint_config_and_log_path(
    *, all_results_folder: Path, name: str
) -> Tuple[Path, Path, Path, Path]:
    """Create result CSV + checkpoint folder + run folder + log path."""

    today = datetime.today().strftime("%Y-%m-%d")
    current_time = datetime.today().strftime("%H%M%S")
    main_resultfile = all_results_folder / f"{today}_{current_time}_{name}.csv"

    file_id = 0
    while main_resultfile.exists():
        file_id += 1
        main_resultfile = (
            all_results_folder / f"{today}_{current_time}_{name}_{file_id}.csv"
        )

    results_folder = main_resultfile.with_suffix("")
    checkpoint_folder = results_folder / "checkpoints"
    mkdir_recursive(checkpoint_folder)
    log_file = results_folder / "main.log"
    return main_resultfile, checkpoint_folder, results_folder, log_file
