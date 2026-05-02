#!/usr/bin/env python3
"""Tail multiple HPC training logs with colored prefixes."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, List


DEFAULT_RESULTS = Path(
    "/home/woody/iwhr/iwhr106h/master-project/results/eve_paper/neurovascular/aorta/gw_only/arch_vmr_94"
)

COLORS = [
    "\033[38;5;39m",
    "\033[38;5;208m",
    "\033[38;5;46m",
    "\033[38;5;199m",
    "\033[38;5;226m",
    "\033[38;5;81m",
    "\033[38;5;213m",
    "\033[38;5;118m",
    "\033[38;5;220m",
]
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


class LogState:
    def __init__(self, job_id: str, run_dir: Path, color: str, initial_lines: int) -> None:
        self.job_id = job_id
        self.run_dir = run_dir
        self.log_path = run_dir / "main.log"
        self.color = color
        self.label = self._build_label()
        self.offset = 0
        self.initial_lines = initial_lines

    def _build_label(self) -> str:
        name = self.run_dir.name
        suffix = f"_{self.job_id}"
        if name.endswith(suffix):
            name = name[: -len(suffix)]
        parts = name.split("_")
        if len(parts) > 2 and parts[0][:4].isdigit():
            name = "_".join(parts[2:])
        return name[-30:]

    def exists(self) -> bool:
        return self.log_path.exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tail multiple HPC training logs with colored prefixes."
    )
    parser.add_argument("job_ids", nargs="+", help="Slurm job IDs to follow.")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Results directory. Default: {DEFAULT_RESULTS}",
    )
    parser.add_argument(
        "--initial-lines",
        type=int,
        default=8,
        help="Show the last N lines from each log before following.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="Polling interval in seconds.",
    )
    return parser.parse_args()


def find_run_dir(results_dir: Path, job_id: str) -> Path:
    matches = sorted(results_dir.glob(f"*_{job_id}"))
    if not matches:
        raise FileNotFoundError(f"No run directory found for job {job_id} in {results_dir}")
    return matches[-1]


def tail_lines(path: Path, count: int) -> List[str]:
    lines: Deque[str] = deque(maxlen=count)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines.append(line.rstrip("\n"))
    return list(lines)


def print_header(states: Iterable[LogState]) -> None:
    print(f"{BOLD}Tracking training logs{RESET}")
    for state in states:
        prefix = f"{state.color}{state.job_id} {state.label}{RESET}"
        print(f"  {prefix}")
        print(f"  {DIM}{state.log_path}{RESET}")
    print()


def print_prefixed(state: LogState, line: str) -> None:
    prefix = f"{state.color}[{state.job_id} {state.label}]{RESET}"
    print(f"{prefix} {line}", flush=True)


def show_initial_lines(states: Iterable[LogState]) -> None:
    for state in states:
        if not state.exists():
            print_prefixed(state, f"{DIM}waiting for {state.log_path}{RESET}")
            continue
        for line in tail_lines(state.log_path, state.initial_lines):
            print_prefixed(state, line)
        state.offset = state.log_path.stat().st_size


def follow(states: List[LogState], poll_seconds: float) -> None:
    keep_running = True

    def stop_handler(signum: int, frame: object) -> None:
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    while keep_running:
        any_output = False
        for state in states:
            if not state.exists():
                continue

            file_size = state.log_path.stat().st_size
            if file_size < state.offset:
                state.offset = 0

            if file_size == state.offset:
                continue

            with state.log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(state.offset)
                for raw_line in handle:
                    print_prefixed(state, raw_line.rstrip("\n"))
                    any_output = True
                state.offset = handle.tell()

        if not any_output:
            time.sleep(poll_seconds)


def main() -> int:
    args = parse_args()

    if not args.results.exists():
        print(f"Results directory does not exist: {args.results}", file=sys.stderr)
        return 1

    states: List[LogState] = []
    for idx, job_id in enumerate(args.job_ids):
        run_dir = find_run_dir(args.results, job_id)
        states.append(
            LogState(
                job_id=job_id,
                run_dir=run_dir,
                color=COLORS[idx % len(COLORS)],
                initial_lines=args.initial_lines,
            )
        )

    print_header(states)
    show_initial_lines(states)
    print()
    follow(states, args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
