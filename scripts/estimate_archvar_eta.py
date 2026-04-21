#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


TS_FMT = "%Y-%m-%d %H:%M:%S,%f"
TS = r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"

HEATUP_RE = re.compile(
    rf"^{TS} - .* - INFO - heatup\s*:\s*"
    r"(?P<dur>[0-9.]+)s \|"
    r"\s*(?P<sps>[0-9.]+) steps/s \|"
    r"\s*(?P<steps>\d+) steps /"
    r"\s*(?P<episodes>\d+) episodes \|"
    r"\s*Total:\s*(?P<total_steps>\d+) steps /"
    r"\s*(?P<total_episodes>\d+) episodes$"
)

UPDATE_RE = re.compile(
    rf"^{TS} - .* - INFO - update / exploration:\s*"
    r"(?P<update_dur>[0-9.]+)/\s*(?P<explore_dur>[0-9.]+) s \|"
    r"\s*(?P<update_sps>[0-9.]+)/\s*(?P<explore_sps>[0-9.]+) steps/s \|"
    r"\s*(?P<update_steps>\d+)/\s*(?P<explore_steps>\d+) steps,"
    r"\s*(?P<explore_episodes>\d+) episodes \|"
    r"\s*(?P<update_total>\d+)/\s*(?P<explore_total>\d+) steps total,"
    r"\s*(?P<episodes_total>\d+) episodes total$"
)

QUALITY_RE = re.compile(
    rf"^{TS} - .* - INFO - Quality:\s*(?P<quality>[^,]+),"
    r"\s*Reward:\s*(?P<reward>[^,]+),"
    r"\s*Exploration steps:\s*(?P<explore_steps>\d+)$"
)


@dataclass
class Heatup:
    ts: datetime
    duration: float


@dataclass
class Update:
    ts: datetime
    update_dur: float
    explore_dur: float
    update_steps: int
    explore_steps: int
    update_total: int
    explore_total: int


@dataclass
class Quality:
    ts: datetime
    explore_steps: int


@dataclass
class Summary:
    run: str
    path: Path
    current_explore: int
    training_steps: int
    progress: float
    wall_rate: Optional[float]
    eval_count: int
    avg_eval_seconds: Optional[float]
    eta_seconds: Optional[float]
    finish_at: Optional[datetime]
    note: str = ""


def parse_ts(text: str) -> datetime:
    return datetime.strptime(text, TS_FMT)


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = int(round(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fmt_finish(dt: Optional[datetime]) -> str:
    return dt.strftime("%m-%d %H:%M") if dt else "unknown"


def parse_log(path: Path):
    heatup: Optional[Heatup] = None
    updates: List[Update] = []
    qualities: List[Quality] = []

    text = path.read_text(encoding="utf-8", errors="ignore")
    for raw_line in text.splitlines():
        line = raw_line.strip()

        m = HEATUP_RE.match(line)
        if m:
            heatup = Heatup(
                ts=parse_ts(m.group("ts")),
                duration=float(m.group("dur")),
            )
            continue

        m = UPDATE_RE.match(line)
        if m:
            updates.append(
                Update(
                    ts=parse_ts(m.group("ts")),
                    update_dur=float(m.group("update_dur")),
                    explore_dur=float(m.group("explore_dur")),
                    update_steps=int(m.group("update_steps")),
                    explore_steps=int(m.group("explore_steps")),
                    update_total=int(m.group("update_total")),
                    explore_total=int(m.group("explore_total")),
                )
            )
            continue

        m = QUALITY_RE.match(line)
        if m:
            qualities.append(
                Quality(
                    ts=parse_ts(m.group("ts")),
                    explore_steps=int(m.group("explore_steps")),
                )
            )

    return heatup, updates, qualities


def summarize_log(
    path: Path, training_steps: int, explore_steps_between_eval: int
) -> Summary:
    heatup, updates, qualities = parse_log(path)
    run = path.parent.name

    if not updates:
        return Summary(
            run=run,
            path=path,
            current_explore=0,
            training_steps=training_steps,
            progress=0.0,
            wall_rate=None,
            eval_count=0,
            avg_eval_seconds=None,
            eta_seconds=None,
            finish_at=None,
            note="no update lines yet",
        )

    last_update = updates[-1]
    current_explore = last_update.explore_total
    if qualities:
        current_explore = max(current_explore, qualities[-1].explore_steps)

    last_ts = last_update.ts
    if qualities:
        last_ts = max(last_ts, qualities[-1].ts)

    if heatup is not None:
        train_start = heatup.ts
    else:
        train_start = updates[0].ts

    elapsed_post_heatup = max(1e-9, (last_ts - train_start).total_seconds())
    wall_rate = current_explore / elapsed_post_heatup if current_explore > 0 else None

    update_by_explore_total: Dict[int, Update] = {u.explore_total: u for u in updates}
    eval_durations = []
    for quality in qualities:
        update = update_by_explore_total.get(quality.explore_steps)
        if update is not None and quality.ts >= update.ts:
            eval_durations.append((quality.ts - update.ts).total_seconds())
    avg_eval = sum(eval_durations) / len(eval_durations) if eval_durations else None

    remaining = max(0, training_steps - current_explore)
    eta = None
    finish_at = None
    if wall_rate and wall_rate > 0:
        eta = remaining / wall_rate
        # Add future eval overhead if we have seen at least one eval.
        if avg_eval is not None:
            completed_intervals = current_explore // explore_steps_between_eval
            total_intervals = math.ceil(training_steps / explore_steps_between_eval)
            remaining_intervals = max(0, total_intervals - completed_intervals)
            eta += remaining_intervals * avg_eval
        finish_at = last_ts + timedelta(seconds=eta)

    return Summary(
        run=run,
        path=path,
        current_explore=current_explore,
        training_steps=training_steps,
        progress=current_explore / training_steps if training_steps else 0.0,
        wall_rate=wall_rate,
        eval_count=len(qualities),
        avg_eval_seconds=avg_eval,
        eta_seconds=eta,
        finish_at=finish_at,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate ArchVar training completion time from main.log files."
    )
    parser.add_argument("logs", nargs="+", help="One or more main.log paths")
    parser.add_argument(
        "--training-steps",
        type=int,
        default=20_000_000,
        help="Target exploration steps for a full run (default: 20000000)",
    )
    parser.add_argument(
        "--explore-steps-between-eval",
        type=int,
        default=250_000,
        help="Eval interval in exploration steps (default: 250000)",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print the resolved log path under each summary row.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summaries = [
        summarize_log(Path(item), args.training_steps, args.explore_steps_between_eval)
        for item in args.logs
    ]

    header = (
        f"{'run':44} {'progress':>16} {'rate':>11} {'evals':>7} "
        f"{'avg_eval':>10} {'eta':>10} {'finish':>11}  note"
    )
    print(header)
    print("-" * len(header))
    for item in summaries:
        progress = f"{item.current_explore:,}/{item.training_steps:,} ({item.progress:.1%})"
        rate = f"{item.wall_rate:,.1f}/s" if item.wall_rate else "unknown"
        avg_eval = fmt_duration(item.avg_eval_seconds)
        eta = fmt_duration(item.eta_seconds)
        finish = fmt_finish(item.finish_at)
        print(
            f"{item.run[:44]:44} {progress:>16} {rate:>11} {item.eval_count:>7} "
            f"{avg_eval:>10} {eta:>10} {finish:>11}  {item.note}"
        )
        if args.details:
            print(f"  {item.path}")


if __name__ == "__main__":
    main()
