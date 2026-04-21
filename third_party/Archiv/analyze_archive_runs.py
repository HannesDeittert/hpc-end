#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml


TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")
HEATUP_RE = re.compile(
    r"heatup\s*:\s*([0-9.]+)s \|\s*([0-9.]+) steps/s \|\s*([0-9]+) steps /\s*([0-9]+) episodes"
)
UPDATE_RE = re.compile(
    r"update / exploration:\s*([0-9.]+)/\s*([0-9.]+) s \|\s*([0-9.]+)/\s*([0-9.]+) steps/s \|\s*([0-9]+)/\s*([0-9]+) steps,\s*([0-9]+) episodes"
)
EVAL_RE = re.compile(
    r"evaluation\s*:\s*([0-9.]+)s \|\s*([0-9.]+) steps/s \|\s*([0-9]+) steps /\s*([0-9]+) episodes"
)
QUALITY_RE = re.compile(
    r"Quality:\s*([0-9.eE+-]+), Reward:\s*([0-9.eE+-]+), Exploration steps:\s*([0-9]+)"
)


@dataclass
class RunSummary:
    run_dir: str
    scenario_guess: str | None
    training_mode: str | None
    result_family: str | None
    env_train_class: str | None
    env_eval_class: str | None
    vessel_tree_train_class: str | None
    vessel_tree_eval_class: str | None
    device_count_train: int | None
    device_count_eval: int | None
    device_names_train: list[str]
    device_names_eval: list[str]
    train_max_steps: int | None
    eval_max_steps: int | None
    n_worker: int | None
    worker_device: str | None
    trainer_device: str | None
    learning_rate: float | None
    hidden_layers: list[int]
    embedder_nodes: int | None
    embedder_layers: int | None
    configured_heatup_steps: float | None
    explore_steps_between_eval: float | None
    consecutive_explore_episodes: int | None
    batch_size: int | None
    update_per_explore_step: float | None
    duration: str | None
    heatup_seconds: float | None
    heatup_steps_per_second: float | None
    heatup_logged_steps: int | None
    heatup_episodes: int | None
    update_steps_per_second_mean: float | None
    update_steps_per_second_median: float | None
    explore_steps_per_second_mean: float | None
    explore_steps_per_second_median: float | None
    steps_per_episode_mean: float | None
    steps_per_episode_median: float | None
    eval_steps_per_second_mean: float | None
    eval_steps_per_second_median: float | None
    first_nonzero_update_steps_per_second: float | None
    last_quality: float | None
    last_reward: float | None
    last_quality_exploration_steps: int | None
    updates_count: int
    evals_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze archived training runs in third_party/Archiv."
    )
    parser.add_argument(
        "archive_root",
        nargs="?",
        default=Path(__file__).resolve().parent,
        help="Archive root folder to scan. Defaults to the folder containing this script.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the human summary.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    # The archive files contain PyYAML-specific tags such as !!python/tuple.
    # These files are local and trusted, so unsafe_load is acceptable here.
    data = yaml.unsafe_load(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not deserialize to a dict")
    return data


def dig(data: Any, *keys: str) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def extract_max_steps(data: Any) -> int | None:
    matches: list[int] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            max_steps = node.get("max_steps")
            if isinstance(max_steps, (int, float)):
                matches.append(int(max_steps))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    return matches[0] if matches else None


def classify_scenario(vessel_tree_class: str | None, device_count: int | None) -> str | None:
    if device_count and device_count >= 2:
        return "DualDeviceNav-style"
    if not vessel_tree_class:
        return None

    normalized = vessel_tree_class.lower()
    if "aorticarchrandom" in normalized:
        return "ArchVariety-style"
    if normalized.endswith(".vmr") or ".vmr" in normalized:
        return "BasicWireNav-style"
    return "Unknown"


def classify_training_mode(run_dir: Path, checkpoint_folder: str | None) -> str | None:
    haystacks = [run_dir.name.lower()]
    if checkpoint_folder:
        haystacks.append(checkpoint_folder.lower())

    if any("optuna" in text or "opti" in text or "hyperparam" in text for text in haystacks):
        return "optimize"
    if any("train" in text for text in haystacks):
        return "train"
    return None


def normalize_result_family(checkpoint_folder: str | None) -> str | None:
    if not checkpoint_folder:
        return None
    marker = "/results/"
    if marker in checkpoint_folder:
        return checkpoint_folder.split(marker, 1)[1].rsplit("/", 2)[0]
    return checkpoint_folder


def parse_duration(first_timestamp: datetime | None, last_timestamp: datetime | None) -> str | None:
    if first_timestamp is None or last_timestamp is None:
        return None
    return str(last_timestamp - first_timestamp)


def parse_main_log(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    heatup: dict[str, Any] | None = None
    updates: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    last_quality: dict[str, Any] | None = None

    for line in lines:
        ts_match = TIMESTAMP_RE.match(line)
        if ts_match:
            ts = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
            if first_timestamp is None:
                first_timestamp = ts
            last_timestamp = ts

        if heatup is None:
            heat_match = HEATUP_RE.search(line)
            if heat_match:
                heatup = {
                    "seconds": float(heat_match.group(1)),
                    "steps_per_second": float(heat_match.group(2)),
                    "steps": int(heat_match.group(3)),
                    "episodes": int(heat_match.group(4)),
                }
                continue

        update_match = UPDATE_RE.search(line)
        if update_match:
            updates.append(
                {
                    "update_seconds": float(update_match.group(1)),
                    "explore_seconds": float(update_match.group(2)),
                    "update_steps_per_second": float(update_match.group(3)),
                    "explore_steps_per_second": float(update_match.group(4)),
                    "update_steps": int(update_match.group(5)),
                    "explore_steps": int(update_match.group(6)),
                    "episodes": int(update_match.group(7)),
                }
            )
            continue

        eval_match = EVAL_RE.search(line)
        if eval_match:
            evals.append(
                {
                    "seconds": float(eval_match.group(1)),
                    "steps_per_second": float(eval_match.group(2)),
                    "steps": int(eval_match.group(3)),
                    "episodes": int(eval_match.group(4)),
                }
            )
            continue

        quality_match = QUALITY_RE.search(line)
        if quality_match:
            last_quality = {
                "quality": float(quality_match.group(1)),
                "reward": float(quality_match.group(2)),
                "exploration_steps": int(quality_match.group(3)),
            }

    nonzero_updates = [row for row in updates if row["update_steps"] > 0]
    steps_per_episode = [
        row["explore_steps"] / row["episodes"] for row in updates if row["episodes"] > 0
    ]
    eval_steps_per_second = [row["steps_per_second"] for row in evals]
    nonzero_update_speeds = [row["update_steps_per_second"] for row in nonzero_updates]
    explore_speeds = [row["explore_steps_per_second"] for row in updates]

    return {
        "duration": parse_duration(first_timestamp, last_timestamp),
        "heatup": heatup,
        "updates": updates,
        "evals": evals,
        "last_quality": last_quality,
        "update_steps_per_second_mean": mean(nonzero_update_speeds) if nonzero_update_speeds else None,
        "update_steps_per_second_median": median(nonzero_update_speeds) if nonzero_update_speeds else None,
        "explore_steps_per_second_mean": mean(explore_speeds) if explore_speeds else None,
        "explore_steps_per_second_median": median(explore_speeds) if explore_speeds else None,
        "steps_per_episode_mean": mean(steps_per_episode) if steps_per_episode else None,
        "steps_per_episode_median": median(steps_per_episode) if steps_per_episode else None,
        "eval_steps_per_second_mean": mean(eval_steps_per_second) if eval_steps_per_second else None,
        "eval_steps_per_second_median": median(eval_steps_per_second) if eval_steps_per_second else None,
        "first_nonzero_update_steps_per_second": (
            nonzero_updates[0]["update_steps_per_second"] if nonzero_updates else None
        ),
    }


def round_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def summarize_run(run_dir: Path) -> RunSummary:
    runner_data = load_yaml(run_dir / "runner.yml")
    env_train_data = load_yaml(run_dir / "env_train.yml")
    env_eval_data = load_yaml(run_dir / "env_eval.yml")
    log_stats = parse_main_log(run_dir / "main.log")

    params = runner_data.get("agent_parameter_for_result_file", {})
    train_devices = dig(env_train_data, "intervention", "devices") or []
    eval_devices = dig(env_eval_data, "intervention", "devices") or []
    checkpoint_folder = runner_data.get("checkpoint_folder")

    return RunSummary(
        run_dir=str(run_dir),
        scenario_guess=classify_scenario(
            dig(env_train_data, "intervention", "vessel_tree", "_class"),
            len(train_devices),
        ),
        training_mode=classify_training_mode(run_dir, checkpoint_folder),
        result_family=normalize_result_family(checkpoint_folder),
        env_train_class=dig(runner_data, "agent", "env_train", "_class"),
        env_eval_class=dig(runner_data, "agent", "env_eval", "_class"),
        vessel_tree_train_class=dig(env_train_data, "intervention", "vessel_tree", "_class"),
        vessel_tree_eval_class=dig(env_eval_data, "intervention", "vessel_tree", "_class"),
        device_count_train=len(train_devices),
        device_count_eval=len(eval_devices),
        device_names_train=[device.get("name", "?") for device in train_devices if isinstance(device, dict)],
        device_names_eval=[device.get("name", "?") for device in eval_devices if isinstance(device, dict)],
        train_max_steps=extract_max_steps(env_train_data),
        eval_max_steps=extract_max_steps(env_eval_data),
        n_worker=dig(runner_data, "agent", "n_worker"),
        worker_device=dig(runner_data, "agent", "worker_device"),
        trainer_device=dig(runner_data, "agent", "trainer_device"),
        learning_rate=params.get("learning_rate"),
        hidden_layers=params.get("hidden_layers", []),
        embedder_nodes=params.get("embedder_nodes"),
        embedder_layers=params.get("embedder_layers"),
        configured_heatup_steps=params.get("HEATUP_STEPS"),
        explore_steps_between_eval=params.get("EXPLORE_STEPS_BTW_EVAL"),
        consecutive_explore_episodes=params.get("CONSECUTIVE_EXPLORE_EPISODES"),
        batch_size=params.get("BATCH_SIZE"),
        update_per_explore_step=params.get("UPDATE_PER_EXPLORE_STEP"),
        duration=log_stats["duration"],
        heatup_seconds=(log_stats["heatup"] or {}).get("seconds"),
        heatup_steps_per_second=(log_stats["heatup"] or {}).get("steps_per_second"),
        heatup_logged_steps=(log_stats["heatup"] or {}).get("steps"),
        heatup_episodes=(log_stats["heatup"] or {}).get("episodes"),
        update_steps_per_second_mean=round_or_none(log_stats["update_steps_per_second_mean"]),
        update_steps_per_second_median=round_or_none(log_stats["update_steps_per_second_median"]),
        explore_steps_per_second_mean=round_or_none(log_stats["explore_steps_per_second_mean"]),
        explore_steps_per_second_median=round_or_none(log_stats["explore_steps_per_second_median"]),
        steps_per_episode_mean=round_or_none(log_stats["steps_per_episode_mean"]),
        steps_per_episode_median=round_or_none(log_stats["steps_per_episode_median"]),
        eval_steps_per_second_mean=round_or_none(log_stats["eval_steps_per_second_mean"]),
        eval_steps_per_second_median=round_or_none(log_stats["eval_steps_per_second_median"]),
        first_nonzero_update_steps_per_second=round_or_none(
            log_stats["first_nonzero_update_steps_per_second"]
        ),
        last_quality=(log_stats["last_quality"] or {}).get("quality"),
        last_reward=(log_stats["last_quality"] or {}).get("reward"),
        last_quality_exploration_steps=(log_stats["last_quality"] or {}).get("exploration_steps"),
        updates_count=len(log_stats["updates"]),
        evals_count=len(log_stats["evals"]),
    )


def find_run_dirs(root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for runner_path in sorted(root.rglob("runner.yml")):
        if "__MACOSX" in runner_path.parts:
            continue
        run_dirs.append(runner_path.parent)
    return run_dirs


def format_list(values: list[Any]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def print_human_summary(summaries: list[RunSummary]) -> None:
    if not summaries:
        print("No archived runs found.")
        return

    for summary in summaries:
        print(f"=== {Path(summary.run_dir).name} ===")
        print(f"run_dir                 : {summary.run_dir}")
        print(f"scenario_guess          : {summary.scenario_guess}")
        print(f"training_mode           : {summary.training_mode}")
        print(f"result_family           : {summary.result_family}")
        print(f"env_train_class         : {summary.env_train_class}")
        print(f"env_eval_class          : {summary.env_eval_class}")
        print(f"vessel_tree_train_class : {summary.vessel_tree_train_class}")
        print(f"vessel_tree_eval_class  : {summary.vessel_tree_eval_class}")
        print(
            f"devices                 : train={summary.device_count_train} {format_list(summary.device_names_train)} | "
            f"eval={summary.device_count_eval} {format_list(summary.device_names_eval)}"
        )
        print(
            f"step_caps               : train={summary.train_max_steps} | eval={summary.eval_max_steps}"
        )
        print(
            f"workers/devices         : n_worker={summary.n_worker} | worker={summary.worker_device} | trainer={summary.trainer_device}"
        )
        print(
            f"architecture            : lr={summary.learning_rate} | hidden={format_list(summary.hidden_layers)} | "
            f"embedder_nodes={summary.embedder_nodes} | embedder_layers={summary.embedder_layers}"
        )
        print(
            f"training_schedule       : heatup_steps={summary.configured_heatup_steps} | explore_steps_between_eval={summary.explore_steps_between_eval} | "
            f"consecutive_explore_episodes={summary.consecutive_explore_episodes} | batch_size={summary.batch_size} | "
            f"update_per_explore_step={summary.update_per_explore_step}"
        )
        print(
            f"log_stats               : duration={summary.duration} | heatup_sps={summary.heatup_steps_per_second} | "
            f"update_sps_mean={summary.update_steps_per_second_mean} | explore_sps_mean={summary.explore_steps_per_second_mean} | "
            f"steps_per_episode_mean={summary.steps_per_episode_mean} | eval_sps_mean={summary.eval_steps_per_second_mean}"
        )
        print(
            f"quality                 : last_quality={summary.last_quality} | last_reward={summary.last_reward} | "
            f"exploration_steps={summary.last_quality_exploration_steps}"
        )
        print(
            f"counts                  : updates={summary.updates_count} | evals={summary.evals_count} | "
            f"first_nonzero_update_sps={summary.first_nonzero_update_steps_per_second}"
        )
        print()


def main() -> None:
    args = parse_args()
    archive_root = Path(args.archive_root).resolve()
    summaries = [summarize_run(run_dir) for run_dir in find_run_dirs(archive_root)]

    if args.json:
        print(json.dumps([asdict(summary) for summary in summaries], indent=2))
        return

    print_human_summary(summaries)


if __name__ == "__main__":
    main()
