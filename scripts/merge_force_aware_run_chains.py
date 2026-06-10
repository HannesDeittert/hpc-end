#!/usr/bin/env python3
"""Merge force-aware training run chains into one synthetic run directory.

Each config directory is expected to contain:
- ``original/``
- zero or more proper working ``*_resume*`` stage directories

The merged output is written to ``merged/`` inside each config directory and
contains:
- combined ``checkpoints/`` with all numeric checkpoints
- a global ``best_checkpoint.everl`` selected from the best stage-best result
- the latest ``latest_replay_buffer.everl`` from the final stage
- concatenated ``main.log``
- linked reward CSVs and per-stage subprocess logs
- a JSON manifest describing the merge
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


QUALITY_RE = re.compile(
    r"Quality:\s*(?P<quality>-?\d+(?:\.\d+)?)\s*,\s*Reward:\s*(?P<reward>-?\d+(?:\.\d+)?)\s*,\s*Exploration steps:\s*(?P<steps>\d+)"
)
UPDATE_RE = re.compile(r"\|\s+\d+/(\d+)\s+steps total")
CHECKPOINT_STEP_RE = re.compile(r"^checkpoint(\d+)\.everl$")
RUN_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_")


@dataclass
class EvalRecord:
    quality: float
    reward: float
    exploration_steps: int
    reward_eval_csv: Optional[str]


@dataclass
class StageSummary:
    name: str
    path: Path
    checkpoint_steps: List[int]
    latest_checkpoint_step: Optional[int]
    latest_checkpoint_path: Optional[Path]
    latest_progress_step: Optional[int]
    latest_replay_buffer_path: Optional[Path]
    best_checkpoint_path: Optional[Path]
    best_eval: Optional[EvalRecord]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/archive/agents/force_aware"),
        help="Force-aware archive root containing wire/alpha config directories.",
    )
    parser.add_argument(
        "--wire",
        action="append",
        default=None,
        help="Optional wire group to restrict to (steve_default, amplatz_super_stiff, universal_ii).",
    )
    parser.add_argument(
        "--alpha",
        action="append",
        default=None,
        help="Optional alpha bucket to restrict to (a001, a005, a05).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing merged directories.",
    )
    return parser.parse_args()


def _iter_config_dirs(root: Path, wires: Optional[Sequence[str]], alphas: Optional[Sequence[str]]) -> Iterable[Path]:
    for wire_dir in sorted(root.iterdir()):
        if not wire_dir.is_dir():
            continue
        if wire_dir.name in {
            "steve_default_standard_j_relu_a001",
            "steve_default_standard_j_relu_a005",
            "steve_default_standard_j_relu_a05",
            "amplatz_super_stiff_standard_j_relu_a001",
            "amplatz_super_stiff_standard_j_relu_a005",
            "amplatz_super_stiff_standard_j_relu_a05",
            "universal_ii_standard_j_relu_a001",
            "universal_ii_standard_j_relu_a005",
            "universal_ii_standard_j_relu_a05",
        }:
            continue
        if wires and wire_dir.name not in wires:
            continue
        for alpha_dir in sorted(wire_dir.iterdir()):
            if not alpha_dir.is_dir():
                continue
            if alphas and alpha_dir.name not in alphas:
                continue
            yield alpha_dir


def _stage_sort_key(path: Path) -> Tuple[int, str]:
    if path.name == "original":
        return (0, "")
    return (1, path.name)


def _list_stage_dirs(config_dir: Path) -> List[Path]:
    stages = [p for p in config_dir.iterdir() if p.is_dir() and (p.name == "original" or RUN_DIR_RE.match(p.name))]
    stages.sort(key=_stage_sort_key)
    return stages


def _try_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _collect_quality_records(main_log: Path, reward_eval_files: Sequence[Path]) -> List[EvalRecord]:
    quality_records: List[Tuple[float, float, int]] = []
    if main_log.exists():
        for line in main_log.read_text(encoding="utf-8", errors="replace").splitlines():
            match = QUALITY_RE.search(line)
            if match:
                quality_records.append(
                    (
                        float(match.group("quality")),
                        float(match.group("reward")),
                        int(match.group("steps")),
                    )
                )

    records: List[EvalRecord] = []
    pair_count = min(len(quality_records), len(reward_eval_files))
    for idx in range(pair_count):
        quality, reward, steps = quality_records[idx]
        records.append(
            EvalRecord(
                quality=quality,
                reward=reward,
                exploration_steps=steps,
                reward_eval_csv=reward_eval_files[idx].name,
            )
        )
    for idx in range(pair_count, len(quality_records)):
        quality, reward, steps = quality_records[idx]
        records.append(
            EvalRecord(
                quality=quality,
                reward=reward,
                exploration_steps=steps,
                reward_eval_csv=None,
            )
        )
    return records


def _latest_progress_step(main_log: Path) -> Optional[int]:
    if not main_log.exists():
        return None
    latest: Optional[int] = None
    for line in main_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = UPDATE_RE.search(line)
        if match:
            latest = int(match.group(1))
    return latest


def _stage_summary(stage_dir: Path) -> StageSummary:
    checkpoint_dir = stage_dir / "checkpoints"
    numbered: List[Tuple[int, Path]] = []
    if checkpoint_dir.exists():
        for path in checkpoint_dir.glob("checkpoint*.everl"):
            match = CHECKPOINT_STEP_RE.match(path.name)
            if match:
                numbered.append((int(match.group(1)), path))
    numbered.sort(key=lambda item: item[0])
    reward_eval_files = sorted(stage_dir.glob("reward_eval_*.csv"))
    eval_records = _collect_quality_records(stage_dir / "main.log", reward_eval_files)
    best_eval = max(
        eval_records,
        key=lambda rec: (rec.quality, rec.reward, rec.exploration_steps),
        default=None,
    )
    latest_rb = checkpoint_dir / "latest_replay_buffer.everl"
    best_ckpt = checkpoint_dir / "best_checkpoint.everl"
    return StageSummary(
        name=stage_dir.name,
        path=stage_dir,
        checkpoint_steps=[step for step, _ in numbered],
        latest_checkpoint_step=numbered[-1][0] if numbered else None,
        latest_checkpoint_path=numbered[-1][1] if numbered else None,
        latest_progress_step=_latest_progress_step(stage_dir / "main.log"),
        latest_replay_buffer_path=latest_rb if latest_rb.exists() else None,
        best_checkpoint_path=best_ckpt if best_ckpt.exists() else None,
        best_eval=best_eval,
    )


def _copy_top_level_csvs(stage_dir: Path, merged_dir: Path, used_names: Dict[str, str]) -> None:
    for path in sorted(stage_dir.glob("*.csv")):
        target_name = path.name
        if target_name in used_names and used_names[target_name] != str(path):
            target_name = f"{stage_dir.name}__{path.name}"
        used_names[target_name] = str(path)
        _try_link(path, merged_dir / target_name)


def _copy_subprocess_logs(stage_dir: Path, merged_dir: Path) -> None:
    src = stage_dir / "logs_subprocesses"
    if not src.exists():
        return
    for log_file in sorted(src.glob("*")):
        if log_file.is_file():
            _try_link(log_file, merged_dir / "logs_subprocesses" / stage_dir.name / log_file.name)


def _merge_main_logs(stages: Sequence[Path], merged_main_log: Path) -> None:
    merged_main_log.parent.mkdir(parents=True, exist_ok=True)
    with merged_main_log.open("w", encoding="utf-8") as out:
        for idx, stage_dir in enumerate(stages):
            if idx:
                out.write("\n")
            out.write(f"===== STAGE {stage_dir.name} =====\n")
            main_log = stage_dir / "main.log"
            if main_log.exists():
                content = main_log.read_text(encoding="utf-8", errors="replace")
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
            else:
                out.write("(missing main.log)\n")


def _copy_numbered_checkpoints(stage_dir: Path, merged_dir: Path, manifest: Dict[str, str]) -> None:
    numbered: List[Tuple[int, Path]] = []
    for path in (stage_dir / "checkpoints").glob("checkpoint*.everl"):
        match = CHECKPOINT_STEP_RE.match(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    numbered.sort(key=lambda item: item[0])
    for _, path in numbered:
        target = merged_dir / "checkpoints" / path.name
        if target.exists():
            if manifest.get(path.name) == str(path):
                continue
            if filecmp_safe(target, path):
                continue
            target = merged_dir / "checkpoints" / f"{stage_dir.name}__{path.name}"
        manifest[target.name] = str(path)
        _try_link(path, target)


def filecmp_safe(a: Path, b: Path) -> bool:
    try:
        if a.stat().st_size != b.stat().st_size:
            return False
        with a.open("rb") as fa, b.open("rb") as fb:
            while True:
                chunk_a = fa.read(1024 * 1024)
                chunk_b = fb.read(1024 * 1024)
                if chunk_a != chunk_b:
                    return False
                if not chunk_a:
                    return True
    except OSError:
        return False


def _write_manifest(config_dir: Path, merged_dir: Path, stage_summaries: Sequence[StageSummary], chosen_best_stage: Optional[StageSummary]) -> None:
    manifest = {
        "config_dir": str(config_dir),
        "merged_dir": str(merged_dir),
        "stages": [
            {
                "name": stage.name,
                "path": str(stage.path),
                "latest_checkpoint_step": stage.latest_checkpoint_step,
                "latest_progress_step": stage.latest_progress_step,
                "best_eval": (
                    {
                        "quality": stage.best_eval.quality,
                        "reward": stage.best_eval.reward,
                        "exploration_steps": stage.best_eval.exploration_steps,
                        "reward_eval_csv": stage.best_eval.reward_eval_csv,
                    }
                    if stage.best_eval
                    else None
                ),
            }
            for stage in stage_summaries
        ],
        "global_best_stage": (
            {
                "name": chosen_best_stage.name,
                "path": str(chosen_best_stage.path),
                "quality": chosen_best_stage.best_eval.quality if chosen_best_stage.best_eval else None,
                "reward": chosen_best_stage.best_eval.reward if chosen_best_stage.best_eval else None,
                "exploration_steps": chosen_best_stage.best_eval.exploration_steps if chosen_best_stage.best_eval else None,
                "reward_eval_csv": chosen_best_stage.best_eval.reward_eval_csv if chosen_best_stage.best_eval else None,
            }
            if chosen_best_stage
            else None
        ),
    }
    (merged_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def merge_config_dir(config_dir: Path, *, force: bool) -> None:
    stages = _list_stage_dirs(config_dir)
    if not stages:
        raise ValueError(f"No stages found in {config_dir}")

    merged_dir = config_dir / "merged"
    if merged_dir.exists():
        if not force:
            raise FileExistsError(f"{merged_dir} already exists. Pass --force to rebuild.")
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    stage_summaries = [_stage_summary(stage_dir) for stage_dir in stages]
    chosen_best_stage = max(
        [stage for stage in stage_summaries if stage.best_eval and stage.best_checkpoint_path],
        key=lambda stage: (
            stage.best_eval.quality,
            stage.best_eval.reward,
            stage.best_eval.exploration_steps,
        ),
        default=None,
    )
    final_stage = stage_summaries[-1]

    for env_name in ("env_train.yml", "env_eval.yml"):
        src = final_stage.path / env_name
        if src.exists():
            _try_link(src, merged_dir / env_name)

    _merge_main_logs(stages, merged_dir / "main.log")

    csv_name_map: Dict[str, str] = {}
    checkpoint_name_map: Dict[str, str] = {}
    for stage_dir in stages:
        _copy_top_level_csvs(stage_dir, merged_dir, csv_name_map)
        _copy_subprocess_logs(stage_dir, merged_dir)
        _copy_numbered_checkpoints(stage_dir, merged_dir, checkpoint_name_map)

    if final_stage.latest_replay_buffer_path is not None:
        _try_link(final_stage.latest_replay_buffer_path, merged_dir / "checkpoints" / "latest_replay_buffer.everl")

    if chosen_best_stage is not None and chosen_best_stage.best_checkpoint_path is not None:
        _try_link(chosen_best_stage.best_checkpoint_path, merged_dir / "checkpoints" / "best_checkpoint.everl")
        _try_link(
            chosen_best_stage.best_checkpoint_path,
            merged_dir / "checkpoints" / f"best_checkpoint__{chosen_best_stage.name}.everl",
        )
        if chosen_best_stage.best_eval and chosen_best_stage.best_eval.reward_eval_csv:
            reward_eval_src = chosen_best_stage.path / chosen_best_stage.best_eval.reward_eval_csv
            if reward_eval_src.exists():
                _try_link(reward_eval_src, merged_dir / "best_checkpoint_eval.csv")

    _write_manifest(config_dir, merged_dir, stage_summaries, chosen_best_stage)


def main() -> int:
    args = _parse_args()
    config_dirs = list(_iter_config_dirs(args.root, args.wire, args.alpha))
    if not config_dirs:
        raise FileNotFoundError(f"No config directories found under {args.root}")
    for config_dir in config_dirs:
        print(f"merging {config_dir}")
        merge_config_dir(config_dir, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
