#!/usr/bin/env python3
"""Merge baseline/non-force-aware training run chains into canonical wire folders.

Output layout:
    data/archive/agents/non_force_aware/<wire_model>/<wire_shape>/merged

Each merged folder contains hardlinked/copied artifacts from one or more source
training stages:
- checkpoints/checkpoint*.everl
- checkpoints/best_checkpoint.everl chosen from the best stage-best evaluation
- checkpoints/latest_replay_buffer.everl from the final stage, when present
- env_train.yml and env_eval.yml from the final stage
- merged main.log
- reward CSVs and subprocess logs
- merge_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_force_aware_run_chains import (  # noqa: E402
    RUN_DIR_RE,
    StageSummary,
    _copy_numbered_checkpoints,
    _copy_subprocess_logs,
    _copy_top_level_csvs,
    _merge_main_logs,
    _stage_summary,
    _try_link,
)

WIRE_SOURCES: Dict[Tuple[str, str], str] = {
    ("amplatz_super_stiff", "gentle"): "complete/complete_amplatz_gentle",
    ("amplatz_super_stiff", "standard_j"): "complete/complete_amplatz_standardj",
    ("amplatz_super_stiff", "straight"): "complete/complete_amplatz_straight",
    ("amplatz_super_stiff", "strong_hook"): "complete/complete_amplatz_stronghook",
    ("amplatz_super_stiff", "tight_j"): "complete/complete_amplatz_tightj",
    ("steve_default", "gentle"): "complete/complete_steve_gentle",
    ("steve_default", "standard_j"): "complete/complete_steve_standard_j",
    ("steve_default", "straight"): "complete/complete_steve_straight",
    ("steve_default", "strong_hook"): "complete/complete_steve_stronghook",
    ("steve_default", "tight_j"): "complete/2026-04-22_070329_crusher_archvar_steve_tight_j_cap150_450_full_nw12_20260422_070328",
    ("universal_ii", "gentle"): "complete/complete_universaalii_gentle",
    ("universal_ii", "standard_j"): "complete/complete_universalii_standard_j",
    ("universal_ii", "straight"): "complete/complete_universalii_straight",
    ("universal_ii", "strong_hook"): "complete/complete_universalii_stronghook",
    ("universal_ii", "tight_j"): "complete/2026-04-24_212216_crusher_archvar_universalii_tight_j_cap150_450_full_nw22_20260424_212215",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, default=Path("data/archive/agents"))
    ap.add_argument("--output-root", type=Path, default=Path("data/archive/agents/non_force_aware"))
    ap.add_argument("--wire-model", action="append", default=None)
    ap.add_argument("--wire-shape", action="append", default=None)
    ap.add_argument("--force", action="store_true", help="Overwrite existing merged folders.")
    return ap.parse_args()


def _stage_sort_key(path: Path) -> Tuple[int, str]:
    # Plain completed one-stage directories sort before timestamped resume stages only when used as children.
    if path.name == "original":
        return (0, "")
    return (1, path.name)


def _list_source_stages(source_dir: Path) -> List[Path]:
    if (source_dir / "checkpoints").is_dir():
        return [source_dir]
    stages = [p for p in source_dir.iterdir() if p.is_dir() and (p.name == "original" or RUN_DIR_RE.match(p.name))]
    if not stages:
        stages = [p for p in source_dir.iterdir() if p.is_dir() and (p / "checkpoints").is_dir()]
    stages.sort(key=_stage_sort_key)
    return stages


def _choose_best_stage(stage_summaries: Sequence[StageSummary]) -> Optional[StageSummary]:
    return max(
        [stage for stage in stage_summaries if stage.best_eval and stage.best_checkpoint_path],
        key=lambda stage: (
            stage.best_eval.quality,
            stage.best_eval.reward,
            stage.best_eval.exploration_steps,
        ),
        default=None,
    )


def _write_merge_manifest(
    *,
    wire_model: str,
    wire_shape: str,
    source_dir: Path,
    merged_dir: Path,
    stage_summaries: Sequence[StageSummary],
    chosen_best_stage: Optional[StageSummary],
) -> None:
    payload = {
        "schema_version": 1,
        "training_type": "non_force_aware",
        "wire_model": wire_model,
        "wire_shape": wire_shape,
        "execution_wire": f"{wire_model}/{wire_shape}",
        "source_dir": str(source_dir),
        "merged_dir": str(merged_dir),
        "stages": [
            {
                "name": stage.name,
                "path": str(stage.path),
                "checkpoint_steps": stage.checkpoint_steps,
                "latest_checkpoint_step": stage.latest_checkpoint_step,
                "latest_progress_step": stage.latest_progress_step,
                "best_eval": asdict(stage.best_eval) if stage.best_eval else None,
            }
            for stage in stage_summaries
        ],
        "global_best_stage": (
            {
                "name": chosen_best_stage.name,
                "path": str(chosen_best_stage.path),
                "best_eval": asdict(chosen_best_stage.best_eval) if chosen_best_stage.best_eval else None,
            }
            if chosen_best_stage
            else None
        ),
    }
    (merged_dir / "merge_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def merge_wire(
    *,
    source_root: Path,
    output_root: Path,
    wire_model: str,
    wire_shape: str,
    rel_source: str,
    force: bool,
) -> Path:
    source_dir = source_root / rel_source
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source for {wire_model}/{wire_shape}: {source_dir}")
    stages = _list_source_stages(source_dir)
    if not stages:
        raise FileNotFoundError(f"No stage directories found for {wire_model}/{wire_shape}: {source_dir}")

    merged_dir = output_root / wire_model / wire_shape / "merged"
    if merged_dir.exists():
        if not force:
            raise FileExistsError(f"{merged_dir} exists. Pass --force to rebuild.")
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    stage_summaries = [_stage_summary(stage) for stage in stages]
    chosen_best_stage = _choose_best_stage(stage_summaries)
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

    _write_merge_manifest(
        wire_model=wire_model,
        wire_shape=wire_shape,
        source_dir=source_dir,
        merged_dir=merged_dir,
        stage_summaries=stage_summaries,
        chosen_best_stage=chosen_best_stage,
    )
    return merged_dir


def main() -> int:
    args = _parse_args()
    selected = []
    for (wire_model, wire_shape), rel_source in WIRE_SOURCES.items():
        if args.wire_model and wire_model not in args.wire_model:
            continue
        if args.wire_shape and wire_shape not in args.wire_shape:
            continue
        selected.append((wire_model, wire_shape, rel_source))

    if not selected:
        raise RuntimeError("No wires selected")

    merged = []
    for wire_model, wire_shape, rel_source in selected:
        print(f"merging {wire_model}/{wire_shape} <- {rel_source}")
        merged.append(
            merge_wire(
                source_root=args.source_root,
                output_root=args.output_root,
                wire_model=wire_model,
                wire_shape=wire_shape,
                rel_source=rel_source,
                force=args.force,
            )
        )

    index = {
        "schema_version": 1,
        "training_type": "non_force_aware",
        "n_wires": len(merged),
        "wires": [str(path) for path in merged],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"merged_wires={len(merged)}")
    print(f"output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
