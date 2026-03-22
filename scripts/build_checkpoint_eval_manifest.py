#!/usr/bin/env python3
"""Build a checkpoint-evaluation manifest from history index files.

This links each selected checkpoint to the exact tool logged during training.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from steve_recommender.analysis.training_history import load_jsonl


def _match(value: str, regex: Optional[re.Pattern]) -> bool:
    if regex is None:
        return True
    return bool(regex.search(value))


def _key_latest(row: Dict[str, Any]) -> tuple[int, float]:
    step = row.get("checkpoint_step")
    step_key = int(step) if isinstance(step, int) else -1
    mtime_key = float(row.get("mtime_epoch") or 0.0)
    return (step_key, mtime_key)


def _select_rows(
    checkpoints: List[Dict[str, Any]],
    *,
    selection: str,
    chain_re: Optional[re.Pattern],
    tool_re: Optional[re.Pattern],
    min_progress_pct: Optional[float],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in checkpoints:
        chain_id = str(row.get("run_chain_id") or row.get("run_id") or "")
        tool = str(row.get("run_tool") or "")
        progress = row.get("checkpoint_progress_pct")

        if not _match(chain_id, chain_re):
            continue
        if not _match(tool, tool_re):
            continue
        if min_progress_pct is not None:
            if progress is None or float(progress) < float(min_progress_pct):
                continue
        filtered.append(row)

    if selection == "all_numeric":
        rows = [r for r in filtered if r.get("checkpoint_step") is not None]
        rows.sort(key=_key_latest, reverse=True)
        return rows

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in filtered:
        chain_id = str(row.get("run_chain_id") or row.get("run_id") or "")
        grouped.setdefault(chain_id, []).append(row)

    out: List[Dict[str, Any]] = []
    for chain_id, items in grouped.items():
        if selection == "best_file_per_chain":
            best_files = [r for r in items if r.get("is_run_best_checkpoint_file")]
            if best_files:
                out.append(max(best_files, key=_key_latest))
                continue
        out.append(max(items, key=_key_latest))

    out.sort(key=_key_latest, reverse=True)
    return out


def _to_agent(row: Dict[str, Any]) -> Dict[str, Any]:
    chain_id = str(row.get("run_chain_id") or row.get("run_id") or "")
    step = row.get("checkpoint_step")
    ckpt_name = str(row.get("checkpoint_name") or "checkpoint")
    suffix = str(step) if step is not None else ckpt_name.replace(".everl", "")
    name = f"{chain_id}__{suffix}"
    return {
        "name": name,
        "chain_id": chain_id,
        "run_id": row.get("run_id"),
        "tool": row.get("run_tool"),
        "checkpoint": row.get("checkpoint_path"),
        "checkpoint_step": row.get("checkpoint_step"),
        "checkpoint_progress_pct": row.get("checkpoint_progress_pct"),
        "run_training_steps_target": row.get("run_training_steps_target"),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--index-dir",
        type=Path,
        default=Path("logs/hpc/history_index"),
        help="Directory produced by export_archvar_history.py",
    )
    p.add_argument(
        "--selection",
        type=str,
        default="latest_per_chain",
        choices=["latest_per_chain", "best_file_per_chain", "all_numeric"],
        help="How checkpoints are selected.",
    )
    p.add_argument(
        "--chain-regex",
        type=str,
        default=None,
        help="Optional regex filter on chain_id.",
    )
    p.add_argument(
        "--tool-regex",
        type=str,
        default=None,
        help="Optional regex filter on tool.",
    )
    p.add_argument(
        "--min-progress-pct",
        type=float,
        default=None,
        help="Optional minimum checkpoint progress percent.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output manifest path (default: <index-dir>/checkpoint_eval_manifest.json).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_path = args.out or (args.index_dir / "checkpoint_eval_manifest.json")

    checkpoints = load_jsonl(args.index_dir / "checkpoints.jsonl")
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints.jsonl found in {args.index_dir}. Run export_archvar_history.py first."
        )

    chain_re = re.compile(args.chain_regex) if args.chain_regex else None
    tool_re = re.compile(args.tool_regex) if args.tool_regex else None

    selected = _select_rows(
        checkpoints,
        selection=args.selection,
        chain_re=chain_re,
        tool_re=tool_re,
        min_progress_pct=args.min_progress_pct,
    )
    agents = [_to_agent(row) for row in selected]

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(sep=" "),
        "index_dir": str(args.index_dir),
        "selection": args.selection,
        "chain_regex": args.chain_regex,
        "tool_regex": args.tool_regex,
        "min_progress_pct": args.min_progress_pct,
        "agents_count": len(agents),
        "agents": agents,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"manifest: {out_path}")
    print(f"agents:   {len(agents)}")
    for agent in agents:
        print(
            f"- {agent['name']} | tool={agent['tool']} | "
            f"step={agent['checkpoint_step']} | ckpt={agent['checkpoint']}"
        )


if __name__ == "__main__":
    main()
