#!/usr/bin/env python3
"""Compare extracted training history chains.

This script reads the index written by ``scripts/export_archvar_history.py``.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from steve_recommender.analysis.training_history import load_jsonl


def _fmt_float(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def _fmt_int(v: Optional[int]) -> str:
    if v is None:
        return "-"
    return str(v)


def _match(value: str, regex: Optional[re.Pattern]) -> bool:
    if regex is None:
        return True
    return bool(regex.search(value))


def _load_chain_rows(index_dir: Path) -> List[Dict[str, Any]]:
    chains = load_jsonl(index_dir / "chains.jsonl")
    if not chains:
        raise FileNotFoundError(
            f"No chains.jsonl found in {index_dir}. Run export_archvar_history.py first."
        )
    return chains


def _sort_chains(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            float(r.get("progress_pct")) if r.get("progress_pct") is not None else -1.0,
            int(r.get("max_progress_step")) if r.get("max_progress_step") is not None else -1,
        ),
        reverse=True,
    )


def _print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        ("chain_id", 34),
        ("runs", 4),
        ("progress", 9),
        ("max_step", 10),
        ("best_q", 8),
        ("latest_q", 8),
        ("restarts", 8),
        ("timeouts", 8),
        ("tools", 30),
    ]

    def cut(text: str, width: int) -> str:
        return text if len(text) <= width else text[: width - 1] + "~"

    header_line = " ".join(name.ljust(width) for name, width in headers)
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        tools = ",".join(row.get("tools") or [])
        cols = {
            "chain_id": str(row.get("chain_id", "")),
            "runs": str(row.get("run_count", "")),
            "progress": _fmt_float(row.get("progress_pct"), 2),
            "max_step": _fmt_int(row.get("max_progress_step")),
            "best_q": _fmt_float(row.get("best_quality"), 4),
            "latest_q": _fmt_float(row.get("latest_quality"), 4),
            "restarts": _fmt_int(row.get("restart_total")),
            "timeouts": _fmt_int(row.get("worker_timeout_kills")),
            "tools": tools,
        }
        print(
            " ".join(cut(cols[name], width).ljust(width) for name, width in headers)
        )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--index-dir",
        type=Path,
        default=Path("logs/hpc/history_index"),
        help="Directory created by export_archvar_history.py",
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
        help="Optional regex filter on the joined tools list.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=20,
        help="How many rows to print after filtering/sorting.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path to write filtered chain rows as CSV.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    rows = _load_chain_rows(args.index_dir)

    chain_re = re.compile(args.chain_regex) if args.chain_regex else None
    tool_re = re.compile(args.tool_regex) if args.tool_regex else None

    filtered: List[Dict[str, Any]] = []
    for row in rows:
        chain_id = str(row.get("chain_id", ""))
        tools = ",".join(row.get("tools") or [])
        if not _match(chain_id, chain_re):
            continue
        if not _match(tools, tool_re):
            continue
        filtered.append(row)

    filtered = _sort_chains(filtered)
    if args.top > 0:
        filtered = filtered[: args.top]

    if not filtered:
        print("No chain rows match the requested filters.")
        return

    _print_table(filtered)

    if args.out_csv:
        _write_csv(args.out_csv, filtered)
        print(f"\nWrote: {args.out_csv}")


if __name__ == "__main__":
    main()
