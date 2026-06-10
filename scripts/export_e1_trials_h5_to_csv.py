#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
HELPER_DIR = PROJECT_ROOT / "experiments" / "master-thesis" / "notebook_helpers"
for path in (PROJECT_ROOT, HELPER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from e1_analysis import load_trials_h5  # noqa: E402


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export every E1 trials.h5 file under a result root to a sibling trials.csv file."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "results" / "master_thesis" / "e1_tinyfat_friction_01_real",
        help="Result tree to scan recursively for trials.h5 files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing trials.csv files.",
    )
    return parser.parse_args(argv)


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    root = Path(args.root).resolve()
    h5_paths = sorted(root.rglob("trials.h5"))
    if not h5_paths:
        print(f"[export_e1_trials_h5_to_csv] no trials.h5 files found under {root}")
        return 0

    converted = 0
    skipped = 0
    failed = 0
    for h5_path in h5_paths:
        csv_path = h5_path.with_suffix(".csv")
        if csv_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            rows = load_trials_h5(h5_path)
            _write_rows_csv(csv_path, rows)
        except Exception as exc:
            failed += 1
            print(f"[export_e1_trials_h5_to_csv] failed: {h5_path} ({type(exc).__name__}: {exc})")
            continue
        converted += 1

    print(
        "[export_e1_trials_h5_to_csv] root={root} converted={converted} skipped={skipped} failed={failed}".format(
            root=root,
            converted=converted,
            skipped=skipped,
            failed=failed,
        )
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
