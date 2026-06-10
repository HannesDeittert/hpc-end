#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import pandas as pd


def _decode(values):
    if getattr(values, "dtype", None) is not None and values.dtype.kind in {"S", "O"}:
        return [v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v for v in values]
    return values


def read_h5_trials(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        group = f["trials"]
        selected = list(columns) if columns else list(group.keys())
        data = {}
        for col in selected:
            if col not in group:
                continue
            data[col] = _decode(group[col][()])
    return pd.DataFrame(data)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert E2 trials.h5 files to sibling trials.csv files.")
    ap.add_argument("root", type=Path, nargs="?", default=Path("results/master_thesis/e2_tinyfat_finished_readable_20260510"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--columns", default=None, help="Comma-separated subset of columns. Default: all columns.")
    args = ap.parse_args()

    columns = [c.strip() for c in args.columns.split(",") if c.strip()] if args.columns else None
    h5_paths = sorted(args.root.glob("runs/config_4/*/trials.h5"))
    print(f"root={args.root}")
    print(f"trials.h5 files={len(h5_paths)}")

    converted = 0
    skipped = 0
    failed = 0
    for i, h5_path in enumerate(h5_paths, start=1):
        csv_path = h5_path.with_suffix(".csv")
        if csv_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            df = read_h5_trials(h5_path, columns=columns)
            tmp_path = csv_path.with_suffix(".csv.tmp")
            df.to_csv(tmp_path, index=False)
            tmp_path.replace(csv_path)
            converted += 1
            if converted % 25 == 0 or converted == 1:
                print(f"converted={converted} latest={csv_path}")
        except Exception as exc:
            failed += 1
            print(f"FAILED {h5_path}: {type(exc).__name__}: {exc}")

    print({"converted": converted, "skipped": skipped, "failed": failed})
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
