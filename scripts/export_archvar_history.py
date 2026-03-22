#!/usr/bin/env python3
"""Extract normalized run/checkpoint history for comparison pipelines.

Example:
  python3 scripts/export_archvar_history.py \
    --runs-root /home/woody/.../results/eve_paper/neurovascular/aorta/gw_only/arch_vmr_94 \
    --out-dir logs/hpc/history_index
"""

from __future__ import annotations

import argparse
from pathlib import Path

from steve_recommender.analysis import extract_history, write_index


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results/eve_paper/neurovascular/aorta/gw_only/arch_vmr_94"),
        help="Directory that contains run folders (each run has a main.log).",
    )
    p.add_argument(
        "--run-glob",
        type=str,
        default="*",
        help="Filter run directory names with fnmatch syntax (default: *).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/hpc/history_index"),
        help="Where to write JSONL/CSV index files.",
    )
    p.add_argument(
        "--target-steps",
        type=int,
        default=20_000_000,
        help="Default training target steps used for progress percentage.",
    )
    p.add_argument(
        "--skip-worker-scan",
        action="store_true",
        help="Skip scanning logs_subprocesses/*.log for timeout/traceback/sigsegv counters.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    result = extract_history(
        runs_root=args.runs_root,
        run_glob=args.run_glob,
        include_worker_scan=not args.skip_worker_scan,
        target_steps_default=args.target_steps,
    )

    manifest = write_index(result, args.out_dir)

    print(f"runs_root: {args.runs_root}")
    print(f"out_dir:   {args.out_dir}")
    print(f"runs:      {manifest['runs_count']}")
    print(f"chains:    {manifest['chains_count']}")
    print(f"checkpoints: {manifest['checkpoints_count']}")
    print(f"events:      {manifest['events_count']}")
    print(f"manifest:    {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
