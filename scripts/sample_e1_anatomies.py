#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steve_recommender.eval_v2.experimental_prep_scripts.sample_anatomies import (  # noqa: E402
    DEFAULT_BRANCHES,
    DEFAULT_POOL_PATH,
    sample_anatomies_from_registry,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample the 12-anatomy input set for E1")
    parser.add_argument("--pool-path", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "experimental_prep" / "sample_12_e1.json",
    )
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--strata", default="none")
    parser.add_argument("--sampling-method", default="random")
    parser.add_argument("--branches", default=",".join(DEFAULT_BRANCHES))
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    payload = sample_anatomies_from_registry(
        pool_path=args.pool_path,
        n=int(args.n),
        seed=int(args.seed),
        strata=str(args.strata),
        sampling_method=str(args.sampling_method),
        branches=tuple(part.strip() for part in str(args.branches).split(",") if part.strip()),
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[sample_e1_anatomies] wrote {len(payload['selected_anatomies'])} anatomies to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
