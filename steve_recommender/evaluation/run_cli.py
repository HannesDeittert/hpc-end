from __future__ import annotations

import argparse
from pathlib import Path

from steve_recommender.evaluation.config import load_config
from steve_recommender.evaluation.pipeline import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained stEVE_rl agents")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to evaluation YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    out = run_evaluation(cfg)
    print(f"[eval] done: {out}")
    print(f"[eval] summary: {out / 'summary.csv'}")
    print(f"[eval] trials: {out / 'trials'}")
    print(f"[eval] report: {out / 'report.md'}")
    print(f"[eval] report_json: {out / 'report.json'}")


if __name__ == "__main__":
    main()
