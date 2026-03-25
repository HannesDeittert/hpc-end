from __future__ import annotations

import argparse
from dataclasses import replace
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
    vis_group = parser.add_mutually_exclusive_group()
    vis_group.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="Override config and open a Sofa window during evaluation runs.",
    )
    vis_group.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Override config and disable Sofa window visualization.",
    )
    parser.set_defaults(visualize=None)
    parser.add_argument(
        "--visualize-trials-per-agent",
        type=int,
        default=None,
        help="If visualization is enabled, render only the first N trials per agent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    if args.visualize is not None:
        cfg = replace(cfg, visualize=bool(args.visualize))
    if args.visualize_trials_per_agent is not None:
        if args.visualize_trials_per_agent < 1:
            raise ValueError("--visualize-trials-per-agent must be >= 1")
        cfg = replace(cfg, visualize_trials_per_agent=int(args.visualize_trials_per_agent))
    out = run_evaluation(cfg)
    print(f"[eval] done: {out}")
    print(f"[eval] summary: {out / 'summary.csv'}")
    print(f"[eval] trials: {out / 'trials'}")
    print(f"[eval] report: {out / 'report.md'}")
    print(f"[eval] report_json: {out / 'report.json'}")


if __name__ == "__main__":
    main()
