from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from steve_recommender.comparison import load_comparison_config
from steve_recommender.comparison.pipeline import (
    resolved_candidates_to_dicts,
    resolve_candidates,
    run_comparison,
)
from steve_recommender.services.library_service import list_agent_refs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tool recommendation comparison")
    parser.add_argument(
        "--config",
        required=False,
        help="Path to comparison YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve candidates only and print selected tool/checkpoint pairs.",
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
    parser.add_argument(
        "--list-agent-refs",
        action="store_true",
        help="Print all available registry refs (model/wire:agent) and exit.",
    )
    args = parser.parse_args()
    if not args.list_agent_refs and not args.config:
        parser.error("--config is required unless --list-agent-refs is used.")
    return args


def main() -> None:
    args = parse_args()

    if args.list_agent_refs:
        refs = list_agent_refs()
        print("[compare] available agent refs")
        if not refs:
            print("(none)")
        else:
            for ref in refs:
                print(ref)
        return

    cfg = load_comparison_config(Path(args.config))
    if args.visualize is not None:
        cfg = replace(cfg, visualize=bool(args.visualize))
    if args.visualize_trials_per_agent is not None:
        if args.visualize_trials_per_agent < 1:
            raise ValueError("--visualize-trials-per-agent must be >= 1")
        cfg = replace(cfg, visualize_trials_per_agent=int(args.visualize_trials_per_agent))

    resolved = resolve_candidates(cfg)

    if args.dry_run:
        print("[compare] resolved candidates")
        print(json.dumps(resolved_candidates_to_dicts(resolved), indent=2))
        return

    out = run_comparison(cfg)
    print(f"[compare] done: {out}")
    print(f"[compare] summary: {out / 'summary.csv'}")
    print(f"[compare] trials: {out / 'trials'}")
    print(f"[compare] report: {out / 'report.md'}")
    print(f"[compare] report_json: {out / 'report.json'}")


if __name__ == "__main__":
    main()
