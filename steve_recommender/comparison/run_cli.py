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
        "--n-trials",
        type=int,
        default=None,
        help="Override config and run exactly N trials per candidate.",
    )
    parser.add_argument(
        "--policy-device",
        default=None,
        help="Override config and run policy inference on this torch device (e.g. cpu, cuda).",
    )
    force_dbg_group = parser.add_mutually_exclusive_group()
    force_dbg_group.add_argument(
        "--visualize-force-debug",
        dest="visualize_force_debug",
        action="store_true",
        help=(
            "When visualizing, show live contact/force telemetry in the "
            "window title and print per-step summaries."
        ),
    )
    force_dbg_group.add_argument(
        "--no-visualize-force-debug",
        dest="visualize_force_debug",
        action="store_false",
        help="Disable live force debug overlay even if enabled in config.",
    )
    parser.set_defaults(visualize_force_debug=None)
    parser.add_argument(
        "--visualize-force-top-k",
        type=int,
        default=None,
        help="How many top force segments to show in debug mode.",
    )
    parser.add_argument(
        "--force-mode",
        choices=["passive", "intrusive_lcp", "constraint_projected_si_validated"],
        default=None,
        help="Override force extraction mode from config.",
    )
    req_group = parser.add_mutually_exclusive_group()
    req_group.add_argument(
        "--force-required",
        dest="force_required",
        action="store_true",
        help="Fail the run if force extraction is unavailable.",
    )
    req_group.add_argument(
        "--force-optional",
        dest="force_required",
        action="store_false",
        help="Allow runs without force extraction (safety term is excluded).",
    )
    parser.set_defaults(force_required=None)
    parser.add_argument(
        "--force-contact-epsilon",
        type=float,
        default=None,
        help="Contact epsilon used in required-force consistency checks.",
    )
    parser.add_argument(
        "--force-plugin-path",
        default=None,
        help="Optional explicit path to libSofaWireForceMonitor.so.",
    )
    parser.add_argument(
        "--list-agent-refs",
        action="store_true",
        help="Print all available registry refs (model/version:agent) and exit.",
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

    force_cfg = cfg.force_extraction
    if args.force_mode is not None:
        force_cfg = replace(force_cfg, mode=args.force_mode)
    if args.force_required is not None:
        force_cfg = replace(force_cfg, required=bool(args.force_required))
    if args.force_contact_epsilon is not None:
        if args.force_contact_epsilon < 0:
            raise ValueError("--force-contact-epsilon must be >= 0")
        force_cfg = replace(force_cfg, contact_epsilon=float(args.force_contact_epsilon))
    if args.force_plugin_path is not None:
        force_cfg = replace(force_cfg, plugin_path=str(args.force_plugin_path))
    if (
        force_cfg.mode == "constraint_projected_si_validated"
        and force_cfg.units is None
    ):
        raise ValueError(
            "force_extraction.units are required for mode='constraint_projected_si_validated'. "
            "Set them in your config file."
        )

    if args.visualize is not None:
        cfg = replace(cfg, visualize=bool(args.visualize))
    if args.visualize_trials_per_agent is not None:
        if args.visualize_trials_per_agent < 1:
            raise ValueError("--visualize-trials-per-agent must be >= 1")
        cfg = replace(cfg, visualize_trials_per_agent=int(args.visualize_trials_per_agent))
    if args.n_trials is not None:
        if args.n_trials < 1:
            raise ValueError("--n-trials must be >= 1")
        cfg = replace(cfg, n_trials=int(args.n_trials))
    if args.policy_device is not None:
        cfg = replace(cfg, policy_device=str(args.policy_device))
    if args.visualize_force_debug is not None:
        cfg = replace(cfg, visualize_force_debug=bool(args.visualize_force_debug))
    if args.visualize_force_top_k is not None:
        if args.visualize_force_top_k < 1:
            raise ValueError("--visualize-force-top-k must be >= 1")
        cfg = replace(cfg, visualize_force_debug_top_k=int(args.visualize_force_top_k))
    cfg = replace(cfg, force_extraction=force_cfg)

    print(
        "[compare] force_extraction mode={mode} required={required} plugin_path={plugin} "
        "visualize_force_debug={vfd} top_k={top_k}".format(
            mode=cfg.force_extraction.mode,
            required=int(cfg.force_extraction.required),
            plugin=cfg.force_extraction.plugin_path or "(auto)",
            vfd=int(cfg.visualize_force_debug),
            top_k=cfg.visualize_force_debug_top_k,
        )
    )

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
