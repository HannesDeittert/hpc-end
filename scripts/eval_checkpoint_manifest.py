#!/usr/bin/env python3
"""Compatibility wrapper for checkpoint-manifest evaluation.

This legacy script now routes through the shared comparison core:
`steve_recommender.comparison` -> `steve_recommender.evaluation`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from steve_recommender.services.comparison_service import (
    comparison_from_dict,
    resolve_candidates,
    run_comparison,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to checkpoint_eval_manifest.json",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/eval_runs"),
        help="Output root for summary/ranking.",
    )
    p.add_argument(
        "--name",
        type=str,
        default="checkpoint_manifest_eval",
        help="Run name suffix in output directory.",
    )
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--base-seed", type=int, default=123)
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated explicit seeds (overrides --n-trials/--base-seed).",
    )
    p.add_argument("--max-episode-steps", type=int, default=1000)
    p.add_argument("--policy-device", type=str, default="cuda")
    p.add_argument(
        "--normalize-actions",
        action="store_true",
        default=True,
        help="Deprecated. Kept for CLI compatibility; ignored in wrapper mode.",
    )
    p.add_argument(
        "--no-normalize-actions",
        action="store_true",
        help="Deprecated. Kept for CLI compatibility; ignored in wrapper mode.",
    )
    p.add_argument(
        "--non-mp",
        action="store_true",
        help="Call intervention.make_non_mp() before evaluation.",
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        help="Open a Sofa window and render the wire during evaluation.",
    )
    p.add_argument(
        "--visualize-trials-per-agent",
        type=int,
        default=1,
        help="How many initial trials per agent are rendered when --visualize is set.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest and print selected agents without simulation.",
    )
    return p.parse_args()


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Manifest root must be an object")
    agents = data.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError("Manifest has no agents")
    return data


def _parse_seeds(raw: str | None, n_trials: int, base_seed: int) -> List[int]:
    if raw:
        seeds = [int(s.strip()) for s in raw.split(",") if s.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but no valid integers were parsed.")
        return seeds
    return [base_seed + i for i in range(n_trials)]


def _ranking_from_summary(summary_path: Path, ranking_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with summary_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(row)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("agent", "")), []).append(row)

    out: List[Dict[str, Any]] = []
    for agent, items in grouped.items():
        success_vals = [float(i.get("success") or 0.0) for i in items]
        reward_vals = [float(i.get("episode_reward") or 0.0) for i in items]
        steps_vals = [float(i.get("steps_total") or 0.0) for i in items]
        out.append(
            {
                "agent": agent,
                "tool": items[0].get("tool"),
                "checkpoint": items[0].get("checkpoint"),
                "n_trials": len(items),
                "quality_success_rate": sum(success_vals) / len(success_vals)
                if success_vals
                else float("nan"),
                "reward_mean": sum(reward_vals) / len(reward_vals)
                if reward_vals
                else float("nan"),
                "steps_mean": sum(steps_vals) / len(steps_vals)
                if steps_vals
                else float("nan"),
            }
        )

    out.sort(
        key=lambda r: (
            float(r["quality_success_rate"]),
            float(r["reward_mean"]),
        ),
        reverse=True,
    )

    with ranking_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(out[0].keys()) if out else []
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        for row in out:
            writer.writerow(row)

    return out


def main() -> None:
    args = _parse_args()
    manifest = _load_manifest(args.manifest)
    agents = manifest["agents"]
    seeds = _parse_seeds(args.seeds, args.n_trials, args.base_seed)

    if args.dry_run:
        print(f"manifest: {args.manifest}")
        print(f"agents:   {len(agents)}")
        print(f"seeds:    {len(seeds)}")
        for agent in agents:
            print(
                f"- {agent.get('name')} | tool={agent.get('tool')} | "
                f"step={agent.get('checkpoint_step')} | ckpt={agent.get('checkpoint')}"
            )
        return

    if args.no_normalize_actions:
        print(
            "[warn] --no-normalize-actions is ignored in wrapper mode "
            "(comparison core uses training-compatible normalization)."
        )

    if args.normalize_actions:
        # Keep silent default compatibility.
        pass

    candidates = []
    for agent in agents:
        candidates.append(
            {
                "name": str(agent.get("name") or "agent"),
                "tool": str(agent.get("tool") or ""),
                "checkpoint": str(agent.get("checkpoint") or ""),
            }
        )

    cfg_payload: Dict[str, Any] = {
        "name": args.name,
        "candidates": candidates,
        "output_root": str(args.out_root),
        "seeds": seeds,
        "n_trials": args.n_trials,
        "base_seed": args.base_seed,
        "max_episode_steps": args.max_episode_steps,
        "policy_device": args.policy_device,
        "use_non_mp_sim": bool(args.non_mp),
        "visualize": bool(args.visualize),
        "visualize_trials_per_agent": int(args.visualize_trials_per_agent),
    }

    cfg = comparison_from_dict(cfg_payload)
    resolved = resolve_candidates(cfg)
    out_dir = run_comparison(cfg)

    summary_path = out_dir / "summary.csv"
    ranking_path = out_dir / "ranking.csv"
    ranking_rows = _ranking_from_summary(summary_path, ranking_path)

    # Keep useful wrapper artifacts.
    (out_dir / "manifest_used.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "resolved_candidates.json").write_text(
        json.dumps(
            [
                {
                    "name": c.name,
                    "tool": c.tool,
                    "checkpoint": str(c.checkpoint),
                    "agent_ref": c.agent_ref,
                    "source": c.source,
                }
                for c in resolved
            ],
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"out_dir:  {out_dir}")
    print(f"summary:  {summary_path}")
    print(f"ranking:  {ranking_path}")
    for row in ranking_rows:
        print(
            f"- {row['agent']}: success_rate={row['quality_success_rate']:.4f}, "
            f"reward_mean={row['reward_mean']:.4f}, steps_mean={row['steps_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
