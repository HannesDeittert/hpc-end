#!/usr/bin/env python3
"""Physically evaluate checkpoint agents from a manifest.

The manifest is produced by ``scripts/build_checkpoint_eval_manifest.py`` and
already links each checkpoint to the tool used in training.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from steve_recommender.adapters import eve_rl


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
        help="Map policy actions from [-1,1] to env.action_space (training-compatible).",
    )
    p.add_argument(
        "--no-normalize-actions",
        action="store_true",
        help="Disable action mapping and pass policy actions directly to env.step.",
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


def _reset_env(env: Any, seed: int) -> Any:
    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def _step_env(env: Any, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminal, truncation, info = out
        return obs, float(reward), bool(terminal), bool(truncation), dict(info or {})
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), False, dict(info or {})
    raise RuntimeError(f"Unexpected step() return signature: {type(out)} / {out}")


def _safe_info_bool(info: Dict[str, Any], key: str) -> bool:
    try:
        return bool(info.get(key, False))
    except Exception:
        return False


def _action_dt_s_from_env(env: Any) -> float:
    try:
        fluoro = env.intervention.fluoroscopy
        image_freq = float(getattr(fluoro, "image_frequency"))
        if image_freq > 0:
            return 1.0 / image_freq
    except Exception:
        pass
    return float("nan")


def _action_space_bounds(env: Any) -> Tuple[np.ndarray, np.ndarray]:
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    return low, high


def _device_velocity_limit(env: Any) -> Tuple[float, float]:
    try:
        dev = env.intervention.devices[0]
        vel = getattr(dev, "velocity_limit", None)
        if vel is None:
            return float("nan"), float("nan")
        return float(vel[0]), float(vel[1])
    except Exception:
        return float("nan"), float("nan")


def _eval_one_agent(
    *,
    agent: Dict[str, Any],
    seeds: List[int],
    max_episode_steps: int,
    policy_device: torch.device,
    non_mp: bool,
    visualize: bool,
    visualize_trials_per_agent: int,
    normalize_actions: bool,
) -> List[Dict[str, Any]]:
    checkpoint = str(agent["checkpoint"])
    algo = eve_rl.algo.AlgoPlayOnly.from_checkpoint(checkpoint)
    algo.to(policy_device)
    env = eve_rl.util.get_env_from_checkpoint(checkpoint, "eval")
    rows: List[Dict[str, Any]] = []
    visualizer = None

    try:
        # SofaPygame requires in-process simulation.
        if non_mp or visualize:
            env.intervention.make_non_mp()
        if visualize:
            from eve.visualisation import SofaPygame  # lazy import

            env.intervention.normalize_action = True
            visualizer = SofaPygame(env.intervention, env.interim_target)
            env.visualisation = visualizer

        if hasattr(env, "truncation") and hasattr(env.truncation, "max_steps"):
            env.truncation.max_steps = int(max_episode_steps)

        dt_s = _action_dt_s_from_env(env)
        action_low, action_high = _action_space_bounds(env)
        vel0, vel1 = _device_velocity_limit(env)
        print(
            "[env] "
            f"dt_s={dt_s:.6f} "
            f"action_low={action_low.tolist()} action_high={action_high.tolist()} "
            f"velocity_limit=({vel0}, {vel1}) "
            f"normalize_actions={normalize_actions}"
        )
        for i, seed in enumerate(seeds):
            t0 = time.time()

            obs = _reset_env(env, seed=seed)
            obs_flat, _ = eve_rl.util.flatten_obs(obs)
            algo.reset()

            steps_total = 0
            episode_reward = 0.0
            success = False
            info_last: Dict[str, Any] = {}

            while True:
                action = algo.get_eval_action(obs_flat)
                env_action = np.asarray(action, dtype=np.float32).reshape(env.action_space.shape)
                if normalize_actions:
                    env_action = (env_action + 1.0) / 2.0 * (
                        env.action_space.high - env.action_space.low
                    ) + env.action_space.low

                obs, reward, terminal, truncation, info = _step_env(env, env_action)
                obs_flat, _ = eve_rl.util.flatten_obs(obs)

                steps_total += 1
                episode_reward += float(reward)
                success = success or _safe_info_bool(info, "success")
                info_last = info

                if visualize and i < int(visualize_trials_per_agent):
                    env.render()

                if terminal or truncation:
                    break

            wall_time_s = time.time() - t0
            sim_time_s = float(steps_total) * dt_s if np.isfinite(dt_s) else float("nan")

            rows.append(
                {
                    "agent": agent["name"],
                    "chain_id": agent.get("chain_id"),
                    "run_id": agent.get("run_id"),
                    "tool": agent.get("tool"),
                    "checkpoint": checkpoint,
                    "checkpoint_step": agent.get("checkpoint_step"),
                    "trial": i + 1,
                    "seed": seed,
                    "success": int(success),
                    "steps_total": int(steps_total),
                    "episode_reward": float(episode_reward),
                    "wall_time_s": float(wall_time_s),
                    "sim_time_s": sim_time_s,
                    "dt_s": dt_s,
                    "action_low_0": float(action_low[0]) if action_low.size > 0 else float("nan"),
                    "action_high_0": float(action_high[0]) if action_high.size > 0 else float("nan"),
                    "action_low_1": float(action_low[1]) if action_low.size > 1 else float("nan"),
                    "action_high_1": float(action_high[1]) if action_high.size > 1 else float("nan"),
                    "device_velocity_limit_0": vel0,
                    "device_velocity_limit_1": vel1,
                    "path_ratio_last": info_last.get("path_ratio"),
                    "trajectory_length_last": info_last.get("trajectory length"),
                    "avg_translation_speed_last": info_last.get("average translation speed"),
                }
            )
            print(
                f"[trial] {agent.get('name')} {i + 1}/{len(seeds)} "
                f"seed={seed} success={int(success)} reward={episode_reward:.4f} steps={steps_total}"
            )
    finally:
        try:
            algo.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        try:
            if visualizer is not None:
                visualizer.close()
        except Exception:
            pass

    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["agent"])].append(row)

    out: List[Dict[str, Any]] = []
    for agent_name, items in grouped.items():
        success = np.asarray([i["success"] for i in items], dtype=np.float64)
        reward = np.asarray([i["episode_reward"] for i in items], dtype=np.float64)
        steps = np.asarray([i["steps_total"] for i in items], dtype=np.float64)

        tool = items[0].get("tool")
        checkpoint = items[0].get("checkpoint")
        chain_id = items[0].get("chain_id")
        run_id = items[0].get("run_id")

        out.append(
            {
                "agent": agent_name,
                "chain_id": chain_id,
                "run_id": run_id,
                "tool": tool,
                "checkpoint": checkpoint,
                "n_trials": len(items),
                "quality_success_rate": float(np.mean(success)) if len(success) else float("nan"),
                "reward_mean": float(np.mean(reward)) if len(reward) else float("nan"),
                "steps_mean": float(np.mean(steps)) if len(steps) else float("nan"),
            }
        )

    out.sort(
        key=lambda r: (
            float(r["quality_success_rate"]),
            float(r["reward_mean"]),
        ),
        reverse=True,
    )
    return out


def main() -> None:
    args = _parse_args()
    manifest = _load_manifest(args.manifest)
    agents = manifest["agents"]
    normalize_actions = bool(args.normalize_actions) and not bool(args.no_normalize_actions)
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but no valid integers were parsed.")
    else:
        seeds = [args.base_seed + i for i in range(args.n_trials)]

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

    out_dir = args.out_root / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{args.name}"
    out_dir.mkdir(parents=True, exist_ok=False)

    all_rows: List[Dict[str, Any]] = []
    device = torch.device(args.policy_device)

    summary_path = out_dir / "summary.csv"
    summary_fields = [
        "agent",
        "chain_id",
        "run_id",
        "tool",
        "checkpoint",
        "checkpoint_step",
        "trial",
        "seed",
        "success",
        "steps_total",
        "episode_reward",
        "wall_time_s",
        "sim_time_s",
        "dt_s",
        "action_low_0",
        "action_high_0",
        "action_low_1",
        "action_high_1",
        "device_velocity_limit_0",
        "device_velocity_limit_1",
        "path_ratio_last",
        "trajectory_length_last",
        "avg_translation_speed_last",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        summary_writer = csv.DictWriter(f, fieldnames=summary_fields, delimiter=";")
        summary_writer.writeheader()

        for agent in agents:
            print(
                f"[eval] {agent.get('name')} | tool={agent.get('tool')} | "
                f"checkpoint={agent.get('checkpoint')}"
            )
            rows = _eval_one_agent(
                agent=agent,
                seeds=seeds,
                max_episode_steps=args.max_episode_steps,
                policy_device=device,
                non_mp=bool(args.non_mp),
                visualize=bool(args.visualize),
                visualize_trials_per_agent=args.visualize_trials_per_agent,
                normalize_actions=normalize_actions,
            )
            all_rows.extend(rows)
            for row in rows:
                summary_writer.writerow(row)
            f.flush()

    ranking_rows = _aggregate(all_rows)

    _write_csv(out_dir / "ranking.csv", ranking_rows)
    (out_dir / "ranking.json").write_text(
        json.dumps(ranking_rows, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "manifest_used.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"out_dir:  {out_dir}")
    print(f"summary:  {out_dir / 'summary.csv'}")
    print(f"ranking:  {out_dir / 'ranking.csv'}")
    for row in ranking_rows:
        print(
            f"- {row['agent']}: success_rate={row['quality_success_rate']:.4f}, "
            f"reward_mean={row['reward_mean']:.4f}, steps_mean={row['steps_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
