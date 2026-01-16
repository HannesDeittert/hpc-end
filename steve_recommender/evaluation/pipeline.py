from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .config import EvaluationConfig
from .info_collectors import SofaWallForceInfo, TipStateInfo
from .intervention_factory import build_aortic_arch_intervention
from steve_recommender.steve_adapter import eve, eve_rl
from .scoring import TrialScore, aggregate_scores, score_trial


def _now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _first_true_index(values: List[Any]) -> int | None:
    """Return 1-based index of first truthy item, else None."""

    for i, v in enumerate(values):
        if bool(v):
            return i + 1
    return None


def _extract_series(
    infos: List[Dict[str, Any]],
    *,
    key: str,
    default: Any,
) -> np.ndarray:
    series = []
    for info in infos:
        if key in info:
            series.append(info[key])
        else:
            series.append(default)
    return np.asarray(series)


def _compute_velocities(pos: np.ndarray, dt_s: float) -> np.ndarray:
    """Compute per-step finite-difference velocities from positions."""

    if pos.size == 0:
        return pos.reshape((0, 3))
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"tip_pos must be (T,3), got {pos.shape}")
    vel = np.zeros_like(pos, dtype=np.float32)
    vel[1:] = (pos[1:] - pos[:-1]) / float(dt_s)
    return vel


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _is_nan(x: Any) -> bool:
    try:
        return bool(np.isnan(x))
    except Exception:
        return False


def _csv_cell(value: Any) -> Any:
    """Convert python values into stable CSV-friendly output.

    We prefer blanks over literal "nan" so spreadsheets load cleanly.
    """

    if value is None:
        return ""
    if isinstance(value, float) and (_is_nan(value) or value == float("inf") or value == float("-inf")):
        return ""
    return value


def _tip_speed_stats(tip_vel: np.ndarray) -> tuple[float, float]:
    if tip_vel.size == 0:
        return float("nan"), float("nan")
    speeds = np.linalg.norm(tip_vel, axis=1).astype(np.float32)
    return float(np.nanmax(speeds)), float(np.nanmean(speeds))


def _finite_or(x: Any, default: float) -> float:
    try:
        x = float(x)
    except Exception:
        return float(default)
    return x if np.isfinite(x) else float(default)


def _write_report_files(run_dir: Path, *, cfg: EvaluationConfig, rows: List[Dict[str, Any]]) -> None:
    """Aggregate trial rows and write report.{json,md,csv}."""

    # Group rows by agent spec (name+tool+checkpoint).
    groups: Dict[tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["agent"]), str(row["tool"]), str(row["checkpoint"]))
        groups.setdefault(key, []).append(row)

    agent_summaries: List[Dict[str, Any]] = []
    for (agent, tool, checkpoint), trials in groups.items():
        scores = np.asarray([t.get("score", np.nan) for t in trials], dtype=np.float64)
        score_mean, score_std = aggregate_scores(scores)

        success = np.asarray([t.get("success", np.nan) for t in trials], dtype=np.float64)
        success_rate = float(np.nanmean(success)) if success.size else float("nan")

        steps_total = np.asarray([t.get("steps_total", np.nan) for t in trials], dtype=np.float64)
        steps_total_mean = float(np.nanmean(steps_total)) if steps_total.size else float("nan")

        # Only count steps_to_success for successful episodes.
        steps_to_success_values: List[float] = []
        for t in trials:
            if bool(t.get("success", 0.0)) and t.get("steps_to_success") not in ("", None):
                try:
                    steps_to_success_values.append(float(t["steps_to_success"]))
                except Exception:
                    pass
        steps_to_success_mean = (
            float(np.mean(steps_to_success_values)) if steps_to_success_values else float("nan")
        )

        tip_speed_max = np.asarray([t.get("tip_speed_max_mm_s", np.nan) for t in trials], dtype=np.float64)
        tip_speed_max_mean = float(np.nanmean(tip_speed_max)) if tip_speed_max.size else float("nan")

        wall_force_max = np.asarray([t.get("wall_force_max", np.nan) for t in trials], dtype=np.float64)
        wall_force_max_mean = float(np.nanmean(wall_force_max)) if wall_force_max.size else float("nan")

        agent_summaries.append(
            {
                "agent": agent,
                "tool": tool,
                "checkpoint": checkpoint,
                "n_trials": len(trials),
                "success_rate": success_rate,
                "score_mean": score_mean,
                "score_std": score_std,
                "steps_total_mean": steps_total_mean,
                "steps_to_success_mean": steps_to_success_mean,
                "tip_speed_max_mean_mm_s": tip_speed_max_mean,
                "wall_force_max_mean": wall_force_max_mean,
            }
        )

    # Sort by score descending (NaNs go last).
    agent_summaries.sort(
        key=lambda r: (
            not np.isfinite(r["score_mean"]),
            -_finite_or(r["score_mean"], float("-inf")),
        ),
    )

    # JSON (for UI / programmatic consumption)
    report_json = {
        "name": cfg.name,
        "generated_at": datetime.now().isoformat(),
        "scoring": asdict(cfg.scoring),
        "n_trials": cfg.n_trials,
        "agents": agent_summaries,
        "summary_csv": str(run_dir / "summary.csv"),
        "trials_dir": str(run_dir / "trials"),
    }
    _write_json(run_dir / "report.json", report_json)

    # CSV (easy to load into pandas)
    report_csv_path = run_dir / "report.csv"
    fields = list(agent_summaries[0].keys()) if agent_summaries else []
    with report_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        for row in agent_summaries:
            writer.writerow({k: _csv_cell(v) for k, v in row.items()})

    # Markdown (human-readable)
    report_md_path = run_dir / "report.md"
    lines: List[str] = []
    lines.append(f"# Evaluation Report: `{cfg.name}`")
    lines.append("")
    lines.append(f"- Trials per agent: `{cfg.n_trials}`")
    lines.append(f"- Scoring mode: `{cfg.scoring.mode}`")
    lines.append("")
    if not agent_summaries:
        lines.append("_No results recorded._")
    else:
        lines.append("## Agent summary (sorted by score)")
        lines.append("")
        lines.append(
            "| agent | success_rate | score_mean | score_std | steps_total_mean | steps_to_success_mean | tip_speed_max_mean (mm/s) | wall_force_max_mean |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        )
        for r in agent_summaries:
            lines.append(
                "| {agent} | {success_rate:.3f} | {score_mean:.3f} | {score_std:.3f} | {steps_total_mean:.1f} | {steps_to_success_mean:.1f} | {tip_speed_max_mean_mm_s:.2f} | {wall_force_max_mean:.3f} |".format(
                    agent=r["agent"],
                    success_rate=_finite_or(r["success_rate"], float("nan")),
                    score_mean=_finite_or(r["score_mean"], float("nan")),
                    score_std=_finite_or(r["score_std"], float("nan")),
                    steps_total_mean=_finite_or(r["steps_total_mean"], float("nan")),
                    steps_to_success_mean=_finite_or(r["steps_to_success_mean"], float("nan")),
                    tip_speed_max_mean_mm_s=_finite_or(r["tip_speed_max_mean_mm_s"], float("nan")),
                    wall_force_max_mean=_finite_or(r["wall_force_max_mean"], float("nan")),
                )
            )
        lines.append("")
        lines.append(f"- Raw trials: `{run_dir / 'summary.csv'}`")
        lines.append(f"- Time series: `{run_dir / 'trials'}`")
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(cfg: EvaluationConfig) -> Path:
    """Run the evaluation and write results to disk.

    Returns the created run directory path (useful for UI integration).
    """

    # Output folders
    out_root = Path(cfg.output_root)
    run_dir = out_root / f"{_now_tag()}_{cfg.name}"
    trials_dir = run_dir / "trials"
    run_dir.mkdir(parents=True, exist_ok=False)
    trials_dir.mkdir(parents=True, exist_ok=False)

    # Save config for reproducibility
    _write_json(run_dir / "config.json", asdict(cfg))

    # Prepare summary CSV
    summary_path = run_dir / "summary.csv"
    summary_fields = [
        "agent",
        "tool",
        "checkpoint",
        "trial",
        "seed",
        "success",
        "steps_total",
        "steps_to_success",
        "episode_reward",
        "path_ratio_last",
        "trajectory_length_last",
        "avg_translation_speed_last",
        "tip_speed_max_mm_s",
        "tip_speed_mean_mm_s",
        "wall_time_s",
        "sim_time_s",
        "wall_lcp_max_abs_max",
        "wall_lcp_sum_abs_mean",
        "wall_wire_force_norm_max",
        "wall_wire_force_norm_mean",
        "wall_collision_force_norm_max",
        "wall_collision_force_norm_mean",
        "wall_force_max",
        "score",
        "score_success",
        "score_efficiency",
        "score_safety",
        "score_smoothness",
        "npz_path",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, delimiter=";")
        writer.writeheader()

        trial_rows: List[Dict[str, Any]] = []

        # Same seeds for each agent for fair comparison.
        seeds = [cfg.base_seed + i for i in range(cfg.n_trials)]

        for agent_spec in cfg.agents:
            # Build intervention/environment for this tool/anatomy.
            if cfg.anatomy.type != "aortic_arch":
                raise ValueError(f"Unsupported anatomy.type: {cfg.anatomy.type}")

            intervention, action_dt_s = build_aortic_arch_intervention(
                tool_ref=agent_spec.tool,
                anatomy=cfg.anatomy,
            )

            # Switch simulation mode.
            if cfg.use_non_mp_sim:
                intervention.make_non_mp()
            else:
                intervention.make_mp()

            # Build an environment equivalent to the training setup, but with extra info.
            # We copy the BenchEnv construction to keep observation/reward identical.
            start = eve.start.InsertionPoint(intervention)
            pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)

            tracking = eve.observation.Tracking2D(intervention, n_points=3, resolution=2)
            tracking = eve.observation.wrapper.NormalizeTracking2DEpisode(tracking, intervention)
            tracking = eve.observation.wrapper.Memory(
                tracking, 2, eve.observation.wrapper.MemoryResetMode.FILL
            )
            target_state = eve.observation.Target2D(intervention)
            target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
                target_state, intervention
            )
            last_action = eve.observation.LastAction(intervention)
            last_action = eve.observation.wrapper.Normalize(last_action)

            observation = eve.observation.ObsDict(
                {
                    "tracking": tracking,
                    "target": target_state,
                    "last_action": last_action,
                }
            )

            target_reward = eve.reward.TargetReached(
                intervention,
                factor=1.0,
                final_only_after_all_interim=False,
            )
            step_reward = eve.reward.Step(factor=-0.005)
            path_delta = eve.reward.PathLengthDelta(pathfinder, 0.001)
            reward = eve.reward.Combination([target_reward, path_delta, step_reward])

            terminal = eve.terminal.TargetReached(intervention)

            max_steps = eve.truncation.MaxSteps(cfg.max_episode_steps)
            vessel_end = eve.truncation.VesselEnd(intervention)
            sim_error = eve.truncation.SimError(intervention)
            truncation = eve.truncation.Combination([max_steps, vessel_end, sim_error])

            # Default metrics (same as training) + extra collectors.
            target_reached = eve.info.TargetReached(intervention, name="success")
            path_ratio = eve.info.PathRatio(pathfinder)
            steps_info = eve.info.Steps()
            trans_speed = eve.info.AverageTranslationSpeed(intervention)
            trajectory_length = eve.info.TrajectoryLength(intervention)
            extra_tip = TipStateInfo(intervention, name_prefix="tip")
            extra_forces = SofaWallForceInfo(intervention)

            info = eve.info.Combination(
                [
                    target_reached,
                    path_ratio,
                    steps_info,
                    trans_speed,
                    trajectory_length,
                    extra_tip,
                    extra_forces,
                ]
            )

            env_eval = eve.Env(
                intervention=intervention,
                observation=observation,
                reward=reward,
                terminal=terminal,
                truncation=truncation,
                start=start,
                pathfinder=pathfinder,
                visualisation=None,
                info=info,
                interim_target=None,
            )

            # Load an evaluation-only agent from checkpoint.
            device = torch.device(cfg.policy_device)
            eval_agent = eve_rl.agent.single.SingleEvalOnly.from_checkpoint(
                agent_spec.checkpoint,
                device=device,
                normalize_actions=True,
                env_eval=env_eval,
            )

            for trial_idx, seed in enumerate(seeds):
                t0 = time.perf_counter()
                episodes = eval_agent.evaluate(episodes=1, seeds=[seed])
                wall_time_s = time.perf_counter() - t0

                episode = episodes[0]
                infos = episode.infos

                # Episode-level metrics
                success_series = [info.get("success", 0.0) for info in infos]
                steps_to_success = _first_true_index(success_series)
                success = float(bool(success_series[-1])) if success_series else 0.0

                episode_reward = float(episode.episode_reward)
                steps_total = int(len(episode))
                sim_time_s = steps_total * float(action_dt_s)

                # Per-step series
                tip_pos = _extract_series(infos, key="tip_pos3d", default=[np.nan, np.nan, np.nan]).astype(np.float32)
                tip_vel = _compute_velocities(tip_pos, dt_s=float(action_dt_s))
                inserted_length = _extract_series(infos, key="tip_inserted_length", default=np.nan).astype(np.float32)
                rotation = _extract_series(infos, key="tip_rotation", default=np.nan).astype(np.float32)
                success_arr = np.asarray(success_series, dtype=np.bool_)
                path_ratio = _extract_series(infos, key="path_ratio", default=np.nan).astype(np.float32)

                wall_lcp_sum_abs = _extract_series(infos, key="wall_lcp_sum_abs", default=np.nan).astype(np.float32)
                wall_lcp_max_abs = _extract_series(infos, key="wall_lcp_max_abs", default=np.nan).astype(np.float32)
                wall_wire_force_norm = _extract_series(infos, key="wall_wire_force_norm", default=np.nan).astype(np.float32)
                wall_collision_force_norm = _extract_series(infos, key="wall_collision_force_norm", default=np.nan).astype(np.float32)

                # Some values are only meaningful as the final scalar.
                last_info = infos[-1] if infos else {}
                path_ratio_last = _safe_float(last_info.get("path_ratio", float("nan")))
                trajectory_length_last = _safe_float(last_info.get("trajectory length", float("nan")))
                avg_translation_speed_last = _safe_float(last_info.get("average translation speed", float("nan")))

                # Aggregate wall force stats (simple but robust).
                wall_lcp_max_abs_max = float(np.nanmax(wall_lcp_max_abs)) if wall_lcp_max_abs.size else float("nan")
                wall_lcp_sum_abs_mean = float(np.nanmean(wall_lcp_sum_abs)) if wall_lcp_sum_abs.size else float("nan")
                wall_wire_force_norm_max = float(np.nanmax(wall_wire_force_norm)) if wall_wire_force_norm.size else float("nan")
                wall_collision_force_norm_max = float(np.nanmax(wall_collision_force_norm)) if wall_collision_force_norm.size else float("nan")
                wall_wire_force_norm_mean = float(np.nanmean(wall_wire_force_norm)) if wall_wire_force_norm.size else float("nan")
                wall_collision_force_norm_mean = float(np.nanmean(wall_collision_force_norm)) if wall_collision_force_norm.size else float("nan")

                wall_force_max = float(
                    np.nanmax([wall_wire_force_norm_max, wall_collision_force_norm_max])
                )

                tip_speed_max, tip_speed_mean = _tip_speed_stats(tip_vel)

                # Score this trial (configurable; defaults are meant for *relative* comparisons).
                trial_score: TrialScore = score_trial(
                    scoring=cfg.scoring,
                    success=bool(success),
                    steps_to_success=steps_to_success,
                    max_episode_steps=cfg.max_episode_steps,
                    tip_speed_max_mm_s=tip_speed_max,
                    wall_wire_force_norm_max=wall_wire_force_norm_max,
                    wall_collision_force_norm_max=wall_collision_force_norm_max,
                    wall_lcp_max_abs_max=wall_lcp_max_abs_max,
                )

                # Save per-trial arrays (for later plotting/analysis).
                npz_name = f"{agent_spec.name}_trial{trial_idx:04d}_seed{seed}.npz"
                npz_path = trials_dir / npz_name
                np.savez_compressed(
                    npz_path,
                    tip_pos3d=tip_pos,
                    tip_vel3d=tip_vel,
                    inserted_length=inserted_length,
                    rotation=rotation,
                    success=success_arr,
                    path_ratio=path_ratio,
                    actions=np.asarray(episode.actions, dtype=np.float32),
                    rewards=np.asarray(episode.rewards, dtype=np.float32),
                    terminals=np.asarray(episode.terminals, dtype=np.bool_),
                    truncations=np.asarray(episode.truncations, dtype=np.bool_),
                    wall_lcp_sum_abs=wall_lcp_sum_abs,
                    wall_lcp_max_abs=wall_lcp_max_abs,
                    wall_wire_force_norm=wall_wire_force_norm,
                    wall_collision_force_norm=wall_collision_force_norm,
                    action_dt_s=float(action_dt_s),
                    seed=int(seed),
                )

                row: Dict[str, Any] = {
                    "agent": agent_spec.name,
                    "tool": agent_spec.tool,
                    "checkpoint": agent_spec.checkpoint,
                    "trial": int(trial_idx),
                    "seed": int(seed),
                    "success": float(success),
                    "steps_total": int(steps_total),
                    "steps_to_success": int(steps_to_success) if steps_to_success is not None else None,
                    "episode_reward": float(episode_reward),
                    "path_ratio_last": float(path_ratio_last),
                    "trajectory_length_last": float(trajectory_length_last),
                    "avg_translation_speed_last": float(avg_translation_speed_last),
                    "tip_speed_max_mm_s": float(tip_speed_max),
                    "tip_speed_mean_mm_s": float(tip_speed_mean),
                    "wall_time_s": float(wall_time_s),
                    "sim_time_s": float(sim_time_s),
                    "wall_lcp_max_abs_max": float(wall_lcp_max_abs_max),
                    "wall_lcp_sum_abs_mean": float(wall_lcp_sum_abs_mean),
                    "wall_wire_force_norm_max": float(wall_wire_force_norm_max),
                    "wall_wire_force_norm_mean": float(wall_wire_force_norm_mean),
                    "wall_collision_force_norm_max": float(wall_collision_force_norm_max),
                    "wall_collision_force_norm_mean": float(wall_collision_force_norm_mean),
                    "wall_force_max": float(wall_force_max),
                    "score": float(trial_score.score),
                    "score_success": float(trial_score.success),
                    "score_efficiency": float(trial_score.efficiency),
                    "score_safety": float(trial_score.safety),
                    "score_smoothness": float(trial_score.smoothness),
                    "npz_path": str(npz_path),
                }

                trial_rows.append(row)

                # Write a CSV-friendly variant of the row (rounding + blanks instead of NaN).
                writer.writerow({k: _csv_cell(row.get(k)) for k in summary_fields})
                f.flush()

            # Cleanup
            try:
                eval_agent.close()
            except Exception:
                pass

        # Report files for the full run (per-agent aggregation + overall ranking).
        _write_report_files(run_dir, cfg=cfg, rows=trial_rows)

    return run_dir
