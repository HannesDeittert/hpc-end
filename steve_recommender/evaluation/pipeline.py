from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import EvaluationConfig
from .force_calibration import calibration_fingerprint, load_force_calibration
from .info_collectors import SofaWallForceInfo, TipStateInfo
from .intervention_factory import build_aortic_arch_intervention
from .torch_checkpoint_compat import legacy_checkpoint_load_context
from steve_recommender.adapters import eve, eve_rl
from steve_recommender.visualisation import ForceDebugSofaPygame
from .scoring import TrialScore, aggregate_scores, score_trial


def _now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _first_true_index(values: List[Any]) -> Optional[int]:
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


def _write_force_gap_report(
    *,
    out_dir: Path,
    agent: str,
    trial: int,
    seed: int,
    active_projected_count: np.ndarray,
    explicit_mapped_count: np.ndarray,
    unmapped_count: np.ndarray,
    class_counts_series: np.ndarray,
    dominant_class_series: np.ndarray,
    contact_mode_series: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps: List[Dict[str, Any]] = []
    class_totals: Dict[str, int] = {}
    n = int(
        min(
            len(active_projected_count),
            len(explicit_mapped_count),
            len(unmapped_count),
            len(class_counts_series),
            len(dominant_class_series),
            len(contact_mode_series),
        )
    )
    for step in range(n):
        active = int(active_projected_count[step])
        explicit = int(explicit_mapped_count[step])
        unmapped = int(unmapped_count[step])
        counts_raw = class_counts_series[step]
        counts = dict(counts_raw) if isinstance(counts_raw, dict) else {}
        for k, v in counts.items():
            key = str(k or "").strip() or "unknown"
            try:
                vv = int(v)
            except Exception:
                vv = 0
            if vv <= 0:
                continue
            class_totals[key] = int(class_totals.get(key, 0)) + int(vv)
        if active <= 0 and explicit <= 0 and unmapped <= 0:
            continue
        steps.append(
            {
                "step": int(step),
                "active_projected_count": active,
                "explicit_mapped_count": explicit,
                "unmapped_count": unmapped,
                "dominant_class": str(dominant_class_series[step] or "none"),
                "contact_mode": str(contact_mode_series[step] or "none"),
                "class_counts": counts,
            }
        )

    summary = {
        "agent": agent,
        "trial": int(trial),
        "seed": int(seed),
        "active_projected_count_sum": int(
            np.sum(np.asarray(active_projected_count, dtype=np.int64))
        ),
        "explicit_mapped_count_sum": int(
            np.sum(np.asarray(explicit_mapped_count, dtype=np.int64))
        ),
        "unmapped_count_sum": int(np.sum(np.asarray(unmapped_count, dtype=np.int64))),
        "class_totals": class_totals,
        "dominant_class": (
            max(class_totals.keys(), key=lambda k: int(class_totals.get(k, 0)))
            if class_totals
            else "none"
        ),
    }

    stem = f"{agent}_trial{int(trial):04d}_seed{int(seed)}"
    _write_json(
        out_dir / f"{stem}.json",
        {
            "summary": summary,
            "steps": steps,
        },
    )
    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "active_projected_count",
                "explicit_mapped_count",
                "unmapped_count",
                "dominant_class",
                "contact_mode",
                "class_counts",
            ],
            delimiter=";",
        )
        writer.writeheader()
        for row in steps:
            writer.writerow(
                {
                    **row,
                    "class_counts": json.dumps(
                        row.get("class_counts", {}), sort_keys=True
                    ),
                }
            )


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


def _stack_maybe(values: List[np.ndarray]) -> np.ndarray:
    """Stack arrays when shapes match, otherwise return object array."""

    if not values:
        return np.asarray([], dtype=np.float32)
    first_shape = values[0].shape
    if all(v.shape == first_shape for v in values):
        try:
            return np.stack(values, axis=0)
        except Exception:
            pass
    return np.asarray(values, dtype=object)


def _reconstruct_dense_wall_segment_forces(
    *,
    wall_segment_count_series: np.ndarray,
    wall_active_segment_ids_series: np.ndarray,
    wall_active_segment_force_vectors_series: np.ndarray,
) -> np.ndarray:
    """Reconstruct dense [T, Nseg, 3] wall-segment forces from sparse per-step data."""

    wall_segment_count_series = np.asarray(wall_segment_count_series, dtype=np.int32).reshape((-1,))
    n_steps = int(wall_segment_count_series.shape[0])
    if n_steps == 0:
        return np.zeros((0, 0, 3), dtype=np.float32)
    n_segments = int(np.max(wall_segment_count_series)) if wall_segment_count_series.size else 0
    n_segments = max(0, n_segments)
    dense = np.zeros((n_steps, n_segments, 3), dtype=np.float32)
    if n_segments == 0:
        return dense

    for t in range(n_steps):
        try:
            ids = np.asarray(wall_active_segment_ids_series[t], dtype=np.int64).reshape((-1,))
        except Exception:
            continue
        try:
            vecs = np.asarray(
                wall_active_segment_force_vectors_series[t], dtype=np.float32
            ).reshape((-1, 3))
        except Exception:
            continue

        m = min(ids.shape[0], vecs.shape[0])
        if m <= 0:
            continue
        ids = ids[:m]
        vecs = vecs[:m]
        valid = (ids >= 0) & (ids < n_segments)
        if not np.any(valid):
            continue
        ids = ids[valid]
        vecs = vecs[valid]
        # Sum duplicates if multiple contacts map to same segment in the same step.
        for i, seg_id in enumerate(ids):
            dense[t, int(seg_id), :] += vecs[i]

    return dense


def _peak_segment_force_metrics(
    segment_force_series: np.ndarray,
) -> tuple[np.ndarray, float, Optional[int], Optional[int]]:
    """Return per-step max segment force norm and global peak location.

    Returns:
    - per_step_max_norm: shape [T]
    - peak_norm
    - peak_step_idx (0-based) or None
    - peak_segment_idx (0-based) or None
    """

    # Fast path for dense [T, Nseg, 3].
    if isinstance(segment_force_series, np.ndarray) and segment_force_series.ndim == 3:
        if segment_force_series.size == 0:
            t = int(segment_force_series.shape[0])
            return np.zeros((t,), dtype=np.float32), float("nan"), None, None
        norms = np.linalg.norm(segment_force_series.astype(np.float32), axis=2)
        if norms.size == 0:
            return np.zeros((norms.shape[0],), dtype=np.float32), float("nan"), None, None
        per_step = np.nanmax(norms, axis=1).astype(np.float32)
        if not np.any(np.isfinite(norms)):
            return per_step, float("nan"), None, None
        flat_idx = int(np.nanargmax(norms))
        peak_step, peak_seg = np.unravel_index(flat_idx, norms.shape)
        peak_norm = float(norms[peak_step, peak_seg])
        return per_step, peak_norm, int(peak_step), int(peak_seg)

    # Generic fallback for object arrays / list-like of [Nseg_t, 3].
    try:
        n_steps = int(len(segment_force_series))
    except Exception:
        n_steps = 0
    per_step = np.zeros((n_steps,), dtype=np.float32)
    peak_norm = float("nan")
    peak_step: Optional[int] = None
    peak_seg: Optional[int] = None

    for t in range(n_steps):
        try:
            vecs = np.asarray(segment_force_series[t], dtype=np.float32).reshape((-1, 3))
        except Exception:
            continue
        if vecs.size == 0:
            continue
        norms = np.linalg.norm(vecs, axis=1)
        if norms.size == 0:
            continue
        step_max = float(np.nanmax(norms))
        per_step[t] = step_max
        if np.isfinite(step_max) and (not np.isfinite(peak_norm) or step_max > peak_norm):
            peak_norm = step_max
            peak_step = int(t)
            peak_seg = int(np.nanargmax(norms))

    return per_step, peak_norm, peak_step, peak_seg


def _validate_force_signal(
    *,
    force_available_series: np.ndarray,
    contact_count_series: np.ndarray,
    contact_detected_series: np.ndarray,
    lcp_active_count_series: np.ndarray,
    total_force_norm_series: np.ndarray,
    contact_epsilon: float,
    active_constraint_step_series: Optional[np.ndarray] = None,
) -> tuple[bool, str]:
    """Check whether force telemetry is present and internally consistent.

    Validation semantics:
    - strict check is tied to *active constraint rows* (post-response contact),
      not raw geometric contact counters.
    - geometric contact without active constraints is treated as non-binding for
      force non-zero checks, because detection can exist without resolved impulse.
    """

    available_any = bool(np.any(force_available_series.astype(bool)))
    if not available_any:
        return False, "force monitor did not report available=true"

    strict_mask: np.ndarray
    if active_constraint_step_series is not None:
        strict_mask = np.asarray(active_constraint_step_series, dtype=np.bool_)
    else:
        active_lcp_mask = lcp_active_count_series > 0
        strict_mask = active_lcp_mask
    if not np.any(strict_mask):
        # Fallback for scenes where LCP active-row telemetry is unavailable.
        strict_mask = (contact_count_series > 0) & contact_detected_series.astype(bool)
    if np.any(strict_mask):
        lcp_force_norm = total_force_norm_series[strict_mask]
        finite = lcp_force_norm[np.isfinite(lcp_force_norm)]
        if finite.size == 0:
            return False, "active contact constraints but force norms missing"
        if np.nanmax(np.abs(finite)) <= float(contact_epsilon):
            return False, (
                "active contact constraints but projected force stayed near zero "
                f"(<= {contact_epsilon})"
            )
    return True, ""


def _validate_force_vector_consistency(
    *,
    segment_force_series: np.ndarray,
    total_force_vector_series: np.ndarray,
    atol: float,
    rtol: float = 1e-3,
) -> tuple[bool, str]:
    """Check total force vector equals sum of segment vectors per step."""

    if total_force_vector_series.size == 0:
        return True, ""

    n_steps = int(total_force_vector_series.shape[0])
    for t in range(n_steps):
        try:
            total = np.asarray(total_force_vector_series[t], dtype=np.float64).reshape((3,))
            seg = np.asarray(segment_force_series[t], dtype=np.float64).reshape((-1, 3))
        except Exception:
            continue
        if seg.size == 0:
            continue
        seg_sum = np.sum(seg, axis=0, dtype=np.float64).reshape((3,))
        if not np.all(np.isfinite(seg_sum)) or not np.all(np.isfinite(total)):
            continue
        # Use vector-norm consistency instead of per-component allclose.
        # This avoids false negatives from tiny orthogonal components caused by
        # floating-point accumulation order while keeping strict aggregate checks.
        diff_norm = float(np.linalg.norm(seg_sum - total))
        ref_norm = float(max(np.linalg.norm(seg_sum), np.linalg.norm(total), 1.0))
        tol = float(max(float(atol), float(rtol) * ref_norm))
        if diff_norm > tol:
            return (
                False,
                "segment-force sum mismatch at step "
                f"{t}: sum={seg_sum.tolist()} total={total.tolist()} "
                f"(diff_norm={diff_norm:.6g}, tol={tol:.6g})",
            )
    return True, ""


def _finite_or(x: Any, default: float) -> float:
    try:
        x = float(x)
    except Exception:
        return float(default)
    return x if np.isfinite(x) else float(default)


def _quality_tier_rank(tier: str) -> int:
    t = str(tier or "").strip().lower()
    if t == "validated":
        return 0
    if t == "degraded":
        return 1
    return 2


def _apply_stochastic_eval_mode(eval_agent: Any, stochastic_eval: bool) -> None:
    """Set deterministic/stochastic eval mode on supported algos."""

    if hasattr(eval_agent, "algo") and hasattr(eval_agent.algo, "stochastic_eval"):
        eval_agent.algo.stochastic_eval = bool(stochastic_eval)


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

        wall_force_max = np.asarray(
            [t.get("wall_force_max_N", t.get("wall_force_max", np.nan)) for t in trials],
            dtype=np.float64,
        )
        wall_force_max_mean = float(np.nanmean(wall_force_max)) if wall_force_max.size else float("nan")
        force_available = np.asarray(
            [t.get("wall_force_available", np.nan) for t in trials], dtype=np.float64
        )
        force_available_rate = (
            float(np.nanmean(force_available)) if force_available.size else float("nan")
        )

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
                "force_available_rate": force_available_rate,
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
        "force_mode": cfg.force_extraction.mode,
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
            "| agent | success_rate | score_mean | score_std | steps_total_mean | steps_to_success_mean | tip_speed_max_mean (mm/s) | wall_force_max_mean | force_available_rate |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for r in agent_summaries:
            lines.append(
                "| {agent} | {success_rate:.3f} | {score_mean:.3f} | {score_std:.3f} | {steps_total_mean:.1f} | {steps_to_success_mean:.1f} | {tip_speed_max_mean_mm_s:.2f} | {wall_force_max_mean:.3f} | {force_available_rate:.3f} |".format(
                    agent=r["agent"],
                    success_rate=_finite_or(r["success_rate"], float("nan")),
                    score_mean=_finite_or(r["score_mean"], float("nan")),
                    score_std=_finite_or(r["score_std"], float("nan")),
                    steps_total_mean=_finite_or(r["steps_total_mean"], float("nan")),
                    steps_to_success_mean=_finite_or(r["steps_to_success_mean"], float("nan")),
                    tip_speed_max_mean_mm_s=_finite_or(r["tip_speed_max_mean_mm_s"], float("nan")),
                    wall_force_max_mean=_finite_or(r["wall_force_max_mean"], float("nan")),
                    force_available_rate=_finite_or(r["force_available_rate"], float("nan")),
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
    gap_reports_dir = run_dir / "force_gap_reports"
    run_dir.mkdir(parents=True, exist_ok=False)
    trials_dir.mkdir(parents=True, exist_ok=False)
    gap_reports_dir.mkdir(parents=True, exist_ok=False)

    # Save config for reproducibility
    _write_json(run_dir / "config.json", asdict(cfg))
    print(
        "[eval] force_extraction mode={mode} required={required} epsilon={eps} plugin_path={plugin}".format(
            mode=cfg.force_extraction.mode,
            required=int(cfg.force_extraction.required),
            eps=cfg.force_extraction.contact_epsilon,
            plugin=cfg.force_extraction.plugin_path or "(auto)",
        )
    )

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
        "wall_total_force_norm_max",
        "wall_total_force_norm_mean",
        "wall_contact_count_max",
        "wall_contact_detected_any",
        "wall_segment_count_max",
        "wall_force_norm_sum_max",
        "wall_force_norm_sum_mean",
        "wall_field_force_norm_max",
        "wall_force_available",
        "wall_force_source",
        "wall_force_channel",
        "wall_force_quality_tier",
        "wall_force_association_method",
        "wall_force_association_explicit_ratio",
        "wall_force_association_coverage",
        "wall_force_association_explicit_force_coverage",
        "wall_force_association_ordering_stable",
        "wall_force_gap_active_projected_count_sum",
        "wall_force_gap_explicit_mapped_count_sum",
        "wall_force_gap_unmapped_count_sum",
        "wall_force_gap_unmapped_ratio",
        "wall_force_gap_dominant_class",
        "wall_force_gap_contact_mode",
        "wall_force_gap_class_counts",
        "wall_force_active_constraint_any",
        "wall_native_contact_export_available",
        "wall_native_contact_export_source",
        "wall_native_contact_export_status",
        "wall_native_contact_export_explicit_coverage",
        "wall_force_status",
        "wall_force_error",
        "wall_force_max",
        "wall_total_force_norm_max_N",
        "wall_total_force_norm_mean_N",
        "wall_force_max_N",
        "wall_peak_segment_force_norm",
        "wall_peak_segment_force_norm_N",
        "wall_peak_segment_force_step",
        "wall_peak_segment_force_segment_id",
        "wall_peak_segment_force_time_s",
        "force_units",
        "unit_converted_si",
        "force_validation_status",
        "force_validation_error",
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
        if cfg.seeds:
            seeds = [int(s) for s in cfg.seeds]
        else:
            seeds = [cfg.base_seed + i for i in range(cfg.n_trials)]

        for agent_spec in cfg.agents:
            calibration_state = {"found": True, "passed": True, "record": None}
            if cfg.force_extraction.mode == "constraint_projected_si_validated":
                fingerprint = calibration_fingerprint(cfg, agent_spec)
                # Stable key is sha256 of the fingerprint payload.
                payload = json.dumps(
                    fingerprint, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                cache_key = hashlib.sha256(payload).hexdigest()
                calibration_state = load_force_calibration(
                    cache_path=cfg.force_extraction.calibration.cache_path,
                    cache_key=cache_key,
                )
                print(
                    "[eval] calibration cache found={found} passed={passed} key={key}".format(
                        found=int(bool(calibration_state.get("found", False))),
                        passed=int(bool(calibration_state.get("passed", False))),
                        key=cache_key[:12],
                    )
                )
                if cfg.force_extraction.calibration.required and (
                    (not calibration_state.get("found", False))
                    or (not calibration_state.get("passed", False))
                ):
                    raise RuntimeError(
                        "Validated SI force mode requires a passing calibration cache entry. "
                        f"cache={cfg.force_extraction.calibration.cache_path} key={cache_key} "
                        "Run: steve-force-calibrate --config <config.yml>"
                    )

            # Build intervention/environment for this tool/anatomy.
            if cfg.anatomy.type != "aortic_arch":
                raise ValueError(f"Unsupported anatomy.type: {cfg.anatomy.type}")

            intervention, action_dt_s = build_aortic_arch_intervention(
                tool_ref=agent_spec.tool,
                anatomy=cfg.anatomy,
                force_extraction=cfg.force_extraction,
            )

            # Switch simulation mode.
            if cfg.visualize:
                intervention.make_non_mp()
            elif cfg.use_non_mp_sim:
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
            extra_forces = SofaWallForceInfo(
                intervention,
                mode=cfg.force_extraction.mode,
                required=cfg.force_extraction.required,
                contact_epsilon=cfg.force_extraction.contact_epsilon,
                plugin_path=cfg.force_extraction.plugin_path,
                units=cfg.force_extraction.units,
                constraint_dt_s=action_dt_s,
            )

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
            with legacy_checkpoint_load_context():
                eval_agent = eve_rl.agent.single.SingleEvalOnly.from_checkpoint(
                    agent_spec.checkpoint,
                    device=device,
                    normalize_actions=True,
                    env_eval=env_eval,
                )
            _apply_stochastic_eval_mode(eval_agent, cfg.stochastic_eval)

            visualizer = None
            # Env.reset() expects a visualisation object with a .reset() method.
            # Keep a dummy visualiser for non-rendered trials instead of None.
            dummy_visualizer = eve.visualisation.VisualisationDummy()

            for trial_idx, seed in enumerate(seeds):
                show_trial = cfg.visualize and trial_idx < int(cfg.visualize_trials_per_agent)
                if show_trial:
                    if visualizer is None:
                        if cfg.visualize_force_debug:
                            visualizer = ForceDebugSofaPygame(
                                env_eval.intervention,
                                env_eval.interim_target,
                                force_info=extra_forces,
                                top_k_segments=int(cfg.visualize_force_debug_top_k),
                            )
                        else:
                            visualizer = eve.visualisation.SofaPygame(
                                env_eval.intervention,
                                env_eval.interim_target,
                            )
                    env_eval.visualisation = visualizer
                else:
                    # Do not keep a stale pygame window open while running hidden trials.
                    # Otherwise the window can appear frozen/unresponsive.
                    if visualizer is not None:
                        try:
                            visualizer.close()
                        except Exception:
                            pass
                        visualizer = None
                    env_eval.visualisation = dummy_visualizer

                print(
                    f"[eval] {agent_spec.name} trial {trial_idx + 1}/{len(seeds)} "
                    f"seed={seed} mode={'visual' if show_trial else 'headless'}"
                )
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
                wall_lcp_active_count = _extract_series(
                    infos, key="wall_lcp_active_count", default=0
                ).astype(np.int32)
                wall_wire_force_norm = _extract_series(infos, key="wall_wire_force_norm", default=np.nan).astype(np.float32)
                wall_collision_force_norm = _extract_series(infos, key="wall_collision_force_norm", default=np.nan).astype(np.float32)
                wall_total_force_norm = _extract_series(
                    infos, key="wall_total_force_norm", default=np.nan
                ).astype(np.float32)
                wall_contact_count = _extract_series(
                    infos, key="wall_contact_count", default=0
                ).astype(np.int32)
                wall_contact_detected = _extract_series(
                    infos, key="wall_contact_detected", default=False
                ).astype(np.bool_)
                wall_segment_count = _extract_series(
                    infos, key="wall_segment_count", default=0
                ).astype(np.int32)
                wall_force_norm_sum = _extract_series(
                    infos, key="wall_force_norm_sum", default=np.nan
                ).astype(np.float32)
                wall_field_force_norm = _extract_series(
                    infos, key="wall_field_force_norm", default=np.nan
                ).astype(np.float32)
                wall_force_available = _extract_series(
                    infos, key="wall_force_available", default=False
                ).astype(np.bool_)
                wall_force_active_constraint_step = _extract_series(
                    infos,
                    key="wall_force_active_constraint_step",
                    default=False,
                ).astype(np.bool_)
                wall_force_gap_active_projected_count = _extract_series(
                    infos,
                    key="wall_force_gap_active_projected_count",
                    default=0,
                ).astype(np.int32)
                wall_force_gap_explicit_mapped_count = _extract_series(
                    infos,
                    key="wall_force_gap_explicit_mapped_count",
                    default=0,
                ).astype(np.int32)
                wall_force_gap_unmapped_count = _extract_series(
                    infos,
                    key="wall_force_gap_unmapped_count",
                    default=0,
                ).astype(np.int32)
                wall_force_gap_dominant_class_series = np.asarray(
                    [
                        str(info.get("wall_force_gap_dominant_class", "none") or "none")
                        for info in infos
                    ],
                    dtype=object,
                )
                wall_force_gap_contact_mode_series = np.asarray(
                    [
                        str(info.get("wall_force_gap_contact_mode", "none") or "none")
                        for info in infos
                    ],
                    dtype=object,
                )
                wall_force_gap_class_counts_series = np.asarray(
                    [
                        dict(info.get("wall_force_gap_class_counts", {}) or {})
                        for info in infos
                    ],
                    dtype=object,
                )

                wall_total_force_vector = _extract_series(
                    infos,
                    key="wall_total_force_vector",
                    default=[np.nan, np.nan, np.nan],
                ).astype(np.float32)
                wall_total_force_vector_N = _extract_series(
                    infos,
                    key="wall_total_force_vector_N",
                    default=[np.nan, np.nan, np.nan],
                ).astype(np.float32)
                wall_total_force_norm_N = _extract_series(
                    infos, key="wall_total_force_norm_N", default=np.nan
                ).astype(np.float32)
                wall_contact_force_vectors = _stack_maybe(
                    [
                        np.asarray(
                            info.get("wall_contact_force_vectors", np.zeros((0, 3), dtype=np.float32)),
                            dtype=np.float32,
                        ).reshape((-1, 3))
                        for info in infos
                    ]
                )
                wall_contact_segment_indices = _stack_maybe(
                    [
                        np.asarray(
                            info.get("wall_contact_segment_indices", np.zeros((0,), dtype=np.int32)),
                            dtype=np.int32,
                        ).reshape((-1,))
                        for info in infos
                    ]
                )
                wall_active_segment_ids = _stack_maybe(
                    [
                        np.asarray(
                            info.get("wall_active_segment_ids", np.zeros((0,), dtype=np.int32)),
                            dtype=np.int32,
                        ).reshape((-1,))
                        for info in infos
                    ]
                )
                wall_active_segment_force_vectors = _stack_maybe(
                    [
                        np.asarray(
                            info.get(
                                "wall_active_segment_force_vectors",
                                np.zeros((0, 3), dtype=np.float32),
                            ),
                            dtype=np.float32,
                        ).reshape((-1, 3))
                        for info in infos
                    ]
                )
                wall_wire_force_vectors = _stack_maybe(
                    [
                        np.asarray(
                            info.get("wall_wire_force_vectors", np.zeros((0, 3), dtype=np.float32)),
                            dtype=np.float32,
                        ).reshape((-1, 3))
                        for info in infos
                    ]
                )
                wall_collision_force_vectors = _stack_maybe(
                    [
                        np.asarray(
                            info.get(
                                "wall_collision_force_vectors",
                                np.zeros((0, 3), dtype=np.float32),
                            ),
                            dtype=np.float32,
                        ).reshape((-1, 3))
                        for info in infos
                    ]
                )
                wall_segment_force_vectors_raw = _stack_maybe(
                    [
                        np.asarray(
                            info.get("wall_segment_force_vectors", np.zeros((0, 3), dtype=np.float32)),
                            dtype=np.float32,
                        ).reshape((-1, 3))
                        for info in infos
                    ]
                )
                wall_segment_force_vectors = _reconstruct_dense_wall_segment_forces(
                    wall_segment_count_series=wall_segment_count,
                    wall_active_segment_ids_series=wall_active_segment_ids,
                    wall_active_segment_force_vectors_series=wall_active_segment_force_vectors,
                )
                if (
                    wall_segment_force_vectors.size == 0
                    and isinstance(wall_segment_force_vectors_raw, np.ndarray)
                    and wall_segment_force_vectors_raw.size > 0
                ):
                    wall_segment_force_vectors = wall_segment_force_vectors_raw
                (
                    wall_peak_segment_force_norm_per_step,
                    wall_peak_segment_force_norm,
                    wall_peak_segment_force_step,
                    wall_peak_segment_force_segment_id,
                ) = _peak_segment_force_metrics(wall_segment_force_vectors)

                # Some values are only meaningful as the final scalar.
                last_info = infos[-1] if infos else {}
                path_ratio_last = _safe_float(last_info.get("path_ratio", float("nan")))
                trajectory_length_last = _safe_float(last_info.get("trajectory length", float("nan")))
                avg_translation_speed_last = _safe_float(last_info.get("average translation speed", float("nan")))

                # Aggregate wall force stats.
                wall_lcp_max_abs_max = float(np.nanmax(wall_lcp_max_abs)) if wall_lcp_max_abs.size else float("nan")
                wall_lcp_sum_abs_mean = float(np.nanmean(wall_lcp_sum_abs)) if wall_lcp_sum_abs.size else float("nan")
                wall_wire_force_norm_max = float(np.nanmax(wall_wire_force_norm)) if wall_wire_force_norm.size else float("nan")
                wall_collision_force_norm_max = float(np.nanmax(wall_collision_force_norm)) if wall_collision_force_norm.size else float("nan")
                wall_wire_force_norm_mean = float(np.nanmean(wall_wire_force_norm)) if wall_wire_force_norm.size else float("nan")
                wall_collision_force_norm_mean = float(np.nanmean(wall_collision_force_norm)) if wall_collision_force_norm.size else float("nan")
                wall_total_force_norm_max = (
                    float(np.nanmax(wall_total_force_norm))
                    if wall_total_force_norm.size
                    else float("nan")
                )
                wall_total_force_norm_mean = (
                    float(np.nanmean(wall_total_force_norm))
                    if wall_total_force_norm.size
                    else float("nan")
                )
                wall_total_force_norm_max_N = (
                    float(np.nanmax(wall_total_force_norm_N))
                    if wall_total_force_norm_N.size
                    else float("nan")
                )
                wall_total_force_norm_mean_N = (
                    float(np.nanmean(wall_total_force_norm_N))
                    if wall_total_force_norm_N.size
                    else float("nan")
                )
                wall_contact_count_max = (
                    int(np.nanmax(wall_contact_count))
                    if wall_contact_count.size
                    else 0
                )
                wall_contact_detected_any = int(
                    bool(np.any(wall_contact_detected.astype(bool)))
                )
                wall_segment_count_max = (
                    int(np.nanmax(wall_segment_count))
                    if wall_segment_count.size
                    else 0
                )
                wall_force_norm_sum_max = (
                    float(np.nanmax(wall_force_norm_sum))
                    if wall_force_norm_sum.size
                    else float("nan")
                )
                wall_force_norm_sum_mean = (
                    float(np.nanmean(wall_force_norm_sum))
                    if wall_force_norm_sum.size
                    else float("nan")
                )
                wall_field_force_norm_max = (
                    float(np.nanmax(wall_field_force_norm))
                    if wall_field_force_norm.size
                    else float("nan")
                )
                wall_force_max = float(
                    np.nanmax(
                        [
                            wall_total_force_norm_max,
                        ]
                    )
                )
                wall_force_max_N = float(
                    np.nanmax(
                        [
                            wall_total_force_norm_max_N,
                        ]
                    )
                )
                wall_peak_segment_force_norm_N = (
                    float(wall_peak_segment_force_norm)
                    if cfg.force_extraction.mode == "constraint_projected_si_validated"
                    else float("nan")
                )
                wall_peak_segment_force_time_s = (
                    float(wall_peak_segment_force_step) * float(action_dt_s)
                    if wall_peak_segment_force_step is not None
                    else float("nan")
                )

                source_values = [
                    str(info.get("wall_force_source", "")).strip()
                    for info in infos
                    if str(info.get("wall_force_source", "")).strip()
                ]
                channel_values = [
                    str(info.get("wall_force_channel", "")).strip()
                    for info in infos
                    if str(info.get("wall_force_channel", "")).strip()
                ]
                quality_values = [
                    str(info.get("wall_force_quality_tier", "unavailable")).strip()
                    or "unavailable"
                    for info in infos
                ]
                association_values = [
                    str(info.get("wall_force_association_method", "none")).strip()
                    or "none"
                    for info in infos
                ]
                association_ratio_values = [
                    _safe_float(info.get("wall_force_association_explicit_ratio", float("nan")))
                    for info in infos
                ]
                association_coverage_values = [
                    _safe_float(info.get("wall_force_association_coverage", float("nan")))
                    for info in infos
                ]
                association_explicit_force_coverage_values = [
                    _safe_float(
                        info.get(
                            "wall_force_association_explicit_force_coverage",
                            float("nan"),
                        )
                    )
                    for info in infos
                ]
                association_ordering_stable_values = [
                    bool(info.get("wall_force_association_ordering_stable", False))
                    for info in infos
                ]
                native_export_available_values = [
                    bool(info.get("wall_native_contact_export_available", False))
                    for info in infos
                ]
                native_export_source_values = [
                    str(info.get("wall_native_contact_export_source", "")).strip()
                    for info in infos
                    if str(info.get("wall_native_contact_export_source", "")).strip()
                ]
                native_export_status_values = [
                    str(info.get("wall_native_contact_export_status", "")).strip()
                    for info in infos
                    if str(info.get("wall_native_contact_export_status", "")).strip()
                ]
                native_export_explicit_coverage_values = [
                    _safe_float(
                        info.get("wall_native_contact_export_explicit_coverage", float("nan"))
                    )
                    for info in infos
                ]
                status_values = [
                    str(info.get("wall_force_status", "")).strip()
                    for info in infos
                    if str(info.get("wall_force_status", "")).strip()
                ]
                error_values = [
                    str(info.get("wall_force_error", "")).strip()
                    for info in infos
                    if str(info.get("wall_force_error", "")).strip()
                ]
                wire_force_source_values = [
                    str(info.get("wall_wire_force_vectors_source", "")).strip()
                    for info in infos
                    if str(info.get("wall_wire_force_vectors_source", "")).strip()
                ]
                collision_force_source_values = [
                    str(info.get("wall_collision_force_vectors_source", "")).strip()
                    for info in infos
                    if str(info.get("wall_collision_force_vectors_source", "")).strip()
                ]
                wall_force_source = source_values[-1] if source_values else "unknown"
                wall_force_channel = channel_values[-1] if channel_values else "none"
                relevant_steps = (
                    wall_contact_detected.astype(bool)
                    | (wall_contact_count > 0)
                    | (wall_lcp_active_count > 0)
                )
                active_constraint_steps = wall_force_active_constraint_step.astype(bool)
                if quality_values:
                    idx_pool = (
                        np.nonzero(relevant_steps)[0].tolist()
                        if np.any(relevant_steps)
                        else list(range(len(quality_values)))
                    )
                    worst_idx = max(
                        idx_pool,
                        key=lambda i: _quality_tier_rank(quality_values[int(i)]),
                    )
                    wall_force_quality_tier = quality_values[int(worst_idx)]
                    wall_force_association_method = association_values[int(worst_idx)]
                    if cfg.force_extraction.mode == "constraint_projected_si_validated":
                        assoc_pool = (
                            np.nonzero(active_constraint_steps)[0].tolist()
                            if np.any(active_constraint_steps)
                            else idx_pool
                        )
                        assoc_candidates = [
                            association_values[int(i)]
                            for i in assoc_pool
                            if str(association_values[int(i)]).strip()
                            and str(association_values[int(i)]).strip().lower() != "none"
                        ]
                        if assoc_candidates:
                            assoc_counts: Dict[str, int] = {}
                            for method in assoc_candidates:
                                assoc_counts[method] = int(assoc_counts.get(method, 0)) + 1
                            wall_force_association_method = max(
                                assoc_counts.keys(),
                                key=lambda k: (int(assoc_counts.get(k, 0)), str(k)),
                            )
                else:
                    wall_force_quality_tier = "unavailable"
                    wall_force_association_method = "none"
                if association_ratio_values:
                    arr = np.asarray(association_ratio_values, dtype=np.float32)
                    if (
                        cfg.force_extraction.mode == "constraint_projected_si_validated"
                        and np.any(active_constraint_steps)
                    ):
                        arr = arr[active_constraint_steps]
                    elif np.any(relevant_steps):
                        arr = arr[relevant_steps]
                    finite = arr[np.isfinite(arr)]
                    wall_force_association_explicit_ratio = (
                        float(np.nanmin(finite)) if finite.size else float("nan")
                    )
                else:
                    wall_force_association_explicit_ratio = float("nan")
                if association_coverage_values:
                    arr_cov = np.asarray(association_coverage_values, dtype=np.float32)
                    if np.any(relevant_steps):
                        arr_cov = arr_cov[relevant_steps]
                    finite_cov = arr_cov[np.isfinite(arr_cov)]
                    wall_force_association_coverage = (
                        float(np.nanmin(finite_cov)) if finite_cov.size else float("nan")
                    )
                else:
                    wall_force_association_coverage = float("nan")
                if association_explicit_force_coverage_values:
                    arr_exp_force_cov = np.asarray(
                        association_explicit_force_coverage_values, dtype=np.float32
                    )
                    if np.any(relevant_steps):
                        arr_exp_force_cov = arr_exp_force_cov[relevant_steps]
                    finite_exp_force_cov = arr_exp_force_cov[
                        np.isfinite(arr_exp_force_cov)
                    ]
                    wall_force_association_explicit_force_coverage = (
                        float(np.nanmin(finite_exp_force_cov))
                        if finite_exp_force_cov.size
                        else float("nan")
                    )
                else:
                    wall_force_association_explicit_force_coverage = float("nan")
                wall_force_association_ordering_stable = bool(
                    np.all(np.asarray(association_ordering_stable_values, dtype=np.bool_))
                ) if association_ordering_stable_values else False
                wall_force_active_constraint_any = int(
                    bool(np.any(wall_force_active_constraint_step))
                )
                wall_force_gap_active_projected_count_sum = int(
                    np.sum(
                        np.asarray(
                            wall_force_gap_active_projected_count,
                            dtype=np.int64,
                        )
                    )
                )
                wall_force_gap_explicit_mapped_count_sum = int(
                    np.sum(
                        np.asarray(
                            wall_force_gap_explicit_mapped_count,
                            dtype=np.int64,
                        )
                    )
                )
                wall_force_gap_unmapped_count_sum = int(
                    np.sum(
                        np.asarray(
                            wall_force_gap_unmapped_count,
                            dtype=np.int64,
                        )
                    )
                )
                wall_force_gap_unmapped_ratio = (
                    float(wall_force_gap_unmapped_count_sum)
                    / float(max(wall_force_gap_active_projected_count_sum, 1))
                )
                gap_class_totals: Dict[str, int] = {}
                for step_counts in wall_force_gap_class_counts_series.tolist():
                    if not isinstance(step_counts, dict):
                        continue
                    for key, value in step_counts.items():
                        k = str(key or "").strip() or "unknown"
                        try:
                            v = int(value)
                        except Exception:
                            v = 0
                        if v <= 0:
                            continue
                        gap_class_totals[k] = int(gap_class_totals.get(k, 0)) + int(v)
                wall_force_gap_dominant_class = (
                    max(gap_class_totals.keys(), key=lambda k: int(gap_class_totals.get(k, 0)))
                    if gap_class_totals
                    else "none"
                )
                gap_contact_mode_values = [
                    str(x or "none")
                    for x in wall_force_gap_contact_mode_series.tolist()
                    if str(x or "none") != "none"
                ]
                if gap_contact_mode_values:
                    mode_counts: Dict[str, int] = {}
                    for m in gap_contact_mode_values:
                        mode_counts[m] = int(mode_counts.get(m, 0)) + 1
                    wall_force_gap_contact_mode = max(
                        mode_counts.keys(), key=lambda k: int(mode_counts.get(k, 0))
                    )
                else:
                    wall_force_gap_contact_mode = "none"
                wall_force_gap_class_counts = json.dumps(
                    gap_class_totals, sort_keys=True
                )
                wall_native_contact_export_available = int(
                    bool(np.any(np.asarray(native_export_available_values, dtype=np.bool_)))
                )
                wall_native_contact_export_source = (
                    native_export_source_values[-1] if native_export_source_values else ""
                )
                wall_native_contact_export_status = (
                    native_export_status_values[-1] if native_export_status_values else ""
                )
                if native_export_explicit_coverage_values:
                    arr_native_cov = np.asarray(
                        native_export_explicit_coverage_values, dtype=np.float32
                    )
                    if np.any(relevant_steps):
                        arr_native_cov = arr_native_cov[relevant_steps]
                    finite_native_cov = arr_native_cov[np.isfinite(arr_native_cov)]
                    wall_native_contact_export_explicit_coverage = (
                        float(np.nanmin(finite_native_cov))
                        if finite_native_cov.size
                        else float("nan")
                    )
                else:
                    wall_native_contact_export_explicit_coverage = float("nan")
                wall_force_status = status_values[-1] if status_values else ""
                wall_force_error = error_values[-1] if error_values else ""
                wall_wire_force_vectors_source = (
                    wire_force_source_values[-1] if wire_force_source_values else ""
                )
                wall_collision_force_vectors_source = (
                    collision_force_source_values[-1]
                    if collision_force_source_values
                    else ""
                )

                force_ok, force_validation_error = _validate_force_signal(
                    force_available_series=wall_force_available,
                    contact_count_series=wall_contact_count,
                    contact_detected_series=wall_contact_detected,
                    lcp_active_count_series=wall_lcp_active_count,
                    total_force_norm_series=wall_total_force_norm,
                    contact_epsilon=cfg.force_extraction.contact_epsilon,
                    active_constraint_step_series=wall_force_active_constraint_step,
                )
                if force_ok:
                    force_consistent, force_consistency_error = _validate_force_vector_consistency(
                        segment_force_series=wall_segment_force_vectors,
                        total_force_vector_series=wall_total_force_vector,
                        atol=max(1e-6, 10.0 * float(cfg.force_extraction.contact_epsilon)),
                    )
                    if not force_consistent:
                        force_ok = False
                        force_validation_error = force_consistency_error
                if cfg.force_extraction.required and not force_ok:
                    raise RuntimeError(
                        "Force extraction required but unavailable for "
                        f"{agent_spec.name} trial={trial_idx} seed={seed}: "
                        f"{force_validation_error or wall_force_error or wall_force_source}"
                    )
                force_available_for_score = bool(
                    force_ok and wall_force_quality_tier == "validated"
                )
                if not force_available_for_score and not wall_force_error:
                    wall_force_error = force_validation_error or wall_force_error
                if cfg.force_extraction.mode == "constraint_projected_si_validated":
                    if cfg.force_extraction.calibration.required:
                        if not calibration_state.get("found", False):
                            wall_force_error = wall_force_error or "missing_calibration_cache"
                        elif not calibration_state.get("passed", False):
                            wall_force_error = wall_force_error or "failed_calibration_cache"

                if cfg.force_extraction.required and not force_available_for_score:
                    raise RuntimeError(
                        "Force extraction required but did not reach validated quality for "
                        f"{agent_spec.name} trial={trial_idx} seed={seed}: "
                        f"quality={wall_force_quality_tier} "
                        f"association={wall_force_association_method} "
                        f"error={wall_force_error or 'n/a'}"
                    )

                if force_available_for_score:
                    force_validation_status = "pass:validated"
                elif force_ok and wall_force_quality_tier == "degraded":
                    force_validation_status = "pass:degraded"
                elif force_ok:
                    force_validation_status = f"pass:{wall_force_quality_tier}"
                else:
                    force_validation_status = "fail:unavailable"
                force_validation_error = wall_force_error

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
                    wall_total_force_norm_max=wall_total_force_norm_max_N
                    if np.isfinite(wall_total_force_norm_max_N)
                    else wall_total_force_norm_max,
                    wall_lcp_max_abs_max=wall_lcp_max_abs_max,
                    force_available=force_available_for_score,
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
                    wall_total_force_vector=wall_total_force_vector,
                    wall_total_force_norm=wall_total_force_norm,
                    wall_total_force_vector_N=wall_total_force_vector_N,
                    wall_total_force_norm_N=wall_total_force_norm_N,
                    wall_contact_count=wall_contact_count,
                    wall_contact_detected=wall_contact_detected,
                    wall_segment_count=wall_segment_count,
                    wall_force_norm_sum=wall_force_norm_sum,
                    wall_field_force_norm=wall_field_force_norm,
                    wall_force_available=wall_force_available,
                    wall_contact_force_vectors=wall_contact_force_vectors,
                    wall_contact_segment_indices=wall_contact_segment_indices,
                    wall_active_segment_ids=wall_active_segment_ids,
                    wall_active_segment_force_vectors=wall_active_segment_force_vectors,
                    wall_active_segment_force_vectors_N=wall_active_segment_force_vectors,
                    wall_wire_force_vectors=wall_wire_force_vectors,
                    wall_collision_force_vectors=wall_collision_force_vectors,
                    wall_wire_force_vectors_source=wall_wire_force_vectors_source,
                    wall_collision_force_vectors_source=wall_collision_force_vectors_source,
                    wall_segment_force_vectors=wall_segment_force_vectors,
                    wall_peak_segment_force_norm_per_step=wall_peak_segment_force_norm_per_step,
                    wall_peak_segment_force_norm=wall_peak_segment_force_norm,
                    wall_peak_segment_force_norm_N=wall_peak_segment_force_norm_N,
                    wall_peak_segment_force_step=(
                        int(wall_peak_segment_force_step)
                        if wall_peak_segment_force_step is not None
                        else -1
                    ),
                    wall_peak_segment_force_segment_id=(
                        int(wall_peak_segment_force_segment_id)
                        if wall_peak_segment_force_segment_id is not None
                        else -1
                    ),
                    wall_peak_segment_force_time_s=wall_peak_segment_force_time_s,
                    wall_force_source=wall_force_source,
                    wall_force_channel=wall_force_channel,
                    wall_force_quality_tier=wall_force_quality_tier,
                    wall_force_association_method=wall_force_association_method,
                    wall_force_association_explicit_ratio=wall_force_association_explicit_ratio,
                    wall_force_association_coverage=wall_force_association_coverage,
                    wall_force_association_explicit_force_coverage=wall_force_association_explicit_force_coverage,
                    wall_force_association_ordering_stable=wall_force_association_ordering_stable,
                    wall_force_active_constraint_step=wall_force_active_constraint_step,
                    wall_force_active_constraint_any=wall_force_active_constraint_any,
                    wall_force_gap_active_projected_count=wall_force_gap_active_projected_count,
                    wall_force_gap_explicit_mapped_count=wall_force_gap_explicit_mapped_count,
                    wall_force_gap_unmapped_count=wall_force_gap_unmapped_count,
                    wall_force_gap_class_counts=wall_force_gap_class_counts_series,
                    wall_force_gap_dominant_class=wall_force_gap_dominant_class_series,
                    wall_force_gap_contact_mode=wall_force_gap_contact_mode_series,
                    wall_native_contact_export_available=wall_native_contact_export_available,
                    wall_native_contact_export_source=wall_native_contact_export_source,
                    wall_native_contact_export_status=wall_native_contact_export_status,
                    wall_native_contact_export_explicit_coverage=wall_native_contact_export_explicit_coverage,
                    wall_force_status=wall_force_status,
                    wall_force_error=wall_force_error,
                    force_units=(
                        asdict(cfg.force_extraction.units)
                        if cfg.force_extraction.units is not None
                        else {}
                    ),
                    unit_converted_si=bool(
                        cfg.force_extraction.mode == "constraint_projected_si_validated"
                    ),
                    force_validation_status=force_validation_status,
                    force_validation_error=force_validation_error,
                    action_dt_s=float(action_dt_s),
                    seed=int(seed),
                )
                _write_force_gap_report(
                    out_dir=gap_reports_dir,
                    agent=agent_spec.name,
                    trial=int(trial_idx),
                    seed=int(seed),
                    active_projected_count=wall_force_gap_active_projected_count,
                    explicit_mapped_count=wall_force_gap_explicit_mapped_count,
                    unmapped_count=wall_force_gap_unmapped_count,
                    class_counts_series=wall_force_gap_class_counts_series,
                    dominant_class_series=wall_force_gap_dominant_class_series,
                    contact_mode_series=wall_force_gap_contact_mode_series,
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
                    "wall_total_force_norm_max": float(wall_total_force_norm_max),
                    "wall_total_force_norm_mean": float(wall_total_force_norm_mean),
                    "wall_total_force_norm_max_N": float(wall_total_force_norm_max_N),
                    "wall_total_force_norm_mean_N": float(wall_total_force_norm_mean_N),
                    "wall_contact_count_max": int(wall_contact_count_max),
                    "wall_contact_detected_any": int(wall_contact_detected_any),
                    "wall_segment_count_max": int(wall_segment_count_max),
                    "wall_force_norm_sum_max": float(wall_force_norm_sum_max),
                    "wall_force_norm_sum_mean": float(wall_force_norm_sum_mean),
                    "wall_field_force_norm_max": float(wall_field_force_norm_max),
                    "wall_force_available": float(force_available_for_score),
                    "wall_force_source": wall_force_source,
                    "wall_force_channel": wall_force_channel,
                    "wall_force_quality_tier": wall_force_quality_tier,
                    "wall_force_association_method": wall_force_association_method,
                    "wall_force_association_explicit_ratio": float(
                        wall_force_association_explicit_ratio
                    ),
                    "wall_force_association_coverage": float(
                        wall_force_association_coverage
                    ),
                    "wall_force_association_explicit_force_coverage": float(
                        wall_force_association_explicit_force_coverage
                    ),
                    "wall_force_association_ordering_stable": int(
                        bool(wall_force_association_ordering_stable)
                    ),
                    "wall_force_gap_active_projected_count_sum": int(
                        wall_force_gap_active_projected_count_sum
                    ),
                    "wall_force_gap_explicit_mapped_count_sum": int(
                        wall_force_gap_explicit_mapped_count_sum
                    ),
                    "wall_force_gap_unmapped_count_sum": int(
                        wall_force_gap_unmapped_count_sum
                    ),
                    "wall_force_gap_unmapped_ratio": float(
                        wall_force_gap_unmapped_ratio
                    ),
                    "wall_force_gap_dominant_class": wall_force_gap_dominant_class,
                    "wall_force_gap_contact_mode": wall_force_gap_contact_mode,
                    "wall_force_gap_class_counts": wall_force_gap_class_counts,
                    "wall_force_active_constraint_any": int(
                        wall_force_active_constraint_any
                    ),
                    "wall_native_contact_export_available": int(
                        wall_native_contact_export_available
                    ),
                    "wall_native_contact_export_source": wall_native_contact_export_source,
                    "wall_native_contact_export_status": wall_native_contact_export_status,
                    "wall_native_contact_export_explicit_coverage": float(
                        wall_native_contact_export_explicit_coverage
                    ),
                    "wall_force_status": wall_force_status,
                    "wall_force_error": wall_force_error,
                    "wall_force_max": float(wall_force_max),
                    "wall_force_max_N": float(wall_force_max_N),
                    "wall_peak_segment_force_norm": float(wall_peak_segment_force_norm),
                    "wall_peak_segment_force_norm_N": float(wall_peak_segment_force_norm_N),
                    "wall_peak_segment_force_step": (
                        int(wall_peak_segment_force_step)
                        if wall_peak_segment_force_step is not None
                        else None
                    ),
                    "wall_peak_segment_force_segment_id": (
                        int(wall_peak_segment_force_segment_id)
                        if wall_peak_segment_force_segment_id is not None
                        else None
                    ),
                    "wall_peak_segment_force_time_s": float(wall_peak_segment_force_time_s),
                    "force_units": json.dumps(
                        asdict(cfg.force_extraction.units)
                        if cfg.force_extraction.units is not None
                        else {},
                        sort_keys=True,
                    ),
                    "unit_converted_si": float(
                        cfg.force_extraction.mode == "constraint_projected_si_validated"
                    ),
                    "force_validation_status": force_validation_status,
                    "force_validation_error": force_validation_error,
                    "score": float(trial_score.score),
                    "score_success": float(trial_score.success),
                    "score_efficiency": float(trial_score.efficiency),
                    "score_safety": float(trial_score.safety),
                    "score_smoothness": float(trial_score.smoothness),
                    "npz_path": str(npz_path),
                }

                trial_rows.append(row)
                print(
                    f"[trial] {agent_spec.name} {trial_idx + 1}/{len(seeds)} "
                    f"seed={seed} success={int(success)} reward={episode_reward:.4f} "
                    f"steps={steps_total} force_available={int(force_available_for_score)} "
                    f"force_max_N={wall_force_max_N:.5g} "
                    f"peak_local={wall_peak_segment_force_norm:.5g}@"
                    f"({wall_peak_segment_force_step},{wall_peak_segment_force_segment_id}) "
                    f"contact={wall_contact_detected_any} channel={wall_force_channel} "
                    f"quality={wall_force_quality_tier} assoc={wall_force_association_method} "
                    f"assoc_ratio={wall_force_association_explicit_ratio:.3g} "
                    f"assoc_cov={wall_force_association_coverage:.3g}"
                )

                # Write a CSV-friendly variant of the row (rounding + blanks instead of NaN).
                writer.writerow({k: _csv_cell(row.get(k)) for k in summary_fields})
                f.flush()

            # Cleanup
            try:
                eval_agent.close()
            except Exception:
                pass
            try:
                if visualizer is not None:
                    visualizer.close()
            except Exception:
                pass

        # Report files for the full run (per-agent aggregation + overall ranking).
        _write_report_files(run_dir, cfg=cfg, rows=trial_rows)

    return run_dir
