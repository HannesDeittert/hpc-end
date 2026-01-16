from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import ScoringConfig


@dataclass(frozen=True)
class TrialScore:
    """Per-trial score and the underlying components.

    All components are designed to be in the range [0, 1] (best = 1).
    The final score is a weighted combination of these components.
    """

    score: float
    success: float
    efficiency: float
    safety: float
    smoothness: float


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _finite_or(x: float, default: float) -> float:
    return float(x) if np.isfinite(x) else float(default)


def compute_default_v1(
    *,
    success: bool,
    steps_to_success: Optional[int],
    max_episode_steps: int,
    tip_speed_max_mm_s: float,
    wall_wire_force_norm_max: float,
    wall_collision_force_norm_max: float,
    wall_lcp_max_abs_max: float,
    scoring: ScoringConfig,
) -> TrialScore:
    """Default scoring function (`mode=default_v1`).

    Intent (relative comparison):
    - prefer success over failure
    - prefer fewer steps to success (efficiency)
    - prefer lower forces (safety)
    - prefer lower peak tip speed (smoothness)

    Notes on units:
    - tip speed is in mm/s (tracking coordinates / image_frequency_hz).
    - force magnitudes depend on scene/unit conventions → controlled via scale factors.
    """

    # 1) Success
    s_success = 1.0 if success else 0.0

    # 2) Efficiency (only meaningful if we reached the target at some point).
    if steps_to_success is None or max_episode_steps <= 0:
        s_eff = 0.0
    else:
        # steps_to_success is 1-based in our pipeline.
        # If success on step 1 → efficiency 1.0, if success at max_episode_steps → ~0.
        s_eff = 1.0 - (float(steps_to_success) - 1.0) / float(max_episode_steps)
        s_eff = _clip01(s_eff)

    # 3) Safety (penalize max forces; treat NaNs as "no penalty" to avoid breaking runs)
    force_max = float(np.nanmax([wall_wire_force_norm_max, wall_collision_force_norm_max]))
    force_max = _finite_or(force_max, 0.0)
    lcp_max = _finite_or(float(wall_lcp_max_abs_max), 0.0)

    # exp(-x/scale) yields 1.0 for x=0 and decays smoothly.
    if scoring.force_scale > 0:
        safety_force = float(np.exp(-force_max / float(scoring.force_scale)))
    else:
        safety_force = 1.0

    if scoring.lcp_scale > 0:
        safety_lcp = float(np.exp(-lcp_max / float(scoring.lcp_scale)))
    else:
        safety_lcp = 1.0

    s_safety = _clip01(safety_force * safety_lcp)

    # 4) Smoothness (penalize peak tip speed)
    tip_speed_max_mm_s = _finite_or(float(tip_speed_max_mm_s), 0.0)
    if scoring.speed_scale_mm_s > 0:
        s_smooth = float(np.exp(-tip_speed_max_mm_s / float(scoring.speed_scale_mm_s)))
    else:
        s_smooth = 1.0
    s_smooth = _clip01(s_smooth)

    # Weighted sum (optionally normalized)
    weights = np.array(
        [scoring.w_success, scoring.w_efficiency, scoring.w_safety, scoring.w_smoothness],
        dtype=np.float64,
    )
    comps = np.array([s_success, s_eff, s_safety, s_smooth], dtype=np.float64)
    score = float(np.sum(weights * comps))

    if scoring.normalize_weights:
        denom = float(np.sum(weights))
        if denom != 0.0:
            score = score / denom

    return TrialScore(
        score=float(score),
        success=float(s_success),
        efficiency=float(s_eff),
        safety=float(s_safety),
        smoothness=float(s_smooth),
    )


def score_trial(
    *,
    scoring: ScoringConfig,
    success: bool,
    steps_to_success: Optional[int],
    max_episode_steps: int,
    tip_speed_max_mm_s: float,
    wall_wire_force_norm_max: float,
    wall_collision_force_norm_max: float,
    wall_lcp_max_abs_max: float,
) -> TrialScore:
    """Dispatch to the configured scoring mode."""

    if scoring.mode == "default_v1":
        return compute_default_v1(
            success=success,
            steps_to_success=steps_to_success,
            max_episode_steps=max_episode_steps,
            tip_speed_max_mm_s=tip_speed_max_mm_s,
            wall_wire_force_norm_max=wall_wire_force_norm_max,
            wall_collision_force_norm_max=wall_collision_force_norm_max,
            wall_lcp_max_abs_max=wall_lcp_max_abs_max,
            scoring=scoring,
        )

    raise ValueError(f"Unsupported scoring.mode: {scoring.mode}")


def aggregate_scores(values: np.ndarray) -> Tuple[float, float]:
    """Return (mean, std) ignoring NaNs."""

    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)), float(np.nanstd(values))

