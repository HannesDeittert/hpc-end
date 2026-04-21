from __future__ import annotations

import math

import numpy as np

from .models import ScoreBreakdown, ScoringSpec, TrialTelemetrySummary


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _finite_or(value: float | None, default: float) -> float:
    if value is None:
        return float(default)
    numeric = float(value)
    return numeric if math.isfinite(numeric) else float(default)


def score_trial(
    *,
    telemetry: TrialTelemetrySummary,
    max_episode_steps: int,
    scoring: ScoringSpec,
) -> ScoreBreakdown:
    """Score one eval_v2 trial using the local `default_v1` definition."""

    if scoring.mode != "default_v1":
        raise ValueError(f"Unsupported scoring.mode: {scoring.mode!r}")

    s_success = 1.0 if telemetry.success else 0.0

    if telemetry.steps_to_success is None or max_episode_steps <= 0:
        s_efficiency = 0.0
    else:
        s_efficiency = 1.0 - (
            (float(telemetry.steps_to_success) - 1.0) / float(max_episode_steps)
        )
        s_efficiency = _clip01(s_efficiency)

    forces = telemetry.forces
    force_available = bool(forces is not None and forces.available_for_score)
    if force_available and forces is not None:
        force_max = _finite_or(forces.total_force_norm_max, 0.0)
        lcp_max = _finite_or(forces.lcp_max_abs_max, 0.0)
        if scoring.scales.force_scale > 0.0:
            safety_force = float(np.exp(-force_max / float(scoring.scales.force_scale)))
        else:
            safety_force = 1.0
        if scoring.scales.lcp_scale > 0.0:
            safety_lcp = float(np.exp(-lcp_max / float(scoring.scales.lcp_scale)))
        else:
            safety_lcp = 1.0
        s_safety: float | None = _clip01(safety_force * safety_lcp)
    else:
        s_safety = None

    tip_speed_max_mm_s = _finite_or(telemetry.tip_speed_max_mm_s, 0.0)
    if scoring.scales.speed_scale_mm_s > 0.0:
        s_smoothness = float(
            np.exp(-tip_speed_max_mm_s / float(scoring.scales.speed_scale_mm_s))
        )
    else:
        s_smoothness = 1.0
    s_smoothness = _clip01(s_smoothness)

    weight_success = float(scoring.weights.success)
    weight_efficiency = float(scoring.weights.efficiency)
    weight_safety = float(scoring.weights.safety if force_available else 0.0)
    weight_smoothness = float(scoring.weights.smoothness)
    weights = np.asarray(
        [weight_success, weight_efficiency, weight_safety, weight_smoothness],
        dtype=np.float64,
    )
    components = np.asarray(
        [
            s_success,
            s_efficiency,
            0.0 if s_safety is None else s_safety,
            s_smoothness,
        ],
        dtype=np.float64,
    )
    total = float(np.sum(weights * components))
    if scoring.weights.normalize:
        denominator = float(np.sum(weights))
        if denominator != 0.0:
            total = total / denominator

    return ScoreBreakdown(
        total=total,
        success=float(s_success),
        efficiency=float(s_efficiency),
        safety=s_safety,
        smoothness=float(s_smoothness),
    )


__all__ = ["score_trial"]
