from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from .models import CandidateSummary, ScoreBreakdown, ScoringSpec, TrialTelemetrySummary


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _finite_or(value: float | None, default: float) -> float:
    if value is None:
        return float(default)
    numeric = float(value)
    return numeric if math.isfinite(numeric) else float(default)


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _active_weights(scoring: ScoringSpec) -> Dict[str, float]:
    return {
        str(name): max(float(weight), 0.0)
        for name, weight in scoring.candidate_score.default_weights.items()
        if str(name) in set(scoring.candidate_score.active_components)
    }


def score_efficiency(*, success: bool, steps_to_success: int | None, max_episode_steps: int) -> float:
    if not success or steps_to_success is None or max_episode_steps <= 1:
        return 0.0
    normalized = 1.0 - ((float(steps_to_success) - 1.0) / (float(max_episode_steps) - 1.0))
    return _clip01(normalized)


def score_safety(*, force_N: float | None, scoring: ScoringSpec) -> float:
    finite_force = _finite_or_none(force_N)
    if finite_force is None:
        return 0.0

    spec = scoring.safety_score
    force_value = max(finite_force, 0.0)

    def g(force: float) -> float:
        return 1.0 / (1.0 + math.exp(float(spec.k) * (float(force) - float(spec.F50_N))))

    g0 = g(0.0)
    gmax = g(float(spec.F_max_N))
    denominator = g0 - gmax
    if abs(denominator) <= 1e-12:
        logistic_term = 0.0
    else:
        logistic_term = (g(force_value) - gmax) / denominator
    polynomial_term = 1.0 - float(spec.c) * (force_value ** float(spec.p))
    return _clip01(polynomial_term * logistic_term)


def score_smoothness(*, tip_jerk_p95: float | None, jerk_scale_mm_s3: float | None) -> float | None:
    jerk_scale = _finite_or_none(jerk_scale_mm_s3)
    jerk_value = _finite_or_none(tip_jerk_p95)
    if jerk_scale is None or jerk_value is None:
        return None
    return _clip01(math.exp(-max(jerk_value, 0.0) / jerk_scale))


def force_within_safety_threshold(*, telemetry: TrialTelemetrySummary, scoring: ScoringSpec) -> bool:
    forces = telemetry.forces
    return bool(
        forces is not None
        and forces.available_for_score
        and forces.total_force_norm_max_newton is not None
        and float(forces.total_force_norm_max_newton) <= float(scoring.force.force_max_N)
    )


def valid_for_ranking(*, telemetry: TrialTelemetrySummary, max_episode_steps: int, scoring: ScoringSpec) -> bool:
    indicator = scoring.trial_indicator
    checks = []
    if indicator.requires_success:
        checks.append(bool(telemetry.success))
    if indicator.requires_steps_within_episode_limit:
        checks.append(
            telemetry.steps_to_success is not None
            and int(telemetry.steps_to_success) <= int(max_episode_steps)
        )
    if indicator.requires_force_available:
        checks.append(
            telemetry.forces is not None and bool(telemetry.forces.available_for_score)
        )
    if indicator.requires_force_within_safety_threshold:
        checks.append(force_within_safety_threshold(telemetry=telemetry, scoring=scoring))
    return bool(all(checks))


def soft_score_total(*, breakdown: ScoreBreakdown, scoring: ScoringSpec) -> float:
    weights = _active_weights(scoring)
    if not weights:
        return 0.0
    components = {
        "score_success": float(breakdown.success),
        "score_efficiency": float(breakdown.efficiency),
        "score_safety": float(breakdown.safety),
        "score_smoothness": float("nan") if breakdown.smoothness is None else float(breakdown.smoothness),
    }
    weighted_sum = 0.0
    total_weight = 0.0
    for name, weight in weights.items():
        component = components.get(name)
        if component is None or not math.isfinite(component):
            continue
        weighted_sum += float(weight) * float(component)
        total_weight += float(weight)
    if total_weight <= 0.0:
        return 0.0
    return weighted_sum / total_weight


def score_trial(
    *,
    telemetry: TrialTelemetrySummary,
    max_episode_steps: int,
    scoring: ScoringSpec,
) -> ScoreBreakdown:
    if scoring.mode != "ranking_v1":
        raise ValueError(f"Unsupported scoring.mode: {scoring.mode!r}")

    success_score = 1.0 if telemetry.success else 0.0
    efficiency = score_efficiency(
        success=telemetry.success,
        steps_to_success=telemetry.steps_to_success,
        max_episode_steps=max_episode_steps,
    )
    forces = telemetry.forces
    if forces is not None and forces.available_for_score:
        safety = score_safety(
            force_N=forces.total_force_norm_max_newton,
            scoring=scoring,
        )
    else:
        safety = 0.0
    smoothness = score_smoothness(
        tip_jerk_p95=telemetry.tip_jerk_p95,
        jerk_scale_mm_s3=scoring.smoothness_score.jerk_scale_mm_s3,
    )
    breakdown = ScoreBreakdown(
        total=0.0,
        success=float(success_score),
        efficiency=float(efficiency),
        safety=float(safety),
        smoothness=smoothness,
    )
    return ScoreBreakdown(
        total=float(soft_score_total(breakdown=breakdown, scoring=scoring)),
        success=breakdown.success,
        efficiency=breakdown.efficiency,
        safety=breakdown.safety,
        smoothness=breakdown.smoothness,
    )


def normalize_success_score(*, success_rate: float | None) -> float:
    return _clip01(_finite_or(success_rate, 0.0))


def normalize_time_score(*, insertion_time: float | None, max_expected_time: float) -> float:
    max_time = max(float(max_expected_time), 1e-9)
    time_value = max(_finite_or(insertion_time, max_time), 0.0)
    return _clip01(1.0 - (time_value / max_time))


def normalize_safety_score(*, max_force: float | None, safe_force_threshold: float) -> float:
    threshold = max(float(safe_force_threshold), 1e-9)
    force_value = max(_finite_or(max_force, threshold), 0.0)
    return _clip01(1.0 - (force_value / threshold))


def calculate_overall_score(
    summary: CandidateSummary,
    *,
    axes_weights: Dict[str, float],
    max_expected_time: float = 1000.0,
    safe_force_threshold: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    success_score = normalize_success_score(success_rate=summary.success_rate)
    insertion_time = summary.steps_to_success_mean
    if insertion_time is None:
        insertion_time = summary.steps_total_mean
    speed_score = normalize_time_score(
        insertion_time=insertion_time,
        max_expected_time=max_expected_time,
    )
    force_value = summary.wall_force_max_mean_newton
    if force_value is None:
        force_value = summary.wall_force_max_mean
    safety_score_value = normalize_safety_score(
        max_force=force_value,
        safe_force_threshold=safe_force_threshold,
    )

    weights = {
        "success": max(float(axes_weights.get("success", 0.0)), 0.0),
        "speed": max(float(axes_weights.get("speed", 0.0)), 0.0),
        "safety": max(float(axes_weights.get("safety", 0.0)), 0.0),
    }
    axes = {
        "success": success_score,
        "speed": speed_score,
        "safety": safety_score_value,
    }

    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        return 0.0, axes

    weighted_total = (
        weights["success"] * axes["success"]
        + weights["speed"] * axes["speed"]
        + weights["safety"] * axes["safety"]
    ) / total_weight
    return _clip01(weighted_total), axes


__all__ = [
    "calculate_overall_score",
    "force_within_safety_threshold",
    "normalize_safety_score",
    "normalize_success_score",
    "normalize_time_score",
    "score_efficiency",
    "score_safety",
    "score_smoothness",
    "score_trial",
    "soft_score_total",
    "valid_for_ranking",
]
