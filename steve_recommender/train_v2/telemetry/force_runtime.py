"""Force telemetry adapter for train_v2 rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from steve_recommender.eval_v2.force_telemetry import EvalV2ForceTelemetryCollector
from steve_recommender.eval_v2.models import ForceTelemetrySpec, ForceUnits

from ..config import RewardSpec

_DEFAULT_UNITS = ForceUnits(length_unit="mm", mass_unit="kg", time_unit="s")


@dataclass(frozen=True)
class ForceRewardSample:
    """One force sample exposed to reward components."""

    total_force_norm_N: float
    tip_force_norm_N: float


class ForceRuntime:
    """Thin adapter around the eval_v2 force collector for training rewards."""

    def __init__(self, *, reward_spec: RewardSpec, action_dt_s: float) -> None:
        needs_units = (
            reward_spec.force_telemetry_mode == "constraint_projected_si_validated"
        )
        spec = ForceTelemetrySpec(
            mode=reward_spec.force_telemetry_mode,
            units=_DEFAULT_UNITS if needs_units else None,
            write_full_trace=False,
            write_diagnostics=False,
        )
        self._collector = EvalV2ForceTelemetryCollector(
            spec=spec,
            action_dt_s=action_dt_s,
        )

    def ensure_runtime(self, *, intervention: Any) -> None:
        self._collector.ensure_runtime(intervention=intervention)

    def sample_step(self, *, intervention: Any, step_index: int) -> ForceRewardSample:
        self._collector.capture_step(intervention=intervention, step_index=step_index)
        summary = self._collector.build_summary()
        return ForceRewardSample(
            total_force_norm_N=float(summary.total_force_norm_max_newton or 0.0),
            tip_force_norm_N=float(summary.tip_force_total_norm_N or 0.0),
        )
