"""Force-aware reward components for train_v2."""

from __future__ import annotations

from eve.reward.reward import Reward

from ..telemetry.force_runtime import ForceRuntime


class ForcePenaltyReward(Reward):
    """Penalty proportional to measured wall force magnitude in Newton."""

    def __init__(
        self,
        *,
        intervention,
        telemetry: ForceRuntime,
        factor: float,
        tip_only: bool,
    ) -> None:
        self.intervention = intervention
        self.telemetry = telemetry
        self.factor = float(factor)
        self.tip_only = bool(tip_only)
        self.reward = 0.0
        self._step_index = 0

    def step(self) -> None:
        sample = self.telemetry.sample_step(
            intervention=self.intervention,
            step_index=self._step_index,
        )
        self._step_index += 1
        magnitude = (
            sample.tip_force_norm_N if self.tip_only else sample.total_force_norm_N
        )
        self.reward = -self.factor * float(magnitude)

    def reset(self, episode_nr: int = 0) -> None:
        self._step_index = 0
        self.reward = 0.0


class ExcessForcePenaltyReward(Reward):
    """Penalty only for force magnitude above one threshold in Newton."""

    def __init__(
        self,
        *,
        intervention,
        telemetry: ForceRuntime,
        threshold_N: float,
        divisor: float,
        tip_only: bool,
    ) -> None:
        self.intervention = intervention
        self.telemetry = telemetry
        self.threshold_N = float(threshold_N)
        self.divisor = float(divisor)
        self.tip_only = bool(tip_only)
        self.reward = 0.0
        self._step_index = 0

    def step(self) -> None:
        sample = self.telemetry.sample_step(
            intervention=self.intervention,
            step_index=self._step_index,
        )
        self._step_index += 1
        magnitude = (
            sample.tip_force_norm_N if self.tip_only else sample.total_force_norm_N
        )
        excess = max(0.0, float(magnitude) - self.threshold_N)
        self.reward = -(excess / self.divisor)

    def reset(self, episode_nr: int = 0) -> None:
        self._step_index = 0
        self.reward = 0.0
