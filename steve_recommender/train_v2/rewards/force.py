"""Force-aware reward components for train_v2."""

from __future__ import annotations

import math

from eve.reward.reward import Reward

from ..telemetry.force_runtime import ForceRuntime


class NormalForcePenaltyReward(Reward):
    """Per-step instantaneous plus one-shot terminal normal-force penalty."""

    def __init__(
        self,
        *,
        intervention,
        telemetry: ForceRuntime,
        terminal,
        truncation,
        alpha: float,
        beta: float,
        force_region: str,
    ) -> None:
        self.intervention = intervention
        self.telemetry = telemetry
        self.terminal = terminal
        self.truncation = truncation
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.force_region = str(force_region)
        self.reward = 0.0
        self._step_index = 0
        self._terminal_penalty_applied = False
        self.last_wire_force_normal_instant_N = 0.0
        self.last_wire_force_normal_trial_max_N = 0.0
        self.last_tip_force_normal_instant_N = 0.0
        self.last_tip_force_normal_trial_max_N = 0.0
        self.last_instant_force_N = 0.0
        self.last_trial_max_force_N = 0.0
        self.last_step_penalty = 0.0
        self.last_terminal_penalty = 0.0

    def _region_values(self, sample) -> tuple[float, float]:
        if self.force_region == "tip_only":
            return (
                float(sample.tip_force_normal_instant_N),
                float(sample.tip_force_normal_trial_max_N),
            )
        return (
            float(sample.wire_force_normal_instant_N),
            float(sample.wire_force_normal_trial_max_N),
        )

    def step(self) -> None:
        sample = self.telemetry.sample_step(
            intervention=self.intervention,
            step_index=self._step_index,
        )
        self._step_index += 1
        self.last_wire_force_normal_instant_N = float(sample.wire_force_normal_instant_N)
        self.last_wire_force_normal_trial_max_N = float(sample.wire_force_normal_trial_max_N)
        self.last_tip_force_normal_instant_N = float(sample.tip_force_normal_instant_N)
        self.last_tip_force_normal_trial_max_N = float(sample.tip_force_normal_trial_max_N)
        instant_force_N, trial_max_force_N = self._region_values(sample)
        per_step_penalty = -self.alpha * instant_force_N
        terminal_penalty = 0.0
        if (
            (bool(self.terminal.terminal) or bool(self.truncation.truncated))
            and not self._terminal_penalty_applied
        ):
            terminal_penalty = -self.beta * math.log1p(max(0.0, trial_max_force_N))
            self._terminal_penalty_applied = True
        self.last_instant_force_N = float(instant_force_N)
        self.last_trial_max_force_N = float(trial_max_force_N)
        self.last_step_penalty = float(per_step_penalty)
        self.last_terminal_penalty = float(terminal_penalty)
        self.reward = per_step_penalty + terminal_penalty

    def reset(self, episode_nr: int = 0) -> None:
        self._step_index = 0
        self._terminal_penalty_applied = False
        self.last_wire_force_normal_instant_N = 0.0
        self.last_wire_force_normal_trial_max_N = 0.0
        self.last_tip_force_normal_instant_N = 0.0
        self.last_tip_force_normal_trial_max_N = 0.0
        self.last_instant_force_N = 0.0
        self.last_trial_max_force_N = 0.0
        self.last_step_penalty = 0.0
        self.last_terminal_penalty = 0.0
        self.reward = 0.0
