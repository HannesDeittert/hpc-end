"""Small local helpers around stEVE rewards."""

from __future__ import annotations

from eve.reward.reward import Reward


class ConstantPenaltyReward(Reward):
    """Simple fixed penalty reward used for smoke validation."""

    def __init__(self, factor: float) -> None:
        self.factor = float(factor)
        self.reward = 0.0

    def step(self) -> None:
        self.reward = self.factor

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
