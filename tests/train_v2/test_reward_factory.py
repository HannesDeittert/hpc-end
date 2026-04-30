import pytest

from steve_recommender.train_v2.config import RewardSpec
from steve_recommender.train_v2.rewards.factory import build_reward
from steve_recommender.train_v2.rewards.force import (
    ExcessForcePenaltyReward,
    ForcePenaltyReward,
)
from steve_recommender.train_v2.telemetry.force_runtime import ForceRewardSample


class DummyReward:
    def __init__(self):
        self.reward = 0.0

    def step(self):
        self.reward = 0.0

    def reset(self, episode_nr=0):
        self.reward = 0.0


class DummyTelemetry:
    def __init__(self):
        self._latest = ForceRewardSample(total_force_norm_N=2.5, tip_force_norm_N=1.5)

    @property
    def latest(self):
        return self._latest

    def sample_step(self, intervention, step_index):
        return self._latest


def test_force_penalty_reward_uses_total_force():
    reward = ForcePenaltyReward(
        intervention=object(),
        telemetry=DummyTelemetry(),
        factor=0.2,
        tip_only=False,
    )
    reward.step()
    assert reward.reward == pytest.approx(-0.5)


def test_force_penalty_reward_uses_tip_force():
    reward = ForcePenaltyReward(
        intervention=object(),
        telemetry=DummyTelemetry(),
        factor=0.2,
        tip_only=True,
    )
    reward.step()
    assert reward.reward == pytest.approx(-0.3)


def test_excess_force_penalty_reward_ignores_force_below_threshold():
    reward = ExcessForcePenaltyReward(
        intervention=object(),
        telemetry=DummyTelemetry(),
        threshold_N=3.0,
        divisor=1000.0,
        tip_only=False,
    )
    reward.step()
    assert reward.reward == pytest.approx(0.0)


def test_excess_force_penalty_reward_penalizes_force_above_threshold():
    reward = ExcessForcePenaltyReward(
        intervention=object(),
        telemetry=DummyTelemetry(),
        threshold_N=0.85,
        divisor=1000.0,
        tip_only=False,
    )
    reward.step()
    assert reward.reward == pytest.approx(-(2.5 - 0.85) / 1000.0)


def test_excess_force_penalty_reward_can_use_tip_force_only():
    reward = ExcessForcePenaltyReward(
        intervention=object(),
        telemetry=DummyTelemetry(),
        threshold_N=0.85,
        divisor=1000.0,
        tip_only=True,
    )
    reward.step()
    assert reward.reward == pytest.approx(-(1.5 - 0.85) / 1000.0)


def test_reward_factory_requires_telemetry_for_force_profile():
    with pytest.raises(ValueError, match="telemetry is required"):
        build_reward(
            intervention=object(),
            pathfinder=object(),
            reward_spec=RewardSpec(
                profile="default_plus_force_penalty",
                force_penalty_factor=0.1,
            ),
            telemetry=None,
        )


def test_reward_factory_requires_telemetry_for_excess_force_profile():
    with pytest.raises(ValueError, match="telemetry is required"):
        build_reward(
            intervention=object(),
            pathfinder=object(),
            reward_spec=RewardSpec(
                profile="default_plus_excess_force_penalty",
                force_threshold_N=0.85,
                force_divisor=1000.0,
            ),
            telemetry=None,
        )
