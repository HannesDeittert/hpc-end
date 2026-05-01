import math

import pytest

from steve_recommender.train_v2.config import RewardSpec
from steve_recommender.train_v2.rewards.factory import build_reward
from steve_recommender.train_v2.rewards.force import NormalForcePenaltyReward
from steve_recommender.train_v2.telemetry.force_runtime import ForceRewardSample


class DummyTerminal:
    def __init__(self):
        self.terminal = False

    def step(self):
        return None

    def reset(self, episode_nr=0):
        self.terminal = False


class DummyTruncation:
    def __init__(self):
        self.truncated = False

    def step(self):
        return None

    def reset(self, episode_nr=0):
        self.truncated = False


class SequenceTelemetry:
    def __init__(self, samples):
        self._samples = list(samples)
        self._index = 0

    def sample_step(self, intervention, step_index):
        _ = intervention, step_index
        if self._index >= len(self._samples):
            return self._samples[-1]
        sample = self._samples[self._index]
        self._index += 1
        return sample


def _sample(*, wire_instant, wire_trial_max, tip_instant=0.0, tip_trial_max=0.0):
    return ForceRewardSample(
        wire_force_normal_instant_N=wire_instant,
        wire_force_normal_trial_max_N=wire_trial_max,
        tip_force_normal_instant_N=tip_instant,
        tip_force_normal_trial_max_N=tip_trial_max,
    )


def test_normal_force_penalty_reward_uses_current_step_force_only():
    terminal = DummyTerminal()
    truncation = DummyTruncation()
    first = _sample(wire_instant=0.9, wire_trial_max=0.9)
    second_low_history = _sample(wire_instant=0.3, wire_trial_max=0.9)
    second_high_history = _sample(wire_instant=0.3, wire_trial_max=1.7)

    low_history_reward = NormalForcePenaltyReward(
        intervention=object(),
        telemetry=SequenceTelemetry([first, second_low_history]),
        terminal=terminal,
        truncation=truncation,
        alpha=0.1,
        beta=1.0,
        force_region="whole_wire",
    )
    high_history_reward = NormalForcePenaltyReward(
        intervention=object(),
        telemetry=SequenceTelemetry([first, second_high_history]),
        terminal=terminal,
        truncation=truncation,
        alpha=0.1,
        beta=1.0,
        force_region="whole_wire",
    )

    low_history_reward.step()
    high_history_reward.step()
    low_history_reward.step()
    high_history_reward.step()

    assert low_history_reward.reward == pytest.approx(-0.03)
    assert high_history_reward.reward == pytest.approx(-0.03)


@pytest.mark.parametrize("end_attr", ["terminal", "truncated"])
def test_normal_force_penalty_reward_applies_terminal_term_once_for_both_episode_end_modes(
    end_attr,
):
    terminal = DummyTerminal()
    truncation = DummyTruncation()
    reward = NormalForcePenaltyReward(
        intervention=object(),
        telemetry=SequenceTelemetry([_sample(wire_instant=0.4, wire_trial_max=0.8)]),
        terminal=terminal,
        truncation=truncation,
        alpha=0.1,
        beta=1.0,
        force_region="whole_wire",
    )

    setattr(terminal if end_attr == "terminal" else truncation, end_attr, True)

    reward.step()
    expected = -0.1 * 0.4 - math.log(1.0 + 0.8)
    assert reward.reward == pytest.approx(expected)

    reward.step()
    assert reward.reward == pytest.approx(-0.1 * 0.4)


def test_zero_alpha_and_beta_reproduces_pure_task_reward():
    terminal = DummyTerminal()
    truncation = DummyTruncation()
    reward = NormalForcePenaltyReward(
        intervention=object(),
        telemetry=SequenceTelemetry([_sample(wire_instant=1.2, wire_trial_max=2.0)]),
        terminal=terminal,
        truncation=truncation,
        alpha=0.0,
        beta=0.0,
        force_region="whole_wire",
    )

    reward.step()
    assert reward.reward == pytest.approx(0.0)


def test_normal_force_penalty_reward_can_use_tip_region():
    terminal = DummyTerminal()
    truncation = DummyTruncation()
    reward = NormalForcePenaltyReward(
        intervention=object(),
        telemetry=SequenceTelemetry(
            [_sample(wire_instant=1.2, wire_trial_max=1.2, tip_instant=0.5, tip_trial_max=0.7)]
        ),
        terminal=terminal,
        truncation=truncation,
        alpha=0.1,
        beta=1.0,
        force_region="tip_only",
    )

    reward.step()
    assert reward.reward == pytest.approx(-0.05)


def test_normal_force_penalty_reward_exposes_debug_components():
    terminal = DummyTerminal()
    truncation = DummyTruncation()
    reward = NormalForcePenaltyReward(
        intervention=object(),
        telemetry=SequenceTelemetry([_sample(wire_instant=0.4, wire_trial_max=0.8)]),
        terminal=terminal,
        truncation=truncation,
        alpha=0.1,
        beta=1.0,
        force_region="whole_wire",
    )
    terminal.terminal = True

    reward.step()

    assert reward.last_instant_force_N == pytest.approx(0.4)
    assert reward.last_trial_max_force_N == pytest.approx(0.8)
    assert reward.last_step_penalty == pytest.approx(-0.04)
    assert reward.last_terminal_penalty == pytest.approx(-math.log(1.0 + 0.8))


def test_reward_factory_requires_telemetry_for_normal_force_profile():
    with pytest.raises(ValueError, match="telemetry is required"):
        build_reward(
            intervention=object(),
            pathfinder=object(),
            reward_spec=RewardSpec(
                profile="default_plus_normal_force_penalty",
                force_alpha=0.1,
                force_beta=1.0,
            ),
            telemetry=None,
        )
