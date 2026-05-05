import csv
from pathlib import Path

import pytest

from steve_recommender.train_v2.rewards.tracker import RewardTracker


class _DummyReward:
    def __init__(self, reward: float = 0.0):
        self.reward = float(reward)
        self.step_calls = 0
        self.reset_calls = 0

    def step(self):
        self.step_calls += 1

    def reset(self, episode_nr=0):
        _ = episode_nr
        self.reset_calls += 1


class _DummyForceReward(_DummyReward):
    def __init__(self, reward: float = 0.0):
        super().__init__(reward=reward)
        self.last_step_penalty = -0.123
        self.last_terminal_penalty = -0.456
        self.last_wire_force_normal_instant_N = 1.25
        self.last_wire_force_normal_trial_max_N = 2.5
        self.last_tip_force_normal_instant_N = 0.75
        self.last_tip_force_normal_trial_max_N = 1.5
        self.last_force_validation_status = "ok"
        self.last_force_source = "passive_monitor:fallback_lcp_dt_mapped"
        self.last_force_channel = "lcp.constraintForces/dt"
        self.last_force_quality_tier = "validated"
        self.last_force_available_for_score = True
        self.last_lcp_mapped_wall_row_count_max = 7


def test_reward_tracker_writes_force_diagnostics_to_csv(tmp_path: Path):
    csv_path = tmp_path / "reward_eval.csv"
    tracker = RewardTracker(
        components=[
            ("target", _DummyReward(reward=1.0)),
            ("force", _DummyForceReward(reward=-0.25)),
        ],
        csv_path=csv_path,
    )

    tracker.step()
    tracker.reset(episode_nr=7)

    resolved = tracker._resolved_path()
    assert resolved is not None
    assert resolved.exists()

    with resolved.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["episode"] == "0"
    assert row["steps"] == "1"
    assert float(row["total"]) == pytest.approx(0.75, rel=0, abs=1e-6)
    assert row["target"] == "1.000000"
    assert row["force"] == "-0.250000"
    assert row["force_step_penalty"] == "-0.123000"
    assert row["force_terminal_penalty"] == "-0.456000"
    assert row["wire_force_normal_instant_N"] == "1.250000"
    assert row["wire_force_normal_trial_max_N"] == "2.500000"
    assert row["tip_force_normal_instant_N"] == "0.750000"
    assert row["tip_force_normal_trial_max_N"] == "1.500000"
    assert row["force_validation_status"] == "ok"
    assert row["force_source"] == "passive_monitor:fallback_lcp_dt_mapped"
    assert row["force_channel"] == "lcp.constraintForces/dt"
    assert row["force_quality_tier"] == "validated"
    assert row["force_available_for_score"] == "1"
    assert row["lcp_mapped_wall_row_count_max"] == "7"
