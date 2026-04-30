from pathlib import Path

import pytest

from steve_recommender.train_v2.config import (
    RewardSpec,
    build_doctor_config,
    parse_eval_seeds,
)


def test_parse_eval_seeds_none_returns_none():
    assert parse_eval_seeds("none") is None


def test_parse_eval_seeds_csv_returns_tuple():
    assert parse_eval_seeds("1, 2,3") == (1, 2, 3)


def test_force_reward_profile_requires_positive_factor():
    with pytest.raises(ValueError, match="force_penalty_factor"):
        RewardSpec(
            profile="default_plus_force_penalty",
            force_penalty_factor=0.0,
        )


def test_excess_force_reward_profile_requires_positive_divisor():
    with pytest.raises(ValueError, match="force_divisor"):
        RewardSpec(
            profile="default_plus_excess_force_penalty",
            force_threshold_N=0.85,
            force_divisor=0.0,
        )


def test_build_doctor_config_converts_paths():
    cfg = build_doctor_config(
        tool_ref="steve_default/standard_j",
        resume_from=Path("checkpoint.everl"),
    )
    assert cfg.resume_from == Path("checkpoint.everl")
