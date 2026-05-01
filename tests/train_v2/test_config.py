from pathlib import Path

import pytest

from steve_recommender.train_v2.config import (
    ARCHVAR_EVAL_SEEDS,
    RewardSpec,
    build_doctor_config,
    build_training_config,
    parse_eval_seeds,
)


def test_parse_eval_seeds_none_returns_none():
    assert parse_eval_seeds("none") is None


def test_parse_eval_seeds_csv_returns_tuple():
    assert parse_eval_seeds("1, 2,3") == (1, 2, 3)


def test_normal_force_reward_profile_requires_positive_alpha():
    with pytest.raises(ValueError, match="force_alpha"):
        RewardSpec(
            profile="default_plus_normal_force_penalty",
            force_alpha=-0.1,
        )


def test_normal_force_reward_profile_requires_non_negative_beta():
    with pytest.raises(ValueError, match="force_beta"):
        RewardSpec(
            profile="default_plus_normal_force_penalty",
            force_beta=-1.0,
        )


def test_build_doctor_config_converts_paths():
    cfg = build_doctor_config(
        tool_ref="steve_default/standard_j",
        resume_from=Path("checkpoint.everl"),
    )
    assert cfg.resume_from == Path("checkpoint.everl")


def test_default_eval_seed_schedule_matches_archvar():
    assert len(ARCHVAR_EVAL_SEEDS) == 98
    assert ARCHVAR_EVAL_SEEDS[:5] == (1, 2, 3, 5, 6)
    assert ARCHVAR_EVAL_SEEDS[-3:] == (168, 171, 175)


def test_training_config_defaults_to_full_eval_seed_batch():
    cfg = build_training_config(name="run", tool_ref="steve_default/standard_j")
    assert cfg.eval_seeds == ARCHVAR_EVAL_SEEDS
    assert cfg.eval_episodes is None
