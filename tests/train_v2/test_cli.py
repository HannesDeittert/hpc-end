from pathlib import Path

import pytest

from steve_recommender.train_v2.cli import build_parser, build_training_config_from_args
from steve_recommender.train_v2.config import ARCHVAR_EVAL_SEEDS


def test_train_parser_builds_preflight_only_config():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--name",
            "smoke",
            "--tool",
            "steve_default/standard_j",
            "--preflight-only",
            "--resume-from",
            "/tmp/checkpoint.everl",
        ]
    )

    cfg = build_training_config_from_args(args)

    assert cfg.name == "smoke"
    assert cfg.preflight_only is True
    assert cfg.preflight is True
    assert cfg.resume_from == Path("/tmp/checkpoint.everl")
    assert cfg.eval_episodes is None


def test_train_parser_supports_normal_force_reward_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--name",
            "force",
            "--tool",
            "steve_default/standard_j",
            "--reward-profile",
            "default_plus_normal_force_penalty",
            "--force-alpha",
            "0.25",
            "--force-beta",
            "1.5",
            "--force-region",
            "tip_only",
        ]
    )

    cfg = build_training_config_from_args(args)

    assert cfg.reward.profile == "default_plus_normal_force_penalty"
    assert cfg.reward.force_alpha == 0.25
    assert cfg.reward.force_beta == 1.5
    assert cfg.reward.force_region == "tip_only"


def test_train_parser_supports_step_trace_h5_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--name",
            "trace",
            "--tool",
            "steve_default/standard_j",
            "--write-step-trace-h5",
            "--step-trace-every-n-steps",
            "7",
        ]
    )

    cfg = build_training_config_from_args(args)

    assert cfg.write_step_trace_h5 is True
    assert cfg.step_trace_every_n_steps == 7


def test_train_parser_rejects_removed_force_flags():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train",
                "--name",
                "legacy-force",
                "--tool",
                "steve_default/standard_j",
                "--force-threshold",
                "0.9",
            ]
        )


def test_train_parser_supports_custom_tool_module_and_class():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--name",
            "custom-tool",
            "--tool",
            "custom",
            "--tool-module",
            "steve_recommender.bench.custom_tools_amplatz_gentle_simple",
            "--tool-class",
            "JShapedAmplatzSuperStiffGentleSimple",
        ]
    )

    cfg = build_training_config_from_args(args)

    assert (
        cfg.runtime.tool_module
        == "steve_recommender.bench.custom_tools_amplatz_gentle_simple"
    )
    assert cfg.runtime.tool_class == "JShapedAmplatzSuperStiffGentleSimple"


def test_train_parser_uses_archvar_eval_seed_schedule_by_default():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--name",
            "default-seeds",
            "--tool",
            "steve_default/standard_j",
        ]
    )

    cfg = build_training_config_from_args(args)

    assert cfg.eval_seeds == ARCHVAR_EVAL_SEEDS
    assert cfg.eval_episodes is None
