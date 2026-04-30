from pathlib import Path

from steve_recommender.train_v2.cli import build_parser, build_training_config_from_args


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


def test_train_parser_supports_force_reward_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--name",
            "force",
            "--tool",
            "steve_default/standard_j",
            "--reward-profile",
            "default_plus_force_penalty",
            "--force-penalty-factor",
            "0.25",
            "--force-tip-only",
        ]
    )

    cfg = build_training_config_from_args(args)

    assert cfg.reward.profile == "default_plus_force_penalty"
    assert cfg.reward.force_penalty_factor == 0.25
    assert cfg.reward.force_tip_only is True


def test_train_parser_supports_excess_force_reward_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--name",
            "excess-force",
            "--tool",
            "steve_default/standard_j",
            "--reward-profile",
            "default_plus_excess_force_penalty",
            "--force-threshold",
            "0.9",
            "--force-divisor",
            "1500",
            "--force-tip-only",
        ]
    )

    cfg = build_training_config_from_args(args)

    assert cfg.reward.profile == "default_plus_excess_force_penalty"
    assert cfg.reward.force_threshold_N == 0.9
    assert cfg.reward.force_divisor == 1500.0
    assert cfg.reward.force_tip_only is True


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
