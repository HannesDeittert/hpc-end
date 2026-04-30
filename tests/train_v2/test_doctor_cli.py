from steve_recommender.train_v2.cli import build_doctor_config_from_args, build_parser


def test_doctor_parser_builds_config():
    parser = build_parser()
    args = parser.parse_args(
        [
            "doctor",
            "--tool",
            "steve_default/standard_j",
            "--force-penalty-factor",
            "0.1",
            "--reward-profile",
            "default_plus_force_penalty",
        ]
    )
    cfg = build_doctor_config_from_args(args)
    assert cfg.runtime.anatomy_id is None
    assert cfg.runtime.tool_ref == "steve_default/standard_j"
    assert cfg.reward.force_penalty_factor == 0.1
