from steve_recommender.train_v2.config import build_doctor_config
from steve_recommender.train_v2.doctor.report import (
    CheckResult,
    exit_code,
    render_report,
)


def test_render_report_formats_levels():
    text = render_report([CheckResult("ok", "code", "message")])
    assert "[OK] code: message" in text


def test_exit_code_returns_error_for_errors():
    code = exit_code([CheckResult("error", "code", "message")], strict=False)
    assert code == 1


def test_exit_code_returns_strict_warning_code():
    code = exit_code([CheckResult("warning", "code", "message")], strict=True)
    assert code == 2


def test_build_doctor_config_defaults_to_boot_env():
    cfg = build_doctor_config(
        tool_ref="steve_default/standard_j",
    )
    assert cfg.boot_env is True
