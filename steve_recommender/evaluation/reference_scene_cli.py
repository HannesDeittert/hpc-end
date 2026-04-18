from __future__ import annotations

import argparse
from pathlib import Path

from .config import ForceUnitsConfig
from .reference_scene import run_reference_scene_suite


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run deterministic SOFA reference scenes (point-vs-triangle and line-vs-triangle) "
            "for strict validated wall-force telemetry checks."
        )
    )
    p.add_argument(
        "--tool",
        default="TestModel_StandardJ035/StandardJ035_PTFE",
        help="Device reference (model/wire).",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to results/force_reference_scene/<timestamp>.",
    )
    p.add_argument("--seed", type=int, default=123, help="Deterministic seed used by both probe runs.")
    p.add_argument(
        "--plugin-path",
        default=None,
        help="Optional explicit path to libSofaWireForceMonitor.so.",
    )
    p.add_argument(
        "--contact-epsilon",
        type=float,
        default=1e-7,
        help="Contact epsilon for active-step and non-zero checks.",
    )
    p.add_argument(
        "--length-unit",
        choices=("mm", "m"),
        default="mm",
        help="Force telemetry length unit metadata.",
    )
    p.add_argument(
        "--mass-unit",
        choices=("kg", "g"),
        default="kg",
        help="Force telemetry mass unit metadata.",
    )
    p.add_argument(
        "--time-unit",
        choices=("s", "ms"),
        default="s",
        help="Force telemetry time unit metadata.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.output_dir).expanduser() if str(args.output_dir).strip() else None
    units = ForceUnitsConfig(
        length_unit=str(args.length_unit),
        mass_unit=str(args.mass_unit),
        time_unit=str(args.time_unit),
    )
    rep = run_reference_scene_suite(
        tool_ref=str(args.tool),
        output_dir=out_dir,
        base_seed=int(args.seed),
        plugin_path=args.plugin_path,
        contact_epsilon=float(args.contact_epsilon),
        units=units,
    )
    print(
        "[force-reference] pass_validated_suite={pass_suite} external_limit={ext} reason={reason}".format(
            pass_suite=int(bool(rep.pass_validated_suite)),
            ext=int(bool(rep.external_limit_detected)),
            reason=rep.external_limit_reason or "none",
        )
    )
    print(f"[force-reference] output_dir={rep.output_dir}")
    for case in rep.case_reports:
        print(
            "[force-reference] case={name} pass={passed} reproducible={repro} "
            "run_a_active={a_act} run_b_active={b_act} "
            "run_a_reason={a_reason} run_b_reason={b_reason} "
            "repro_error={repro_err} div_step={div_step} div_domain={div_domain} "
            "div_kind={div_kind} nonsemantic={nonsemantic}".format(
                name=case.config.name,
                passed=int(bool(case.pass_validated_case)),
                repro=int(bool(case.reproducible)),
                a_act=int(case.run_a.active_constraint_steps),
                b_act=int(case.run_b.active_constraint_steps),
                a_reason=case.run_a.dominant_failure_reason or "none",
                b_reason=case.run_b.dominant_failure_reason or "none",
                repro_err=case.reproducibility_error or "none",
                div_step=int(case.first_divergence_step),
                div_domain=case.first_divergence_domain or "none",
                div_kind=case.first_divergence_kind or "none",
                nonsemantic=case.first_nonsemantic_divergence or "none",
            )
        )


if __name__ == "__main__":
    main()
