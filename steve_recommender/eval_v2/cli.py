from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, TextIO, Tuple

from .models import (
    BranchEndTarget,
    BranchIndexTarget,
    EvaluationCandidate,
    EvaluationJob,
    EvaluationScenario,
    ExecutionPlan,
    FluoroscopySpec,
    ManualTarget,
    PolicySpec,
    WireRef,
)
from .service import DefaultEvaluationService, EvaluationService


def _parse_wire_ref(text: str) -> WireRef:
    parts = tuple(part.strip() for part in str(text).split("/", maxsplit=1))
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"wire ref must look like 'model/wire', got {text!r}")
    return WireRef(model=parts[0], wire=parts[1])


def _parse_vector3(text: str) -> tuple[float, float, float]:
    parts = tuple(part.strip() for part in str(text).split(","))
    if len(parts) != 3:
        raise ValueError(f"manual target must look like 'x,y,z', got {text!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _parse_branch_names(text: str) -> Tuple[str, ...]:
    branches = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if not branches:
        raise ValueError("--target-branches must include at least one branch")
    return branches


def _format_optional(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.4g}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Researcher CLI for the clean-room eval_v2 evaluation pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-anatomies", help="List discoverable anatomies")

    wires_parser = subparsers.add_parser("list-wires", help="List execution wires")
    wires_parser.add_argument(
        "--startable-only",
        action="store_true",
        help="Show only wires that currently have at least one loadable agent",
    )

    policies_parser = subparsers.add_parser("list-policies", help="List discoverable policies")
    policies_parser.add_argument(
        "--execution-wire",
        default=None,
        help="Optional wire ref filter formatted as 'model/wire'",
    )

    candidates_parser = subparsers.add_parser("list-candidates", help="List candidate options")
    candidates_parser.add_argument(
        "--execution-wire",
        required=True,
        help="Execution wire formatted as 'model/wire'",
    )
    candidates_parser.add_argument(
        "--no-cross-wire",
        action="store_true",
        help="Exclude policies trained on different wires",
    )

    branches_parser = subparsers.add_parser("list-branches", help="List branches for one anatomy")
    branches_parser.add_argument(
        "--anatomy",
        required=True,
        help="Stable anatomy record id",
    )

    subparsers.add_parser("list-target-modes", help="List supported target modes")

    run_parser = subparsers.add_parser("run", help="Build and execute one eval_v2 job")
    run_parser.add_argument("--job-name", default="eval_v2_job")
    run_parser.add_argument("--scenario-name", default="scenario")
    run_parser.add_argument("--candidate-label", default=None)
    run_parser.add_argument(
        "--anatomy",
        required=True,
        help="Stable anatomy record id",
    )
    run_parser.add_argument(
        "--execution-wire",
        required=True,
        help="Execution wire formatted as 'model/wire'",
    )

    policy_group = run_parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument(
        "--candidate-name",
        default=None,
        help="Select one pre-synthesized candidate name from service.list_candidates(...)",
    )
    policy_group.add_argument(
        "--policy-name",
        default=None,
        help="Select one discoverable policy by PolicySpec.name",
    )
    policy_group.add_argument(
        "--policy-checkpoint",
        default=None,
        help="Use an explicit checkpoint path not yet registered in the policy discovery layer",
    )
    run_parser.add_argument(
        "--policy-label",
        default=None,
        help="Override the policy name when using --policy-checkpoint",
    )
    run_parser.add_argument(
        "--policy-trained-on-wire",
        default=None,
        help="Optional trained-on wire ref for explicit checkpoint usage",
    )

    run_parser.add_argument(
        "--target-mode",
        required=True,
        choices=("branch_end", "branch_index", "manual"),
    )
    run_parser.add_argument(
        "--target-branches",
        default=None,
        help="Comma-separated branches for branch_end mode",
    )
    run_parser.add_argument(
        "--target-branch",
        default=None,
        help="Single branch name for branch_index mode",
    )
    run_parser.add_argument(
        "--target-index",
        type=int,
        default=None,
        help="Centerline index for branch_index mode",
    )
    run_parser.add_argument(
        "--manual-target",
        action="append",
        default=None,
        help="Manual vessel-space target formatted as 'x,y,z'; repeat for multiple points",
    )
    run_parser.add_argument("--threshold-mm", type=float, default=5.0)

    run_parser.add_argument("--trial-count", type=int, default=10)
    run_parser.add_argument("--base-seed", type=int, default=123)
    run_parser.add_argument("--max-episode-steps", type=int, default=1000)
    run_parser.add_argument("--policy-device", default="cpu")
    run_parser.add_argument(
        "--policy-mode",
        choices=("deterministic", "stochastic"),
        default="deterministic",
    )

    run_parser.add_argument("--friction", type=float, default=0.001)
    run_parser.add_argument("--image-frequency-hz", type=float, default=7.5)
    run_parser.add_argument("--image-rot-z-deg", type=float, default=20.0)
    run_parser.add_argument("--image-rot-x-deg", type=float, default=5.0)
    run_parser.add_argument("--output-root", default="results/eval_runs")

    normalize_group = run_parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--normalize-action",
        dest="normalize_action",
        action="store_true",
        help="Normalize actions before feeding them into the intervention",
    )
    normalize_group.add_argument(
        "--no-normalize-action",
        dest="normalize_action",
        action="store_false",
        help="Disable action normalization",
    )
    run_parser.set_defaults(normalize_action=True)

    tree_end_group = run_parser.add_mutually_exclusive_group()
    tree_end_group.add_argument(
        "--stop-device-at-tree-end",
        dest="stop_device_at_tree_end",
        action="store_true",
        help="Stop device advancement at the end of the vessel tree",
    )
    tree_end_group.add_argument(
        "--allow-device-past-tree-end",
        dest="stop_device_at_tree_end",
        action="store_false",
        help="Allow advancement logic to continue beyond tree-end checks",
    )
    run_parser.set_defaults(stop_device_at_tree_end=True)

    return parser


def _write(stdout: TextIO, line: str) -> None:
    stdout.write(f"{line}\n")


def _select_policy_for_run(
    *,
    args: argparse.Namespace,
    service: EvaluationService,
    execution_wire: WireRef,
) -> EvaluationCandidate | PolicySpec:
    if args.candidate_name is not None:
        candidates = service.list_candidates(
            execution_wire=execution_wire,
            include_cross_wire=True,
        )
        matches = tuple(candidate for candidate in candidates if candidate.name == args.candidate_name)
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one candidate named {args.candidate_name!r}, found {len(matches)}"
            )
        return matches[0]

    if args.policy_name is not None:
        policies = service.list_registry_policies() + service.list_explicit_policies()
        matches = tuple(policy for policy in policies if policy.name == args.policy_name)
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one policy named {args.policy_name!r}, found {len(matches)}"
            )
        return matches[0]

    checkpoint_path = Path(args.policy_checkpoint)
    trained_on_wire = (
        None
        if args.policy_trained_on_wire is None
        else _parse_wire_ref(args.policy_trained_on_wire)
    )
    return PolicySpec(
        name=args.policy_label or checkpoint_path.stem,
        checkpoint_path=checkpoint_path,
        source="explicit",
        trained_on_wire=trained_on_wire,
    )


def _build_target_spec(
    *,
    args: argparse.Namespace,
    service: EvaluationService,
    anatomy: AorticArchAnatomy,
):
    available_modes = {mode.kind for mode in service.list_target_modes()}
    if args.target_mode not in available_modes:
        raise ValueError(f"Unsupported target mode: {args.target_mode!r}")

    if args.target_mode == "branch_end":
        if args.target_branches is None:
            raise ValueError("--target-branches is required for target_mode='branch_end'")
        branches = _parse_branch_names(args.target_branches)
        for branch_name in branches:
            service.get_branch(anatomy, branch_name=branch_name)
        return BranchEndTarget(
            threshold_mm=float(args.threshold_mm),
            branches=branches,
        )

    if args.target_mode == "branch_index":
        if args.target_branch is None:
            raise ValueError("--target-branch is required for target_mode='branch_index'")
        if args.target_index is None:
            raise ValueError("--target-index is required for target_mode='branch_index'")
        service.get_branch(anatomy, branch_name=args.target_branch)
        return BranchIndexTarget(
            branch=args.target_branch,
            index=int(args.target_index),
            threshold_mm=float(args.threshold_mm),
        )

    if not args.manual_target:
        raise ValueError("--manual-target is required for target_mode='manual'")
    return ManualTarget(
        targets_vessel_cs=tuple(_parse_vector3(item) for item in args.manual_target),
        threshold_mm=float(args.threshold_mm),
    )


def _handle_list_anatomies(service: EvaluationService, stdout: TextIO) -> int:
    for anatomy in service.list_anatomies():
        _write(
            stdout,
            "record_id={record_id} arch_type={arch_type} seed={seed}".format(
                record_id=anatomy.record_id or "<none>",
                arch_type=anatomy.arch_type,
                seed=anatomy.seed,
            ),
        )
    return 0


def _handle_list_wires(
    args: argparse.Namespace,
    service: EvaluationService,
    stdout: TextIO,
) -> int:
    wires = service.list_startable_wires() if args.startable_only else service.list_execution_wires()
    for wire in wires:
        _write(stdout, f"{wire.tool_ref}")
    return 0


def _handle_list_policies(
    args: argparse.Namespace,
    service: EvaluationService,
    stdout: TextIO,
) -> int:
    execution_wire = None if args.execution_wire is None else _parse_wire_ref(args.execution_wire)
    policies = service.list_registry_policies(execution_wire=execution_wire) + service.list_explicit_policies(
        execution_wire=execution_wire
    )
    for policy in policies:
        _write(
            stdout,
            "name={name} source={source} trained_on={trained_on} checkpoint={checkpoint}".format(
                name=policy.name,
                source=policy.source,
                trained_on=policy.trained_on_wire.tool_ref if policy.trained_on_wire else "unknown",
                checkpoint=policy.checkpoint_path,
            ),
        )
    return 0


def _handle_list_candidates(
    args: argparse.Namespace,
    service: EvaluationService,
    stdout: TextIO,
) -> int:
    execution_wire = _parse_wire_ref(args.execution_wire)
    candidates = service.list_candidates(
        execution_wire=execution_wire,
        include_cross_wire=not bool(args.no_cross_wire),
    )
    for candidate in candidates:
        _write(
            stdout,
            "name={name} execution={execution} trained_on={trained_on} cross_wire={cross}".format(
                name=candidate.name,
                execution=candidate.execution_wire.tool_ref,
                trained_on=(
                    candidate.policy.trained_on_wire.tool_ref
                    if candidate.policy.trained_on_wire is not None
                    else "unknown"
                ),
                cross=int(candidate.is_cross_wire),
            ),
        )
    return 0


def _handle_list_branches(
    args: argparse.Namespace,
    service: EvaluationService,
    stdout: TextIO,
) -> int:
    anatomy = service.get_anatomy(record_id=args.anatomy)
    for branch in service.list_branches(anatomy):
        _write(
            stdout,
            "name={name} point_count={point_count} terminal_index={terminal_index} length_mm={length_mm}".format(
                name=branch.name,
                point_count=branch.point_count,
                terminal_index=branch.terminal_index,
                length_mm=branch.length_mm,
            ),
        )
    return 0


def _handle_list_target_modes(service: EvaluationService, stdout: TextIO) -> int:
    for mode in service.list_target_modes():
        _write(
            stdout,
            (
                "kind={kind} label={label} branch={branch} index={index} "
                "multi_branch={multi_branch} manual_points={manual_points}"
            ).format(
                kind=mode.kind,
                label=mode.label,
                branch=int(mode.requires_branch_selection),
                index=int(mode.requires_index_selection),
                multi_branch=int(mode.allows_multi_branch_selection),
                manual_points=int(mode.requires_manual_points),
            ),
        )
    return 0


def _handle_run(
    args: argparse.Namespace,
    service: EvaluationService,
    stdout: TextIO,
) -> int:
    anatomy = service.get_anatomy(record_id=args.anatomy)
    execution_wire = _parse_wire_ref(args.execution_wire)
    target = _build_target_spec(args=args, service=service, anatomy=anatomy)
    selected = _select_policy_for_run(
        args=args,
        service=service,
        execution_wire=execution_wire,
    )
    if isinstance(selected, EvaluationCandidate):
        candidate = selected
    else:
        candidate = service.build_candidate(
            name=args.candidate_label or selected.name,
            execution_wire=execution_wire,
            policy=selected,
        )

    scenario = EvaluationScenario(
        name=args.scenario_name,
        anatomy=anatomy,
        target=target,
        fluoroscopy=FluoroscopySpec(
            image_frequency_hz=float(args.image_frequency_hz),
            image_rot_zx_deg=(float(args.image_rot_z_deg), float(args.image_rot_x_deg)),
        ),
        friction=float(args.friction),
        stop_device_at_tree_end=bool(args.stop_device_at_tree_end),
        normalize_action=bool(args.normalize_action),
    )
    job = EvaluationJob(
        name=args.job_name,
        scenarios=(scenario,),
        candidates=(candidate,),
        execution=ExecutionPlan(
            trials_per_candidate=int(args.trial_count),
            base_seed=int(args.base_seed),
            max_episode_steps=int(args.max_episode_steps),
            policy_device=str(args.policy_device),
            policy_mode=str(args.policy_mode),
        ),
        output_root=Path(args.output_root),
    )
    report = service.run_evaluation_job(job)
    _write(stdout, f"[eval_v2] job={report.job_name} generated_at={report.generated_at}")
    _write(stdout, f"[eval_v2] summaries={len(report.summaries)} trials={len(report.trials)}")
    if report.artifacts is not None:
        _write(stdout, f"[eval_v2] output_dir={report.artifacts.output_dir}")
    for summary in report.summaries:
        _write(
            stdout,
            (
                "[eval_v2] summary scenario={scenario} candidate={candidate} "
                "success_rate={success_rate} score_mean={score_mean} trials={trials}"
            ).format(
                scenario=summary.scenario_name,
                candidate=summary.candidate_name,
                success_rate=_format_optional(summary.success_rate),
                score_mean=_format_optional(summary.score_mean),
                trials=summary.trial_count,
            ),
        )
    return 0


def run_cli(
    argv: Optional[Sequence[str]] = None,
    *,
    service: Optional[EvaluationService] = None,
    stdout: Optional[TextIO] = None,
) -> int:
    active_service = service or DefaultEvaluationService()
    active_stdout = stdout or sys.stdout
    args = build_parser().parse_args(argv)

    if args.command == "list-anatomies":
        return _handle_list_anatomies(active_service, active_stdout)
    if args.command == "list-wires":
        return _handle_list_wires(args, active_service, active_stdout)
    if args.command == "list-policies":
        return _handle_list_policies(args, active_service, active_stdout)
    if args.command == "list-candidates":
        return _handle_list_candidates(args, active_service, active_stdout)
    if args.command == "list-branches":
        return _handle_list_branches(args, active_service, active_stdout)
    if args.command == "list-target-modes":
        return _handle_list_target_modes(active_service, active_stdout)
    if args.command == "run":
        return _handle_run(args, active_service, active_stdout)
    raise ValueError(f"Unsupported command: {args.command!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
