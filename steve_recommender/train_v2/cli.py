"""CLI entrypoint for train_v2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONSECUTIVE_ACTION_STEPS,
    DEFAULT_EMBEDDER_LAYERS,
    DEFAULT_EMBEDDER_NODES,
    DEFAULT_EVAL_EVERY,
    DEFAULT_EXPLORE_EPISODES_BETWEEN_UPDATES,
    DEFAULT_GAMMA,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_HEATUP_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LR_END_FACTOR,
    DEFAULT_LR_LINEAR_END_STEPS,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_REWARD_PROFILE,
    DEFAULT_REPLAY_BUFFER_SIZE,
    DEFAULT_REWARD_SCALING,
    DEFAULT_TRAINING_STEPS,
    DEFAULT_UPDATE_PER_EXPLORE_STEP,
    DoctorConfig,
    build_doctor_config,
    build_training_config,
)
from .doctor.checks import run_doctor
from .doctor.report import exit_code, render_report
from .runtime.execution import execute_training_run


def build_parser() -> argparse.ArgumentParser:
    """Build the root train_v2 parser."""

    parser = argparse.ArgumentParser(description="train_v2 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Run a local train_v2 training job.",
    )
    train_parser.add_argument("--name", required=True)
    train_parser.add_argument("--anatomy", default=None)
    train_parser.add_argument("--tool", required=True)
    train_parser.add_argument("--tool-module", default=None)
    train_parser.add_argument("--tool-class", default=None)
    train_parser.add_argument(
        "--reward-profile",
        default=DEFAULT_REWARD_PROFILE,
        choices=(
            "default",
            "default_plus_force_penalty",
            "default_plus_excess_force_penalty",
        ),
    )
    train_parser.add_argument("--force-penalty-factor", type=float, default=0.0)
    train_parser.add_argument("--force-threshold", type=float, default=0.85)
    train_parser.add_argument("--force-divisor", type=float, default=1000.0)
    train_parser.add_argument("--force-tip-only", action="store_true")
    train_parser.add_argument("--trainer-device", default="cpu")
    train_parser.add_argument("--worker-device", default="cpu")
    train_parser.add_argument("--replay-device", default="cpu")
    train_parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    train_parser.add_argument("--worker-count", type=int, default=2)
    train_parser.add_argument("--heatup-steps", type=int, default=DEFAULT_HEATUP_STEPS)
    train_parser.add_argument(
        "--training-steps", type=int, default=DEFAULT_TRAINING_STEPS
    )
    train_parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    train_parser.add_argument("--eval-episodes", type=int, default=1)
    train_parser.add_argument(
        "--explore-episodes-between-updates",
        type=int,
        default=DEFAULT_EXPLORE_EPISODES_BETWEEN_UPDATES,
    )
    train_parser.add_argument(
        "--consecutive-action-steps",
        type=int,
        default=DEFAULT_CONSECUTIVE_ACTION_STEPS,
    )
    train_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    train_parser.add_argument(
        "--replay-buffer-size", type=int, default=DEFAULT_REPLAY_BUFFER_SIZE
    )
    train_parser.add_argument(
        "--update-per-explore-step",
        type=float,
        default=DEFAULT_UPDATE_PER_EXPLORE_STEP,
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    train_parser.add_argument(
        "--hidden",
        nargs="+",
        type=int,
        default=list(DEFAULT_HIDDEN_LAYERS),
    )
    train_parser.add_argument(
        "--embedder-nodes",
        type=int,
        default=DEFAULT_EMBEDDER_NODES,
    )
    train_parser.add_argument(
        "--embedder-layers",
        type=int,
        default=DEFAULT_EMBEDDER_LAYERS,
    )
    train_parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    train_parser.add_argument(
        "--reward-scaling",
        type=float,
        default=DEFAULT_REWARD_SCALING,
    )
    train_parser.add_argument(
        "--lr-end-factor",
        type=float,
        default=DEFAULT_LR_END_FACTOR,
    )
    train_parser.add_argument(
        "--lr-linear-end-steps",
        type=int,
        default=DEFAULT_LR_LINEAR_END_STEPS,
    )
    train_parser.add_argument("--resume-from", type=Path, default=None)
    train_parser.add_argument("--resume-skip-heatup", action="store_true")
    train_parser.add_argument("--save-latest-replay-buffer", action="store_true")
    train_parser.add_argument("--resume-replay-buffer-from", type=Path, default=None)
    train_parser.add_argument("--train-max-steps", type=int, default=None)
    train_parser.add_argument("--eval-max-steps", type=int, default=None)
    train_parser.add_argument("--eval-seeds", type=str, default="none")
    train_parser.add_argument("--stochastic-eval", action="store_true")
    train_parser.add_argument("--no-preflight", action="store_true")
    train_parser.add_argument("--preflight-only", action="store_true")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Validate runtime, files, and environment boot before training.",
    )
    doctor_parser.add_argument("--anatomy", default=None)
    doctor_parser.add_argument("--tool", required=True)
    doctor_parser.add_argument("--tool-module", default=None)
    doctor_parser.add_argument("--tool-class", default=None)
    doctor_parser.add_argument(
        "--reward-profile",
        default=DEFAULT_REWARD_PROFILE,
        choices=(
            "default",
            "default_plus_force_penalty",
            "default_plus_excess_force_penalty",
        ),
    )
    doctor_parser.add_argument("--force-penalty-factor", type=float, default=0.0)
    doctor_parser.add_argument("--force-threshold", type=float, default=0.85)
    doctor_parser.add_argument("--force-divisor", type=float, default=1000.0)
    doctor_parser.add_argument("--force-tip-only", action="store_true")
    doctor_parser.add_argument("--resume-from", type=Path, default=None)
    doctor_parser.add_argument("--resume-replay-buffer-from", type=Path, default=None)
    doctor_parser.add_argument("--trainer-device", default="cpu")
    doctor_parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    doctor_parser.add_argument("--strict", action="store_true")
    doctor_parser.add_argument("--no-boot-env", action="store_true")
    return parser


def build_doctor_config_from_args(args: argparse.Namespace) -> DoctorConfig:
    """Convert parsed doctor args into a validated config."""

    return build_doctor_config(
        anatomy_id=args.anatomy,
        tool_ref=args.tool,
        tool_module=args.tool_module,
        tool_class=args.tool_class,
        reward_profile=args.reward_profile,
        force_penalty_factor=args.force_penalty_factor,
        force_threshold_N=args.force_threshold,
        force_divisor=args.force_divisor,
        force_tip_only=args.force_tip_only,
        resume_from=args.resume_from,
        resume_replay_buffer_from=args.resume_replay_buffer_from,
        trainer_device=args.trainer_device,
        output_root=args.output_root,
        strict=args.strict,
        boot_env=not args.no_boot_env,
    )


def build_training_config_from_args(args: argparse.Namespace):
    """Convert parsed train args into a validated config."""

    return build_training_config(
        name=args.name,
        anatomy_id=args.anatomy,
        tool_ref=args.tool,
        tool_module=args.tool_module,
        tool_class=args.tool_class,
        reward_profile=args.reward_profile,
        force_penalty_factor=args.force_penalty_factor,
        force_threshold_N=args.force_threshold,
        force_divisor=args.force_divisor,
        force_tip_only=args.force_tip_only,
        trainer_device=args.trainer_device,
        worker_device=args.worker_device,
        replay_device=args.replay_device,
        output_root=args.output_root,
        worker_count=args.worker_count,
        heatup_steps=args.heatup_steps,
        training_steps=args.training_steps,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_seeds=(
            None
            if args.eval_seeds == "none"
            else tuple(int(v) for v in args.eval_seeds.split(",") if v.strip())
        ),
        explore_episodes_between_updates=args.explore_episodes_between_updates,
        consecutive_action_steps=args.consecutive_action_steps,
        batch_size=args.batch_size,
        replay_buffer_size=args.replay_buffer_size,
        update_per_explore_step=args.update_per_explore_step,
        learning_rate=args.learning_rate,
        hidden_layers=args.hidden,
        embedder_nodes=args.embedder_nodes,
        embedder_layers=args.embedder_layers,
        gamma=args.gamma,
        reward_scaling=args.reward_scaling,
        lr_end_factor=args.lr_end_factor,
        lr_linear_end_steps=args.lr_linear_end_steps,
        resume_from=args.resume_from,
        resume_skip_heatup=args.resume_skip_heatup,
        save_latest_replay_buffer=args.save_latest_replay_buffer,
        resume_replay_buffer_from=args.resume_replay_buffer_from,
        train_max_steps=args.train_max_steps,
        eval_max_steps=args.eval_max_steps,
        preflight=not args.no_preflight,
        preflight_only=args.preflight_only,
        stochastic_eval=args.stochastic_eval,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one train_v2 CLI command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        cfg = build_training_config_from_args(args)
        results_path = execute_training_run(cfg)
        print(results_path)
        return
    if args.command == "doctor":
        cfg = build_doctor_config_from_args(args)
        results = run_doctor(cfg)
        print(render_report(results))
        raise SystemExit(exit_code(results, strict=cfg.strict))
    raise SystemExit(f"Unsupported command: {args.command}")
