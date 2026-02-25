"""Paper-style SAC training with BenchAgentSingle (no multiprocessing).

This mirrors stEVE_training/training_scripts but uses the
single-agent variant, and loads a local device from
data/<tool_name>/tool.py.

Example:
  python -m steve_recommender.rl.train_paper_arch_single \
    --tool Device1 -d cuda -n device1_single
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import List

import torch
import torch.multiprocessing as mp

from steve_recommender.devices import make_device
from steve_recommender.rl.bench_env import BenchEnv
from steve_recommender.rl.paper_agent_factory import (
    PaperAgentConfig,
    make_single_agent,
)
from steve_recommender.rl.results_paths import (
    get_result_checkpoint_config_and_log_path,
)
from steve_recommender.rl.runner import Runner
from steve_recommender.adapters import eve
from steve_recommender.storage import repo_root


REPO_ROOT = repo_root()


# Defaults adapted from stEVE_training BasicWireNav_train.
HEATUP_STEPS = 5e5
TRAINING_STEPS = 2e7
CONSECUTIVE_EXPLORE_EPISODES = 100
EXPLORE_STEPS_BTW_EVAL = 2.5e5

GAMMA = 0.99
REWARD_SCALING = 1
REPLAY_BUFFER_SIZE = 1e4
CONSECUTIVE_ACTION_STEPS = 1
BATCH_SIZE = 32
UPDATE_PER_EXPLORE_STEP = 1 / 20

LR_END_FACTOR = 0.15
LR_LINEAR_END_STEPS = 6e6

DEBUG_LEVEL = logging.INFO


def build_intervention(tool_name: str, seed: int = 30) -> eve.intervention.MonoPlaneStatic:
    vessel_tree = eve.intervention.vesseltree.AorticArch(seed=seed)
    device = make_device(tool_name)

    simulation = eve.intervention.simulation.sofabeamadapter.SofaBeamAdapter(
        friction=0.001
    )
    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=7.5,
        image_rot_zx=[20, 5],
    )
    target = eve.intervention.target.CenterlineRandom(
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
        threshold=5,
        branches=["lcca", "rcca", "lsa", "rsa", "bct", "co"],
    )
    return eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
        stop_device_at_tree_end=True,
        normalize_action=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper SAC training (single agent)")
    parser.add_argument(
        "--tool",
        required=True,
        help="Wire name, or 'model/wire' for model-scoped wires.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name (prepended if --tool has no '/').",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device for NN updates.",
        choices=["cpu", "cuda:0", "cuda:1", "cuda", "mps"],
    )
    parser.add_argument(
        "-n", "--name", type=str, default="paper_single", help="Name of the training run"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=3.2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden",
        nargs="+",
        type=int,
        default=[900, 900, 900, 900],
        help="Hidden layers for policy/Q networks",
    )
    parser.add_argument(
        "-en",
        "--embedder_nodes",
        type=int,
        default=500,
        help="Nodes per embedder layer",
    )
    parser.add_argument(
        "-el",
        "--embedder_layers",
        type=int,
        default=1,
        help="Number of embedder layers",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "results" / "paper_runs_single"),
        help="Base output folder for results/checkpoints/configs",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also log to stdout (console).",
    )
    parser.add_argument(
        "--heatup-steps",
        type=float,
        default=HEATUP_STEPS,
        help="Heatup steps (debug: lower this to see progress sooner).",
    )
    parser.add_argument(
        "--training-steps",
        type=float,
        default=TRAINING_STEPS,
        help="Training steps (debug: lower this to run a short test).",
    )
    parser.add_argument(
        "--eval-every",
        type=float,
        default=EXPLORE_STEPS_BTW_EVAL,
        help="Exploration steps between evals (debug: lower this for more frequent logs/checkpoints).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes per eval call.",
    )
    parser.add_argument(
        "--replay-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "cuda:0", "cuda:1", "mps"],
        help="Device for replay buffer sampling (recommend cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Replay buffer batch size (lower -> updates start earlier).",
    )
    parser.add_argument(
        "--replay-buffer-size",
        type=float,
        default=REPLAY_BUFFER_SIZE,
        help="Replay buffer capacity.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1000,
        help="Max steps per episode (lower -> faster feedback; matches BenchEnv n_max_steps).",
    )
    parser.add_argument(
        "--progress-every-s",
        type=float,
        default=600.0,
        help="Console heartbeat interval in seconds (0 disables).",
    )
    parser.add_argument(
        "--progress-chunk-steps",
        type=int,
        default=200,
        help="Train/explore in small chunks to emit progress (larger -> less overhead).",
    )
    parser.add_argument(
        "--explore-episodes-between-updates",
        type=int,
        default=CONSECUTIVE_EXPLORE_EPISODES,
        help="Episodes per explore/update cycle (lower -> faster feedback, more overhead).",
    )
    parser.add_argument(
        "--update-per-explore-step",
        type=float,
        default=UPDATE_PER_EXPLORE_STEP,
        help="Update steps per exploration step (paper default is 1/20 = 0.05).",
    )
    parser.add_argument(
        "-se",
        "--stochastic_eval",
        action="store_true",
        help="Use stochastic eval variant of SAC.",
    )
    parser.add_argument(
        "--ff-only",
        action="store_true",
        help="Use feed-forward embedder instead of LSTM.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Write TensorBoard scalars (loss/reward/quality) into <run>/tb.",
    )
    parser.add_argument(
        "--tb-logdir",
        type=str,
        default=None,
        help="Optional TensorBoard log dir (default: <run>/tb).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tool_ref = args.tool
    if args.model and "/" not in tool_ref:
        tool_ref = f"{args.model}/{tool_ref}"

    trainer_device = torch.device(args.device)
    if trainer_device.type == "cuda":
        mp.set_start_method("spawn", force=True)
    print(f"[train] tool={tool_ref} device={trainer_device}")
    try:
        print(
            f"[train] torch={torch.__version__} cuda_available={torch.cuda.is_available()}"
        )
        if trainer_device.type == "cuda" and torch.cuda.is_available():
            idx = trainer_device.index if trainer_device.index is not None else 0
            print(f"[train] cuda_device={idx} name={torch.cuda.get_device_name(idx)}")
    except Exception:
        pass

    custom_parameters = {
        "lr": args.learning_rate,
        "hidden_layers": args.hidden,
        "embedder_nodes": args.embedder_nodes,
        "embedder_layers": args.embedder_layers,
        "ff_only": args.ff_only,
        "HEATUP_STEPS": HEATUP_STEPS,
        "EXPLORE_STEPS_BTW_EVAL": EXPLORE_STEPS_BTW_EVAL,
        "CONSECUTIVE_EXPLORE_EPISODES": CONSECUTIVE_EXPLORE_EPISODES,
        "BATCH_SIZE": args.batch_size,
        "UPDATE_PER_EXPLORE_STEP": args.update_per_explore_step,
    }

    results_root = Path(args.out)
    results_file, checkpoint_folder, config_folder, log_file = (
        get_result_checkpoint_config_and_log_path(
            all_results_folder=str(results_root), name=args.name
        )
    )
    handlers: List[logging.Handler] = [logging.FileHandler(log_file)]
    if args.stdout:
        handlers.append(logging.StreamHandler(stream=sys.stdout))
    logging.basicConfig(
        level=DEBUG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    print(f"[train] logs={log_file} results={results_file} checkpoints={checkpoint_folder}")

    intervention = build_intervention(tool_ref)
    intervention_eval = deepcopy(intervention)

    env_train = BenchEnv(intervention=intervention, mode="train", visualisation=False)
    env_train.truncation = eve.truncation.Combination(
        [
            eve.truncation.MaxSteps(args.max_episode_steps),
            eve.truncation.VesselEnd(intervention),
            eve.truncation.SimError(intervention),
        ]
    )

    env_eval = BenchEnv(intervention=intervention_eval, mode="eval", visualisation=False)
    env_eval.truncation = eve.truncation.MaxSteps(args.max_episode_steps)

    agent_cfg = PaperAgentConfig(
        hidden_layers=args.hidden,
        embedder_nodes=args.embedder_nodes,
        embedder_layers=args.embedder_layers,
        ff_only=args.ff_only,
        lr=args.learning_rate,
        lr_end_factor=LR_END_FACTOR,
        lr_linear_end_steps=int(LR_LINEAR_END_STEPS),
        gamma=GAMMA,
        reward_scaling=REWARD_SCALING,
        batch_size=args.batch_size,
        replay_buffer_size=int(args.replay_buffer_size),
        stochastic_eval=args.stochastic_eval,
    )
    replay_device = torch.device(args.replay_device)
    agent = make_single_agent(
        agent_cfg,
        env_train=env_train,
        env_eval=env_eval,
        trainer_device=trainer_device,
        replay_device=replay_device,
        consecutive_action_steps=CONSECUTIVE_ACTION_STEPS,
    )

    env_train_config = os.path.join(config_folder, "env_train.yml")
    env_train.save_config(env_train_config)
    env_eval_config = os.path.join(config_folder, "env_eval.yml")
    env_eval.save_config(env_eval_config)

    infos = list(env_eval.info.info.keys())
    tb_logdir = None
    if args.tensorboard:
        tb_logdir = args.tb_logdir or os.path.join(config_folder, "tb")
    runner = Runner(
        agent=agent,
        heatup_action_low=[-10.0, -1.0],
        heatup_action_high=[25, 3.14],
        agent_parameter_for_result_file=custom_parameters,
        checkpoint_folder=checkpoint_folder,
        results_file=results_file,
        info_results=infos,
        quality_info="success",
        tensorboard_logdir=tb_logdir,
    )
    runner_config = os.path.join(config_folder, "runner.yml")
    runner.save_config(runner_config)

    def heartbeat(prefix: str) -> None:
        if args.progress_every_s <= 0:
            return
        h = agent.step_counter.heatup
        e = agent.step_counter.exploration
        u = agent.step_counter.update
        try:
            rb_len = len(agent.replay_buffer)
            rb_bs = agent.replay_buffer.batch_size
            rb_txt = f" replay={rb_len}/{rb_bs}"
        except Exception:
            rb_txt = ""
        print(
            f"[{prefix}] heatup={h:.0f} explore={e:.0f} update={u:.0f}{rb_txt}",
            flush=True,
        )

    try:
        import time

        # Heatup in small chunks so we can show progress.
        target_heatup = int(args.heatup_steps)
        last_print = 0.0
        while agent.step_counter.heatup < target_heatup:
            remaining = target_heatup - int(agent.step_counter.heatup)
            runner.heatup(steps=min(args.progress_chunk_steps, remaining))
            now = time.time()
            if args.progress_every_s > 0 and now - last_print >= args.progress_every_s:
                heartbeat("heatup")
                last_print = now

        # Training with periodic evals and heartbeats.
        next_eval_step_limit = int(agent.step_counter.exploration + args.eval_every)
        training_steps = int(args.training_steps)
        last_print = time.time()
        while agent.step_counter.exploration < training_steps:
            next_chunk_limit = int(
                min(
                    agent.step_counter.exploration + args.progress_chunk_steps,
                    next_eval_step_limit,
                    training_steps,
                )
            )
            # Explore up to the chunk limit, then run the required number of updates.
            # This makes progress (and GPU usage) much more visible than the internal
            # explore_and_update helper when running with very small debug limits.
            before = int(agent.step_counter.exploration)
            while int(agent.step_counter.exploration) < next_chunk_limit:
                runner.explore(n_episodes=args.explore_episodes_between_updates)
                if int(agent.step_counter.exploration) == before:
                    break
                before = int(agent.step_counter.exploration)
            desired_updates = int(
                agent.step_counter.exploration * args.update_per_explore_step
            )
            missing_updates = desired_updates - int(agent.step_counter.update)
            # Updates only start once the replay buffer has enough data.
            try:
                replay_ready = len(agent.replay_buffer) > agent.replay_buffer.batch_size
            except Exception:
                replay_ready = True

            if missing_updates > 0 and replay_ready:
                runner.update(missing_updates)

            now = time.time()
            if args.progress_every_s > 0 and now - last_print >= args.progress_every_s:
                heartbeat("train")
                last_print = now

            if agent.step_counter.exploration >= next_eval_step_limit:
                runner.eval(episodes=args.eval_episodes, seeds=None)
                next_eval_step_limit += int(args.eval_every)
    except KeyboardInterrupt:
        print("[train] interrupted (Ctrl+C).", flush=True)
    finally:
        runner.close()
        try:
            agent.close()
        except Exception as e:
            print(f"[train] warning: failed to close cleanly: {e}", flush=True)


if __name__ == "__main__":
    main()
