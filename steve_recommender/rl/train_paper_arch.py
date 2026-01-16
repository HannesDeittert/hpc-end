"""Paper-style SAC training (stEVE_rl) from your own codebase.

This mirrors stEVE_training/training_scripts/BasicWireNav_train.py
but loads a device from data/<tool_name>/tool.py.

Example:
  steve-train --tool Device1 -d cuda -nw 8 -n device1_paper
"""

from __future__ import annotations

import argparse
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import torch
import torch.multiprocessing as mp

from steve_recommender.devices import make_device
from steve_recommender.rl.bench_env import BenchEnv
from steve_recommender.rl.paper_agent_factory import (
    PaperAgentConfig,
    make_synchron_agent,
)
from steve_recommender.rl.results_paths import (
    get_result_checkpoint_config_and_log_path,
)
from steve_recommender.rl.runner import Runner
from steve_recommender.steve_adapter import eve
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
    """Minimal wire-nav intervention using a stored guidewire."""
    vessel_tree = eve.intervention.vesseltree.AorticArch(seed=seed)
    device = make_device(tool_name)

    simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.001)
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
    intervention = eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
        stop_device_at_tree_end=True,
        normalize_action=True,
    )
    return intervention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper SAC training from local devices")
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
        "-nw", "--n_worker", type=int, default=2, help="Number of workers"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Trainer device for NN updates.",
        choices=["cpu", "cuda:0", "cuda:1", "cuda", "mps"],
    )
    parser.add_argument(
        "-n", "--name", type=str, default="paper_run", help="Name of the training run"
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
        help="Nodes per embedder layer (LSTM in paper)",
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
        default=str(REPO_ROOT / "results" / "paper_runs"),
        help="Base output folder for results/checkpoints/configs",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a .everl checkpoint to resume training from.",
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
        "--consecutive-action-steps",
        type=int,
        default=CONSECUTIVE_ACTION_STEPS,
        help="How many simulator steps each action is repeated for.",
    )
    parser.add_argument(
        "--log-interval-s",
        type=float,
        default=600.0,
        help="Progress heartbeat interval (seconds).",
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
        "--replay-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "cuda:0", "cuda:1", "mps"],
        help="Device for replay buffer sampling (recommend cpu).",
    )
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=1,
        help="Sets OMP/MKL/OPENBLAS/NUMEXPR threads (recommended 1 with many workers).",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Sets torch intra-op threads (recommended 1 with many workers).",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=1,
        help="Sets torch inter-op threads (recommended 1 with many workers).",
    )
    parser.add_argument(
        "-se",
        "--stochastic_eval",
        action="store_true",
        help="Use stochastic eval variant of SAC.",
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
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    # Threading: with multiple worker processes, keep BLAS/OMP and torch CPU threads low
    # to avoid oversubscription.
    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.omp_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.omp_threads)

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    if str(args.device).startswith("cuda"):
        # Allow TF32 matmuls on Ampere+ for speed (Ada supports it too).
        torch.set_float32_matmul_precision("high")

    tool_ref = args.tool
    if args.model and "/" not in tool_ref:
        tool_ref = f"{args.model}/{tool_ref}"

    resume_path: Optional[Path] = None
    if args.resume_from:
        resume_path = Path(args.resume_from).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume-from not found: {resume_path}")

    trainer_device = torch.device(args.device)
    worker_device = torch.device("cpu")
    print(f"[train] tool={tool_ref} trainer_device={trainer_device} workers={args.n_worker}")
    print(
        "[train] "
        f"heatup_steps={int(args.heatup_steps)} training_steps={int(args.training_steps)} "
        f"eval_every={int(args.eval_every)} eval_episodes={args.eval_episodes} "
        f"batch_size={args.batch_size} replay_buffer_size={int(args.replay_buffer_size)} "
        f"explore_episodes_between_updates={args.explore_episodes_between_updates} "
        f"update_per_explore_step={args.update_per_explore_step} "
        f"consecutive_action_steps={args.consecutive_action_steps} "
        f"replay_device={args.replay_device} "
        f"omp_threads={args.omp_threads} torch_threads={args.torch_threads}/{args.torch_interop_threads}"
    )

    replay_device = torch.device(args.replay_device)
    if replay_device.type == "cuda":
        # Keep replay sampling on CPU unless you know what you're doing.
        print("[train] warning: replay buffer on CUDA can cause CUDA IPC warnings.")

    custom_parameters = {
        "lr": args.learning_rate,
        "hidden_layers": args.hidden,
        "embedder_nodes": args.embedder_nodes,
        "embedder_layers": args.embedder_layers,
        "HEATUP_STEPS": args.heatup_steps,
        "TRAINING_STEPS": args.training_steps,
        "EXPLORE_STEPS_BTW_EVAL": args.eval_every,
        "EVAL_EPISODES": args.eval_episodes,
        "CONSECUTIVE_EXPLORE_EPISODES": args.explore_episodes_between_updates,
        "UPDATE_PER_EXPLORE_STEP": args.update_per_explore_step,
        "CONSECUTIVE_ACTION_STEPS": args.consecutive_action_steps,
        "BATCH_SIZE": args.batch_size,
        "REPLAY_BUFFER_SIZE": args.replay_buffer_size,
        "REPLAY_DEVICE": args.replay_device,
    }

    results_root = Path(args.out)
    results_file, checkpoint_folder, config_folder, log_file = (
        get_result_checkpoint_config_and_log_path(
            all_results_folder=str(results_root), name=args.name
        )
    )
    handlers: List[logging.Handler] = [logging.FileHandler(log_file)]
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
    env_eval = BenchEnv(intervention=intervention_eval, mode="eval", visualisation=False)

    # Heuristic sanity checks: these combinations commonly lead to "no updates yet"
    # because a single explore chunk overshoots the (too small) eval/training step limits.
    max_steps_per_episode = 1000  # BenchEnv default
    expected_steps_per_explore_chunk = (
        args.explore_episodes_between_updates * max_steps_per_episode
    )
    if args.training_steps <= expected_steps_per_explore_chunk:
        print(
            "[train] warning: training_steps is very small compared to one explore chunk "
            f"({int(args.training_steps)} <= {expected_steps_per_explore_chunk}). "
            "This often finishes before any update happens. Increase --training-steps "
            "or reduce --explore-episodes-between-updates.",
            flush=True,
        )
    if args.eval_every <= expected_steps_per_explore_chunk:
        print(
            "[train] warning: eval_every is very small compared to one explore chunk "
            f"({int(args.eval_every)} <= {expected_steps_per_explore_chunk}). "
            "This can cause evals before updates start. Increase --eval-every "
            "or reduce --explore-episodes-between-updates.",
            flush=True,
        )

    agent_cfg = PaperAgentConfig(
        hidden_layers=args.hidden,
        embedder_nodes=args.embedder_nodes,
        embedder_layers=args.embedder_layers,
        ff_only=False,
        lr=args.learning_rate,
        lr_end_factor=LR_END_FACTOR,
        lr_linear_end_steps=int(LR_LINEAR_END_STEPS),
        gamma=GAMMA,
        reward_scaling=REWARD_SCALING,
        batch_size=args.batch_size,
        replay_buffer_size=int(args.replay_buffer_size),
        stochastic_eval=args.stochastic_eval,
    )
    agent = make_synchron_agent(
        agent_cfg,
        env_train=env_train,
        env_eval=env_eval,
        trainer_device=trainer_device,
        worker_device=worker_device,
        replay_device=replay_device,
        consecutive_action_steps=args.consecutive_action_steps,
        n_worker=args.n_worker,
    )

    heatup_steps_to_run = args.heatup_steps
    if resume_path is not None:
        print(f"[train] resuming from checkpoint: {resume_path}", flush=True)
        agent.load_checkpoint(str(resume_path))
        print(
            "[train] loaded checkpoint: "
            f"heatup={int(agent.step_counter.heatup)} "
            f"explore={int(agent.step_counter.exploration)} "
            f"update={int(agent.step_counter.update)} "
            f"eval={int(agent.step_counter.evaluation)}",
            flush=True,
        )
        # The checkpoint already contains heatup steps; do not heatup again.
        heatup_steps_to_run = 0

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

    # Lightweight progress heartbeat.
    import threading
    import time

    stop_evt = threading.Event()

    def heartbeat():
        last_h = 0
        last_e = 0
        last_u = 0
        last_t = time.time()
        while not stop_evt.wait(args.log_interval_s):
            now = time.time()
            dt = max(1e-6, now - last_t)
            h = int(agent.step_counter.heatup)
            e = int(agent.step_counter.exploration)
            u = int(agent.step_counter.update)
            dh = h - last_h
            de = e - last_e
            du = u - last_u
            stage = "heatup" if e == 0 and u == 0 and h > 0 else "train"
            msg = (
                f"heartbeat[{stage}]: heatup={h} (+{dh}/ {dt:.0f}s) "
                f"explore={e} (+{de}/ {dt:.0f}s) update={u} (+{du}/ {dt:.0f}s)"
            )
            logging.getLogger("gw.paper").info(msg)
            if args.stdout:
                print(f"[train] {msg}", flush=True)
            last_h, last_e, last_u, last_t = h, e, u, now

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()

    try:
        runner.training_run(
            heatup_steps_to_run,
            args.training_steps,
            args.eval_every,
            args.explore_episodes_between_updates,
            args.update_per_explore_step,
            eval_episodes=args.eval_episodes,
            eval_seeds=None,
        )
    except KeyboardInterrupt:
        print("[train] interrupted (Ctrl+C) - shutting down cleanly...", flush=True)
        raise
    finally:
        stop_evt.set()
        runner.close()
        try:
            agent.close()
        except Exception as e:  # noqa: BLE001
            print(f"[train] warning: failed to close cleanly: {e}", flush=True)


if __name__ == "__main__":
    main()
