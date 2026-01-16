"""Fast smoke test for the training stack (SOFA + env + agent + runner).

Goal: verify that we can
1) build the intervention and step the simulation
2) fill the replay buffer
3) run at least one update (NN training) on the selected device
4) run at least one evaluation and write CSV + checkpoint

Example:
  CUDA_VISIBLE_DEVICES=0 python -m steve_recommender.rl.smoke_train \
    --tool TestModel_StandardJ035/StandardJ035_PTFE --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.multiprocessing as mp

from steve_recommender.rl.bench_env import BenchEnv
from steve_recommender.rl.paper_agent_factory import (
    PaperAgentConfig,
    make_single_agent,
)
from steve_recommender.rl.results_paths import (
    get_result_checkpoint_config_and_log_path,
)
from steve_recommender.rl.runner import Runner
from steve_recommender.steve_adapter import eve
from steve_recommender.storage import repo_root


REPO_ROOT = repo_root()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test training run")
    p.add_argument("--tool", required=True, help="Wire name or 'model/wire'")
    p.add_argument(
        "-d",
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "cuda:0", "cuda:1", "mps"],
        help="Trainer device for NN updates.",
    )
    p.add_argument(
        "--out",
        default=str(REPO_ROOT / "results" / "smoke_runs"),
        help="Base output folder for results/checkpoints/configs",
    )
    return p.parse_args()


def build_intervention(tool_ref: str, seed: int = 30) -> eve.intervention.MonoPlaneStatic:
    from steve_recommender.devices import make_device

    vessel_tree = eve.intervention.vesseltree.AorticArch(seed=seed)
    device = make_device(tool_ref)

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
    return eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
        stop_device_at_tree_end=True,
        normalize_action=True,
    )


def main() -> None:
    args = parse_args()
    trainer_device = torch.device(args.device)

    # VanillaEpisodeShared uses a subprocess; CUDA requires spawn (not fork).
    if trainer_device.type == "cuda":
        mp.set_start_method("spawn", force=True)

    print(f"[smoke] tool={args.tool} device={trainer_device}")
    print(f"[smoke] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if trainer_device.type == "cuda" and torch.cuda.is_available():
        idx = trainer_device.index if trainer_device.index is not None else 0
        print(f"[smoke] cuda_device={idx} name={torch.cuda.get_device_name(idx)}")

    results_root = Path(args.out)
    results_file, checkpoint_folder, config_folder, log_file = (
        get_result_checkpoint_config_and_log_path(
            all_results_folder=str(results_root), name="smoke"
        )
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(stream=sys.stdout)],
        force=True,
    )
    print(f"[smoke] logs={log_file} results={results_file} checkpoints={checkpoint_folder}")

    intervention = build_intervention(args.tool)
    intervention_eval = deepcopy(intervention)

    env_train = BenchEnv(intervention=intervention, mode="train", visualisation=False)
    env_eval = BenchEnv(intervention=intervention_eval, mode="eval", visualisation=False)

    # Make it fast: shorter episodes.
    max_episode_steps = 100
    env_train.truncation = eve.truncation.Combination(
        [
            eve.truncation.MaxSteps(max_episode_steps),
            eve.truncation.VesselEnd(intervention),
            eve.truncation.SimError(intervention),
        ]
    )
    env_eval.truncation = eve.truncation.MaxSteps(max_episode_steps)

    # Small batch size so updates start early.
    batch_size = 2
    replay_buffer_size = 200

    agent_cfg = PaperAgentConfig(
        hidden_layers=[128, 128],
        embedder_nodes=0,
        embedder_layers=0,
        ff_only=False,
        lr=3.2e-4,
        lr_end_factor=0.15,
        lr_linear_end_steps=10_000,
        gamma=0.99,
        reward_scaling=1.0,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        stochastic_eval=False,
    )
    agent = make_single_agent(
        agent_cfg,
        env_train=env_train,
        env_eval=env_eval,
        trainer_device=trainer_device,
        replay_device=torch.device("cpu"),
        consecutive_action_steps=1,
    )

    env_train.save_config(os.path.join(config_folder, "env_train.yml"))
    env_eval.save_config(os.path.join(config_folder, "env_eval.yml"))

    runner = Runner(
        agent=agent,
        heatup_action_low=[-10.0, -1.0],
        heatup_action_high=[25, 3.14],
        agent_parameter_for_result_file={
            "batch_size": batch_size,
            "replay_buffer_size": replay_buffer_size,
            "max_episode_steps": max_episode_steps,
        },
        checkpoint_folder=checkpoint_folder,
        results_file=results_file,
        info_results=list(env_eval.info.info.keys()),
        quality_info="success",
    )
    runner.save_config(os.path.join(config_folder, "runner.yml"))

    try:
        runner.heatup(steps=400)
        print(
            f"[smoke] after heatup: heatup={agent.step_counter.heatup} replay_len={len(agent.replay_buffer)}"
        )

        # Explore a bit and force at least one update.
        runner.explore(n_episodes=2)
        runner.update(10)
        print(
            f"[smoke] after update: explore={agent.step_counter.exploration} update={agent.step_counter.update}"
        )

        # One eval => writes CSV + checkpoint.
        runner.eval(episodes=1)
        print("[smoke] eval done; check CSV + checkpoints for output.")
    finally:
        try:
            agent.close()
        except Exception as e:
            print(f"[smoke] warning: failed to close cleanly: {e}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
