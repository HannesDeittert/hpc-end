"""Paper-style SAC training with BenchAgentSingle (no multiprocessing).

This mirrors third_party/stEVE_training/training_scripts but uses the
single-agent variant, and loads a local device from
dev/gw_tool_recommender/data/<tool_name>/tool.py.

Example:
  python3 dev/gw_tool_recommender/rl/train_paper_arch_single.py \
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


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]

# Make third_party packages importable without installation.
sys.path.insert(0, str(REPO_ROOT / "third_party" / "stEVE_training" / "eve"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "stEVE_training" / "eve_bench"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "stEVE_training" / "training_scripts"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "stEVE_rl"))

import eve  # noqa: E402
from eve_rl import Runner  # noqa: E402
from util.env import BenchEnv  # noqa: E402
from util.util import get_result_checkpoint_config_and_log_path  # noqa: E402
from util.agent import BenchAgentSingle  # noqa: E402

from dev.gw_tool_recommender.devices import make_device  # noqa: E402


# Defaults adapted from third_party BasicWireNav_train.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper SAC training (single agent)")
    parser.add_argument("--tool", required=True, help="device folder under data/<tool>/tool.py")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trainer_device = torch.device(args.device)

    custom_parameters = {
        "lr": args.learning_rate,
        "hidden_layers": args.hidden,
        "embedder_nodes": args.embedder_nodes,
        "embedder_layers": args.embedder_layers,
        "ff_only": args.ff_only,
        "HEATUP_STEPS": HEATUP_STEPS,
        "EXPLORE_STEPS_BTW_EVAL": EXPLORE_STEPS_BTW_EVAL,
        "CONSECUTIVE_EXPLORE_EPISODES": CONSECUTIVE_EXPLORE_EPISODES,
        "BATCH_SIZE": BATCH_SIZE,
        "UPDATE_PER_EXPLORE_STEP": UPDATE_PER_EXPLORE_STEP,
    }

    results_root = Path(args.out)
    results_file, checkpoint_folder, config_folder, log_file = (
        get_result_checkpoint_config_and_log_path(
            all_results_folder=str(results_root), name=args.name
        )
    )
    logging.basicConfig(
        filename=log_file,
        level=DEBUG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    intervention = build_intervention(args.tool)
    intervention_eval = deepcopy(intervention)

    env_train = BenchEnv(intervention=intervention, mode="train", visualisation=False)
    env_eval = BenchEnv(intervention=intervention_eval, mode="eval", visualisation=False)

    agent = BenchAgentSingle(
        trainer_device,
        args.learning_rate,
        LR_END_FACTOR,
        LR_LINEAR_END_STEPS,
        args.hidden,
        args.embedder_nodes,
        args.embedder_layers,
        GAMMA,
        BATCH_SIZE,
        REWARD_SCALING,
        REPLAY_BUFFER_SIZE,
        env_train,
        env_eval,
        CONSECUTIVE_ACTION_STEPS,
        args.stochastic_eval,
        args.ff_only,
    )

    env_train_config = os.path.join(config_folder, "env_train.yml")
    env_train.save_config(env_train_config)
    env_eval_config = os.path.join(config_folder, "env_eval.yml")
    env_eval.save_config(env_eval_config)

    infos = list(env_eval.info.info.keys())
    runner = Runner(
        agent=agent,
        heatup_action_low=[-10.0, -1.0],
        heatup_action_high=[25, 3.14],
        agent_parameter_for_result_file=custom_parameters,
        checkpoint_folder=checkpoint_folder,
        results_file=results_file,
        info_results=infos,
        quality_info="success",
    )
    runner_config = os.path.join(config_folder, "runner.yml")
    runner.save_config(runner_config)

    runner.training_run(
        HEATUP_STEPS,
        TRAINING_STEPS,
        EXPLORE_STEPS_BTW_EVAL,
        CONSECUTIVE_EXPLORE_EPISODES,
        UPDATE_PER_EXPLORE_STEP,
        eval_seeds=None,
    )
    agent.close()


if __name__ == "__main__":
    main()

