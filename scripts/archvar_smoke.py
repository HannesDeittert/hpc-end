#!/usr/bin/env python3
from copy import deepcopy
import argparse
import logging
import os
import sys
import torch.multiprocessing as mp
import torch

# Make stEVE_training scripts importable (util.*, eve_rl, eve_bench)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAINING_SCRIPTS_DIR = os.path.join(
    REPO_ROOT, "third_party", "stEVE_training", "training_scripts"
)
if TRAINING_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, TRAINING_SCRIPTS_DIR)

from util.env import BenchEnv  # noqa: E402
from util.util import get_result_checkpoint_config_and_log_path  # noqa: E402
from util.agent import BenchAgentSynchron  # noqa: E402
from eve_rl import Runner  # noqa: E402
from eve_bench import ArchVariety  # noqa: E402


# Very short smoke-test settings (fast sanity check)
RESULTS_FOLDER = os.path.join(os.getcwd(), "results", "smoke")
EVAL_SEEDS = list(range(5))
HEATUP_STEPS = 1e3
TRAINING_STEPS = 1e4
CONSECUTIVE_EXPLORE_EPISODES = 5
EXPLORE_STEPS_BTW_EVAL = 1e3

GAMMA = 0.99
REWARD_SCALING = 1
REPLAY_BUFFER_SIZE = 1e4
CONSECUTIVE_ACTION_STEPS = 1
BATCH_SIZE = 32
UPDATE_PER_EXPLORE_STEP = 1 / 20

LR_END_FACTOR = 0.15
LR_LINEAR_END_STEPS = 6e6

DEBUG_LEVEL = logging.INFO


def main() -> None:
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="ArchVariety smoke training")
    parser.add_argument(
        "-nw", "--n_worker", type=int, default=2, help="Number of workers"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Trainer device for NN updates",
        choices=["cpu", "cuda:0", "cuda:1", "cuda"],
    )
    parser.add_argument(
        "-n", "--name", type=str, default="archvar_smoke", help="Run name"
    )
    args = parser.parse_args()

    trainer_device = torch.device(args.device)
    n_worker = args.n_worker
    trial_name = args.name
    worker_device = torch.device("cpu")

    custom_parameters = {
        "HEATUP_STEPS": HEATUP_STEPS,
        "EXPLORE_STEPS_BTW_EVAL": EXPLORE_STEPS_BTW_EVAL,
        "CONSECUTIVE_EXPLORE_EPISODES": CONSECUTIVE_EXPLORE_EPISODES,
        "BATCH_SIZE": BATCH_SIZE,
        "UPDATE_PER_EXPLORE_STEP": UPDATE_PER_EXPLORE_STEP,
    }

    (
        results_file,
        checkpoint_folder,
        config_folder,
        log_file,
    ) = get_result_checkpoint_config_and_log_path(
        all_results_folder=RESULTS_FOLDER, name=trial_name
    )
    logging.basicConfig(
        filename=log_file,
        level=DEBUG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    intervention = ArchVariety()
    intervention2 = deepcopy(intervention)

    env_train = BenchEnv(intervention=intervention, mode="train", visualisation=False)
    env_eval = BenchEnv(intervention=intervention2, mode="eval", visualisation=False)

    agent = BenchAgentSynchron(
        trainer_device,
        worker_device,
        0.0003217978434614328,
        LR_END_FACTOR,
        LR_LINEAR_END_STEPS,
        [400, 400, 400],
        700,
        1,
        GAMMA,
        BATCH_SIZE,
        REWARD_SCALING,
        REPLAY_BUFFER_SIZE,
        env_train,
        env_eval,
        CONSECUTIVE_ACTION_STEPS,
        n_worker,
        False,
        False,
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
        eval_seeds=EVAL_SEEDS,
    )
    agent.close()


if __name__ == "__main__":
    main()
