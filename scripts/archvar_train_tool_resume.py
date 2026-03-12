#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import argparse
import logging
import os
import sys
from typing import List

import torch
import torch.multiprocessing as mp

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPTS = (
    REPO_ROOT / "third_party" / "stEVE_training" / "training_scripts"
)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TRAINING_SCRIPTS))

from util.env import BenchEnv  # noqa: E402
from util.util import get_result_checkpoint_config_and_log_path  # noqa: E402
from util.agent import BenchAgentSynchron  # noqa: E402
from eve_rl import Runner  # noqa: E402
from steve_recommender.bench import (  # noqa: E402
    build_archvar_intervention,
    list_tools,
    resolve_device,
)


RESULTS_FOLDER = (
    os.getcwd() + "/results/eve_paper/neurovascular/aorta/gw_only/arch_vmr_94"
)

EVAL_SEEDS = (
    "1,2,3,5,6,7,8,9,10,12,13,14,16,17,18,21,22,23,27,31,34,35,37,39,42,43,"
    "44,47,48,50,52,55,56,58,61,62,63,68,69,70,71,73,79,80,81,84,89,91,92,93,"
    "95,97,102,103,108,109,110,115,116,117,118,120,122,123,124,126,127,128,"
    "129,130,131,132,134,136,138,139,140,141,142,143,144,147,148,149,150,151,"
    "152,154,155,156,158,159,161,162,167,168,171,175"
)
EVAL_SEEDS = [int(seed) for seed in EVAL_SEEDS.split(",")]

HEATUP_STEPS = int(5e5)
TRAINING_STEPS = int(2e7)
CONSECUTIVE_EXPLORE_EPISODES = 100
EXPLORE_STEPS_BTW_EVAL = int(2.5e5)

SHORT_HEATUP_STEPS = 20000
SHORT_TRAINING_STEPS = 20000
SHORT_CONSECUTIVE_EXPLORE_EPISODES = 20
SHORT_EXPLORE_STEPS_BTW_EVAL = 20000

GAMMA = 0.99
REWARD_SCALING = 1
REPLAY_BUFFER_SIZE = 1e4
CONSECUTIVE_ACTION_STEPS = 1
BATCH_SIZE = 32
UPDATE_PER_EXPLORE_STEP = 1 / 20

LR_END_FACTOR = 0.15
LR_LINEAR_END_STEPS = 6e6

DEBUG_LEVEL = logging.INFO


def _select_steps(args) -> List[int]:
    if args.short:
        heatup = args.heatup_steps or SHORT_HEATUP_STEPS
        training = args.training_steps or SHORT_TRAINING_STEPS
        explore_steps = args.explore_steps_between_eval or SHORT_EXPLORE_STEPS_BTW_EVAL
        explore_episodes = (
            args.explore_episodes_between_updates or SHORT_CONSECUTIVE_EXPLORE_EPISODES
        )
    else:
        heatup = args.heatup_steps or HEATUP_STEPS
        training = args.training_steps or TRAINING_STEPS
        explore_steps = args.explore_steps_between_eval or EXPLORE_STEPS_BTW_EVAL
        explore_episodes = (
            args.explore_episodes_between_updates or CONSECUTIVE_EXPLORE_EPISODES
        )

    return [int(heatup), int(training), int(explore_steps), int(explore_episodes)]


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="ArchVar training with selectable tools",
    )
    parser.add_argument("-nw", "--n_worker", type=int, default=2)
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda:0", "cuda:1", "cuda"],
    )
    parser.add_argument("-n", "--name", type=str, default="test")
    parser.add_argument("-se", "--stochastic_eval", action="store_true")
    parser.add_argument("--short", action="store_true", help="Use short step counts")

    parser.add_argument(
        "--tool",
        type=str,
        default="jshaped_default",
        help="Tool key or module:Class (see --list-tools)",
    )
    parser.add_argument("--tool-module", type=str, default=None)
    parser.add_argument("--tool-class", type=str, default=None)
    parser.add_argument("--list-tools", action="store_true")
    parser.add_argument(
        "--debug-device",
        action="store_true",
        help="Log key device/sofa parameters for debugging.",
    )

    parser.add_argument("--heatup-steps", type=int, default=None)
    parser.add_argument("--training-steps", type=int, default=None)
    parser.add_argument("--explore-steps-between-eval", type=int, default=None)
    parser.add_argument("--explore-episodes-between-updates", type=int, default=None)
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional path to a .everl checkpoint to resume from.",
    )
    parser.add_argument(
        "--resume-skip-heatup",
        action="store_true",
        help="If set with --resume-from, skip additional heatup steps.",
    )
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=None,
        help="Override SimulationMP step timeout in seconds.",
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.0003217978434614328,
    )
    parser.add_argument("--hidden", nargs="+", type=int, default=[400, 400, 400])
    parser.add_argument("-en", "--embedder_nodes", type=int, default=700)
    parser.add_argument("-el", "--embedder_layers", type=int, default=1)
    parser.add_argument(
        "--results-folder",
        type=str,
        default=RESULTS_FOLDER,
    )

    args = parser.parse_args()

    if args.list_tools:
        for name in list_tools():
            print(name)
        raise SystemExit(0)

    (
        heatup_steps,
        training_steps,
        explore_steps_between_eval,
        explore_episodes_between_updates,
    ) = _select_steps(args)

    trainer_device = torch.device(args.device)
    worker_device = torch.device("cpu")

    custom_parameters = {
        "lr": args.learning_rate,
        "hidden_layers": args.hidden,
        "embedder_nodes": args.embedder_nodes,
        "embedder_layers": args.embedder_layers,
        "HEATUP_STEPS": heatup_steps,
        "EXPLORE_STEPS_BTW_EVAL": explore_steps_between_eval,
        "CONSECUTIVE_EXPLORE_EPISODES": explore_episodes_between_updates,
        "BATCH_SIZE": BATCH_SIZE,
        "UPDATE_PER_EXPLORE_STEP": UPDATE_PER_EXPLORE_STEP,
        "TOOL": args.tool,
    }

    (
        results_file,
        checkpoint_folder,
        config_folder,
        log_file,
    ) = get_result_checkpoint_config_and_log_path(
        all_results_folder=args.results_folder, name=args.name
    )

    logging.basicConfig(
        filename=log_file,
        level=DEBUG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    device, tool_label = resolve_device(
        args.tool, args.tool_module, args.tool_class
    )
    logging.getLogger(__name__).info("Using tool: %s", tool_label)
    if args.debug_device:
        logger = logging.getLogger(__name__)
        logger.info("Device class: %s", device.__class__.__name__)
        for field in (
            "length",
            "tip_length",
            "tip_radius",
            "tip_angle",
            "tip_outer_diameter",
            "tip_inner_diameter",
            "straight_outer_diameter",
            "straight_inner_diameter",
            "poisson_ratio",
            "young_modulus_tip",
            "young_modulus_straight",
            "mass_density_tip",
            "mass_density_straight",
            "visu_edges_per_mm",
            "collis_edges_per_mm_tip",
            "collis_edges_per_mm_straight",
            "beams_per_mm_tip",
            "beams_per_mm_straight",
        ):
            if hasattr(device, field):
                logger.info("device.%s=%s", field, getattr(device, field))
        sofa_device = getattr(device, "sofa_device", None)
        if sofa_device is not None:
            for field in (
                "length",
                "straight_length",
                "spire_diameter",
                "spire_height",
                "num_edges",
                "num_edges_collis",
                "density_of_beams",
                "key_points",
                "young_modulus",
                "young_modulus_extremity",
                "mass_density",
                "mass_density_extremity",
            ):
                if hasattr(sofa_device, field):
                    logger.info("sofa_device.%s=%s", field, getattr(sofa_device, field))
            print("[device.debug] class:", device.__class__.__name__, flush=True)
            print("[device.debug] sofa_device:", sofa_device, flush=True)

    intervention = build_archvar_intervention(device=device)
    intervention2 = deepcopy(intervention)

    env_train = BenchEnv(intervention=intervention, mode="train", visualisation=False)
    env_eval = BenchEnv(intervention=intervention2, mode="eval", visualisation=False)

    if args.step_timeout is not None:
        for env in (env_train, env_eval):
            sim = getattr(env.intervention, "simulation", None)
            if sim is not None and hasattr(sim, "step_timeout"):
                sim.step_timeout = args.step_timeout
            fluoro = getattr(env.intervention, "fluoroscopy", None)
            if fluoro is not None and hasattr(fluoro, "simulation"):
                f_sim = fluoro.simulation
                if hasattr(f_sim, "step_timeout"):
                    f_sim.step_timeout = args.step_timeout

    agent = BenchAgentSynchron(
        trainer_device,
        worker_device,
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
        args.n_worker,
        args.stochastic_eval,
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

    heatup_steps_to_run = heatup_steps
    training_steps_target = training_steps
    if args.resume_from:
        resume_path = os.path.abspath(args.resume_from)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--resume-from not found: {resume_path}")
        logging.getLogger(__name__).info("Loading checkpoint: %s", resume_path)
        agent.load_checkpoint(resume_path)
        current_explore = int(agent.step_counter.exploration)
        # Runner.training_run expects an absolute exploration-step target.
        # When resuming, treat a smaller/equal value as "additional steps".
        if training_steps <= current_explore:
            additional_steps = max(training_steps, 1)
            training_steps_target = current_explore + additional_steps
        if args.resume_skip_heatup:
            heatup_steps_to_run = 0
        logging.getLogger(__name__).info(
            (
                "Loaded counters heatup=%d explore=%d update=%d eval=%d "
                "| heatup_steps_run=%d training_steps_target=%d"
            ),
            int(agent.step_counter.heatup),
            current_explore,
            int(agent.step_counter.update),
            int(agent.step_counter.evaluation),
            heatup_steps_to_run,
            training_steps_target,
        )

    runner.training_run(
        heatup_steps_to_run,
        training_steps_target,
        explore_steps_between_eval,
        explore_episodes_between_updates,
        UPDATE_PER_EXPLORE_STEP,
        eval_seeds=EVAL_SEEDS,
    )
    agent.close()
