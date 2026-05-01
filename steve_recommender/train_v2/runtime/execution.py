"""Execution helpers for train_v2 doctor and training commands."""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch.multiprocessing as mp

from ..agents.factory import build_agent, maybe_resume_agent
from ..config import ARCHVAR_EVAL_SEEDS, DoctorConfig, TrainingRunConfig
from ..doctor.checks import run_doctor
from ..doctor.report import exit_code, render_report
from ..paths import LATEST_REPLAY_BUFFER_NAME, get_result_checkpoint_config_and_log_path
from .env_factory import build_env
from .intervention_factory import build_intervention
from .tracking_runner import TrackingRunner


HEATUP_ACTION_LOW = (-10.0, -1.0)
HEATUP_ACTION_HIGH = (25.0, 3.14)


def compute_resume_targets(
    *,
    requested_heatup_steps: int,
    requested_training_steps: int,
    current_explore_steps: int,
    resume_skip_heatup: bool,
) -> Tuple[int, int]:
    """Resolve runner targets for one resumed training run."""

    training_target = requested_training_steps
    if requested_training_steps <= current_explore_steps:
        training_target = current_explore_steps + max(requested_training_steps, 1)
    heatup_steps = 0 if resume_skip_heatup else requested_heatup_steps
    return heatup_steps, training_target


def training_parameters_for_results(config: TrainingRunConfig) -> Dict[str, object]:
    """Return the result-file parameter payload recorded by the runner."""

    return {
        "tool_ref": config.runtime.tool_ref,
        "anatomy_id": config.runtime.anatomy_id,
        "reward_profile": config.reward.profile,
        "force_alpha": config.reward.force_alpha,
        "force_beta": config.reward.force_beta,
        "force_region": config.reward.force_region,
        "learning_rate": config.learning_rate,
        "hidden_layers": list(config.hidden_layers),
        "embedder_nodes": config.embedder_nodes,
        "embedder_layers": config.embedder_layers,
        "heatup_steps": config.heatup_steps,
        "training_steps": config.training_steps,
        "eval_every": config.eval_every,
        "train_max_steps": config.train_max_steps,
        "eval_max_steps": config.eval_max_steps,
        "write_step_trace_h5": config.write_step_trace_h5,
        "step_trace_every_n_steps": config.step_trace_every_n_steps,
    }


def _build_doctor_config_from_train(config: TrainingRunConfig) -> DoctorConfig:
    return DoctorConfig(
        runtime=config.runtime,
        reward=config.reward,
        trainer_device=config.trainer_device,
        output_root=config.output_root,
        resume_from=config.resume_from,
        resume_replay_buffer_from=config.resume_replay_buffer_from,
        strict=False,
        boot_env=True,
    )


def execute_training_run(config: TrainingRunConfig) -> Path:
    """Build environments, run preflight, and execute one training job."""

    if config.preflight:
        doctor_cfg = _build_doctor_config_from_train(config)
        results = run_doctor(doctor_cfg)
        if exit_code(results, strict=doctor_cfg.strict) != 0:
            raise RuntimeError(f"train_v2 preflight failed\n{render_report(results)}")
        if config.preflight_only:
            return config.output_root

    mp.set_start_method("spawn", force=True)

    results_file, checkpoint_folder, config_folder, log_file = (
        get_result_checkpoint_config_and_log_path(
            all_results_folder=config.output_root,
            name=config.name,
        )
    )
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Match the legacy ArchVar script: build one randomized scene, then reuse a
    # deep-copied twin for evaluation so train/eval start from the same scene state.
    intervention_train = build_intervention(runtime_spec=config.runtime)
    intervention_eval = deepcopy(intervention_train)
    env_train = build_env(
        intervention=intervention_train,
        reward_spec=config.reward,
        mode="train",
        n_max_steps=config.train_max_steps or 1000,
        reward_csv_path=config_folder / "reward_train.csv",
        step_trace_path=(
            config_folder / "step_trace_train.h5"
            if config.write_step_trace_h5
            else None
        ),
        step_trace_every_n_steps=config.step_trace_every_n_steps,
    )
    env_eval = build_env(
        intervention=intervention_eval,
        reward_spec=config.reward,
        mode="eval",
        n_max_steps=config.eval_max_steps or 1000,
        reward_csv_path=config_folder / "reward_eval.csv",
        step_trace_path=(
            config_folder / "step_trace_eval.h5"
            if config.write_step_trace_h5
            else None
        ),
        step_trace_every_n_steps=config.step_trace_every_n_steps,
    )

    env_train_config = config_folder / "env_train.yml"
    env_eval_config = config_folder / "env_eval.yml"
    env_train.save_config(str(env_train_config))
    env_eval.save_config(str(env_eval_config))

    agent = build_agent(config=config, env_train=env_train, env_eval=env_eval)
    if config.save_latest_replay_buffer:
        agent.configure_replay_buffer_checkpointing(
            latest_replay_buffer_path=checkpoint_folder / LATEST_REPLAY_BUFFER_NAME,
            resume_replay_buffer_path=config.resume_replay_buffer_from,
            embed_in_checkpoint=True,
        )
    elif config.resume_replay_buffer_from is not None:
        agent.configure_replay_buffer_checkpointing(
            latest_replay_buffer_path=None,
            resume_replay_buffer_path=config.resume_replay_buffer_from,
            embed_in_checkpoint=True,
        )

    infos = list(env_eval.info.info.keys())
    runner = TrackingRunner(
        agent=agent,
        heatup_action_low=list(HEATUP_ACTION_LOW),
        heatup_action_high=list(HEATUP_ACTION_HIGH),
        agent_parameter_for_result_file=training_parameters_for_results(config),
        checkpoint_folder=str(checkpoint_folder),
        results_file=str(results_file),
        info_results=infos,
        quality_info="success",
    )
    heatup_steps = config.heatup_steps
    training_target = config.training_steps
    if config.resume_from is not None:
        maybe_resume_agent(agent=agent, checkpoint_path=config.resume_from)
        heatup_steps, training_target = compute_resume_targets(
            requested_heatup_steps=config.heatup_steps,
            requested_training_steps=config.training_steps,
            current_explore_steps=int(agent.step_counter.exploration),
            resume_skip_heatup=config.resume_skip_heatup,
        )

    try:
        runner.training_run(
            heatup_steps,
            training_target,
            config.eval_every,
            config.explore_episodes_between_updates,
            config.update_per_explore_step,
            eval_episodes=config.eval_episodes,
            eval_seeds=list(config.eval_seeds or ARCHVAR_EVAL_SEEDS),
        )
    finally:
        agent.close()

    return results_file
