from __future__ import annotations

from typing import Any, Dict, Optional

import eve
import eve_rl
import numpy as np
import torch
from torch import optim

from .replaybuffer import ResumableVanillaEpisodeShared


class _CheckpointMixin:
    def save_checkpoint(
        self, file_path: str, additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        checkpoint_dict = {
            "algo": self.algo.get_config_dict(),
            "replay_buffer": self.replay_buffer.get_config_dict(),
            "replay_buffer_state": self.replay_buffer.state_dict(),
            "env_train": (
                self.env_train.get_config_dict()
                if hasattr(self.env_train, "get_config_dict")
                else None
            ),
            "env_eval": (
                self.env_eval.get_config_dict()
                if hasattr(self.env_eval, "get_config_dict")
                else None
            ),
            "steps": {
                "heatup": self.step_counter.heatup,
                "exploration": self.step_counter.exploration,
                "update": self.step_counter.update,
                "evaluation": self.step_counter.evaluation,
            },
            "episodes": {
                "heatup": self.episode_counter.heatup,
                "exploration": self.episode_counter.exploration,
                "evaluation": self.episode_counter.evaluation,
            },
            "network_state_dicts": self.algo.state_dicts_network(),
            "optimizer_state_dicts": self.algo.state_dicts_optimizer(),
            "scheduler_state_dicts": self.algo.state_dicts_scheduler(),
            "additional_info": additional_info,
        }
        torch.save(checkpoint_dict, file_path)

    def _load_checkpoint_state(self, checkpoint: Dict[str, Any]) -> None:
        self.algo.load_state_dicts_network(checkpoint["network_state_dicts"])
        if "optimizer_state_dicts" in checkpoint:
            self.algo.load_state_dicts_optimizer(checkpoint["optimizer_state_dicts"])
        if "scheduler_state_dicts" in checkpoint:
            self.algo.load_state_dicts_scheduler(checkpoint["scheduler_state_dicts"])
        if "replay_buffer_state" in checkpoint:
            if not hasattr(self.replay_buffer, "load_state_dict"):
                raise TypeError("Replay buffer does not support state restoration")
            self.replay_buffer.load_state_dict(checkpoint["replay_buffer_state"])

        self.step_counter.heatup = checkpoint["steps"]["heatup"]
        self.step_counter.exploration = checkpoint["steps"]["exploration"]
        self.step_counter.evaluation = checkpoint["steps"]["evaluation"]
        self.step_counter.update = checkpoint["steps"]["update"]

        self.episode_counter.heatup = checkpoint["episodes"]["heatup"]
        self.episode_counter.exploration = checkpoint["episodes"]["exploration"]
        self.episode_counter.evaluation = checkpoint["episodes"]["evaluation"]


class ResumableSingle(_CheckpointMixin, eve_rl.agent.Single):
    def load_checkpoint(self, file_path: str) -> None:
        checkpoint = torch.load(file_path, map_location="cpu")
        self._load_checkpoint_state(checkpoint)


class ResumableSynchron(_CheckpointMixin, eve_rl.agent.Synchron):
    def load_checkpoint(self, file_path: str) -> None:
        checkpoint = torch.load(file_path, map_location="cpu")
        self._load_checkpoint_state(checkpoint)
        self._worker_load_state_dicts_network(self.algo.state_dicts_network())
        self.trainer.load_state_dicts_network(self.algo.state_dicts_network())
        self.trainer.load_state_dicts_optimizer(self.algo.state_dicts_optimizer())
        self.trainer.load_state_dicts_scheduler(self.algo.state_dicts_scheduler())


def _build_algo(
    lr: float,
    lr_end_factor: float,
    lr_linear_end_steps: float,
    hidden_layers,
    embedder_nodes: int,
    embedder_layers: int,
    gamma: float,
    reward_scaling: float,
    stochastic_eval: bool,
    ff_only: bool,
    env_train: eve.Env,
) -> eve_rl.algo.SAC:
    obs_dict = env_train.observation_space.sample()
    obs_list = [obs.flatten() for obs in obs_dict.values()]
    obs_np = np.concatenate(obs_list)

    n_observations = obs_np.shape[0]
    n_actions = env_train.action_space.sample().flatten().shape[0]
    if embedder_layers and embedder_nodes and not ff_only:
        q1_embedder = eve_rl.network.component.LSTM(
            n_layer=embedder_layers, n_nodes=embedder_nodes
        )
    elif embedder_layers and embedder_nodes and ff_only:
        hidden_layers = [embedder_nodes] * embedder_layers
        q1_embedder = eve_rl.network.component.MLP(hidden_layers=hidden_layers)
    else:
        q1_embedder = eve_rl.network.component.ComponentDummy()

    q1_base = eve_rl.network.component.MLP(hidden_layers)
    q2_base = eve_rl.network.component.MLP(hidden_layers)
    policy_base = eve_rl.network.component.MLP(hidden_layers)

    q1 = eve_rl.network.QNetwork(q1_base, n_observations, n_actions, q1_embedder)
    q1_optim = eve_rl.optim.Adam(
        q1,
        lr=lr,
    )
    q1_scheduler = optim.lr_scheduler.LinearLR(
        q1_optim,
        start_factor=1.0,
        end_factor=lr_end_factor,
        total_iters=lr_linear_end_steps,
    )

    q2 = eve_rl.network.QNetwork(q2_base, n_observations, n_actions, q1_embedder)
    q2_optim = eve_rl.optim.Adam(
        q2_base,
        lr=lr,
    )
    q2_scheduler = optim.lr_scheduler.LinearLR(
        q2_optim,
        start_factor=1.0,
        end_factor=lr_end_factor,
        total_iters=lr_linear_end_steps,
    )

    policy = eve_rl.network.GaussianPolicy(
        policy_base, n_observations, n_actions, q1_embedder
    )
    policy_optim = eve_rl.optim.Adam(
        policy_base,
        lr=lr,
    )
    policy_scheduler = optim.lr_scheduler.LinearLR(
        policy_optim,
        start_factor=1.0,
        end_factor=lr_end_factor,
        total_iters=lr_linear_end_steps,
    )

    sac_model = eve_rl.model.SACModel(
        lr_alpha=lr,
        q1=q1,
        q2=q2,
        policy=policy,
        q1_optimizer=q1_optim,
        q2_optimizer=q2_optim,
        policy_optimizer=policy_optim,
        q1_scheduler=q1_scheduler,
        q2_scheduler=q2_scheduler,
        policy_scheduler=policy_scheduler,
    )

    return eve_rl.algo.SAC(
        sac_model,
        n_actions=n_actions,
        gamma=gamma,
        reward_scaling=reward_scaling,
        stochastic_eval=stochastic_eval,
    )


class BenchAgentSingle(ResumableSingle):
    def __init__(
        self,
        device,
        lr,
        lr_end_factor,
        lr_linear_end_steps,
        hidden_layers,
        embedder_nodes,
        embedder_layers,
        gamma,
        batch_size,
        reward_scaling,
        replay_buffer_size,
        env_train: eve.Env,
        env_eval: eve.Env,
        consecutive_action_steps,
        stochastic_eval: bool = False,
        ff_only: bool = False,
    ):
        algo = _build_algo(
            lr,
            lr_end_factor,
            lr_linear_end_steps,
            hidden_layers,
            embedder_nodes,
            embedder_layers,
            gamma,
            reward_scaling,
            stochastic_eval,
            ff_only,
            env_train,
        )
        replay_buffer = ResumableVanillaEpisodeShared(
            replay_buffer_size, batch_size, device
        )

        super().__init__(
            algo,
            env_train,
            env_eval,
            replay_buffer,
            consecutive_action_steps=consecutive_action_steps,
            device=device,
            normalize_actions=True,
        )


class BenchAgentSynchron(ResumableSynchron):
    def __init__(
        self,
        trainer_device,
        worker_device,
        lr,
        lr_end_factor,
        lr_linear_end_steps,
        hidden_layers,
        embedder_nodes,
        embedder_layers,
        gamma,
        batch_size,
        reward_scaling,
        replay_buffer_size,
        env_train: eve.Env,
        env_eval: eve.Env,
        consecutive_action_steps,
        n_worker,
        stochastic_eval: bool = False,
        ff_only: bool = False,
    ):
        algo = _build_algo(
            lr,
            lr_end_factor,
            lr_linear_end_steps,
            hidden_layers,
            embedder_nodes,
            embedder_layers,
            gamma,
            reward_scaling,
            stochastic_eval,
            ff_only,
            env_train,
        )
        replay_buffer = ResumableVanillaEpisodeShared(
            replay_buffer_size, batch_size, trainer_device
        )

        super().__init__(
            algo,
            env_train,
            env_eval,
            replay_buffer,
            consecutive_action_steps=consecutive_action_steps,
            trainer_device=trainer_device,
            worker_device=worker_device,
            n_worker=n_worker,
            normalize_actions=True,
            timeout_worker_after_reaching_limit=180,
        )
