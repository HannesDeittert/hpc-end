from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import eve
import eve_rl
import numpy as np
import torch
from torch import optim

from .replaybuffer import ResumableVanillaEpisodeShared

LATEST_REPLAY_BUFFER_NAME = "latest_replay_buffer.everl"


class _CheckpointMixin:
    embed_replay_buffer_state_in_checkpoint: bool = True
    latest_replay_buffer_path: Optional[Path] = None
    resume_replay_buffer_path: Optional[Path] = None

    def configure_replay_buffer_checkpointing(
        self,
        *,
        latest_replay_buffer_path: Optional[Union[str, Path]] = None,
        resume_replay_buffer_path: Optional[Union[str, Path]] = None,
        embed_in_checkpoint: bool = True,
    ) -> None:
        self.embed_replay_buffer_state_in_checkpoint = embed_in_checkpoint
        self.latest_replay_buffer_path = (
            Path(latest_replay_buffer_path)
            if latest_replay_buffer_path is not None
            else None
        )
        self.resume_replay_buffer_path = (
            Path(resume_replay_buffer_path)
            if resume_replay_buffer_path is not None
            else None
        )

    def save_checkpoint(
        self, file_path: str, additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        checkpoint_dict = {
            "algo": self.algo.get_config_dict(),
            "replay_buffer": self.replay_buffer.get_config_dict(),
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

        if self.embed_replay_buffer_state_in_checkpoint:
            checkpoint_dict["replay_buffer_state"] = self.replay_buffer.state_dict()

        if self.latest_replay_buffer_path is not None:
            checkpoint_dict["replay_buffer_sidecar"] = self.latest_replay_buffer_path.name
            self.save_replay_buffer_state(self.latest_replay_buffer_path)

        torch.save(checkpoint_dict, file_path)

    def save_replay_buffer_state(self, file_path: Union[str, Path]) -> None:
        if not hasattr(self.replay_buffer, "state_dict"):
            raise TypeError("Replay buffer does not support state export")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        payload = {
            "replay_buffer_state": self.replay_buffer.state_dict(),
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
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def _load_replay_buffer_state(self, state: Dict[str, Any]) -> None:
        if not hasattr(self.replay_buffer, "load_state_dict"):
            raise TypeError("Replay buffer does not support state restoration")
        self.replay_buffer.load_state_dict(state)

    def _load_replay_buffer_sidecar(self, checkpoint_path: Optional[Path], checkpoint):
        candidates = []
        if self.resume_replay_buffer_path is not None:
            candidates.append(self.resume_replay_buffer_path)

        sidecar = checkpoint.get("replay_buffer_sidecar")
        if sidecar and checkpoint_path is not None:
            sidecar_path = Path(sidecar)
            if not sidecar_path.is_absolute():
                sidecar_path = checkpoint_path.parent / sidecar_path
            candidates.append(sidecar_path)

        if checkpoint_path is not None:
            candidates.append(checkpoint_path.parent / LATEST_REPLAY_BUFFER_NAME)

        for candidate in candidates:
            if candidate.is_file():
                payload = _load_trusted_checkpoint(
                    str(candidate), map_location=torch.device("cpu")
                )
                state = payload.get("replay_buffer_state", payload)
                self._load_replay_buffer_state(state)
                return candidate
        return None

    def _load_checkpoint_state(
        self, checkpoint: Dict[str, Any], checkpoint_path: Optional[Path] = None
    ) -> None:
        self.algo.load_state_dicts_network(checkpoint["network_state_dicts"])
        if "optimizer_state_dicts" in checkpoint:
            self.algo.load_state_dicts_optimizer(checkpoint["optimizer_state_dicts"])
        if "scheduler_state_dicts" in checkpoint:
            self.algo.load_state_dicts_scheduler(checkpoint["scheduler_state_dicts"])
        if "replay_buffer_state" in checkpoint:
            self._load_replay_buffer_state(checkpoint["replay_buffer_state"])
        else:
            self._load_replay_buffer_sidecar(checkpoint_path, checkpoint)

        self.step_counter.heatup = checkpoint["steps"]["heatup"]
        self.step_counter.exploration = checkpoint["steps"]["exploration"]
        self.step_counter.evaluation = checkpoint["steps"]["evaluation"]
        self.step_counter.update = checkpoint["steps"]["update"]

        self.episode_counter.heatup = checkpoint["episodes"]["heatup"]
        self.episode_counter.exploration = checkpoint["episodes"]["exploration"]
        self.episode_counter.evaluation = checkpoint["episodes"]["evaluation"]


def _checkpoint_map_location(device: torch.device) -> torch.device:
    # Keep optimizer tensors on the active training device when possible.
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _load_trusted_checkpoint(file_path: str, map_location: torch.device):
    # These checkpoints are produced by our training jobs and contain numpy
    # scalars in additional_info, so PyTorch's weights_only loader rejects them.
    try:
        return torch.load(file_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(file_path, map_location=map_location)


class ResumableSingle(_CheckpointMixin, eve_rl.agent.Single):
    def load_checkpoint(self, file_path: str) -> None:
        checkpoint = _load_trusted_checkpoint(
            file_path,
            map_location=_checkpoint_map_location(self.device),
        )
        self._load_checkpoint_state(checkpoint, checkpoint_path=Path(file_path))


class ResumableSynchron(_CheckpointMixin, eve_rl.agent.Synchron):
    def load_checkpoint(self, file_path: str) -> None:
        checkpoint = _load_trusted_checkpoint(
            file_path,
            map_location=_checkpoint_map_location(self.trainer_device),
        )
        self._load_checkpoint_state(checkpoint, checkpoint_path=Path(file_path))
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
