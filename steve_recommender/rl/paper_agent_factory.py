from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from steve_recommender.adapters import eve_rl

@dataclass(frozen=True)
class PaperAgentConfig:
    hidden_layers: List[int]
    embedder_nodes: int
    embedder_layers: int
    ff_only: bool
    lr: float
    lr_end_factor: float
    lr_linear_end_steps: int
    gamma: float
    reward_scaling: float
    batch_size: int
    replay_buffer_size: int
    stochastic_eval: bool


def _validate_hidden_layers(hidden_layers: List[int]) -> None:
    if len(hidden_layers) < 2:
        raise ValueError(
            f"hidden_layers must have >=2 entries for stEVE_rl MLP, got {hidden_layers}"
        )


def _make_embedder(cfg: PaperAgentConfig):
    if cfg.embedder_layers and cfg.embedder_nodes and not cfg.ff_only:
        return eve_rl.network.component.LSTM(
            n_layer=cfg.embedder_layers, n_nodes=cfg.embedder_nodes
        )
    if cfg.embedder_layers and cfg.embedder_nodes and cfg.ff_only:
        # stEVE_rl MLP requires >=2 entries; duplicate if needed.
        layers = [cfg.embedder_nodes] * max(2, cfg.embedder_layers)
        return eve_rl.network.component.MLP(hidden_layers=layers)
    return eve_rl.network.component.ComponentDummy()


def make_sac_algo(
    cfg: PaperAgentConfig, n_observations: int, n_actions: int
):
    from torch import optim

    _validate_hidden_layers(cfg.hidden_layers)

    q_embedder = _make_embedder(cfg)

    q1_base = eve_rl.network.component.MLP(cfg.hidden_layers)
    q2_base = eve_rl.network.component.MLP(cfg.hidden_layers)
    policy_base = eve_rl.network.component.MLP(cfg.hidden_layers)

    q1 = eve_rl.network.QNetwork(q1_base, n_observations, n_actions, q_embedder)
    q1_opt = eve_rl.optim.Adam(q1, lr=cfg.lr)
    q1_sched = optim.lr_scheduler.LinearLR(
        q1_opt,
        start_factor=1.0,
        end_factor=cfg.lr_end_factor,
        total_iters=cfg.lr_linear_end_steps,
    )

    q2 = eve_rl.network.QNetwork(q2_base, n_observations, n_actions, q_embedder)
    q2_opt = eve_rl.optim.Adam(q2_base, lr=cfg.lr)
    q2_sched = optim.lr_scheduler.LinearLR(
        q2_opt,
        start_factor=1.0,
        end_factor=cfg.lr_end_factor,
        total_iters=cfg.lr_linear_end_steps,
    )

    policy = eve_rl.network.GaussianPolicy(
        policy_base, n_observations, n_actions, q_embedder
    )
    policy_opt = eve_rl.optim.Adam(policy_base, lr=cfg.lr)
    policy_sched = optim.lr_scheduler.LinearLR(
        policy_opt,
        start_factor=1.0,
        end_factor=cfg.lr_end_factor,
        total_iters=cfg.lr_linear_end_steps,
    )

    model = eve_rl.model.SACModel(
        lr_alpha=cfg.lr,
        q1=q1,
        q2=q2,
        policy=policy,
        q1_optimizer=q1_opt,
        q2_optimizer=q2_opt,
        policy_optimizer=policy_opt,
        q1_scheduler=q1_sched,
        q2_scheduler=q2_sched,
        policy_scheduler=policy_sched,
    )

    return eve_rl.algo.SAC(
        model,
        n_actions=n_actions,
        gamma=cfg.gamma,
        reward_scaling=cfg.reward_scaling,
        stochastic_eval=cfg.stochastic_eval,
    )


def _infer_obs_act_sizes(env_train) -> tuple[int, int]:
    obs_dict = env_train.observation_space.sample()
    obs_list = [obs.flatten() for obs in obs_dict.values()]
    n_observations = sum(o.shape[0] for o in obs_list)
    n_actions = env_train.action_space.sample().flatten().shape[0]
    return n_observations, n_actions


def make_replay_buffer(cfg: PaperAgentConfig, replay_device: torch.device):
    return eve_rl.replaybuffer.VanillaEpisodeShared(
        cfg.replay_buffer_size, cfg.batch_size, replay_device
    )


def make_single_agent(
    cfg: PaperAgentConfig,
    env_train,
    env_eval,
    trainer_device: torch.device,
    replay_device: torch.device,
    consecutive_action_steps: int,
):
    n_observations, n_actions = _infer_obs_act_sizes(env_train)
    algo = make_sac_algo(cfg, n_observations=n_observations, n_actions=n_actions)
    replay_buffer = make_replay_buffer(cfg, replay_device=replay_device)
    return eve_rl.agent.Single(
        algo=algo,
        env_train=env_train,
        env_eval=env_eval,
        replay_buffer=replay_buffer,
        consecutive_action_steps=consecutive_action_steps,
        device=trainer_device,
        normalize_actions=True,
    )


def make_synchron_agent(
    cfg: PaperAgentConfig,
    env_train,
    env_eval,
    trainer_device: torch.device,
    worker_device: torch.device,
    replay_device: torch.device,
    consecutive_action_steps: int,
    n_worker: int,
    timeout_worker_after_reaching_limit: int = 600,
):
    n_observations, n_actions = _infer_obs_act_sizes(env_train)
    algo = make_sac_algo(cfg, n_observations=n_observations, n_actions=n_actions)
    replay_buffer = make_replay_buffer(cfg, replay_device=replay_device)
    return eve_rl.agent.Synchron(
        algo=algo,
        env_train=env_train,
        env_eval=env_eval,
        replay_buffer=replay_buffer,
        consecutive_action_steps=consecutive_action_steps,
        trainer_device=trainer_device,
        worker_device=worker_device,
        n_worker=n_worker,
        normalize_actions=True,
        timeout_worker_after_reaching_limit=timeout_worker_after_reaching_limit,
    )
