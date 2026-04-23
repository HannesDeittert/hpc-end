from __future__ import annotations

from pathlib import Path
from time import sleep, time

import gymnasium.spaces as spaces
import numpy as np
import torch
from eve_rl.replaybuffer import Episode

from steve_recommender.rl.paper_agent_factory import (
    PaperAgentConfig,
    make_single_agent,
)


class _StubEnv:
    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(2,),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def close(self) -> None:
        return None


def _make_episode(seed: int, steps: int = 3) -> Episode:
    reset_obs = {"obs": np.array([seed, seed + 0.5], dtype=np.float32)}
    episode = Episode(
        reset_obs=reset_obs,
        reset_flat_obs=reset_obs["obs"],
        flat_obs_to_obs=None,
        seed=seed,
        options={"seed": seed},
    )
    for i in range(steps):
        obs = {"obs": np.array([seed + i + 1, seed + i + 1.5], dtype=np.float32)}
        episode.add_transition(
            obs=obs,
            flat_obs=obs["obs"],
            action=np.array([seed + i], dtype=np.float32),
            reward=float(seed + i),
            terminal=i == steps - 1,
            truncation=False,
            info={"step": i},
        )
    return episode


def _wait_for(predicate, timeout: float = 3.0) -> None:
    deadline = time() + timeout
    while time() < deadline:
        if predicate():
            return
        sleep(0.01)
    raise AssertionError("Timed out waiting for async replay buffer state")


def test_paper_factory_single_resume_restores_replay_buffer(tmp_path: Path) -> None:
    cfg = PaperAgentConfig(
        hidden_layers=[4, 4],
        embedder_nodes=0,
        embedder_layers=0,
        ff_only=False,
        lr=1e-3,
        lr_end_factor=0.9,
        lr_linear_end_steps=10,
        gamma=0.99,
        reward_scaling=1.0,
        batch_size=2,
        replay_buffer_size=16,
        stochastic_eval=False,
    )

    checkpoint_path = tmp_path / "paper_resume.everl"
    agent = make_single_agent(
        cfg,
        env_train=_StubEnv(),
        env_eval=_StubEnv(),
        trainer_device=torch.device("cpu"),
        replay_device=torch.device("cpu"),
        consecutive_action_steps=1,
    )
    restored = make_single_agent(
        cfg,
        env_train=_StubEnv(),
        env_eval=_StubEnv(),
        trainer_device=torch.device("cpu"),
        replay_device=torch.device("cpu"),
        consecutive_action_steps=1,
    )

    try:
        for seed in range(4):
            agent.replay_buffer.push(_make_episode(seed))
        _wait_for(lambda: len(agent.replay_buffer) == 4)
        agent.step_counter.heatup = 10
        agent.step_counter.exploration = 20
        agent.step_counter.update = 30
        agent.step_counter.evaluation = 40

        agent.save_checkpoint(str(checkpoint_path))
        restored.load_checkpoint(str(checkpoint_path))
        _wait_for(lambda: len(restored.replay_buffer) == 4)

        assert len(restored.replay_buffer) == 4
        assert restored.step_counter.heatup == 10
        assert restored.step_counter.exploration == 20
        assert restored.step_counter.update == 30
        assert restored.step_counter.evaluation == 40
    finally:
        agent.close()
        restored.close()
