from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import sleep, time
from types import SimpleNamespace

import gymnasium.spaces as spaces
import numpy as np
import torch
from torch import optim

import eve_rl
from eve_rl.replaybuffer import Episode

from steve_recommender.training.bench_agents import ResumableSingle
from steve_recommender.training.replaybuffer import (
    ResumableVanillaEpisode,
    ResumableVanillaEpisodeShared,
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


def _assert_nested_equal(actual, expected) -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
        return
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
        return
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
        return
    assert actual == expected


def _prime_model_state(model: eve_rl.model.SACModel) -> None:
    torch.manual_seed(0)

    for network, optimizer_, scheduler in (
        (model.q1, model.q1_optimizer, model.q1_scheduler),
        (model.q2, model.q2_optimizer, model.q2_scheduler),
        (model.policy, model.policy_optimizer, model.policy_scheduler),
    ):
        optimizer_.zero_grad()
        loss = sum(parameter.sum() for parameter in network.parameters())
        loss.backward()
        optimizer_.step()
        scheduler.step()

    model.alpha_optimizer.zero_grad()
    model.log_alpha.backward(torch.ones_like(model.log_alpha))
    model.alpha_optimizer.step()


def _make_agent(replay_buffer: eve_rl.replaybuffer.ReplayBuffer) -> ResumableSingle:
    env_train = _StubEnv()
    env_eval = _StubEnv()

    q1_base = eve_rl.network.component.MLP([4, 4])
    q1 = eve_rl.network.QNetwork(q1_base, 2, 1)
    q1_optimizer = eve_rl.optim.Adam(q1, lr=1e-3)
    q1_scheduler = optim.lr_scheduler.LinearLR(
        q1_optimizer, start_factor=1.0, end_factor=0.5, total_iters=5
    )

    q2_base = eve_rl.network.component.MLP([4, 4])
    q2 = eve_rl.network.QNetwork(q2_base, 2, 1)
    q2_optimizer = eve_rl.optim.Adam(q2, lr=1e-3)
    q2_scheduler = optim.lr_scheduler.LinearLR(
        q2_optimizer, start_factor=1.0, end_factor=0.5, total_iters=5
    )

    policy_base = eve_rl.network.component.MLP([4, 4])
    policy = eve_rl.network.GaussianPolicy(policy_base, 2, 1)
    policy_optimizer = eve_rl.optim.Adam(policy, lr=1e-3)
    policy_scheduler = optim.lr_scheduler.LinearLR(
        policy_optimizer, start_factor=1.0, end_factor=0.5, total_iters=5
    )

    sac_model = eve_rl.model.SACModel(
        lr_alpha=1e-3,
        q1=q1,
        q2=q2,
        policy=policy,
        q1_optimizer=q1_optimizer,
        q2_optimizer=q2_optimizer,
        policy_optimizer=policy_optimizer,
        q1_scheduler=q1_scheduler,
        q2_scheduler=q2_scheduler,
        policy_scheduler=policy_scheduler,
    )
    algo = eve_rl.algo.SAC(
        sac_model,
        n_actions=1,
        gamma=0.99,
    )
    return ResumableSingle(
        algo,
        env_train,
        env_eval,
        replay_buffer,
        device=torch.device("cpu"),
        consecutive_action_steps=1,
        normalize_actions=True,
    )


def test_vanilla_episode_shared_state_roundtrip() -> None:
    replay = ResumableVanillaEpisodeShared(
        capacity=16,
        batch_size=8,
        sample_device=torch.device("cpu"),
    )
    clone = ResumableVanillaEpisodeShared(
        capacity=16,
        batch_size=8,
        sample_device=torch.device("cpu"),
    )
    try:
        for seed in range(4):
            replay.push(_make_episode(seed))

        _wait_for(lambda: len(replay) == 4)
        state = replay.state_dict()

        clone.load_state_dict(state)
        _wait_for(lambda: len(clone) == 4)
        _assert_nested_equal(clone.state_dict(), state)
    finally:
        replay.close()
        clone.close()


def test_agent_checkpoint_restores_weights_counters_and_replay_buffer(tmp_path: Path) -> None:
    replay = ResumableVanillaEpisodeShared(
        capacity=16,
        batch_size=8,
        sample_device=torch.device("cpu"),
    )
    restored_replay = ResumableVanillaEpisodeShared(
        capacity=16,
        batch_size=8,
        sample_device=torch.device("cpu"),
    )
    agent = _make_agent(replay)
    restored_agent = _make_agent(restored_replay)
    checkpoint_path = tmp_path / "agent_resume.everl"

    try:
        for seed in range(5):
            agent.replay_buffer.push(_make_episode(seed))
        _wait_for(lambda: len(agent.replay_buffer) == 5)

        _prime_model_state(agent.algo.model)

        agent.step_counter.heatup = 123
        agent.step_counter.exploration = 456
        agent.step_counter.update = 78
        agent.step_counter.evaluation = 9
        agent.episode_counter.heatup = 11
        agent.episode_counter.exploration = 22
        agent.episode_counter.evaluation = 33

        expected_network = deepcopy(agent.algo.state_dicts_network())
        expected_optimizer = deepcopy(agent.algo.state_dicts_optimizer())
        expected_scheduler = deepcopy(agent.algo.state_dicts_scheduler())
        expected_replay_state = deepcopy(agent.replay_buffer.state_dict())

        agent.save_checkpoint(str(checkpoint_path))
        restored_agent.load_checkpoint(str(checkpoint_path))
        _wait_for(
            lambda: len(restored_agent.replay_buffer) == len(agent.replay_buffer)
        )

        _assert_nested_equal(restored_agent.algo.state_dicts_network(), expected_network)
        _assert_nested_equal(
            restored_agent.algo.state_dicts_optimizer(), expected_optimizer
        )
        _assert_nested_equal(
            restored_agent.algo.state_dicts_scheduler(), expected_scheduler
        )
        _assert_nested_equal(
            restored_agent.replay_buffer.state_dict(),
            expected_replay_state,
        )

        assert restored_agent.step_counter.heatup == 123
        assert restored_agent.step_counter.exploration == 456
        assert restored_agent.step_counter.update == 78
        assert restored_agent.step_counter.evaluation == 9
        assert restored_agent.episode_counter.heatup == 11
        assert restored_agent.episode_counter.exploration == 22
        assert restored_agent.episode_counter.evaluation == 33
    finally:
        agent.close()
        restored_agent.close()


def test_agent_checkpoint_can_store_latest_replay_buffer_sidecar(
    tmp_path: Path,
) -> None:
    replay = ResumableVanillaEpisodeShared(
        capacity=16,
        batch_size=8,
        sample_device=torch.device("cpu"),
    )
    restored_replay = ResumableVanillaEpisodeShared(
        capacity=16,
        batch_size=8,
        sample_device=torch.device("cpu"),
    )
    agent = _make_agent(replay)
    restored_agent = _make_agent(restored_replay)
    checkpoint_path = tmp_path / "checkpoint.everl"
    sidecar_path = tmp_path / "latest_replay_buffer.everl"

    try:
        agent.configure_replay_buffer_checkpointing(
            latest_replay_buffer_path=sidecar_path,
            embed_in_checkpoint=False,
        )
        for seed in range(2):
            agent.replay_buffer.push(_make_episode(seed))
        _wait_for(lambda: len(agent.replay_buffer) == 2)
        agent.save_checkpoint(str(checkpoint_path))

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        assert "replay_buffer_state" not in checkpoint
        assert checkpoint["replay_buffer_sidecar"] == sidecar_path.name
        assert sidecar_path.is_file()
        first_sidecar = torch.load(
            sidecar_path, map_location="cpu", weights_only=False
        )
        assert len(first_sidecar["replay_buffer_state"]["buffer"]) == 2

        agent.replay_buffer.push(_make_episode(9))
        _wait_for(lambda: len(agent.replay_buffer) == 3)
        agent.save_checkpoint(str(checkpoint_path))

        second_sidecar = torch.load(
            sidecar_path, map_location="cpu", weights_only=False
        )
        assert len(second_sidecar["replay_buffer_state"]["buffer"]) == 3

        restored_agent.configure_replay_buffer_checkpointing(
            resume_replay_buffer_path=sidecar_path,
            embed_in_checkpoint=False,
        )
        restored_agent.load_checkpoint(str(checkpoint_path))
        _wait_for(lambda: len(restored_agent.replay_buffer) == 3)
        assert len(restored_agent.replay_buffer) == 3
    finally:
        agent.close()
        restored_agent.close()


def test_restored_agent_can_continue_updating(tmp_path: Path) -> None:
    replay = ResumableVanillaEpisode(capacity=16, batch_size=2)
    restored_replay = ResumableVanillaEpisode(capacity=16, batch_size=2)
    agent = _make_agent(replay)
    restored_agent = _make_agent(restored_replay)
    checkpoint_path = tmp_path / "agent_continue.everl"

    try:
        for seed in range(4):
            agent.replay_buffer.push(_make_episode(seed))

        _prime_model_state(agent.algo.model)
        agent.step_counter.update = 17

        agent.save_checkpoint(str(checkpoint_path))
        restored_agent.load_checkpoint(str(checkpoint_path))

        update_results = restored_agent.update(steps=1)

        assert len(update_results) == 1
        assert restored_agent.step_counter.update == 18
    finally:
        agent.close()
        restored_agent.close()


def test_resumable_single_load_checkpoint_uses_agent_device(
    tmp_path: Path, monkeypatch
) -> None:
    replay = ResumableVanillaEpisode(capacity=16, batch_size=2)
    agent = _make_agent(replay)
    checkpoint_path = tmp_path / "agent_device_map.everl"

    try:
        agent.save_checkpoint(str(checkpoint_path))
        real_torch_load = torch.load
        observed = {}

        def _spy_torch_load(*args, **kwargs):
            observed["map_location"] = kwargs.get("map_location")
            return real_torch_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", _spy_torch_load)
        agent.load_checkpoint(str(checkpoint_path))
        assert observed["map_location"] == agent.device
    finally:
        agent.close()


def test_resumable_synchron_load_checkpoint_uses_trainer_device(monkeypatch) -> None:
    observed = {}

    class _Algo:
        @staticmethod
        def state_dicts_network():
            return {"network": 1}

        @staticmethod
        def state_dicts_optimizer():
            return {"optim": 1}

        @staticmethod
        def state_dicts_scheduler():
            return {"sched": 1}

    class _Trainer:
        def load_state_dicts_network(self, state):
            observed["trainer_network"] = state

        def load_state_dicts_optimizer(self, state):
            observed["trainer_optimizer"] = state

        def load_state_dicts_scheduler(self, state):
            observed["trainer_scheduler"] = state

    checkpoint = {
        "network_state_dicts": {},
        "optimizer_state_dicts": {},
        "scheduler_state_dicts": {},
        "steps": {"heatup": 0, "exploration": 0, "update": 0, "evaluation": 0},
        "episodes": {"heatup": 0, "exploration": 0, "evaluation": 0},
    }

    def _spy_torch_load(*args, **kwargs):
        observed["map_location"] = kwargs.get("map_location")
        return checkpoint

    def _load_checkpoint_state(_checkpoint, checkpoint_path=None):
        observed["checkpoint_loaded"] = _checkpoint
        observed["checkpoint_path"] = checkpoint_path

    def _worker_load_state_dicts_network(state):
        observed["worker_network"] = state

    fake_agent = SimpleNamespace(
        trainer_device=torch.device("cpu"),
        algo=_Algo(),
        trainer=_Trainer(),
        _load_checkpoint_state=_load_checkpoint_state,
        _worker_load_state_dicts_network=_worker_load_state_dicts_network,
    )

    monkeypatch.setattr(torch, "load", _spy_torch_load)
    from steve_recommender.training.bench_agents import ResumableSynchron

    ResumableSynchron.load_checkpoint(fake_agent, "/tmp/fake.everl")
    assert observed["map_location"] == fake_agent.trainer_device
    assert observed["checkpoint_loaded"] is checkpoint
    assert str(observed["checkpoint_path"]) == "/tmp/fake.everl"
    assert observed["worker_network"] == {"network": 1}
    assert observed["trainer_network"] == {"network": 1}
    assert observed["trainer_optimizer"] == {"optim": 1}
    assert observed["trainer_scheduler"] == {"sched": 1}
