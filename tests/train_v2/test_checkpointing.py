from pathlib import Path

import pytest
import torch

from steve_recommender.train_v2.agents.checkpointing import (
    LATEST_REPLAY_BUFFER_NAME,
    ReplayBufferCheckpointMixin,
)
from steve_recommender.train_v2.agents.replaybuffer import ResumableVanillaEpisode


class DummyAlgo:
    def __init__(self):
        self.network_loaded = None
        self.optimizer_loaded = None
        self.scheduler_loaded = None

    def get_config_dict(self):
        return {"algo": "dummy"}

    def state_dicts_network(self):
        return {"net": 1}

    def state_dicts_optimizer(self):
        return {"opt": 2}

    def state_dicts_scheduler(self):
        return {"sched": 3}

    def load_state_dicts_network(self, state):
        self.network_loaded = state

    def load_state_dicts_optimizer(self, state):
        self.optimizer_loaded = state

    def load_state_dicts_scheduler(self, state):
        self.scheduler_loaded = state


class DummyCounter:
    def __init__(self, heatup=0, exploration=0, update=0, evaluation=0):
        self.heatup = heatup
        self.exploration = exploration
        self.update = update
        self.evaluation = evaluation


class DummyCheckpointHost(ReplayBufferCheckpointMixin):
    def __init__(self):
        self.algo = DummyAlgo()
        self.replay_buffer = ResumableVanillaEpisode(16, 2)
        self.step_counter = DummyCounter(1, 2, 3, 4)
        self.episode_counter = DummyCounter(5, 6, 0, 7)
        self.env_train = None
        self.env_eval = None


def _push_dummy_transition(buffer: ResumableVanillaEpisode) -> None:
    buffer.buffer = [("obs", "action", "reward")]
    buffer.position = 1


def test_resumable_episode_round_trip_state_dict():
    buffer = ResumableVanillaEpisode(16, 2)
    _push_dummy_transition(buffer)

    state = buffer.state_dict()

    restored = ResumableVanillaEpisode(1, 1)
    restored.load_state_dict(state)

    assert restored.capacity == 16
    assert restored.batch_size == 2
    assert restored.position == 1
    assert restored.buffer == [("obs", "action", "reward")]


def test_save_replay_buffer_state_writes_payload(tmp_path: Path):
    host = DummyCheckpointHost()
    _push_dummy_transition(host.replay_buffer)

    out_path = tmp_path / "rb.everl"
    host.save_replay_buffer_state(out_path)

    payload = torch.load(out_path, map_location="cpu", weights_only=False)
    assert payload["replay_buffer_state"]["position"] == 1
    assert payload["steps"]["exploration"] == 2
    assert payload["episodes"]["evaluation"] == 7


def test_load_checkpoint_state_uses_embedded_replay_buffer_state():
    host = DummyCheckpointHost()
    checkpoint = {
        "network_state_dicts": {"net": 9},
        "optimizer_state_dicts": {"opt": 8},
        "scheduler_state_dicts": {"sched": 7},
        "replay_buffer_state": {
            "capacity": 16,
            "batch_size": 2,
            "buffer": [("x",)],
            "position": 1,
        },
        "steps": {"heatup": 10, "exploration": 11, "update": 12, "evaluation": 13},
        "episodes": {"heatup": 20, "exploration": 21, "evaluation": 22},
    }

    host._load_checkpoint_state(checkpoint)

    assert host.algo.network_loaded == {"net": 9}
    assert host.replay_buffer.buffer == [("x",)]
    assert host.step_counter.exploration == 11
    assert host.episode_counter.evaluation == 22


def test_load_checkpoint_state_falls_back_to_sidecar(tmp_path: Path):
    host = DummyCheckpointHost()
    sidecar_path = tmp_path / LATEST_REPLAY_BUFFER_NAME
    torch.save(
        {
            "replay_buffer_state": {
                "capacity": 16,
                "batch_size": 2,
                "buffer": [("sidecar",)],
                "position": 1,
            }
        },
        sidecar_path,
    )
    checkpoint_path = tmp_path / "checkpoint.everl"
    checkpoint_path.write_bytes(b"placeholder")

    checkpoint = {
        "network_state_dicts": {"net": 1},
        "optimizer_state_dicts": {"opt": 2},
        "scheduler_state_dicts": {"sched": 3},
        "steps": {"heatup": 1, "exploration": 2, "update": 3, "evaluation": 4},
        "episodes": {"heatup": 5, "exploration": 6, "evaluation": 7},
    }

    used_path = host._load_replay_buffer_sidecar(checkpoint_path, checkpoint)

    assert used_path == sidecar_path
    assert host.replay_buffer.buffer == [("sidecar",)]


def test_load_replay_buffer_state_requires_supported_buffer():
    host = DummyCheckpointHost()
    host.replay_buffer = object()

    with pytest.raises(
        TypeError, match="Replay buffer does not support state restoration"
    ):
        host._load_replay_buffer_state({})
