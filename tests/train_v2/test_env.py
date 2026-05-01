import numpy as np
import pytest

from steve_recommender.train_v2.runtime.env_factory import TrainV2Env
from steve_recommender.eval_v2.force_telemetry import ForceRuntimeStatus


class _CallRecorder:
    def __init__(self, name, events, attr_name=None):
        self.name = name
        self.events = events
        self.attr_name = attr_name
        if attr_name is not None:
            setattr(self, attr_name, False)

    def step(self):
        self.events.append(self.name)

    def reset(self, episode_nr=0):
        _ = episode_nr


class _Observation:
    def __init__(self, events):
        self.events = events

    def step(self):
        self.events.append("observation")

    def __call__(self):
        return np.asarray([1.0], dtype=np.float32)

    def reset(self, episode_nr=0):
        _ = episode_nr


class _Reward:
    def __init__(self, events):
        self.events = events
        self.reward = 0.0

    def step(self):
        self.events.append("reward")
        self.reward = 1.0

    def reset(self, episode_nr=0):
        _ = episode_nr
        self.reward = 0.0


class _Info:
    def __init__(self, events):
        self.events = events
        self.info = {}

    def step(self):
        self.events.append("info")

    def reset(self, episode_nr=0):
        _ = episode_nr

    def __call__(self):
        return self.info


class _Intervention:
    action_space = None

    def __init__(self, events):
        self.events = events
        self.simulation = type("Simulation", (), {"root": None})()

    def step(self, action):
        _ = action
        self.events.append("intervention")

    def reset(self, episode_nr, seed=None, options=None):
        _ = episode_nr, seed, options
        self.simulation.root = object()


class _Resettable:
    def reset(self, episode_nr=0):
        _ = episode_nr


class _ForceTelemetry:
    def __init__(self, configured=True):
        self.calls = 0
        self.configured = configured

    def ensure_runtime(self, *, intervention):
        self.calls += 1
        root = getattr(getattr(intervention, "simulation", None), "root", None)
        if self.configured and root is not None:
            return ForceRuntimeStatus(True, "ok", "")
        return ForceRuntimeStatus(False, "simulation_root_not_ready", "root missing")


def test_train_v2_env_computes_terminal_and_truncation_before_reward():
    events = []
    env = TrainV2Env(
        intervention=_Intervention(events),
        observation=_Observation(events),
        reward=_Reward(events),
        terminal=_CallRecorder("terminal", events, attr_name="terminal"),
        truncation=_CallRecorder("truncation", events, attr_name="truncated"),
        info=_Info(events),
        start=None,
        pathfinder=_CallRecorder("pathfinder", events),
        interim_target=_CallRecorder("interim_target", events),
        visualisation=None,
    )

    env.step(np.asarray([0.0], dtype=np.float32))

    assert events == [
        "intervention",
        "pathfinder",
        "interim_target",
        "observation",
        "terminal",
        "truncation",
        "reward",
        "info",
    ]


def test_train_v2_env_binds_force_telemetry_after_reset():
    events = []
    telemetry = _ForceTelemetry(configured=True)
    env = TrainV2Env(
        intervention=_Intervention(events),
        observation=_Observation(events),
        reward=_Reward(events),
        terminal=_CallRecorder("terminal", events, attr_name="terminal"),
        truncation=_CallRecorder("truncation", events, attr_name="truncated"),
        info=_Info(events),
        start=_Resettable(),
        pathfinder=_CallRecorder("pathfinder", events),
        interim_target=_CallRecorder("interim_target", events),
        visualisation=_Resettable(),
        force_telemetry=telemetry,
    )

    env.reset()

    assert telemetry.calls == 1


def test_train_v2_env_raises_when_force_telemetry_remains_detached_after_reset():
    events = []
    telemetry = _ForceTelemetry(configured=False)
    env = TrainV2Env(
        intervention=_Intervention(events),
        observation=_Observation(events),
        reward=_Reward(events),
        terminal=_CallRecorder("terminal", events, attr_name="terminal"),
        truncation=_CallRecorder("truncation", events, attr_name="truncated"),
        info=_Info(events),
        start=_Resettable(),
        pathfinder=_CallRecorder("pathfinder", events),
        interim_target=_CallRecorder("interim_target", events),
        visualisation=_Resettable(),
        force_telemetry=telemetry,
    )

    with pytest.raises(RuntimeError, match="force telemetry failed to bind"):
        env.reset()
