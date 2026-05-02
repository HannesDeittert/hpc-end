from types import SimpleNamespace

import pytest

from steve_recommender.eval_v2.force_telemetry import ForceRuntimeStatus
from steve_recommender.train_v2.config import RewardSpec
from steve_recommender.train_v2.telemetry.force_runtime import ForceRuntime


class _CollectorStub:
    def __init__(self):
        self.ensure_calls = 0
        self.capture_calls = 0

    def ensure_runtime(self, *, intervention):
        _ = intervention
        self.ensure_calls += 1
        return ForceRuntimeStatus(True, "ok", "")

    def capture_step(self, *, intervention, step_index):
        _ = intervention, step_index
        self.capture_calls += 1

    def build_summary(self):
        return SimpleNamespace(
            wire_force_normal_instant_N=0.4,
            wire_force_normal_trial_max_N=0.8,
            tip_force_normal_instant_N=0.1,
            tip_force_normal_trial_max_N=0.2,
        )


def test_force_runtime_retries_runtime_setup_during_sampling():
    runtime = ForceRuntime(
        reward_spec=RewardSpec(profile="default_plus_normal_force_penalty"),
        action_dt_s=1.0 / 7.5,
    )
    stub = _CollectorStub()
    runtime._collector = stub
    intervention = SimpleNamespace(simulation=SimpleNamespace(root=object()))

    sample = runtime.sample_step(intervention=intervention, step_index=3)

    assert stub.ensure_calls == 1
    assert stub.capture_calls == 1
    assert sample.wire_force_normal_instant_N == 0.4
    assert sample.wire_force_normal_trial_max_N == 0.8


def test_force_runtime_raises_when_runtime_stays_unavailable_during_live_step():
    runtime = ForceRuntime(
        reward_spec=RewardSpec(profile="default_plus_normal_force_penalty"),
        action_dt_s=1.0 / 7.5,
    )

    class _BrokenCollector:
        def ensure_runtime(self, *, intervention):
            _ = intervention
            return ForceRuntimeStatus(False, "broken", "no live root")

        def capture_step(self, *, intervention, step_index):
            raise AssertionError("capture_step should not run when ensure_runtime failed")

    intervention = SimpleNamespace(simulation=SimpleNamespace(root=object()))
    runtime._collector = _BrokenCollector()

    with pytest.raises(RuntimeError, match="force telemetry is unavailable"):
        runtime.sample_step(intervention=intervention, step_index=0)


def test_force_runtime_learns_anatomy_mesh_path_from_intervention():
    runtime = ForceRuntime(
        reward_spec=RewardSpec(profile="default_plus_normal_force_penalty"),
        action_dt_s=1.0 / 7.5,
    )

    class _CollectorStub:
        def __init__(self):
            self.ensure_calls = 0
            self._anatomy_mesh_path = None

        def ensure_runtime(self, *, intervention):
            _ = intervention
            self.ensure_calls += 1
            return ForceRuntimeStatus(True, "ok", "")

    stub = _CollectorStub()
    runtime._collector = stub
    intervention = SimpleNamespace(
        simulation=SimpleNamespace(root=object()),
        vessel_tree=SimpleNamespace(
            mesh_path="/tmp/archvar_mesh.vtu",
            visu_mesh_path="/tmp/archvar_visu.stl",
        ),
    )

    status = runtime.ensure_runtime(intervention=intervention)

    assert status.configured is True
    assert stub.ensure_calls == 1
    assert str(stub._anatomy_mesh_path) == "/tmp/archvar_mesh.vtu"
