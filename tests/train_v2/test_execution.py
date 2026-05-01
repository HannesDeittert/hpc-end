from types import SimpleNamespace

from steve_recommender.train_v2.config import build_training_config
from steve_recommender.train_v2.runtime.execution import compute_resume_targets
from steve_recommender.train_v2.runtime import execution as execution_module


def test_compute_resume_targets_keeps_requested_absolute_target_when_larger():
    heatup_steps, training_target = compute_resume_targets(
        requested_heatup_steps=100,
        requested_training_steps=500,
        current_explore_steps=300,
        resume_skip_heatup=False,
    )
    assert heatup_steps == 100
    assert training_target == 500


def test_compute_resume_targets_treats_smaller_training_steps_as_additional():
    heatup_steps, training_target = compute_resume_targets(
        requested_heatup_steps=100,
        requested_training_steps=200,
        current_explore_steps=300,
        resume_skip_heatup=False,
    )
    assert heatup_steps == 100
    assert training_target == 500


def test_compute_resume_targets_can_skip_heatup():
    heatup_steps, training_target = compute_resume_targets(
        requested_heatup_steps=100,
        requested_training_steps=200,
        current_explore_steps=300,
        resume_skip_heatup=True,
    )
    assert heatup_steps == 0
    assert training_target == 500


def test_execute_training_run_reuses_one_randomized_intervention_for_train_and_eval(
    monkeypatch, tmp_path
):
    built_interventions = []
    built_envs = []
    training_run_calls = []

    class DummyEnv:
        def __init__(self, intervention):
            self.intervention = intervention
            self.info = SimpleNamespace(info={"success": None})

        def save_config(self, *_args, **_kwargs):
            return None

    class DummyAgent:
        def close(self):
            return None

    class DummyRunner:
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs

        def training_run(self, *args, **kwargs):
            training_run_calls.append((args, kwargs))

    monkeypatch.setattr(
        execution_module,
        "get_result_checkpoint_config_and_log_path",
        lambda **_kwargs: (
            tmp_path / "results.csv",
            tmp_path / "checkpoints",
            tmp_path / "config",
            tmp_path / "main.log",
        ),
    )
    monkeypatch.setattr(
        execution_module,
        "build_intervention",
        lambda **_kwargs: built_interventions.append(SimpleNamespace(tag="scene"))
        or built_interventions[-1],
    )
    monkeypatch.setattr(
        execution_module,
        "build_env",
        lambda *, intervention, **_kwargs: built_envs.append(intervention)
        or DummyEnv(intervention),
    )
    monkeypatch.setattr(execution_module, "build_agent", lambda **_kwargs: DummyAgent())
    monkeypatch.setattr(execution_module, "TrackingRunner", DummyRunner)
    monkeypatch.setattr(execution_module.mp, "set_start_method", lambda *args, **kwargs: None)

    cfg = build_training_config(
        name="run",
        tool_ref="steve_default/standard_j",
        preflight=False,
    )

    execution_module.execute_training_run(cfg)

    assert len(built_interventions) == 1
    assert built_envs[0] is not built_envs[1]
    assert built_envs[0] is built_interventions[0]
    assert built_envs[1] is not built_interventions[0]
    assert len(training_run_calls) == 1
    _, kwargs = training_run_calls[0]
    assert kwargs["eval_episodes"] is None
    assert len(kwargs["eval_seeds"]) == 98
