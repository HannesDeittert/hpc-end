import csv
from pathlib import Path

from steve_recommender.train_v2.runtime.tracking_runner import TrackingRunner


class _DummyAgent:
    def __init__(self):
        self.step_counter = type("StepCounter", (), {"exploration": 0, "update": 0})()
        self.episode_counter = type("EpisodeCounter", (), {"exploration": 0})()

    def evaluate(self, episodes=None, seeds=None):
        _ = episodes, seeds
        episode = type(
            "Episode",
            (),
            {
                "episode_reward": 1.25,
                "infos": [{"success": 0.75, "path_ratio": 0.5}],
                "seed": 123,
                "options": None,
            },
        )()
        return [episode]

    def save_checkpoint(self, *_args, **_kwargs):
        return None


def test_tracking_runner_rewrites_results_file_as_clean_csv(tmp_path: Path):
    results_file = tmp_path / "results.csv"
    runner = TrackingRunner(
        agent=_DummyAgent(),
        heatup_action_low=[-1.0, -1.0],
        heatup_action_high=[1.0, 1.0],
        agent_parameter_for_result_file={
            "tool_ref": "jshaped_default",
            "hidden_layers": [400, 400, 400],
            "reward_profile": "default_plus_normal_force_penalty",
        },
        checkpoint_folder=str(tmp_path / "checkpoints"),
        results_file=str(results_file),
        quality_info="success",
        info_results=["success", "path_ratio"],
    )

    runner.eval()

    with results_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["episodes_explore"] == "0"
    assert row["steps_explore"] == "0"
    assert row["quality"] == "0.75"
    assert row["success"] == "0.75"
    assert row["path_ratio"] == "0.5"
    assert row["reward"] == "1.25"
    assert row["tool_ref"] == "jshaped_default"
    assert row["hidden_layers"] == "[400, 400, 400]"
    assert row["reward_profile"] == "default_plus_normal_force_penalty"
