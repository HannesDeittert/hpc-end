from pathlib import Path

import h5py
import pytest

from steve_recommender.train_v2.telemetry.step_trace import StepTraceRecorder


def test_step_trace_recorder_writes_every_nth_step_and_terminal_step(tmp_path: Path):
    recorder = StepTraceRecorder(
        base_path=tmp_path / "step_trace_train.h5",
        mode="train",
        every_n_steps=3,
    )
    recorder.reset(episode_nr=4)

    for step_index in range(5):
        recorder.record_step(
            step_index=step_index,
            terminated=False,
            truncated=False,
            reward_snapshot={
                "reward_total": float(step_index),
                "reward_target": 0.0,
                "reward_path_delta": 0.1,
                "reward_step": -0.005,
                "reward_force": -0.2,
                "force_step_penalty": -0.2,
                "force_terminal_penalty": 0.0,
                "wire_force_normal_instant_N": 2.0,
                "wire_force_normal_trial_max_N": 3.0,
                "tip_force_normal_instant_N": 0.5,
                "tip_force_normal_trial_max_N": 0.75,
            },
            info_snapshot={
                "success": False,
                "path_ratio": 0.25,
                "trajectory_length": 12.0,
                "average_translation_speed": 1.5,
                "steps": step_index + 1,
            },
        )
    recorder.record_step(
        step_index=5,
        terminated=True,
        truncated=False,
        reward_snapshot={
            "reward_total": 5.0,
            "reward_target": 1.0,
            "reward_path_delta": 0.2,
            "reward_step": -0.005,
            "reward_force": -0.8,
            "force_step_penalty": -0.3,
            "force_terminal_penalty": -0.5,
            "wire_force_normal_instant_N": 3.0,
            "wire_force_normal_trial_max_N": 4.0,
            "tip_force_normal_instant_N": 0.6,
            "tip_force_normal_trial_max_N": 0.9,
        },
        info_snapshot={
            "success": True,
            "path_ratio": 0.8,
            "trajectory_length": 14.0,
            "average_translation_speed": 1.7,
            "steps": 6,
        },
    )
    recorder.close()

    output_path = tmp_path / f"step_trace_train_{recorder.pid}.h5"
    assert output_path.exists()

    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["mode"] == "train"
        assert int(handle.attrs["every_n_steps"]) == 3
        assert list(handle["step_index"][:]) == [2, 5]
        assert list(handle["episode"][:]) == [4, 4]
        assert list(handle["terminated"][:]) == [False, True]
        assert list(handle["reward_force"][:]) == pytest.approx([-0.2, -0.8])
        assert list(handle["force_terminal_penalty"][:]) == pytest.approx([0.0, -0.5])
        assert list(handle["wire_force_normal_trial_max_N"][:]) == pytest.approx([3.0, 4.0])
