"""Per-step HDF5 trace recorder for local train_v2 smoke investigations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

import h5py
import numpy as np


class StepTraceRecorder:
    """Append step-level reward and force diagnostics to one per-process HDF5."""

    _FLOAT_FIELDS = (
        "reward_total",
        "reward_target",
        "reward_path_delta",
        "reward_step",
        "reward_force",
        "force_step_penalty",
        "force_terminal_penalty",
        "wire_force_normal_instant_N",
        "wire_force_normal_trial_max_N",
        "tip_force_normal_instant_N",
        "tip_force_normal_trial_max_N",
        "path_ratio",
        "trajectory_length",
        "average_translation_speed",
    )
    _INT_FIELDS = ("episode", "step_index", "steps")
    _BOOL_FIELDS = ("terminated", "truncated", "success")

    def __init__(
        self,
        *,
        base_path: Path,
        mode: str,
        every_n_steps: int,
    ) -> None:
        self._base_path = Path(base_path)
        self._mode = str(mode)
        self._every_n_steps = int(every_n_steps)
        if self._every_n_steps <= 0:
            raise ValueError("every_n_steps must be > 0")
        self._pid = int(os.getpid())
        self._path = self._base_path.parent / (
            f"{self._base_path.stem}_{self._pid}{self._base_path.suffix or '.h5'}"
        )
        self._handle: Optional[h5py.File] = None
        self._current_episode = 0

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def path(self) -> Path:
        return self._path

    def reset(self, *, episode_nr: int) -> None:
        self._current_episode = int(episode_nr)

    def _ensure_open(self) -> h5py.File:
        if self._handle is not None:
            return self._handle
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = h5py.File(self._path, "a")
        handle.attrs.setdefault("schema_version", "train_v2_step_trace_v1")
        handle.attrs.setdefault("mode", self._mode)
        handle.attrs.setdefault("pid", self._pid)
        handle.attrs.setdefault("every_n_steps", self._every_n_steps)
        for name in self._FLOAT_FIELDS:
            if name not in handle:
                handle.create_dataset(
                    name,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.float32,
                    chunks=True,
                )
        for name in self._INT_FIELDS:
            if name not in handle:
                handle.create_dataset(
                    name,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.int32,
                    chunks=True,
                )
        for name in self._BOOL_FIELDS:
            if name not in handle:
                handle.create_dataset(
                    name,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.bool_,
                    chunks=True,
                )
        self._handle = handle
        return handle

    def _append_scalar(self, handle: h5py.File, name: str, value) -> None:
        dataset = handle[name]
        index = int(dataset.shape[0])
        dataset.resize((index + 1,))
        dataset[index] = value

    def record_step(
        self,
        *,
        step_index: int,
        terminated: bool,
        truncated: bool,
        reward_snapshot: Mapping[str, float],
        info_snapshot: Mapping[str, object],
    ) -> None:
        one_based_step = int(step_index) + 1
        should_record = (
            (one_based_step % self._every_n_steps) == 0
            or bool(terminated)
            or bool(truncated)
        )
        if not should_record:
            return
        handle = self._ensure_open()
        row = {
            "episode": self._current_episode,
            "step_index": int(step_index),
            "steps": int(info_snapshot.get("steps", one_based_step)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(info_snapshot.get("success", False)),
            "path_ratio": float(info_snapshot.get("path_ratio", 0.0) or 0.0),
            "trajectory_length": float(
                info_snapshot.get("trajectory_length", 0.0) or 0.0
            ),
            "average_translation_speed": float(
                info_snapshot.get("average_translation_speed", 0.0) or 0.0
            ),
        }
        row.update(
            {
                name: float(reward_snapshot.get(name, 0.0) or 0.0)
                for name in self._FLOAT_FIELDS
                if name.startswith("reward_")
                or name.startswith("force_")
                or name.endswith("_N")
            }
        )
        for name in self._INT_FIELDS:
            self._append_scalar(handle, name, int(row[name]))
        for name in self._BOOL_FIELDS:
            self._append_scalar(handle, name, bool(row[name]))
        for name in self._FLOAT_FIELDS:
            self._append_scalar(handle, name, float(row.get(name, 0.0)))
        handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
