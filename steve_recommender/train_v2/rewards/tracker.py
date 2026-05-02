"""Per-episode reward component tracker for train_v2."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eve.reward.reward import Reward

logger = logging.getLogger(__name__)

INFO_KEY_PREFIX = "reward_"


class RewardTracker(Reward):
    """Wraps a list of named reward components.

    On each episode boundary (reset) logs a one-line summary to the Python
    logger and appends a row to a CSV file (one file per process so parallel
    workers never contend on the same file handle).

    CSV path: <base_csv_path> with '.csv' replaced by '_<pid>.csv'.
    Example: reward_train.csv → reward_train_12345.csv

    last_episode_totals is updated just before the accumulators are cleared,
    so RewardComponentInfo.reset() can read it regardless of call ordering.
    """

    def __init__(
        self,
        components: List[Tuple[str, Reward]],
        csv_path: Optional[Path] = None,
    ) -> None:
        self._names = [n for n, _ in components]
        self._rewards = [r for _, r in components]
        self._csv_base = Path(csv_path) if csv_path is not None else None
        self._csv_resolved: Optional[Path] = None
        self._episode_totals = [0.0] * len(self._rewards)
        self._last_episode_totals: Dict[str, float] = {}
        self._last_step_components: Dict[str, float] = {
            name: 0.0 for name in self._names
        }
        self._step_count = 0
        self._episode_nr = 0
        self.reward = 0.0

    @property
    def component_names(self) -> List[str]:
        return list(self._names)

    @property
    def last_episode_totals(self) -> Dict[str, float]:
        """Episode totals from the most recently completed episode."""
        return dict(self._last_episode_totals)

    @property
    def last_step_components(self) -> Dict[str, float]:
        return dict(self._last_step_components)

    def _resolved_path(self) -> Optional[Path]:
        if self._csv_base is None:
            return None
        if self._csv_resolved is None:
            stem = self._csv_base.stem
            suffix = self._csv_base.suffix or ".csv"
            self._csv_resolved = self._csv_base.parent / f"{stem}_{os.getpid()}{suffix}"
        return self._csv_resolved

    def _ensure_header(self) -> None:
        path = self._resolved_path()
        if path is None or path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("episode,steps,total," + ",".join(self._names) + "\n")

    def step(self) -> None:
        total = 0.0
        for i, r in enumerate(self._rewards):
            r.step()
            self._last_step_components[self._names[i]] = float(r.reward)
            self._episode_totals[i] += r.reward
            total += r.reward
        self.reward = total
        self._step_count += 1

    def reset(self, episode_nr: int = 0) -> None:
        if self._step_count > 0:
            self._flush_episode()
        self._episode_totals = [0.0] * len(self._rewards)
        self._last_step_components = {name: 0.0 for name in self._names}
        self._step_count = 0
        self._episode_nr = episode_nr
        self.reward = 0.0
        for r in self._rewards:
            r.reset(episode_nr)

    def debug_snapshot(self) -> Dict[str, float]:
        snapshot: Dict[str, float] = {
            "reward_total": float(self.reward),
        }
        for name, value in self._last_step_components.items():
            snapshot[f"reward_{name}"] = float(value)
        force_component = None
        for name, reward in zip(self._names, self._rewards):
            if name == "force":
                force_component = reward
                break
        if force_component is not None:
            snapshot["force_step_penalty"] = float(
                getattr(force_component, "last_step_penalty", 0.0)
            )
            snapshot["force_terminal_penalty"] = float(
                getattr(force_component, "last_terminal_penalty", 0.0)
            )
            snapshot["wire_force_normal_instant_N"] = float(
                getattr(force_component, "last_wire_force_normal_instant_N", 0.0)
            )
            snapshot["wire_force_normal_trial_max_N"] = float(
                getattr(force_component, "last_wire_force_normal_trial_max_N", 0.0)
            )
            snapshot["tip_force_normal_instant_N"] = float(
                getattr(force_component, "last_tip_force_normal_instant_N", 0.0)
            )
            snapshot["tip_force_normal_trial_max_N"] = float(
                getattr(force_component, "last_tip_force_normal_trial_max_N", 0.0)
            )
        return snapshot

    def _flush_episode(self) -> None:
        self._last_episode_totals = dict(zip(self._names, self._episode_totals))
        total = sum(self._episode_totals)
        parts = "  ".join(
            f"{n}={v:+.3f}" for n, v in zip(self._names, self._episode_totals)
        )
        logger.info(
            "[reward] ep=%05d steps=%3d total=%+.3f  |  %s",
            self._episode_nr,
            self._step_count,
            total,
            parts,
        )
        path = self._resolved_path()
        if path is not None:
            self._ensure_header()
            row = (
                f"{self._episode_nr},{self._step_count},{total:.6f},"
                + ",".join(f"{v:.6f}" for v in self._episode_totals)
                + "\n"
            )
            with open(path, "a") as f:
                f.write(row)


class RewardComponentInfo:
    """Exposes RewardTracker episode totals as eve.info-compatible info dict.

    Reads last_episode_totals from the tracker at each episode reset so the
    values flow through the Runner's result collection back to the main
    process, where they appear in the results CSV and in main.log.

    Keys are prefixed with 'reward_' (e.g. 'reward_target', 'reward_force').
    """

    def __init__(self, tracker: RewardTracker) -> None:
        self._tracker = tracker
        self.info: Dict[str, float] = {
            f"{INFO_KEY_PREFIX}{n}": 0.0 for n in tracker.component_names
        }

    def step(self) -> None:
        pass

    def reset(self, episode_nr: int = 0) -> None:
        totals = self._tracker.last_episode_totals
        for name, value in totals.items():
            self.info[f"{INFO_KEY_PREFIX}{name}"] = float(value)
