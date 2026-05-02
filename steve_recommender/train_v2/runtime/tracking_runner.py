"""Runner subclass that logs reward component breakdown after each eval."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import List, Optional

from eve_rl import Runner

from ..rewards.tracker import INFO_KEY_PREFIX


class TrackingRunner(Runner):
    """Extends eve_rl.Runner to log per-component reward breakdown after eval.

    After the standard 'Quality: X, Reward: Y' line it emits:
        [reward breakdown]  target=+0.000  path_delta=+0.031  step=-0.090  force=-0.041

    component_keys should be the reward component names (without prefix),
    e.g. ['target', 'path_delta', 'step', 'force'].
    These must match the keys added to env_eval's info by RewardComponentInfo.
    """

    def __init__(
        self, *args, component_keys: Optional[List[str]] = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._component_keys = component_keys or []
        self._results_fieldnames = [
            "episodes_explore",
            "steps_explore",
            "quality",
            *self.info_results,
            "reward",
            "best_quality",
            "best_explore_steps",
            *self.agent_parameter_for_result_file.keys(),
        ]
        self._rewrite_results_file_header()

    def eval(self, *, episodes=None, seeds=None):
        original_results_file = self.results_file
        try:
            self.results_file = os.devnull
            quality, reward = super().eval(episodes=episodes, seeds=seeds)
        finally:
            self.results_file = original_results_file
        self._append_clean_results_row()
        if self._component_keys:
            parts = "  ".join(
                f"{k}={self._results.get(f'{INFO_KEY_PREFIX}{k}', 0.0):+.3f}"
                for k in self._component_keys
            )
            self.logger.info("[reward breakdown]  %s", parts)
        return quality, reward

    def _rewrite_results_file_header(self) -> None:
        path = Path(self.results_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self._results_fieldnames)
            writer.writeheader()

    def _append_clean_results_row(self) -> None:
        row = {
            "episodes_explore": self._results["episodes explore"],
            "steps_explore": self._results["steps explore"],
            "quality": self._results["quality"],
            "reward": self._results["reward"],
            "best_quality": self._results["best quality"],
            "best_explore_steps": self._results["best explore steps"],
        }
        for info_name in self.info_results:
            row[info_name] = self._results.get(info_name)
        for key, value in self.agent_parameter_for_result_file.items():
            row[key] = self._serialize_parameter_value(value)
        with Path(self.results_file).open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self._results_fieldnames)
            writer.writerow(row)

    @staticmethod
    def _serialize_parameter_value(value):
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value)
        return value
