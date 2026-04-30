"""Runner subclass that logs reward component breakdown after each eval."""

from __future__ import annotations

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

    def eval(self, *, episodes=None, seeds=None):
        quality, reward = super().eval(episodes=episodes, seeds=seeds)
        if self._component_keys:
            parts = "  ".join(
                f"{k}={self._results.get(f'{INFO_KEY_PREFIX}{k}', 0.0):+.3f}"
                for k in self._component_keys
            )
            self.logger.info("[reward breakdown]  %s", parts)
        return quality, reward
