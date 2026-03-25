from __future__ import annotations

import unittest

from steve_recommender.evaluation.pipeline import _apply_stochastic_eval_mode


class _Algo:
    def __init__(self) -> None:
        self.stochastic_eval = False


class _AgentWithAlgo:
    def __init__(self) -> None:
        self.algo = _Algo()


class _AgentWithoutAlgo:
    pass


class StochasticEvalModeTests(unittest.TestCase):
    def test_applies_mode_when_supported(self) -> None:
        agent = _AgentWithAlgo()
        _apply_stochastic_eval_mode(agent, True)
        self.assertTrue(agent.algo.stochastic_eval)
        _apply_stochastic_eval_mode(agent, False)
        self.assertFalse(agent.algo.stochastic_eval)

    def test_noop_when_not_supported(self) -> None:
        agent = _AgentWithoutAlgo()
        _apply_stochastic_eval_mode(agent, True)
        # no exception expected
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
