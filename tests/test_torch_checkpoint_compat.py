from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.optim as optim

from steve_recommender.evaluation.torch_checkpoint_compat import (
    legacy_checkpoint_load_context,
)


class TorchCheckpointCompatTests(unittest.TestCase):
    def test_context_sets_weights_only_false_by_default(self) -> None:
        seen = []

        def _fake_load(*args, **kwargs):
            seen.append(kwargs.copy())
            return {"ok": True}

        with patch.object(torch, "load", _fake_load):
            with legacy_checkpoint_load_context():
                torch.load("a.everl")
                torch.load("b.everl", weights_only=True)

        self.assertEqual(seen[0].get("weights_only"), False)
        self.assertEqual(seen[1].get("weights_only"), True)

    def test_context_accepts_legacy_verbose_scheduler_kwarg(self) -> None:
        model = nn.Linear(2, 2)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        with legacy_checkpoint_load_context():
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=1.0,
                total_iters=1,
                verbose=False,
            )
        self.assertIsNotNone(scheduler)


if __name__ == "__main__":
    unittest.main()
