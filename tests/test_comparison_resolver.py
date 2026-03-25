from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from steve_recommender.comparison.config import ComparisonCandidateSpec, ComparisonConfig
from steve_recommender.comparison.pipeline import _to_evaluation_config, resolve_candidates


def _touch(path: Path, *, sleep_s: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    if sleep_s > 0:
        time.sleep(sleep_s)


class ComparisonResolverTests(unittest.TestCase):
    def test_registry_resolution_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agents_root = root / "data" / "M" / "wires" / "W" / "agents"
            agent_dir = agents_root / "A1"
            agent_dir.mkdir(parents=True, exist_ok=True)

            override_ckpt = root / "override.everl"
            _touch(override_ckpt)
            metadata_ckpt = agent_dir / "meta.everl"
            _touch(metadata_ckpt)
            best_ckpt = agent_dir / "best_checkpoint.everl"
            _touch(best_ckpt)
            _touch(agent_dir / "checkpoint10.everl")
            _touch(agent_dir / "checkpoint20.everl")
            latest_ckpt = agent_dir / "latest.everl"
            _touch(latest_ckpt, sleep_s=0.01)

            (agent_dir / "agent.json").write_text(
                '{"tool": "M/W", "checkpoint": "meta.everl"}',
                encoding="utf-8",
            )

            with patch(
                "steve_recommender.comparison.pipeline.wire_agents_dir",
                return_value=agents_root,
            ):
                cfg_override = ComparisonConfig(
                    name="x",
                    candidates=[
                        ComparisonCandidateSpec(
                            name="c1",
                            agent_ref="M/W:A1",
                            checkpoint_override=str(override_ckpt),
                        )
                    ],
                )
                resolved = resolve_candidates(cfg_override)
                self.assertEqual(resolved[0].checkpoint, override_ckpt.resolve())
                self.assertEqual(resolved[0].source, "checkpoint_override")

                cfg_metadata = ComparisonConfig(
                    name="x",
                    candidates=[ComparisonCandidateSpec(name="c2", agent_ref="M/W:A1")],
                )
                resolved = resolve_candidates(cfg_metadata)
                self.assertEqual(resolved[0].checkpoint, metadata_ckpt.resolve())
                self.assertEqual(resolved[0].source, "agent.json")

                # Remove metadata pointer -> best_checkpoint fallback.
                (agent_dir / "agent.json").write_text("{}", encoding="utf-8")
                resolved = resolve_candidates(cfg_metadata)
                self.assertEqual(resolved[0].checkpoint, best_ckpt.resolve())
                self.assertEqual(resolved[0].source, "best_checkpoint.everl")

                # Remove best checkpoint -> highest checkpointN fallback.
                best_ckpt.unlink()
                resolved = resolve_candidates(cfg_metadata)
                self.assertEqual(
                    resolved[0].checkpoint, (agent_dir / "checkpoint20.everl").resolve()
                )
                self.assertEqual(resolved[0].source, "highest_checkpointN")

                # Remove numeric checkpoints -> latest *.everl fallback.
                (agent_dir / "checkpoint10.everl").unlink()
                (agent_dir / "checkpoint20.everl").unlink()
                resolved = resolve_candidates(cfg_metadata)
                self.assertEqual(resolved[0].checkpoint, latest_ckpt.resolve())
                self.assertEqual(resolved[0].source, "latest_everl")

    def test_explicit_candidate_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "a.everl"
            _touch(ckpt)
            cfg = ComparisonConfig(
                name="x",
                candidates=[
                    ComparisonCandidateSpec(
                        name="explicit",
                        tool="M/W",
                        checkpoint=str(ckpt),
                    )
                ],
            )
            resolved = resolve_candidates(cfg)
            self.assertEqual(resolved[0].checkpoint, ckpt.resolve())
            self.assertEqual(resolved[0].tool, "M/W")
            self.assertEqual(resolved[0].source, "explicit_checkpoint")

    def test_invalid_agent_ref_raises(self) -> None:
        cfg = ComparisonConfig(
            name="x",
            candidates=[ComparisonCandidateSpec(name="bad", agent_ref="invalid_ref")],
        )
        with self.assertRaises(ValueError):
            resolve_candidates(cfg)

    def test_to_evaluation_config_keeps_seeds_and_stochastic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "a.everl"
            _touch(ckpt)
            cfg = ComparisonConfig(
                name="x",
                candidates=[
                    ComparisonCandidateSpec(
                        name="c",
                        tool="M/W",
                        checkpoint=str(ckpt),
                    )
                ],
                seeds=[123, 127, 175],
                stochastic_eval=True,
            )
            resolved = resolve_candidates(cfg)
            eval_cfg = _to_evaluation_config(cfg, resolved)
            self.assertEqual(eval_cfg.seeds, [123, 127, 175])
            self.assertTrue(eval_cfg.stochastic_eval)


if __name__ == "__main__":
    unittest.main()
