"""Run an evaluation from the example config."""

from __future__ import annotations

from pathlib import Path

from steve_recommender.evaluation import load_config, run_evaluation


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "docs" / "eval_example.yml"
    cfg = load_config(cfg_path)
    run_dir = run_evaluation(cfg)
    print(f"[example] finished: {run_dir}")


if __name__ == "__main__":
    main()
