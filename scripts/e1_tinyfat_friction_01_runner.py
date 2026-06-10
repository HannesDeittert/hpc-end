from __future__ import annotations

import os
import runpy
import sys


def _argv_with_env_friction(argv: list[str]) -> list[str]:
    """Backwards-compatible runner for old sbatch files.

    Newer `run_e1_cell.py` has a real `--friction` argument. Older generated
    TinyFat sbatch files only exported E1_FRICTION, so inject it here when the
    caller did not pass `--friction` explicitly.
    """

    if "--friction" in argv:
        return argv
    return [*argv, "--friction", os.environ.get("E1_FRICTION", "0.1")]


if __name__ == "__main__":
    module_ns = runpy.run_path("experiments/master-thesis/run_e1_cell.py")
    raise SystemExit(module_ns["main"](_argv_with_env_friction(sys.argv[1:])))
