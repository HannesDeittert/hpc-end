from __future__ import annotations

import os
import runpy
from dataclasses import replace

module_ns = runpy.run_path("experiments/master-thesis/run_e1_cell.py")
_original_build_scenario = module_ns["_build_scenario"]


def _build_scenario_with_friction(**kwargs):
    scenario = _original_build_scenario(**kwargs)
    friction = float(os.environ.get("E1_FRICTION", "0.1"))
    return replace(scenario, friction=friction)


module_ns["_build_scenario"] = _build_scenario_with_friction


if __name__ == "__main__":
    raise SystemExit(module_ns["main"]())
