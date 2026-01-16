from __future__ import annotations

from typing import Any, Dict

import numpy as np


class TipStateInfo:
    """Collect basic per-step tip state from the intervention.

    This is implemented as an *eve Info* compatible object, but we keep it local
    to avoid editing upstream stEVE packages.

    Important:
    - stEVE's `eve.Env` only requires an "Info-like" object with:
        * `info` property (dict)
        * `step()` andi.e. called after every env.step()
        * `reset(episode_nr)` called at the start of every episode
        * `__call__()` returning the dict

    We intentionally do *not* subclass `eve.info.Info` here so importing this
    module never relies on custom sys.path hacks.
    """

    def __init__(self, intervention, name_prefix: str = "tip") -> None:
        if not isinstance(name_prefix, str) or not name_prefix:
            raise ValueError("name_prefix must be a non-empty string")

        self.name = name_prefix
        self._intervention = intervention
        self._name_prefix = name_prefix

        self.reset()

    def reset(self, episode_nr: int = 0) -> None:
        _ = episode_nr
        # Cached values updated on every step().
        self._pos3d = np.full((3,), np.nan, dtype=np.float32)
        self._inserted_length = float("nan")
        self._rotation = float("nan")

        # Best-effort: try to capture an initial state after the intervention reset.
        try:
            self.step()
        except Exception:
            pass

    def step(self) -> None:
        # tracking3d[0] is the tip (distal-most point) in tracking coordinates.
        try:
            self._pos3d = np.asarray(
                self._intervention.fluoroscopy.tracking3d[0],
                dtype=np.float32,
            )
        except Exception:
            self._pos3d = np.full((3,), np.nan, dtype=np.float32)

        # Device state is accessible via intervention properties.
        try:
            self._inserted_length = float(self._intervention.device_lengths_inserted[0])
        except Exception:
            self._inserted_length = float("nan")

        try:
            self._rotation = float(self._intervention.device_rotations[0])
        except Exception:
            self._rotation = float("nan")

    @property
    def info(self) -> Dict[str, Any]:
        return {
            f"{self._name_prefix}_pos3d": self._pos3d,
            f"{self._name_prefix}_inserted_length": self._inserted_length,
            f"{self._name_prefix}_rotation": self._rotation,
        }

    def __call__(self) -> Dict[str, Any]:
        return self.info


class SofaWallForceInfo:
    """Collects approximate wall/contact forces from a non-multiprocessing SOFA scene.

    Notes:
    - This requires *non-mp* simulation (`intervention.make_non_mp()`), otherwise
      we cannot access the SOFA scene graph from Python.
    - SOFA exposes multiple force concepts; here we record a few robust scalars
      that are available with the current scene setup:
        * LCP constraint forces (global contact constraints)
        * MechanicalObject force norms for the wire DOFs and collision DOFs
    """

    def __init__(self, intervention, name: str = "wall_forces") -> None:
        self.name = name
        self._intervention = intervention

        self.reset()

    def reset(self, episode_nr: int = 0) -> None:
        _ = episode_nr
        self._lcp_sum_abs = float("nan")
        self._lcp_max_abs = float("nan")
        self._wire_force_norm = float("nan")
        self._collis_force_norm = float("nan")

        # Best-effort: capture forces at the initial pose.
        try:
            self.step()
        except Exception:
            pass

    @staticmethod
    def _safe_norm(arr: Any) -> float:
        try:
            return float(np.linalg.norm(np.asarray(arr)))
        except Exception:
            return float("nan")

    def step(self) -> None:
        sim = getattr(self._intervention, "simulation", None)

        # If the intervention is still in mp mode, we cannot access SOFA objects.
        if sim is None or not hasattr(sim, "root"):
            self._lcp_sum_abs = float("nan")
            self._lcp_max_abs = float("nan")
            self._wire_force_norm = float("nan")
            self._collis_force_norm = float("nan")
            return

        # 1) Constraint forces from LCP solver (global list of constraint magnitudes).
        try:
            lcp = sim.root.LCP
            forces = lcp.constraintForces
            if hasattr(forces, "value"):
                forces = forces.value
            forces_arr = np.asarray(forces, dtype=np.float32)
            if forces_arr.size == 0:
                self._lcp_sum_abs = 0.0
                self._lcp_max_abs = 0.0
            else:
                self._lcp_sum_abs = float(np.sum(np.abs(forces_arr)))
                self._lcp_max_abs = float(np.max(np.abs(forces_arr)))
        except Exception:
            self._lcp_sum_abs = float("nan")
            self._lcp_max_abs = float("nan")

        # 2) Mechanical forces on the wire rigid DOFs and the collision DOFs.
        try:
            wire_forces = sim._instruments_combined.DOFs.force.value  # noqa: SLF001
            self._wire_force_norm = self._safe_norm(wire_forces)
        except Exception:
            self._wire_force_norm = float("nan")

        try:
            coll_forces = sim._instruments_combined.CollisionModel.CollisionDOFs.force.value  # noqa: SLF001
            self._collis_force_norm = self._safe_norm(coll_forces)
        except Exception:
            self._collis_force_norm = float("nan")

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "wall_lcp_sum_abs": self._lcp_sum_abs,
            "wall_lcp_max_abs": self._lcp_max_abs,
            "wall_wire_force_norm": self._wire_force_norm,
            "wall_collision_force_norm": self._collis_force_norm,
        }

    def __call__(self) -> Dict[str, Any]:
        return self.info
