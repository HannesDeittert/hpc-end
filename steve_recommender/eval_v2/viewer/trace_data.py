"""Trace-data wrapper for the Phase F replay viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from steve_recommender.eval_v2.force_trace_persistence import TraceReader


ANATOMY_SIMULATION_MESH_BASE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "anatomy_registry" / "anatomies"
)
DEFAULT_MAX_FORCE_PERCENTILE = 95.0
EMPTY_TRACE_FALLBACK_MAX_FORCE_N = 1.0


@dataclass(frozen=True)
class TraceFrame:
    """One replay frame with wire positions and triangle-force scalars."""

    step_index: int
    sim_time_s: float
    wire_positions_mm: np.ndarray
    triangle_force_indices: np.ndarray
    triangle_force_magnitudes_N: np.ndarray


@dataclass
class TraceData:
    """Reader-backed replay data accessor for one persisted eval_v2 trace."""

    vessel_mesh_path: Path
    n_steps: int
    n_dofs: int
    metadata: dict[str, Any]
    is_partial: bool
    recommended_max_display_force_N: float

    def __init__(self, trace_path: Path) -> None:
        self._trace_path = Path(trace_path)
        self._reader = TraceReader(self._trace_path)
        self._reader.__enter__()
        self._reader._require_open()
        file_handle = self._reader._file
        if file_handle is None:
            raise RuntimeError(f"TraceReader failed to open trace: {self._trace_path}")

        scenario = dict(file_handle["scenario"].attrs)
        meta = dict(file_handle["meta"].attrs)
        anatomy_id = str(scenario["anatomy_id"])
        self.vessel_mesh_path = (
            ANATOMY_SIMULATION_MESH_BASE_PATH
            / anatomy_id
            / "mesh"
            / "simulationmesh.obj"
        )
        if not self.vessel_mesh_path.exists():
            self.close()
            raise FileNotFoundError(
                f"Missing simulation mesh for anatomy '{anatomy_id}': {self.vessel_mesh_path}"
            )

        self.n_steps = int(file_handle["steps/step_index"].shape[0])
        self.n_dofs = int(file_handle["steps/wire_positions"].shape[1])
        self.metadata = {
            "meta": meta,
            "scenario": scenario,
        }
        self.is_partial = bool(self._reader.is_partial)
        self.recommended_max_display_force_N = (
            self._compute_recommended_max_display_force_N()
        )

    def frame(self, step: int) -> TraceFrame:
        """Return one replay frame for the requested simulation step."""

        step_index = int(step)
        if step_index < 0 or step_index >= self.n_steps:
            raise ValueError(
                f"step must be in [0, {self.n_steps - 1}], got {step_index}"
            )
        step_data = self._reader.step(step_index)
        triangle_contacts = self._reader.triangle_contacts_for_step(step_index)
        triangle_force_indices = np.asarray(
            [record.triangle_id for record in triangle_contacts],
            dtype=np.int32,
        )
        triangle_force_magnitudes_N = np.asarray(
            [
                np.linalg.norm([record.fx_N, record.fy_N, record.fz_N])
                for record in triangle_contacts
            ],
            dtype=np.float32,
        )
        return TraceFrame(
            step_index=int(step_data.step_index),
            sim_time_s=float(step_data.sim_time_s),
            wire_positions_mm=np.asarray(step_data.wire_positions_mm, dtype=np.float32),
            triangle_force_indices=triangle_force_indices,
            triangle_force_magnitudes_N=triangle_force_magnitudes_N,
        )

    def close(self) -> None:
        """Close the underlying trace reader."""
        self._reader.__exit__(None, None, None)

    def _compute_recommended_max_display_force_N(self) -> float:
        """Return the percentile-based force scale recommendation in Newtons."""

        file_handle = self._reader._file
        if file_handle is None:
            raise RuntimeError(
                f"TraceReader is not open while reading {self._trace_path}"
            )
        vectors = np.asarray(
            file_handle["contacts/triangle/triangle_force_xyz_N"],
            dtype=np.float32,
        )
        if vectors.size == 0:
            return EMPTY_TRACE_FALLBACK_MAX_FORCE_N
        magnitudes_N = np.linalg.norm(vectors, axis=1)
        return float(np.percentile(magnitudes_N, DEFAULT_MAX_FORCE_PERCENTILE))
