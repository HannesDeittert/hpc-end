"""Trace persistence helpers keep pure CSR logic and HDF5/v1 I/O together."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

TRACE_SCHEMA_VERSION = 2
LEGACY_TRACE_SCHEMA_VERSION = 1
SCHEMA_VERSION = LEGACY_TRACE_SCHEMA_VERSION
DEFAULT_FLUSH_INTERVAL_STEPS = 50
DEFAULT_GZIP_LEVEL = 4
STEP_DATASET_CHUNK_LEADING_DIM = 1
SMALL_TIMESERIES_CHUNK_LENGTH = 256

if TYPE_CHECKING:
    import h5py


@dataclass(frozen=True)
class TriangleContactRecord:
    """Triangle-side wall-contact record emitted by the collector."""

    timestep: int
    triangle_id: int
    fx_N: float
    fy_N: float
    fz_N: float
    norm_N: float
    contributing_rows: int
    mapped: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestep", int(self.timestep))
        object.__setattr__(self, "triangle_id", int(self.triangle_id))
        object.__setattr__(self, "fx_N", float(np.float32(self.fx_N)))
        object.__setattr__(self, "fy_N", float(np.float32(self.fy_N)))
        object.__setattr__(self, "fz_N", float(np.float32(self.fz_N)))
        object.__setattr__(self, "norm_N", float(np.float32(self.norm_N)))
        object.__setattr__(self, "contributing_rows", int(self.contributing_rows))
        object.__setattr__(self, "mapped", bool(self.mapped))


@dataclass(frozen=True)
class WireContactRecord:
    """Wire-side collision-DOF force record emitted by the collector."""

    timestep: int
    wire_collision_dof: int
    row_idx: int
    fx_N: float
    fy_N: float
    fz_N: float
    norm_N: float
    arc_length_from_distal_mm: float | None = None
    is_tip: bool = False
    world_pos: Tuple[float, float, float] | None = None
    fx_scene: float | None = None
    fy_scene: float | None = None
    fz_scene: float | None = None
    norm_scene: float | None = None
    mapped: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestep", int(self.timestep))
        object.__setattr__(self, "wire_collision_dof", int(self.wire_collision_dof))
        object.__setattr__(self, "row_idx", int(self.row_idx))
        object.__setattr__(self, "fx_N", float(np.float32(self.fx_N)))
        object.__setattr__(self, "fy_N", float(np.float32(self.fy_N)))
        object.__setattr__(self, "fz_N", float(np.float32(self.fz_N)))
        object.__setattr__(self, "norm_N", float(np.float32(self.norm_N)))
        object.__setattr__(
            self,
            "arc_length_from_distal_mm",
            (
                None
                if self.arc_length_from_distal_mm is None
                else float(np.float32(self.arc_length_from_distal_mm))
            ),
        )
        object.__setattr__(self, "is_tip", bool(self.is_tip))
        object.__setattr__(
            self,
            "world_pos",
            None if self.world_pos is None else _as_float32_triplet(self.world_pos),
        )
        for field_name in ("fx_scene", "fy_scene", "fz_scene", "norm_scene"):
            value = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                None if value is None else float(np.float32(value)),
            )
        object.__setattr__(self, "mapped", bool(self.mapped))


@dataclass(frozen=True)
class CSREncodedWire:
    """CSR-style ragged wire-contact storage for per-step random access."""

    step_offsets: NDArray[np.int32]
    wire_dof_index: NDArray[np.int32]
    wire_dof_force_xyz_N: NDArray[np.float32]
    arc_length_from_distal_mm: NDArray[np.float32]
    is_tip: NDArray[np.bool_]

    @property
    def n_steps(self) -> int:
        """Return the number of encoded simulation steps."""

        return max(int(self.step_offsets.shape[0]) - 1, 0)


@dataclass(frozen=True)
class CSREncodedTriangle:
    """CSR-style ragged triangle-contact storage for per-step random access."""

    step_offsets: NDArray[np.int32]
    triangle_id: NDArray[np.int32]
    triangle_force_xyz_N: NDArray[np.float32]

    @property
    def n_steps(self) -> int:
        """Return the number of encoded simulation steps."""

        return max(int(self.step_offsets.shape[0]) - 1, 0)


@dataclass(frozen=True)
class ScenarioConfig:
    """Recorder-ready immutable trial metadata for one trace file."""

    anatomy_id: str
    wire_id: str
    target_spec_json: str
    env_seed: int
    policy_seed: Optional[int]
    dt_s: float
    friction_mu: float
    tip_threshold_mm: float
    max_episode_steps: int
    mesh_ref: str
    eval_v2_sha: str = "unknown"
    sofa_version: str = "unknown"
    created_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "anatomy_id", str(self.anatomy_id))
        object.__setattr__(self, "wire_id", str(self.wire_id))
        object.__setattr__(self, "target_spec_json", str(self.target_spec_json))
        object.__setattr__(self, "env_seed", int(self.env_seed))
        object.__setattr__(
            self,
            "policy_seed",
            None if self.policy_seed is None else int(self.policy_seed),
        )
        object.__setattr__(self, "dt_s", float(self.dt_s))
        object.__setattr__(self, "friction_mu", float(self.friction_mu))
        object.__setattr__(self, "tip_threshold_mm", float(self.tip_threshold_mm))
        object.__setattr__(self, "max_episode_steps", int(self.max_episode_steps))
        object.__setattr__(self, "mesh_ref", str(self.mesh_ref))
        object.__setattr__(self, "eval_v2_sha", str(self.eval_v2_sha))
        object.__setattr__(self, "sofa_version", str(self.sofa_version))
        object.__setattr__(
            self,
            "created_at",
            (
                str(self.created_at)
                if str(self.created_at).strip()
                else datetime.now(timezone.utc).isoformat()
            ),
        )


@dataclass(frozen=True)
class SceneStaticState:
    """Static scene state captured once per trial in schema units."""

    wire_initial_position_mm: NDArray[np.float32]
    wire_initial_rotation_quat: NDArray[np.float32]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "wire_initial_position_mm",
            np.asarray(self.wire_initial_position_mm, dtype=np.float32).reshape(
                (-1, 3)
            ),
        )
        object.__setattr__(
            self,
            "wire_initial_rotation_quat",
            np.asarray(self.wire_initial_rotation_quat, dtype=np.float32).reshape((4,)),
        )


@dataclass(frozen=True)
class StepData:
    """One persisted simulation step in schema units."""

    step_index: int
    sim_time_s: float
    wire_positions_mm: NDArray[np.float32]
    wire_collision_positions_mm: NDArray[np.float32]
    action: NDArray[np.float32]
    total_wall_force_N: float
    tip_force_norm_N: float
    contact_count: int
    scoreable: bool
    wire_contacts: Tuple[WireContactRecord, ...] = ()
    triangle_contacts: Tuple[TriangleContactRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_index", int(self.step_index))
        object.__setattr__(self, "sim_time_s", float(np.float32(self.sim_time_s)))
        object.__setattr__(
            self,
            "wire_positions_mm",
            np.asarray(self.wire_positions_mm, dtype=np.float32).reshape((-1, 3)),
        )
        object.__setattr__(
            self,
            "wire_collision_positions_mm",
            np.asarray(self.wire_collision_positions_mm, dtype=np.float32).reshape(
                (-1, 3)
            ),
        )
        object.__setattr__(
            self, "action", np.asarray(self.action, dtype=np.float32).reshape((-1,))
        )
        object.__setattr__(
            self,
            "total_wall_force_N",
            float(np.float32(self.total_wall_force_N)),
        )
        object.__setattr__(
            self,
            "tip_force_norm_N",
            float(np.float32(self.tip_force_norm_N)),
        )
        object.__setattr__(self, "contact_count", int(self.contact_count))
        object.__setattr__(self, "scoreable", bool(self.scoreable))
        object.__setattr__(self, "wire_contacts", tuple(self.wire_contacts))
        object.__setattr__(self, "triangle_contacts", tuple(self.triangle_contacts))


def _records_to_array(
    records: Sequence[dict[str, Any]], key: str, *, dtype: Any
) -> np.ndarray:
    return np.asarray([record.get(key) for record in records], dtype=dtype).reshape(
        (-1,)
    )


def _records_to_json(records: Sequence[dict[str, Any]]) -> str:
    return json.dumps([dict(record) for record in records], sort_keys=True)


def _as_float32_triplet(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float32).reshape((3,))
    return (float(array[0]), float(array[1]), float(array[2]))


def _build_step_offsets(step_lengths: Sequence[int]) -> NDArray[np.int32]:
    offsets = [0]
    running_total = 0
    for length in step_lengths:
        running_total += int(length)
        offsets.append(running_total)
    return np.asarray(offsets, dtype=np.int32)


def encode_wire_contacts_csr(
    per_step_records: list[list[WireContactRecord]],
) -> CSREncodedWire:
    """Encode ragged per-step wire-contact rows into CSR arrays."""

    if not per_step_records:
        empty_int = np.zeros((0,), dtype=np.int32)
        empty_vec = np.zeros((0, 3), dtype=np.float32)
        empty_bool = np.zeros((0,), dtype=np.bool_)
        return CSREncodedWire(
            step_offsets=np.asarray([0], dtype=np.int32),
            wire_dof_index=empty_int,
            wire_dof_force_xyz_N=empty_vec,
            arc_length_from_distal_mm=np.zeros((0,), dtype=np.float32),
            is_tip=empty_bool,
        )

    wire_dof_index = [
        int(record.wire_collision_dof)
        for step_records in per_step_records
        for record in step_records
    ]
    wire_dof_force_xyz_N = [
        _as_float32_triplet((record.fx_N, record.fy_N, record.fz_N))
        for step_records in per_step_records
        for record in step_records
    ]
    arc_length_from_distal_mm = [
        (
            0.0
            if record.arc_length_from_distal_mm is None
            else float(record.arc_length_from_distal_mm)
        )
        for step_records in per_step_records
        for record in step_records
    ]
    is_tip = [
        bool(record.is_tip)
        for step_records in per_step_records
        for record in step_records
    ]
    return CSREncodedWire(
        step_offsets=_build_step_offsets([len(step) for step in per_step_records]),
        wire_dof_index=np.asarray(wire_dof_index, dtype=np.int32).reshape((-1,)),
        wire_dof_force_xyz_N=np.asarray(wire_dof_force_xyz_N, dtype=np.float32).reshape(
            (-1, 3)
        ),
        arc_length_from_distal_mm=np.asarray(
            arc_length_from_distal_mm, dtype=np.float32
        ).reshape((-1,)),
        is_tip=np.asarray(is_tip, dtype=np.bool_).reshape((-1,)),
    )


def decode_wire_step_from_csr(
    encoded: CSREncodedWire,
    step_index: int,
) -> list[WireContactRecord]:
    """Decode one step's wire-contact rows from CSR arrays."""

    if step_index < 0 or step_index >= encoded.n_steps:
        raise IndexError(
            f"step_index out of range: {step_index} for {encoded.n_steps} steps"
        )
    start = int(encoded.step_offsets[step_index])
    end = int(encoded.step_offsets[step_index + 1])
    return [
        WireContactRecord(
            timestep=step_index,
            wire_collision_dof=int(encoded.wire_dof_index[index]),
            row_idx=-1,
            fx_N=float(encoded.wire_dof_force_xyz_N[index, 0]),
            fy_N=float(encoded.wire_dof_force_xyz_N[index, 1]),
            fz_N=float(encoded.wire_dof_force_xyz_N[index, 2]),
            norm_N=float(np.linalg.norm(encoded.wire_dof_force_xyz_N[index])),
            arc_length_from_distal_mm=float(encoded.arc_length_from_distal_mm[index]),
            is_tip=bool(encoded.is_tip[index]),
        )
        for index in range(start, end)
    ]


def encode_triangle_contacts_csr(
    per_step_records: list[list[TriangleContactRecord]],
) -> CSREncodedTriangle:
    """Encode ragged per-step triangle-contact rows into CSR arrays."""

    if not per_step_records:
        empty_int = np.zeros((0,), dtype=np.int32)
        empty_vec = np.zeros((0, 3), dtype=np.float32)
        return CSREncodedTriangle(
            step_offsets=np.asarray([0], dtype=np.int32),
            triangle_id=empty_int,
            triangle_force_xyz_N=empty_vec,
        )

    triangle_id = [
        int(record.triangle_id)
        for step_records in per_step_records
        for record in step_records
    ]
    triangle_force_xyz_N = [
        _as_float32_triplet((record.fx_N, record.fy_N, record.fz_N))
        for step_records in per_step_records
        for record in step_records
    ]
    return CSREncodedTriangle(
        step_offsets=_build_step_offsets([len(step) for step in per_step_records]),
        triangle_id=np.asarray(triangle_id, dtype=np.int32).reshape((-1,)),
        triangle_force_xyz_N=np.asarray(triangle_force_xyz_N, dtype=np.float32).reshape(
            (-1, 3)
        ),
    )


def decode_triangle_step_from_csr(
    encoded: CSREncodedTriangle,
    step_index: int,
) -> list[TriangleContactRecord]:
    """Decode one step's triangle-contact rows from CSR arrays."""

    if step_index < 0 or step_index >= encoded.n_steps:
        raise IndexError(
            f"step_index out of range: {step_index} for {encoded.n_steps} steps"
        )
    start = int(encoded.step_offsets[step_index])
    end = int(encoded.step_offsets[step_index + 1])
    return [
        TriangleContactRecord(
            timestep=step_index,
            triangle_id=int(encoded.triangle_id[index]),
            fx_N=float(encoded.triangle_force_xyz_N[index, 0]),
            fy_N=float(encoded.triangle_force_xyz_N[index, 1]),
            fz_N=float(encoded.triangle_force_xyz_N[index, 2]),
            norm_N=float(np.linalg.norm(encoded.triangle_force_xyz_N[index])),
            contributing_rows=1,
        )
        for index in range(start, end)
    ]


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for trial trace persistence; install h5py>=3.0"
        ) from exc
    return h5py


def _validate_scenario_config(scenario: ScenarioConfig) -> None:
    required_text_fields = {
        "anatomy_id": scenario.anatomy_id,
        "wire_id": scenario.wire_id,
        "target_spec_json": scenario.target_spec_json,
        "mesh_ref": scenario.mesh_ref,
        "eval_v2_sha": scenario.eval_v2_sha,
        "sofa_version": scenario.sofa_version,
        "created_at": scenario.created_at,
    }
    for field_name, value in required_text_fields.items():
        if not str(value).strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if scenario.dt_s <= 0.0:
        raise ValueError(f"dt_s must be > 0, got {scenario.dt_s}")
    if scenario.tip_threshold_mm <= 0.0:
        raise ValueError(
            f"tip_threshold_mm must be > 0, got {scenario.tip_threshold_mm}"
        )
    if scenario.max_episode_steps < 1:
        raise ValueError(
            f"max_episode_steps must be >= 1, got {scenario.max_episode_steps}"
        )


def _scalar_chunk_shape() -> tuple[int]:
    return (SMALL_TIMESERIES_CHUNK_LENGTH,)


class TrialTraceRecorder:
    """Context-managed writer for one trial trace file in schema v2."""

    def __init__(
        self,
        path: Path,
        scenario: ScenarioConfig,
        scene_static: SceneStaticState,
        flush_interval_steps: int = DEFAULT_FLUSH_INTERVAL_STEPS,
    ) -> None:
        self._h5py = _import_h5py()
        self._path = Path(path)
        self._scenario = scenario
        self._scene_static = scene_static
        self._flush_interval_steps = int(flush_interval_steps)
        if self._flush_interval_steps < 1:
            raise ValueError(
                "flush_interval_steps must be >= 1, "
                f"got {self._flush_interval_steps}"
            )
        _validate_scenario_config(self._scenario)

        self._buffered_steps: list[StepData] = []
        self._file: Optional[h5py.File] = None
        self._step_count = 0
        self._wire_contact_count = 0
        self._triangle_contact_count = 0
        self._n_dofs: Optional[int] = None
        self._n_collision_dofs: Optional[int] = None
        self._action_dim: Optional[int] = None
        self._steps_group: Any = None
        self._contacts_group: Any = None

    def __enter__(self) -> "TrialTraceRecorder":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._h5py.File(self._path, "w")
        self._file.create_group("meta")
        self._file.create_group("scenario")
        self._file.create_group("scene_static")
        self._file.create_group("steps")
        self._file.create_group("contacts")
        self._file.create_group("diagnostics")
        self._steps_group = self._file["steps"]
        self._contacts_group = self._file["contacts"]
        self._write_attributes()
        self._write_scene_static()
        return self

    def add_step(self, step_data: StepData) -> None:
        """Buffer one step and flush when the configured interval is reached."""

        self._validate_step_shape(step_data)
        self._buffered_steps.append(step_data)
        if len(self._buffered_steps) >= self._flush_interval_steps:
            self._flush_buffer()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._file is None:
            return
        try:
            if self._buffered_steps:
                self._flush_buffer()
            if exc_type is None:
                self._file["meta"].attrs["trial_status"] = "complete"
        finally:
            self._file.close()
            self._file = None

    def _write_attributes(self) -> None:
        assert self._file is not None
        meta = self._file["meta"]
        meta.attrs["schema_version"] = TRACE_SCHEMA_VERSION
        meta.attrs["eval_v2_sha"] = self._scenario.eval_v2_sha
        meta.attrs["sofa_version"] = self._scenario.sofa_version
        meta.attrs["created_at"] = self._scenario.created_at
        meta.attrs["trial_status"] = "partial"

        scenario = self._file["scenario"]
        scenario.attrs["anatomy_id"] = self._scenario.anatomy_id
        scenario.attrs["wire_id"] = self._scenario.wire_id
        scenario.attrs["target_spec_json"] = self._scenario.target_spec_json
        scenario.attrs["env_seed"] = self._scenario.env_seed
        if self._scenario.policy_seed is not None:
            scenario.attrs["policy_seed"] = self._scenario.policy_seed
        scenario.attrs["dt_s"] = self._scenario.dt_s
        scenario.attrs["friction_mu"] = self._scenario.friction_mu
        scenario.attrs["tip_threshold_mm"] = self._scenario.tip_threshold_mm
        scenario.attrs["max_episode_steps"] = self._scenario.max_episode_steps
        scenario.attrs["mesh_ref"] = self._scenario.mesh_ref

    def _write_scene_static(self) -> None:
        assert self._file is not None
        scene_static = self._file["scene_static"]
        scene_static.create_dataset(
            "wire_initial_position",
            data=self._scene_static.wire_initial_position_mm.astype(np.float32),
            dtype=np.float32,
        )
        scene_static.create_dataset(
            "wire_initial_rotation",
            data=self._scene_static.wire_initial_rotation_quat.astype(np.float32),
            dtype=np.float32,
        )

    def _validate_step_shape(self, step_data: StepData) -> None:
        if self._n_dofs is None:
            self._n_dofs = int(step_data.wire_positions_mm.shape[0])
            self._n_collision_dofs = int(step_data.wire_collision_positions_mm.shape[0])
            self._action_dim = int(step_data.action.shape[0])
            return
        if step_data.wire_positions_mm.shape != (self._n_dofs, 3):
            raise ValueError(
                "wire_positions_mm shape mismatch: "
                f"expected {(self._n_dofs, 3)}, got {step_data.wire_positions_mm.shape}"
            )
        if step_data.wire_collision_positions_mm.shape != (
            self._n_collision_dofs,
            3,
        ):
            raise ValueError(
                "wire_collision_positions_mm shape mismatch: "
                f"expected {(self._n_collision_dofs, 3)}, got "
                f"{step_data.wire_collision_positions_mm.shape}"
            )
        if step_data.action.shape != (self._action_dim,):
            raise ValueError(
                "action shape mismatch: "
                f"expected {(self._action_dim,)}, got {step_data.action.shape}"
            )

    def _ensure_step_datasets(self) -> None:
        assert self._steps_group is not None
        assert self._n_dofs is not None
        assert self._n_collision_dofs is not None
        assert self._action_dim is not None
        if "step_index" in self._steps_group:
            return
        self._steps_group.create_dataset(
            "step_index",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=_scalar_chunk_shape(),
        )
        self._steps_group.create_dataset(
            "sim_time_s",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=_scalar_chunk_shape(),
        )
        self._steps_group.create_dataset(
            "wire_positions",
            shape=(0, self._n_dofs, 3),
            maxshape=(None, self._n_dofs, 3),
            dtype=np.float32,
            chunks=(STEP_DATASET_CHUNK_LEADING_DIM, self._n_dofs, 3),
            compression="gzip",
            compression_opts=DEFAULT_GZIP_LEVEL,
        )
        self._steps_group.create_dataset(
            "wire_collision_positions",
            shape=(0, self._n_collision_dofs, 3),
            maxshape=(None, self._n_collision_dofs, 3),
            dtype=np.float32,
            chunks=(STEP_DATASET_CHUNK_LEADING_DIM, self._n_collision_dofs, 3),
            compression="gzip",
            compression_opts=DEFAULT_GZIP_LEVEL,
        )
        self._steps_group.create_dataset(
            "action",
            shape=(0, self._action_dim),
            maxshape=(None, self._action_dim),
            dtype=np.float32,
            chunks=(STEP_DATASET_CHUNK_LEADING_DIM, self._action_dim),
        )
        for name, dtype in (
            ("total_wall_force_N", np.float32),
            ("tip_force_norm_N", np.float32),
            ("contact_count", np.int16),
            ("scoreable", np.bool_),
        ):
            self._steps_group.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=_scalar_chunk_shape(),
            )

    def _ensure_contact_datasets(self) -> None:
        assert self._contacts_group is not None
        if "wire" in self._contacts_group and "triangle" in self._contacts_group:
            return
        wire_group = self._contacts_group.create_group("wire")
        triangle_group = self._contacts_group.create_group("triangle")

        wire_group.create_dataset(
            "step_offsets",
            data=np.asarray([0], dtype=np.int32),
            maxshape=(None,),
            dtype=np.int32,
            chunks=_scalar_chunk_shape(),
            compression="gzip",
            compression_opts=DEFAULT_GZIP_LEVEL,
        )
        wire_group.create_dataset(
            "wire_dof_index",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=_scalar_chunk_shape(),
            compression="gzip",
            compression_opts=DEFAULT_GZIP_LEVEL,
        )
        wire_group.create_dataset(
            "wire_dof_force_xyz_N",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float32,
            chunks=(STEP_DATASET_CHUNK_LEADING_DIM, 3),
            compression="gzip",
            compression_opts=DEFAULT_GZIP_LEVEL,
        )
        wire_group.create_dataset(
            "arc_length_from_distal_mm",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=_scalar_chunk_shape(),
            compression="gzip",
            compression_opts=DEFAULT_GZIP_LEVEL,
        )
        wire_group.create_dataset(
            "is_tip",
            shape=(0,),
            maxshape=(None,),
            dtype=np.bool_,
            chunks=_scalar_chunk_shape(),
            compression="gzip",
            compression_opts=DEFAULT_GZIP_LEVEL,
        )
        triangle_group.create_dataset(
            "step_offsets",
            data=np.asarray([0], dtype=np.int32),
            maxshape=(None,),
            dtype=np.int32,
            chunks=_scalar_chunk_shape(),
            compression="gzip",
            compression_opts=DEFAULT_GZIP_LEVEL,
        )
        triangle_group.create_dataset(
            "triangle_id",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=_scalar_chunk_shape(),
            compression="gzip",
            compression_opts=DEFAULT_GZIP_LEVEL,
        )
        triangle_group.create_dataset(
            "triangle_force_xyz_N",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float32,
            chunks=(STEP_DATASET_CHUNK_LEADING_DIM, 3),
            compression="gzip",
            compression_opts=DEFAULT_GZIP_LEVEL,
        )

    def _flush_buffer(self) -> None:
        if not self._buffered_steps:
            return
        self._ensure_step_datasets()
        self._ensure_contact_datasets()
        assert self._steps_group is not None
        assert self._contacts_group is not None

        chunk_steps = tuple(self._buffered_steps)
        n_new_steps = len(chunk_steps)
        start = self._step_count
        end = start + n_new_steps

        step_index = np.asarray(
            [step.step_index for step in chunk_steps], dtype=np.int32
        ).reshape((-1,))
        sim_time_s = np.asarray(
            [step.sim_time_s for step in chunk_steps], dtype=np.float32
        ).reshape((-1,))
        wire_positions = np.stack(
            [step.wire_positions_mm for step in chunk_steps], axis=0
        ).astype(np.float32)
        wire_collision_positions = np.stack(
            [step.wire_collision_positions_mm for step in chunk_steps], axis=0
        ).astype(np.float32)
        action = np.stack([step.action for step in chunk_steps], axis=0).astype(
            np.float32
        )
        total_wall_force_N = np.asarray(
            [step.total_wall_force_N for step in chunk_steps], dtype=np.float32
        )
        tip_force_norm_N = np.asarray(
            [step.tip_force_norm_N for step in chunk_steps], dtype=np.float32
        )
        contact_count = np.asarray(
            [step.contact_count for step in chunk_steps], dtype=np.int16
        )
        scoreable = np.asarray([step.scoreable for step in chunk_steps], dtype=np.bool_)

        for name, values in (
            ("step_index", step_index),
            ("sim_time_s", sim_time_s),
            ("wire_positions", wire_positions),
            ("wire_collision_positions", wire_collision_positions),
            ("action", action),
            ("total_wall_force_N", total_wall_force_N),
            ("tip_force_norm_N", tip_force_norm_N),
            ("contact_count", contact_count),
            ("scoreable", scoreable),
        ):
            dataset = self._steps_group[name]
            dataset.resize((end,) + dataset.shape[1:])
            dataset[start:end] = values

        wire_group = self._contacts_group["wire"]
        triangle_group = self._contacts_group["triangle"]

        wire_encoded = encode_wire_contacts_csr(
            [list(step.wire_contacts) for step in chunk_steps]
        )
        wire_contact_start = self._wire_contact_count
        wire_contact_end = wire_contact_start + int(
            wire_encoded.wire_dof_index.shape[0]
        )
        for name, values in (
            ("wire_dof_index", wire_encoded.wire_dof_index),
            ("wire_dof_force_xyz_N", wire_encoded.wire_dof_force_xyz_N),
            ("arc_length_from_distal_mm", wire_encoded.arc_length_from_distal_mm),
            ("is_tip", wire_encoded.is_tip),
        ):
            dataset = wire_group[name]
            dataset.resize((wire_contact_end,) + dataset.shape[1:])
            dataset[wire_contact_start:wire_contact_end] = values
        wire_step_offsets = wire_group["step_offsets"]
        wire_existing_offsets = int(wire_step_offsets.shape[0])
        wire_step_offsets.resize((wire_existing_offsets + n_new_steps,))
        wire_step_offsets[-n_new_steps:] = wire_encoded.step_offsets[1:] + np.int32(
            wire_contact_start
        )

        triangle_encoded = encode_triangle_contacts_csr(
            [list(step.triangle_contacts) for step in chunk_steps]
        )
        triangle_contact_start = self._triangle_contact_count
        triangle_contact_end = triangle_contact_start + int(
            triangle_encoded.triangle_id.shape[0]
        )
        for name, values in (
            ("triangle_id", triangle_encoded.triangle_id),
            ("triangle_force_xyz_N", triangle_encoded.triangle_force_xyz_N),
        ):
            dataset = triangle_group[name]
            dataset.resize((triangle_contact_end,) + dataset.shape[1:])
            dataset[triangle_contact_start:triangle_contact_end] = values
        triangle_step_offsets = triangle_group["step_offsets"]
        triangle_existing_offsets = int(triangle_step_offsets.shape[0])
        triangle_step_offsets.resize((triangle_existing_offsets + n_new_steps,))
        triangle_step_offsets[-n_new_steps:] = triangle_encoded.step_offsets[
            1:
        ] + np.int32(triangle_contact_start)

        self._step_count = end
        self._wire_contact_count = wire_contact_end
        self._triangle_contact_count = triangle_contact_end
        self._buffered_steps.clear()


class TraceFileCorruptError(RuntimeError):
    """Raised when a trace file cannot be opened as a valid schema-v2 HDF5 file."""


class TraceReader:
    """Context-managed HDF5 reader with full-load and per-step access modes."""

    def __init__(self, path: Path) -> None:
        self._h5py = _import_h5py()
        self._path = Path(path)
        self._file: Optional[h5py.File] = None
        self._steps_group: Any = None
        self._contacts_group: Any = None
        self.is_partial = False

    def __enter__(self) -> "TraceReader":
        try:
            self._file = self._h5py.File(self._path, "r")
            self._steps_group = self._file["steps"]
            self._contacts_group = self._file["contacts"]
            self.is_partial = (
                str(self._file["meta"].attrs.get("trial_status", "partial"))
                == "partial"
            )
        except Exception as exc:
            raise TraceFileCorruptError(
                f"corrupt or unreadable trace file: {self._path}"
            ) from exc
        return self

    def load_all(self) -> dict[str, Any]:
        """Load the full trace into memory with bulk dataset reads."""

        self._require_open()
        assert self._file is not None
        return {
            "meta": dict(self._file["meta"].attrs),
            "scenario": dict(self._file["scenario"].attrs),
            "scene_static": {
                "wire_initial_position": self._read_all_dataset(
                    "scene_static/wire_initial_position"
                ),
                "wire_initial_rotation": self._read_all_dataset(
                    "scene_static/wire_initial_rotation"
                ),
            },
            "steps": {
                "step_index": self._read_all_dataset("steps/step_index"),
                "sim_time_s": self._read_all_dataset("steps/sim_time_s"),
                "wire_positions": self._read_all_dataset("steps/wire_positions"),
                "wire_collision_positions": self._read_all_dataset(
                    "steps/wire_collision_positions"
                ),
                "action": self._read_all_dataset("steps/action"),
                "total_wall_force_N": self._read_all_dataset(
                    "steps/total_wall_force_N"
                ),
                "tip_force_norm_N": self._read_all_dataset("steps/tip_force_norm_N"),
                "contact_count": self._read_all_dataset("steps/contact_count"),
                "scoreable": self._read_all_dataset("steps/scoreable"),
            },
            "contacts": {
                "wire": {
                    "step_offsets": self._read_all_dataset(
                        "contacts/wire/step_offsets"
                    ),
                    "wire_dof_index": self._read_all_dataset(
                        "contacts/wire/wire_dof_index"
                    ),
                    "wire_dof_force_xyz_N": self._read_all_dataset(
                        "contacts/wire/wire_dof_force_xyz_N"
                    ),
                    "arc_length_from_distal_mm": self._read_all_dataset(
                        "contacts/wire/arc_length_from_distal_mm"
                    ),
                    "is_tip": self._read_all_dataset("contacts/wire/is_tip"),
                },
                "triangle": {
                    "step_offsets": self._read_all_dataset(
                        "contacts/triangle/step_offsets"
                    ),
                    "triangle_id": self._read_all_dataset(
                        "contacts/triangle/triangle_id"
                    ),
                    "triangle_force_xyz_N": self._read_all_dataset(
                        "contacts/triangle/triangle_force_xyz_N"
                    ),
                },
            },
            "is_partial": self.is_partial,
        }

    def step(self, k: int) -> StepData:
        """Load one step's dense data and ragged contacts without full-file reads."""

        self._require_open()
        return StepData(
            step_index=int(k),
            sim_time_s=float(self._read_step_dataset("sim_time_s", k)),
            wire_positions_mm=self._read_step_dataset("wire_positions", k),
            wire_collision_positions_mm=self._read_step_dataset(
                "wire_collision_positions", k
            ),
            action=self._read_step_dataset("action", k),
            total_wall_force_N=float(self._read_step_dataset("total_wall_force_N", k)),
            tip_force_norm_N=float(self._read_step_dataset("tip_force_norm_N", k)),
            contact_count=int(self._read_step_dataset("contact_count", k)),
            scoreable=bool(self._read_step_dataset("scoreable", k)),
            wire_contacts=tuple(self.wire_contacts_for_step(k)),
            triangle_contacts=tuple(self.triangle_contacts_for_step(k)),
        )

    def wire_contacts_for_step(self, k: int) -> list[WireContactRecord]:
        """Return the ragged wire-contact rows for step `k` using CSR offsets."""

        self._require_open()
        assert self._contacts_group is not None
        wire_group = self._contacts_group["wire"]
        step_offsets = wire_group["step_offsets"]
        n_steps = int(step_offsets.shape[0] - 1)
        if k < 0 or k >= n_steps:
            raise IndexError(f"step_index out of range: {k} for {n_steps} steps")
        start = int(step_offsets[k])
        end = int(step_offsets[k + 1])
        wire_dof_index = np.asarray(wire_group["wire_dof_index"][start:end])
        wire_dof_force_xyz_N = np.asarray(wire_group["wire_dof_force_xyz_N"][start:end])
        arc_length_from_distal_mm = np.asarray(
            wire_group["arc_length_from_distal_mm"][start:end]
        )
        is_tip = np.asarray(wire_group["is_tip"][start:end])
        return [
            WireContactRecord(
                timestep=k,
                wire_collision_dof=int(wire_dof_index[index]),
                row_idx=-1,
                fx_N=float(wire_dof_force_xyz_N[index, 0]),
                fy_N=float(wire_dof_force_xyz_N[index, 1]),
                fz_N=float(wire_dof_force_xyz_N[index, 2]),
                norm_N=float(np.linalg.norm(wire_dof_force_xyz_N[index])),
                arc_length_from_distal_mm=float(arc_length_from_distal_mm[index]),
                is_tip=bool(is_tip[index]),
            )
            for index in range(end - start)
        ]

    def triangle_contacts_for_step(self, k: int) -> list[TriangleContactRecord]:
        """Return the ragged triangle-contact rows for step `k` using CSR offsets."""

        self._require_open()
        assert self._contacts_group is not None
        triangle_group = self._contacts_group["triangle"]
        step_offsets = triangle_group["step_offsets"]
        n_steps = int(step_offsets.shape[0] - 1)
        if k < 0 or k >= n_steps:
            raise IndexError(f"step_index out of range: {k} for {n_steps} steps")
        start = int(step_offsets[k])
        end = int(step_offsets[k + 1])
        triangle_id = np.asarray(triangle_group["triangle_id"][start:end])
        triangle_force_xyz_N = np.asarray(
            triangle_group["triangle_force_xyz_N"][start:end]
        )
        return [
            TriangleContactRecord(
                timestep=k,
                triangle_id=int(triangle_id[index]),
                fx_N=float(triangle_force_xyz_N[index, 0]),
                fy_N=float(triangle_force_xyz_N[index, 1]),
                fz_N=float(triangle_force_xyz_N[index, 2]),
                norm_N=float(np.linalg.norm(triangle_force_xyz_N[index])),
                contributing_rows=1,
            )
            for index in range(end - start)
        ]

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _require_open(self) -> None:
        if self._file is None:
            raise RuntimeError("TraceReader must be opened with a context manager")

    def _read_all_dataset(self, dataset_path: str) -> np.ndarray:
        assert self._file is not None
        return np.asarray(self._file[dataset_path][:])

    def _read_step_dataset(self, dataset_name: str, step_index: int) -> np.ndarray:
        assert self._steps_group is not None
        return np.asarray(self._steps_group[dataset_name][step_index])


def write_anatomy_mesh(
    path: Path,
    triangle_indices: NDArray[np.int32],
    vertex_positions: NDArray[np.float32],
    anatomy_id: str,
    *,
    overwrite: bool = False,
) -> None:
    """Write one anatomy mesh HDF5 file with atomic publish semantics."""

    h5py = _import_h5py()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"mesh file already exists: {out_path}")

    triangle_array = np.asarray(triangle_indices, dtype=np.int32).reshape((-1, 3))
    vertex_array = np.asarray(vertex_positions, dtype=np.float32).reshape((-1, 3))
    temp_path = out_path.with_name(f"{out_path.name}.tmp_{os.getpid()}")

    try:
        with h5py.File(temp_path, "w") as handle:
            handle.create_dataset(
                "triangle_indices",
                data=triangle_array,
                dtype=np.int32,
            )
            handle.create_dataset(
                "vertex_positions",
                data=vertex_array,
                dtype=np.float32,
            )
            handle["/"].attrs["anatomy_id"] = str(anatomy_id)

        if overwrite:
            os.replace(temp_path, out_path)
            return

        os.link(temp_path, out_path)
        os.unlink(temp_path)
    except FileExistsError:
        if temp_path.exists():
            temp_path.unlink()
        raise
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def write_force_trace_npz(
    path: Path | str,
    *,
    triangle_records: Sequence[dict[str, Any]] = (),
    wire_records: Sequence[dict[str, Any]] = (),
    metadata: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    triangle_records = tuple(triangle_records)
    wire_records = tuple(wire_records)
    np.savez_compressed(
        out_path,
        schema_version=np.asarray([LEGACY_TRACE_SCHEMA_VERSION], dtype=np.int64),
        triangle_timestep=_records_to_array(
            triangle_records, "timestep", dtype=np.int64
        ),
        triangle_id=_records_to_array(triangle_records, "triangle_id", dtype=np.int64),
        triangle_fx_N=_records_to_array(triangle_records, "fx_N", dtype=np.float64),
        triangle_fy_N=_records_to_array(triangle_records, "fy_N", dtype=np.float64),
        triangle_fz_N=_records_to_array(triangle_records, "fz_N", dtype=np.float64),
        triangle_norm_N=_records_to_array(triangle_records, "norm_N", dtype=np.float64),
        triangle_contributing_rows=_records_to_array(
            triangle_records, "contributing_rows", dtype=np.int64
        ),
        wire_timestep=_records_to_array(wire_records, "timestep", dtype=np.int64),
        wire_collision_dof=_records_to_array(
            wire_records, "wire_collision_dof", dtype=np.int64
        ),
        wire_row_idx=_records_to_array(wire_records, "row_idx", dtype=np.int64),
        wire_fx_N=_records_to_array(wire_records, "fx_N", dtype=np.float64),
        wire_fy_N=_records_to_array(wire_records, "fy_N", dtype=np.float64),
        wire_fz_N=_records_to_array(wire_records, "fz_N", dtype=np.float64),
        wire_norm_N=_records_to_array(wire_records, "norm_N", dtype=np.float64),
        triangle_records_json=np.asarray(_records_to_json(triangle_records)),
        wire_records_json=np.asarray(_records_to_json(wire_records)),
        metadata_json=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    )
    return out_path


def read_force_trace_npz(path: Path | str) -> dict[str, Any]:
    in_path = Path(path)
    with np.load(in_path, allow_pickle=True) as data:
        schema_version = int(np.asarray(data["schema_version"]).reshape((-1,))[0])
        triangle_records = json.loads(
            str(np.asarray(data["triangle_records_json"]).reshape(()))
        )
        wire_records = json.loads(
            str(np.asarray(data["wire_records_json"]).reshape(()))
        )
        metadata = json.loads(str(np.asarray(data["metadata_json"]).reshape(())))
        return {
            "schema_version": schema_version,
            "triangle_records": triangle_records,
            "wire_records": wire_records,
            "metadata": metadata,
            "triangle_timestep": np.asarray(data["triangle_timestep"]),
            "wire_timestep": np.asarray(data["wire_timestep"]),
        }


def write_force_trace_jsonl(
    path: Path | str,
    records: Iterable[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "record_type": "header",
                    "schema_version": LEGACY_TRACE_SCHEMA_VERSION,
                    "metadata": metadata or {},
                },
                sort_keys=True,
            )
            + "\n"
        )
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
    return out_path


def read_force_trace_jsonl(path: Path | str) -> dict[str, Any]:
    in_path = Path(path)
    header: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("record_type") == "header" and header is None:
                header = payload
                continue
            records.append(payload)
    return {
        "schema_version": (
            int(header.get("schema_version", LEGACY_TRACE_SCHEMA_VERSION))
            if header
            else LEGACY_TRACE_SCHEMA_VERSION
        ),
        "metadata": (header or {}).get("metadata", {}),
        "records": records,
    }


__all__ = [
    "CSREncodedTriangle",
    "CSREncodedWire",
    "DEFAULT_FLUSH_INTERVAL_STEPS",
    "DEFAULT_GZIP_LEVEL",
    "LEGACY_TRACE_SCHEMA_VERSION",
    "SceneStaticState",
    "SCHEMA_VERSION",
    "STEP_DATASET_CHUNK_LEADING_DIM",
    "ScenarioConfig",
    "StepData",
    "TraceFileCorruptError",
    "TraceReader",
    "TRACE_SCHEMA_VERSION",
    "TriangleContactRecord",
    "TrialTraceRecorder",
    "WireContactRecord",
    "decode_triangle_step_from_csr",
    "decode_wire_step_from_csr",
    "encode_triangle_contacts_csr",
    "encode_wire_contacts_csr",
    "read_force_trace_jsonl",
    "read_force_trace_npz",
    "write_anatomy_mesh",
    "write_force_trace_jsonl",
    "write_force_trace_npz",
]
