from __future__ import annotations

import json
import multiprocessing as mp
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from steve_recommender.eval_v2.force_trace_persistence import (
    DEFAULT_FLUSH_INTERVAL_STEPS,
    DEFAULT_GZIP_LEVEL,
    LEGACY_TRACE_SCHEMA_VERSION,
    STEP_DATASET_CHUNK_LEADING_DIM,
    CSREncodedTriangle,
    CSREncodedWire,
    SceneStaticState,
    ScenarioConfig,
    StepData,
    TRACE_SCHEMA_VERSION,
    TraceFileCorruptError,
    TraceReader,
    TriangleContactRecord,
    TrialTraceRecorder,
    WireContactRecord,
    decode_triangle_step_from_csr,
    decode_wire_step_from_csr,
    encode_triangle_contacts_csr,
    encode_wire_contacts_csr,
    read_force_trace_jsonl,
    read_force_trace_npz,
    write_anatomy_mesh,
    write_force_trace_jsonl,
    write_force_trace_npz,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "steve_recommender"
    / "eval_v2"
    / "tests"
    / "fixtures"
    / "trace_smoke_seed123.json"
)


def _worker_scenario(worker_id: int) -> ScenarioConfig:
    return ScenarioConfig(
        anatomy_id=f"Tree_{worker_id:02d}",
        wire_id="steve_default/standard_j",
        target_spec_json='{"kind": "branch_end", "branches": ["lcca"]}',
        env_seed=100 + worker_id,
        policy_seed=1000 + worker_id,
        dt_s=0.1,
        friction_mu=0.001,
        tip_threshold_mm=3.0,
        max_episode_steps=10,
        mesh_ref=f"../meshes/anatomy_Tree_{worker_id:02d}.h5",
        eval_v2_sha="deadbeef",
        sofa_version="23.06",
        created_at="2026-04-29T10:00:00+00:00",
    )


def _worker_scene_static() -> SceneStaticState:
    return SceneStaticState(
        wire_initial_position_mm=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32
        ),
        wire_initial_rotation_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _worker_wire_contact(worker_id: int, step_index: int) -> WireContactRecord:
    force_xyz = np.asarray([worker_id + 0.4, step_index + 0.5, 0.6], dtype=np.float32)
    return WireContactRecord(
        timestep=step_index,
        wire_collision_dof=step_index % 2,
        row_idx=-1,
        fx_N=float(force_xyz[0]),
        fy_N=float(force_xyz[1]),
        fz_N=float(force_xyz[2]),
        norm_N=float(np.linalg.norm(force_xyz)),
        arc_length_from_distal_mm=float(worker_id + step_index + 1.5),
        is_tip=bool((worker_id + step_index) % 2),
    )


def _worker_triangle_contact(worker_id: int, step_index: int) -> TriangleContactRecord:
    force_xyz = np.asarray([worker_id + 0.1, step_index + 0.2, 0.3], dtype=np.float32)
    return TriangleContactRecord(
        timestep=step_index,
        triangle_id=worker_id * 100 + step_index,
        fx_N=float(force_xyz[0]),
        fy_N=float(force_xyz[1]),
        fz_N=float(force_xyz[2]),
        norm_N=float(np.linalg.norm(force_xyz)),
        contributing_rows=1,
    )


def _worker_step(worker_id: int, step_index: int) -> StepData:
    return StepData(
        step_index=step_index,
        sim_time_s=0.1 * step_index,
        wire_positions_mm=np.asarray(
            [
                [worker_id, step_index, 0.0],
                [worker_id, step_index, 1.0],
            ],
            dtype=np.float32,
        ),
        wire_collision_positions_mm=np.asarray(
            [[worker_id, step_index, 2.0]],
            dtype=np.float32,
        ),
        action=np.asarray([worker_id, -step_index], dtype=np.float32),
        total_wall_force_N=np.float32(worker_id + step_index),
        tip_force_norm_N=np.float32(worker_id + 0.5 * step_index),
        contact_count=2,
        scoreable=True,
        wire_contacts=(_worker_wire_contact(worker_id, step_index),),
        triangle_contacts=(_worker_triangle_contact(worker_id, step_index),),
    )


def _write_worker_trace(
    path_str: str, worker_id: int, crash_after_steps: int | None = None
) -> int:
    path = Path(path_str)
    with TrialTraceRecorder(
        path=path,
        scenario=_worker_scenario(worker_id),
        scene_static=_worker_scene_static(),
        flush_interval_steps=2,
    ) as recorder:
        for step_index in range(5):
            recorder.add_step(_worker_step(worker_id, step_index))
            if crash_after_steps is not None and (step_index + 1) >= crash_after_steps:
                raise RuntimeError(f"simulated worker crash {worker_id}")
    return worker_id


def _write_worker_mesh(path_str: str, worker_id: int, overwrite: bool = False) -> int:
    write_anatomy_mesh(
        Path(path_str),
        triangle_indices=np.asarray([[0, 1, 2]], dtype=np.int32),
        vertex_positions=np.asarray(
            [
                [worker_id, 0.0, 0.0],
                [worker_id, 1.0, 0.0],
                [worker_id, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        anatomy_id=f"Tree_{worker_id:02d}",
        overwrite=overwrite,
    )
    return worker_id


class ForceTracePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.triangle_records = [
            {
                "timestep": 1,
                "triangle_id": 12,
                "fx_N": 0.0,
                "fy_N": 0.0,
                "fz_N": 0.004,
                "norm_N": 0.004,
                "contributing_rows": 1,
            },
        ]
        self.wire_records = [
            {
                "timestep": 1,
                "wire_collision_dof": 7,
                "row_idx": 0,
                "fx_N": 0.0,
                "fy_N": 0.0,
                "fz_N": 0.004,
                "norm_N": 0.004,
            },
        ]

    def test_persistence_roundtrip_preserves_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "trace.npz"
            jsonl_path = Path(tmp) / "trace.jsonl"
            write_force_trace_npz(
                npz_path,
                triangle_records=self.triangle_records,
                wire_records=self.wire_records,
                metadata={"seed": 123},
            )
            write_force_trace_jsonl(
                jsonl_path,
                self.triangle_records + self.wire_records,
                metadata={"seed": 123},
            )

            npz = read_force_trace_npz(npz_path)
            jsonl = read_force_trace_jsonl(jsonl_path)

            self.assertEqual(npz["schema_version"], 1)
            self.assertEqual(npz["metadata"], {"seed": 123})
            self.assertEqual(npz["triangle_records"], self.triangle_records)
            self.assertEqual(npz["wire_records"], self.wire_records)
            self.assertEqual(jsonl["schema_version"], 1)
            self.assertEqual(jsonl["metadata"], {"seed": 123})
            self.assertEqual(
                jsonl["records"], self.triangle_records + self.wire_records
            )

    def test_trace_schema_version_constant_pinned_to_2(self) -> None:
        self.assertEqual(TRACE_SCHEMA_VERSION, 2)

    def test_legacy_schema_version_constant_pinned_to_1(self) -> None:
        self.assertEqual(LEGACY_TRACE_SCHEMA_VERSION, 1)

    def test_v1_reader_still_loads_existing_fixtures(self) -> None:
        fixture_payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(fixture_payload["schema_version"], LEGACY_TRACE_SCHEMA_VERSION)

        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "trace.npz"
            jsonl_path = Path(tmp) / "trace.jsonl"
            write_force_trace_npz(
                npz_path,
                triangle_records=self.triangle_records,
                wire_records=self.wire_records,
                metadata={"seed": 123},
            )
            write_force_trace_jsonl(
                jsonl_path,
                self.triangle_records + self.wire_records,
                metadata={"seed": 123},
            )

            npz = read_force_trace_npz(npz_path)
            jsonl = read_force_trace_jsonl(jsonl_path)

        self.assertEqual(npz["schema_version"], LEGACY_TRACE_SCHEMA_VERSION)
        self.assertEqual(jsonl["schema_version"], LEGACY_TRACE_SCHEMA_VERSION)
        self.assertEqual(npz["triangle_records"], self.triangle_records)
        self.assertEqual(npz["wire_records"], self.wire_records)
        self.assertEqual(jsonl["records"], self.triangle_records + self.wire_records)

    def test_h5py_dependency_importable(self) -> None:
        import h5py

        self.assertTrue(hasattr(h5py, "File"))

    def test_persistence_schema_version_in_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "trace.npz"
            write_force_trace_npz(
                npz_path,
                triangle_records=self.triangle_records,
                wire_records=self.wire_records,
            )
            with npz_path.open("rb") as handle:
                payload = handle.read(16)
            self.assertTrue(
                payload.startswith(b"PK"), "NPZ bundle should be a zip archive"
            )

            npz = read_force_trace_npz(npz_path)
            self.assertEqual(npz["schema_version"], LEGACY_TRACE_SCHEMA_VERSION)

    def test_jsonl_header_contains_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "trace.jsonl"
            write_force_trace_jsonl(
                jsonl_path, self.triangle_records, metadata={"seed": 123}
            )
            header = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(header["record_type"], "header")
            self.assertEqual(header["schema_version"], LEGACY_TRACE_SCHEMA_VERSION)


class ContactCSRTests(unittest.TestCase):
    def _wire_record(
        self,
        *,
        timestep: int,
        wire_collision_dof: int,
        force_xyz_N: tuple[float, float, float] = (0.4, 0.5, 0.6),
        arc_length_from_distal_mm: float = 3.0,
        is_tip: bool = False,
    ) -> WireContactRecord:
        force_xyz = np.asarray(force_xyz_N, dtype=np.float32)
        return WireContactRecord(
            timestep=timestep,
            wire_collision_dof=wire_collision_dof,
            row_idx=-1,
            fx_N=float(force_xyz[0]),
            fy_N=float(force_xyz[1]),
            fz_N=float(force_xyz[2]),
            norm_N=float(np.linalg.norm(force_xyz)),
            arc_length_from_distal_mm=arc_length_from_distal_mm,
            is_tip=is_tip,
        )

    def _triangle_record(
        self,
        *,
        timestep: int,
        triangle_id: int,
        force_xyz_N: tuple[float, float, float] = (0.1, 0.2, 0.3),
    ) -> TriangleContactRecord:
        force_xyz = np.asarray(force_xyz_N, dtype=np.float32)
        return TriangleContactRecord(
            timestep=timestep,
            triangle_id=triangle_id,
            fx_N=float(force_xyz[0]),
            fy_N=float(force_xyz[1]),
            fz_N=float(force_xyz[2]),
            norm_N=float(np.linalg.norm(force_xyz)),
            contributing_rows=1,
        )

    def test_wire_csr_round_trip_preserves_records(self) -> None:
        per_step_records = [
            [self._wire_record(timestep=0, wire_collision_dof=1)],
            [],
            [
                self._wire_record(timestep=2, wire_collision_dof=2, is_tip=True),
                self._wire_record(
                    timestep=2,
                    wire_collision_dof=3,
                    force_xyz_N=(4.0, 5.0, 6.0),
                    arc_length_from_distal_mm=1.5,
                ),
            ],
        ]
        encoded = encode_wire_contacts_csr(per_step_records)
        self.assertEqual(
            [decode_wire_step_from_csr(encoded, index) for index in range(3)],
            [
                [self._wire_record(timestep=0, wire_collision_dof=1)],
                [],
                [
                    self._wire_record(timestep=2, wire_collision_dof=2, is_tip=True),
                    self._wire_record(
                        timestep=2,
                        wire_collision_dof=3,
                        force_xyz_N=(4.0, 5.0, 6.0),
                        arc_length_from_distal_mm=1.5,
                    ),
                ],
            ],
        )

    def test_triangle_csr_round_trip_preserves_records(self) -> None:
        per_step_records = [
            [self._triangle_record(timestep=0, triangle_id=10)],
            [],
            [
                self._triangle_record(timestep=2, triangle_id=20),
                self._triangle_record(
                    timestep=2,
                    triangle_id=21,
                    force_xyz_N=(1.0, 2.0, 3.0),
                ),
            ],
        ]
        encoded = encode_triangle_contacts_csr(per_step_records)
        self.assertEqual(
            [decode_triangle_step_from_csr(encoded, index) for index in range(3)],
            [
                [self._triangle_record(timestep=0, triangle_id=10)],
                [],
                [
                    self._triangle_record(timestep=2, triangle_id=20),
                    self._triangle_record(
                        timestep=2,
                        triangle_id=21,
                        force_xyz_N=(1.0, 2.0, 3.0),
                    ),
                ],
            ],
        )

    def test_wire_and_triangle_step_offsets_are_independent(self) -> None:
        wire = encode_wire_contacts_csr(
            [
                [
                    self._wire_record(timestep=0, wire_collision_dof=index)
                    for index in range(5)
                ]
            ]
        )
        triangle = encode_triangle_contacts_csr(
            [
                [
                    self._triangle_record(timestep=0, triangle_id=index)
                    for index in range(2)
                ]
            ]
        )
        self.assertEqual(wire.step_offsets.tolist(), [0, 5])
        self.assertEqual(triangle.step_offsets.tolist(), [0, 2])
        self.assertEqual(len(decode_wire_step_from_csr(wire, 0)), 5)
        self.assertEqual(len(decode_triangle_step_from_csr(triangle, 0)), 2)

    def test_csr_zero_contact_steps_have_zero_length_slices(self) -> None:
        encoded = encode_wire_contacts_csr(
            [
                [self._wire_record(timestep=0, wire_collision_dof=1)],
                [],
                [self._wire_record(timestep=2, wire_collision_dof=2)],
            ]
        )
        self.assertEqual(decode_wire_step_from_csr(encoded, 1), [])
        self.assertEqual(encoded.step_offsets.shape, (4,))
        self.assertEqual(int(encoded.step_offsets[1]), int(encoded.step_offsets[2]))

    def test_csr_offsets_array_length_is_n_plus_1(self) -> None:
        encoded = encode_triangle_contacts_csr(
            [
                [],
                [self._triangle_record(timestep=1, triangle_id=1)],
                [self._triangle_record(timestep=2, triangle_id=2)],
                [],
            ]
        )
        self.assertEqual(encoded.step_offsets.shape[0], 5)

    def test_csr_handles_empty_trial_zero_steps(self) -> None:
        wire = encode_wire_contacts_csr([])
        triangle = encode_triangle_contacts_csr([])

        self.assertIsInstance(wire, CSREncodedWire)
        self.assertEqual(wire.step_offsets.shape, (1,))
        self.assertEqual(wire.wire_dof_index.shape, (0,))
        self.assertEqual(wire.wire_dof_force_xyz_N.shape, (0, 3))
        self.assertEqual(wire.arc_length_from_distal_mm.shape, (0,))
        self.assertEqual(wire.is_tip.shape, (0,))
        self.assertIsInstance(triangle, CSREncodedTriangle)
        self.assertEqual(triangle.step_offsets.shape, (1,))
        self.assertEqual(triangle.triangle_id.shape, (0,))
        self.assertEqual(triangle.triangle_force_xyz_N.shape, (0, 3))

    def test_csr_dtypes_locked(self) -> None:
        wire = encode_wire_contacts_csr(
            [[self._wire_record(timestep=0, wire_collision_dof=2, is_tip=True)]]
        )
        triangle = encode_triangle_contacts_csr(
            [[self._triangle_record(timestep=0, triangle_id=1)]]
        )

        self.assertEqual(wire.step_offsets.dtype, np.int32)
        self.assertEqual(wire.wire_dof_index.dtype, np.int32)
        self.assertEqual(wire.wire_dof_force_xyz_N.dtype, np.float32)
        self.assertEqual(wire.arc_length_from_distal_mm.dtype, np.float32)
        self.assertEqual(wire.is_tip.dtype, np.bool_)
        self.assertEqual(triangle.step_offsets.dtype, np.int32)
        self.assertEqual(triangle.triangle_id.dtype, np.int32)
        self.assertEqual(triangle.triangle_force_xyz_N.dtype, np.float32)


class _TrialTraceTestHelpers:
    def _scenario(self, **overrides: object) -> ScenarioConfig:
        payload: dict[str, object] = {
            "anatomy_id": "Tree_00",
            "wire_id": "steve_default/standard_j",
            "target_spec_json": '{"kind": "branch_end", "branches": ["lcca"]}',
            "env_seed": 123,
            "policy_seed": 1000,
            "dt_s": 0.1,
            "friction_mu": 0.001,
            "tip_threshold_mm": 3.0,
            "max_episode_steps": 100,
            "mesh_ref": "../meshes/anatomy_Tree_00.h5",
            "eval_v2_sha": "deadbeef",
            "sofa_version": "23.06",
            "created_at": "2026-04-29T10:00:00+00:00",
        }
        payload.update(overrides)
        return ScenarioConfig(**payload)

    def _scene_static(self) -> SceneStaticState:
        return SceneStaticState(
            wire_initial_position_mm=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32
            ),
            wire_initial_rotation_quat=np.asarray(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32
            ),
        )

    def _wire_contact(self, *, step_index: int) -> WireContactRecord:
        force_xyz = np.asarray((0.4, 0.5 * step_index, 0.6), dtype=np.float32)
        return WireContactRecord(
            timestep=step_index,
            wire_collision_dof=step_index % 4,
            row_idx=-1,
            fx_N=float(force_xyz[0]),
            fy_N=float(force_xyz[1]),
            fz_N=float(force_xyz[2]),
            norm_N=float(np.linalg.norm(force_xyz)),
            arc_length_from_distal_mm=1.5 + step_index,
            is_tip=bool(step_index % 2),
        )

    def _triangle_contact(self, *, step_index: int) -> TriangleContactRecord:
        force_xyz = np.asarray((0.1 * step_index, 0.2, 0.3), dtype=np.float32)
        return TriangleContactRecord(
            timestep=step_index,
            triangle_id=10 + step_index,
            fx_N=float(force_xyz[0]),
            fy_N=float(force_xyz[1]),
            fz_N=float(force_xyz[2]),
            norm_N=float(np.linalg.norm(force_xyz)),
            contributing_rows=1,
        )

    def _step(self, step_index: int, *, with_contact: bool = True) -> StepData:
        wire_contacts = (
            [self._wire_contact(step_index=step_index)] if with_contact else []
        )
        triangle_contacts = (
            [self._triangle_contact(step_index=step_index)] if with_contact else []
        )
        return StepData(
            step_index=step_index,
            sim_time_s=0.1 * step_index,
            wire_positions_mm=np.asarray(
                [
                    [step_index, 0.0, 0.0],
                    [step_index, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            wire_collision_positions_mm=np.asarray(
                [
                    [step_index, 0.0, 1.0],
                    [step_index, 1.0, 1.0],
                    [step_index, 2.0, 1.0],
                ],
                dtype=np.float32,
            ),
            action=np.asarray(
                [0.01 * step_index, -0.01 * step_index], dtype=np.float32
            ),
            total_wall_force_N=np.float32(0.5 * step_index),
            tip_force_norm_N=np.float32(0.25 * step_index),
            contact_count=len(wire_contacts) + len(triangle_contacts),
            scoreable=bool(with_contact),
            wire_contacts=tuple(wire_contacts),
            triangle_contacts=tuple(triangle_contacts),
        )

    def _large_step(self, step_index: int) -> StepData:
        return StepData(
            step_index=step_index,
            sim_time_s=0.1 * step_index,
            wire_positions_mm=np.full((128, 3), float(step_index), dtype=np.float32),
            wire_collision_positions_mm=np.full(
                (64, 3), float(step_index) + 1.0, dtype=np.float32
            ),
            action=np.asarray(
                [0.01 * step_index, -0.01 * step_index], dtype=np.float32
            ),
            total_wall_force_N=np.float32(0.5 * step_index),
            tip_force_norm_N=np.float32(0.25 * step_index),
            contact_count=2,
            scoreable=True,
            wire_contacts=(self._wire_contact(step_index=step_index),),
            triangle_contacts=(self._triangle_contact(step_index=step_index),),
        )

    def _write_trace(
        self,
        path: Path,
        *,
        n_steps: int,
        flush_interval_steps: int = DEFAULT_FLUSH_INTERVAL_STEPS,
        with_contact: bool = True,
        scenario: ScenarioConfig | None = None,
    ) -> None:
        with TrialTraceRecorder(
            path=path,
            scenario=self._scenario() if scenario is None else scenario,
            scene_static=self._scene_static(),
            flush_interval_steps=flush_interval_steps,
        ) as recorder:
            for step_index in range(n_steps):
                recorder.add_step(self._step(step_index, with_contact=with_contact))


class TrialTraceRecorderTests(_TrialTraceTestHelpers, unittest.TestCase):
    def test_recorder_writes_all_required_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            self._write_trace(path, n_steps=30)

            with h5py.File(path, "r") as handle:
                self.assertIn("meta", handle)
                self.assertIn("scenario", handle)
                self.assertIn("scene_static", handle)
                self.assertIn("steps", handle)
                self.assertIn("contacts", handle)
                self.assertIn("diagnostics", handle)
                self.assertEqual(
                    handle["scene_static/wire_initial_position"].shape, (2, 3)
                )
                self.assertEqual(
                    handle["scene_static/wire_initial_rotation"].shape, (4,)
                )
                self.assertEqual(handle["steps/step_index"].shape, (30,))
                self.assertEqual(handle["steps/sim_time_s"].shape, (30,))
                self.assertEqual(handle["steps/wire_positions"].shape, (30, 2, 3))
                self.assertEqual(
                    handle["steps/wire_collision_positions"].shape, (30, 3, 3)
                )
                self.assertEqual(handle["steps/action"].shape, (30, 2))
                self.assertEqual(handle["steps/total_wall_force_N"].shape, (30,))
                self.assertEqual(handle["steps/tip_force_norm_N"].shape, (30,))
                self.assertEqual(handle["steps/contact_count"].shape, (30,))
                self.assertEqual(handle["steps/scoreable"].shape, (30,))
                self.assertIn("wire", handle["contacts"])
                self.assertIn("triangle", handle["contacts"])
                self.assertEqual(handle["contacts/wire/step_offsets"].shape, (31,))
                self.assertEqual(handle["contacts/wire/wire_dof_index"].shape, (30,))
                self.assertEqual(
                    handle["contacts/wire/wire_dof_force_xyz_N"].shape, (30, 3)
                )
                self.assertEqual(
                    handle["contacts/wire/arc_length_from_distal_mm"].shape, (30,)
                )
                self.assertEqual(handle["contacts/wire/is_tip"].shape, (30,))
                self.assertEqual(handle["contacts/triangle/step_offsets"].shape, (31,))
                self.assertEqual(handle["contacts/triangle/triangle_id"].shape, (30,))
                self.assertEqual(
                    handle["contacts/triangle/triangle_force_xyz_N"].shape, (30, 3)
                )

    def test_recorder_attributes_match_scenario_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            scenario = self._scenario(policy_seed=1111, friction_mu=0.005)
            self._write_trace(path, n_steps=3, scenario=scenario)

            with h5py.File(path, "r") as handle:
                self.assertEqual(
                    handle["meta"].attrs["schema_version"], TRACE_SCHEMA_VERSION
                )
                self.assertEqual(handle["meta"].attrs["eval_v2_sha"], "deadbeef")
                self.assertEqual(handle["meta"].attrs["sofa_version"], "23.06")
                self.assertEqual(
                    handle["meta"].attrs["created_at"], "2026-04-29T10:00:00+00:00"
                )
                self.assertEqual(handle["meta"].attrs["trial_status"], "complete")
                self.assertEqual(
                    handle["scenario"].attrs["anatomy_id"], scenario.anatomy_id
                )
                self.assertEqual(handle["scenario"].attrs["wire_id"], scenario.wire_id)
                self.assertEqual(
                    handle["scenario"].attrs["target_spec_json"],
                    scenario.target_spec_json,
                )
                self.assertEqual(
                    handle["scenario"].attrs["env_seed"], scenario.env_seed
                )
                self.assertEqual(
                    handle["scenario"].attrs["policy_seed"], scenario.policy_seed
                )
                self.assertEqual(handle["scenario"].attrs["dt_s"], scenario.dt_s)
                self.assertEqual(
                    handle["scenario"].attrs["friction_mu"], scenario.friction_mu
                )
                self.assertEqual(
                    handle["scenario"].attrs["tip_threshold_mm"],
                    scenario.tip_threshold_mm,
                )
                self.assertEqual(
                    handle["scenario"].attrs["max_episode_steps"],
                    scenario.max_episode_steps,
                )
                self.assertEqual(
                    handle["scenario"].attrs["mesh_ref"], scenario.mesh_ref
                )

    def test_recorder_chunking_one_step_per_chunk_for_dense_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            self._write_trace(path, n_steps=5)

            with h5py.File(path, "r") as handle:
                self.assertEqual(
                    handle["steps/wire_positions"].chunks,
                    (STEP_DATASET_CHUNK_LEADING_DIM, 2, 3),
                )
                self.assertEqual(
                    handle["steps/wire_collision_positions"].chunks,
                    (STEP_DATASET_CHUNK_LEADING_DIM, 3, 3),
                )

    def test_recorder_compression_applied_to_bulk_arrays_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            self._write_trace(path, n_steps=5)

            with h5py.File(path, "r") as handle:
                self.assertEqual(handle["steps/wire_positions"].compression, "gzip")
                self.assertEqual(
                    handle["steps/wire_positions"].compression_opts, DEFAULT_GZIP_LEVEL
                )
                self.assertEqual(
                    handle["steps/wire_collision_positions"].compression, "gzip"
                )
                self.assertEqual(
                    handle["contacts/wire/wire_dof_index"].compression, "gzip"
                )
                self.assertEqual(
                    handle["contacts/wire/wire_dof_force_xyz_N"].compression, "gzip"
                )
                self.assertEqual(
                    handle["contacts/triangle/triangle_id"].compression, "gzip"
                )
                self.assertEqual(
                    handle["contacts/triangle/triangle_force_xyz_N"].compression,
                    "gzip",
                )
                self.assertEqual(handle["steps/total_wall_force_N"].compression, None)
                self.assertEqual(handle["steps/tip_force_norm_N"].compression, None)
                self.assertEqual(handle["steps/contact_count"].compression, None)
                self.assertEqual(handle["steps/scoreable"].compression, None)

    def test_recorder_flush_boundary_does_not_drop_steps(self) -> None:
        for n_steps in (47, 50, 53):
            with self.subTest(n_steps=n_steps):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "trace.h5"
                    self._write_trace(path, n_steps=n_steps, flush_interval_steps=10)

                    with h5py.File(path, "r") as handle:
                        step_index = handle["steps/step_index"][:]
                        self.assertEqual(step_index.shape, (n_steps,))
                        np.testing.assert_array_equal(
                            step_index, np.arange(n_steps, dtype=np.int32)
                        )

    def test_recorder_close_finalizes_status_to_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            self._write_trace(path, n_steps=2)

            with h5py.File(path, "r") as handle:
                self.assertEqual(handle["meta"].attrs["trial_status"], "complete")

    def test_recorder_unclosed_file_has_partial_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            try:
                with TrialTraceRecorder(
                    path=path,
                    scenario=self._scenario(),
                    scene_static=self._scene_static(),
                    flush_interval_steps=2,
                ) as recorder:
                    recorder.add_step(self._step(0))
                    recorder.add_step(self._step(1))
                    raise RuntimeError("simulated crash")
            except RuntimeError:
                pass

            with h5py.File(path, "r") as handle:
                self.assertEqual(handle["meta"].attrs["trial_status"], "partial")
                np.testing.assert_array_equal(
                    handle["steps/step_index"][:], np.asarray([0, 1], dtype=np.int32)
                )

    def test_recorder_rejects_missing_scenario_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            with self.assertRaisesRegex(ValueError, "anatomy_id"):
                TrialTraceRecorder(
                    path=path,
                    scenario=self._scenario(anatomy_id=""),
                    scene_static=self._scene_static(),
                )

    def test_recorder_dtypes_match_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            self._write_trace(path, n_steps=4)

            with h5py.File(path, "r") as handle:
                self.assertEqual(
                    handle["scene_static/wire_initial_position"].dtype, np.float32
                )
                self.assertEqual(
                    handle["scene_static/wire_initial_rotation"].dtype, np.float32
                )
                self.assertEqual(handle["steps/step_index"].dtype, np.int32)
                self.assertEqual(handle["steps/sim_time_s"].dtype, np.float32)
                self.assertEqual(handle["steps/wire_positions"].dtype, np.float32)
                self.assertEqual(
                    handle["steps/wire_collision_positions"].dtype, np.float32
                )
                self.assertEqual(handle["steps/action"].dtype, np.float32)
                self.assertEqual(handle["steps/total_wall_force_N"].dtype, np.float32)
                self.assertEqual(handle["steps/tip_force_norm_N"].dtype, np.float32)
                self.assertEqual(handle["steps/contact_count"].dtype, np.int16)
                self.assertEqual(handle["steps/scoreable"].dtype, np.bool_)
                self.assertEqual(handle["contacts/wire/step_offsets"].dtype, np.int32)
                self.assertEqual(
                    handle["contacts/triangle/step_offsets"].dtype, np.int32
                )
                self.assertEqual(
                    handle["contacts/triangle/triangle_id"].dtype, np.int32
                )
                self.assertEqual(
                    handle["contacts/triangle/triangle_force_xyz_N"].dtype, np.float32
                )
                self.assertEqual(handle["contacts/wire/wire_dof_index"].dtype, np.int32)
                self.assertEqual(
                    handle["contacts/wire/wire_dof_force_xyz_N"].dtype, np.float32
                )
                self.assertEqual(
                    handle["contacts/wire/arc_length_from_distal_mm"].dtype,
                    np.float32,
                )
                self.assertEqual(handle["contacts/wire/is_tip"].dtype, np.bool_)


class TraceReaderTests(_TrialTraceTestHelpers, unittest.TestCase):
    def test_reader_round_trip_full_load_preserves_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            self._write_trace(path, n_steps=100)

            with TraceReader(path) as reader:
                payload = reader.load_all()

            self.assertFalse(payload["is_partial"])
            np.testing.assert_array_equal(
                payload["steps"]["step_index"], np.arange(100, dtype=np.int32)
            )
            self.assertEqual(payload["steps"]["wire_positions"].shape, (100, 2, 3))
            self.assertEqual(
                payload["steps"]["wire_collision_positions"].shape, (100, 3, 3)
            )
            self.assertEqual(payload["contacts"]["wire"]["step_offsets"].shape, (101,))
            self.assertEqual(
                payload["contacts"]["wire"]["wire_dof_index"].shape, (100,)
            )
            self.assertEqual(
                payload["contacts"]["triangle"]["step_offsets"].shape, (101,)
            )
            self.assertEqual(
                payload["contacts"]["triangle"]["triangle_id"].shape, (100,)
            )

    def test_reader_step_access_returns_correct_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            self._write_trace(path, n_steps=100)

            with TraceReader(path) as reader:
                step = reader.step(47)

            self.assertEqual(step.step_index, 47)
            self.assertAlmostEqual(step.sim_time_s, np.float32(4.7), places=6)
            np.testing.assert_array_equal(
                step.wire_positions_mm,
                np.asarray([[47.0, 0.0, 0.0], [47.0, 1.0, 0.0]], dtype=np.float32),
            )
            self.assertEqual(step.wire_contacts, (self._wire_contact(step_index=47),))
            self.assertEqual(
                step.triangle_contacts, (self._triangle_contact(step_index=47),)
            )

    def test_random_step_access_does_not_load_full_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            with TrialTraceRecorder(
                path=path,
                scenario=self._scenario(),
                scene_static=self._scene_static(),
                flush_interval_steps=25,
            ) as recorder:
                for step_index in range(500):
                    recorder.add_step(self._large_step(step_index))

            with TraceReader(path) as reader:
                with mock.patch.object(
                    reader,
                    "_read_step_dataset",
                    wraps=reader._read_step_dataset,
                ) as read_step_dataset, mock.patch.object(
                    reader,
                    "_read_all_dataset",
                    wraps=reader._read_all_dataset,
                ) as read_all_dataset:
                    step = reader.step(47)

            self.assertEqual(step.step_index, 47)
            self.assertEqual(read_all_dataset.call_count, 0)
            self.assertGreater(read_step_dataset.call_count, 0)
            self.assertTrue(
                all(call.args[1] == 47 for call in read_step_dataset.call_args_list)
            )

    def test_sequential_full_scan_is_chunk_efficient(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            self._write_trace(path, n_steps=20)

            with TraceReader(path) as reader:
                with mock.patch.object(
                    reader,
                    "_read_step_dataset",
                    wraps=reader._read_step_dataset,
                ) as read_step_dataset, mock.patch.object(
                    reader,
                    "_read_all_dataset",
                    wraps=reader._read_all_dataset,
                ) as read_all_dataset:
                    for step_index in range(20):
                        step = reader.step(step_index)
                        self.assertEqual(step.step_index, step_index)

            self.assertEqual(read_all_dataset.call_count, 0)
            self.assertEqual(read_step_dataset.call_count, 20 * 8)

    def test_reader_handles_partial_trace_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            try:
                with TrialTraceRecorder(
                    path=path,
                    scenario=self._scenario(),
                    scene_static=self._scene_static(),
                    flush_interval_steps=2,
                ) as recorder:
                    recorder.add_step(self._step(0))
                    recorder.add_step(self._step(1))
                    raise RuntimeError("simulated crash")
            except RuntimeError:
                pass

            with TraceReader(path) as reader:
                self.assertTrue(reader.is_partial)
                step = reader.step(1)

            self.assertEqual(step.step_index, 1)
            self.assertEqual(step.wire_contacts, (self._wire_contact(step_index=1),))
            self.assertEqual(
                step.triangle_contacts, (self._triangle_contact(step_index=1),)
            )

    def test_reader_handles_corrupt_file_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corrupt.h5"
            path.write_bytes(b"not an hdf5 file")

            with self.assertRaisesRegex(TraceFileCorruptError, "corrupt"):
                with TraceReader(path):
                    pass

    def test_reader_wire_contacts_for_step_returns_only_wire_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            self._write_trace(path, n_steps=10)

            with TraceReader(path) as reader:
                contacts = reader.wire_contacts_for_step(5)

            self.assertEqual(contacts, [self._wire_contact(step_index=5)])

    def test_reader_triangle_contacts_for_step_returns_only_triangle_records(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            self._write_trace(path, n_steps=10)

            with TraceReader(path) as reader:
                contacts = reader.triangle_contacts_for_step(5)

            self.assertEqual(contacts, [self._triangle_contact(step_index=5)])

    def test_reader_contacts_for_empty_step_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.h5"
            with TrialTraceRecorder(
                path=path,
                scenario=self._scenario(),
                scene_static=self._scene_static(),
                flush_interval_steps=2,
            ) as recorder:
                recorder.add_step(self._step(0, with_contact=False))
                recorder.add_step(self._step(1, with_contact=True))

            with TraceReader(path) as reader:
                wire_contacts = reader.wire_contacts_for_step(0)
                triangle_contacts = reader.triangle_contacts_for_step(0)

            self.assertEqual(wire_contacts, [])
            self.assertEqual(triangle_contacts, [])


class ParallelTracePersistenceTests(unittest.TestCase):
    def test_four_workers_write_distinct_trial_files_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = mp.get_context("spawn")
            paths = [
                Path(tmp) / f"trial_worker_{worker_id}.h5" for worker_id in range(4)
            ]
            with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
                futures = [
                    pool.submit(_write_worker_trace, str(path), worker_id)
                    for worker_id, path in enumerate(paths)
                ]
                results = [future.result(timeout=10) for future in futures]

            self.assertEqual(results, [0, 1, 2, 3])
            for path in paths:
                self.assertTrue(path.exists())
                with h5py.File(path, "r") as handle:
                    self.assertEqual(handle["meta"].attrs["trial_status"], "complete")

    def test_concurrent_writes_produce_correct_per_worker_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = mp.get_context("spawn")
            paths = [
                Path(tmp) / f"trial_worker_{worker_id}.h5" for worker_id in range(4)
            ]
            with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
                futures = [
                    pool.submit(_write_worker_trace, str(path), worker_id)
                    for worker_id, path in enumerate(paths)
                ]
                [future.result(timeout=10) for future in futures]

            for worker_id, path in enumerate(paths):
                with TraceReader(path) as reader:
                    step = reader.step(4)
                    wire_contacts = reader.wire_contacts_for_step(4)
                    triangle_contacts = reader.triangle_contacts_for_step(4)

                self.assertEqual(step.step_index, 4)
                np.testing.assert_array_equal(
                    step.wire_positions_mm,
                    _worker_step(worker_id, 4).wire_positions_mm,
                )
                np.testing.assert_array_equal(
                    step.action,
                    _worker_step(worker_id, 4).action,
                )
                self.assertEqual(wire_contacts, [_worker_wire_contact(worker_id, 4)])
                self.assertEqual(
                    triangle_contacts, [_worker_triangle_contact(worker_id, 4)]
                )

    def test_worker_crash_leaves_partial_status_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial_trace.h5"
            ctx = mp.get_context("spawn")
            process = ctx.Process(
                target=_write_worker_trace,
                args=(str(path), 7, 2),
            )
            process.start()
            process.join(timeout=10)

            self.assertNotEqual(process.exitcode, 0)
            self.assertTrue(path.exists())
            with h5py.File(path, "r") as handle:
                self.assertEqual(handle["meta"].attrs["trial_status"], "partial")
                np.testing.assert_array_equal(
                    handle["steps/step_index"][:],
                    np.asarray([0, 1], dtype=np.int32),
                )

    def test_write_anatomy_mesh_creates_valid_hdf5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meshes" / "anatomy_Tree_00.h5"
            write_anatomy_mesh(
                path,
                triangle_indices=np.asarray([[0, 1, 2]], dtype=np.int32),
                vertex_positions=np.asarray(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                ),
                anatomy_id="Tree_00",
            )

            with h5py.File(path, "r") as handle:
                self.assertEqual(handle.attrs["anatomy_id"], "Tree_00")
                self.assertEqual(handle["triangle_indices"].shape, (1, 3))
                self.assertEqual(handle["vertex_positions"].shape, (3, 3))

    def test_write_anatomy_mesh_rejects_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mesh.h5"
            write_anatomy_mesh(
                path,
                triangle_indices=np.asarray([[0, 1, 2]], dtype=np.int32),
                vertex_positions=np.asarray(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                ),
                anatomy_id="Tree_00",
            )

            with self.assertRaises(FileExistsError):
                write_anatomy_mesh(
                    path,
                    triangle_indices=np.asarray([[0, 1, 2]], dtype=np.int32),
                    vertex_positions=np.asarray(
                        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                        dtype=np.float32,
                    ),
                    anatomy_id="Tree_00",
                )

    def test_write_anatomy_mesh_atomic_rename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mesh.h5"
            write_anatomy_mesh(
                path,
                triangle_indices=np.asarray([[0, 1, 2]], dtype=np.int32),
                vertex_positions=np.asarray(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                ),
                anatomy_id="Tree_00",
            )

            self.assertTrue(path.exists())
            self.assertEqual(list(path.parent.glob("*.tmp_*")), [])

    def test_two_processes_writing_same_mesh_only_one_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mesh.h5"
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
                futures = [
                    pool.submit(_write_worker_mesh, str(path), worker_id)
                    for worker_id in (1, 2)
                ]
                results: list[int] = []
                errors: list[BaseException] = []
                for future in futures:
                    try:
                        results.append(future.result(timeout=10))
                    except BaseException as exc:
                        errors.append(exc)

            self.assertEqual(len(results), 1)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], FileExistsError)
            with h5py.File(path, "r") as handle:
                self.assertIn(handle.attrs["anatomy_id"], {"Tree_01", "Tree_02"})
                self.assertEqual(handle["triangle_indices"].shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
