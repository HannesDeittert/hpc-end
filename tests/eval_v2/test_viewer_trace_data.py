from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from steve_recommender.eval_v2.viewer.trace_data import (
    ANATOMY_SIMULATION_MESH_BASE_PATH,
    EMPTY_TRACE_FALLBACK_MAX_FORCE_N,
    TraceFrame,
    TraceData,
)
from steve_recommender.eval_v2.force_trace_persistence import (
    TraceReader,
    TrialTraceRecorder,
)
from tests.eval_v2.test_force_trace_persistence import _TrialTraceTestHelpers


class TraceDataLoaderTests(_TrialTraceTestHelpers, unittest.TestCase):
    def test_loader_resolves_mesh_path_from_anatomy_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=2)

            loader = TraceData(trace_path)
            try:
                self.assertEqual(
                    loader.vessel_mesh_path,
                    ANATOMY_SIMULATION_MESH_BASE_PATH
                    / "Tree_00"
                    / "mesh"
                    / "simulationmesh.obj",
                )
            finally:
                loader.close()

    def test_loader_frame_shapes_match_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=3)

            loader = TraceData(trace_path)
            try:
                frame = loader.frame(1)
                self.assertIsInstance(frame, TraceFrame)
                self.assertEqual(frame.step_index, 1)
                self.assertAlmostEqual(frame.sim_time_s, 0.1, places=6)
                self.assertEqual(frame.wire_positions_mm.shape, (2, 3))
                self.assertEqual(frame.wire_positions_mm.dtype, np.float32)
                self.assertEqual(frame.triangle_force_indices.shape, (1,))
                self.assertEqual(frame.triangle_force_indices.dtype, np.int32)
                self.assertEqual(frame.triangle_force_magnitudes_N.shape, (1,))
                self.assertEqual(frame.triangle_force_magnitudes_N.dtype, np.float32)
            finally:
                loader.close()

    def test_loader_frame_zero_contacts_returns_empty_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=2, with_contact=False)

            loader = TraceData(trace_path)
            try:
                frame = loader.frame(0)
                self.assertEqual(frame.triangle_force_indices.shape, (0,))
                self.assertEqual(frame.triangle_force_magnitudes_N.shape, (0,))
            finally:
                loader.close()

    def test_loader_handles_partial_trace_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "partial.h5"
            try:
                with TrialTraceRecorder(
                    path=trace_path,
                    scenario=self._scenario(),
                    scene_static=self._scene_static(),
                    flush_interval_steps=2,
                ) as recorder:
                    recorder.add_step(self._step(0))
                    recorder.add_step(self._step(1))
                    raise RuntimeError("simulated crash")
            except RuntimeError:
                pass

            loader = TraceData(trace_path)
            try:
                self.assertTrue(loader.is_partial)
                frame = loader.frame(1)
                self.assertEqual(frame.step_index, 1)
            finally:
                loader.close()

    def test_loader_force_magnitudes_match_persisted_xyz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=3)

            loader = TraceData(trace_path)
            try:
                frame = loader.frame(2)
                with h5py.File(trace_path, "r") as handle:
                    start = int(handle["contacts/triangle/step_offsets"][2])
                    end = int(handle["contacts/triangle/step_offsets"][3])
                    vectors = np.asarray(
                        handle["contacts/triangle/triangle_force_xyz_N"][start:end],
                        dtype=np.float32,
                    )
                expected = np.linalg.norm(vectors, axis=1).astype(np.float32)
                np.testing.assert_allclose(frame.triangle_force_magnitudes_N, expected)
            finally:
                loader.close()

    def test_loader_random_step_access_does_not_load_full_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            with TrialTraceRecorder(
                path=trace_path,
                scenario=self._scenario(),
                scene_static=self._scene_static(),
                flush_interval_steps=25,
            ) as recorder:
                for step_index in range(100):
                    recorder.add_step(self._large_step(step_index))

            with mock.patch.object(
                TraceReader, "load_all", wraps=TraceReader.load_all
            ) as load_all:
                loader = TraceData(trace_path)
                try:
                    with mock.patch.object(
                        loader._reader,
                        "_read_step_dataset",
                        wraps=loader._reader._read_step_dataset,
                    ) as read_step_dataset:
                        frame = loader.frame(47)
                    self.assertEqual(frame.step_index, 47)
                    self.assertEqual(load_all.call_count, 0)
                    self.assertGreater(read_step_dataset.call_count, 0)
                    self.assertTrue(
                        all(
                            call.args[1] == 47
                            for call in read_step_dataset.call_args_list
                        )
                    )
                finally:
                    loader.close()

    def test_loader_raises_clear_error_for_missing_mesh_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            scenario = self._scenario(anatomy_id="missing_anatomy")
            self._write_trace(trace_path, n_steps=1, scenario=scenario)

            with self.assertRaisesRegex(FileNotFoundError, "simulation mesh"):
                TraceData(trace_path)

    def test_loader_computes_force_p95_across_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=5)

            loader = TraceData(trace_path)
            try:
                expected = float(
                    np.percentile(
                        [
                            np.linalg.norm([0.1 * step_index, 0.2, 0.3])
                            for step_index in range(5)
                        ],
                        95.0,
                    )
                )
                self.assertAlmostEqual(
                    loader.recommended_max_display_force_N,
                    expected,
                    places=6,
                )
            finally:
                loader.close()

    def test_loader_force_p95_handles_zero_contact_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=3, with_contact=False)

            loader = TraceData(trace_path)
            try:
                self.assertEqual(
                    loader.recommended_max_display_force_N,
                    EMPTY_TRACE_FALLBACK_MAX_FORCE_N,
                )
            finally:
                loader.close()
