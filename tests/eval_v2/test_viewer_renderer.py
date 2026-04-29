from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv

from steve_recommender.eval_v2.force_trace_persistence import TrialTraceRecorder
from steve_recommender.eval_v2.viewer.renderer import (
    BOTTOM_RIGHT_SCALAR_BAR_HEIGHT,
    BOTTOM_RIGHT_SCALAR_BAR_POSITION_X,
    BOTTOM_RIGHT_SCALAR_BAR_POSITION_Y,
    BOTTOM_RIGHT_SCALAR_BAR_WIDTH,
    TraceRenderer,
)
from steve_recommender.eval_v2.viewer.trace_data import TraceData
from tests.eval_v2.test_force_trace_persistence import _TrialTraceTestHelpers


class TraceRendererTests(_TrialTraceTestHelpers, unittest.TestCase):
    def _write_mixed_contact_trace(self, path: Path) -> None:
        with TrialTraceRecorder(
            path=path,
            scenario=self._scenario(),
            scene_static=self._scene_static(),
            flush_interval_steps=2,
        ) as recorder:
            recorder.add_step(self._step(0, with_contact=False))
            recorder.add_step(self._step(1, with_contact=True))

    def test_renderer_opens_with_real_trace_without_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=2)

            trace = TraceData(trace_path)
            try:
                renderer = TraceRenderer(trace)
                try:
                    renderer.render_frame(0)
                finally:
                    renderer.close()
            finally:
                trace.close()

    def test_renderer_renders_step_zero_with_wire_actor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=2)

            trace = TraceData(trace_path)
            try:
                renderer = TraceRenderer(trace)
                try:
                    renderer.render_frame(0)
                    self.assertIsNotNone(renderer._wire_actor)
                    self.assertIsNotNone(renderer._text_actor)
                    poly_data = renderer._wire_actor.mapper.dataset
                finally:
                    renderer.close()
            finally:
                trace.close()

        self.assertEqual(poly_data.n_points, 2)

    def test_renderer_step_change_updates_wire_points_and_contact_actor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_mixed_contact_trace(trace_path)

            trace = TraceData(trace_path)
            try:
                renderer = TraceRenderer(trace)
                try:
                    renderer.render_frame(0)
                    self.assertIsNone(renderer._contact_actor)
                    points_a = np.asarray(renderer._wire_actor.mapper.dataset.points)
                    renderer.render_frame(1)
                    self.assertIsNotNone(renderer._contact_actor)
                    points_b = np.asarray(renderer._wire_actor.mapper.dataset.points)
                    scalars = np.asarray(
                        renderer._contact_actor.mapper.dataset.cell_data[
                            "triangle_force_magnitudes_N"
                        ]
                    )
                finally:
                    renderer.close()
            finally:
                trace.close()

        self.assertFalse(np.array_equal(points_a, points_b))
        self.assertEqual(scalars.shape, (1,))
        self.assertGreater(float(scalars[0]), 0.0)

    def test_renderer_handles_zero_contact_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=2, with_contact=False)

            trace = TraceData(trace_path)
            try:
                renderer = TraceRenderer(trace)
                try:
                    renderer.render_frame(0)
                    self.assertIsNone(renderer._contact_actor)
                    self.assertIsNotNone(renderer._wire_actor)
                finally:
                    renderer.close()
            finally:
                trace.close()

    def test_renderer_uses_injected_plotter_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=2)

            plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
            trace = TraceData(trace_path)
            try:
                renderer = TraceRenderer(trace, plotter=plotter)
                try:
                    self.assertIs(renderer.get_plotter(), plotter)
                    renderer.render_frame(0)
                finally:
                    renderer.close()
            finally:
                trace.close()

    def test_renderer_standalone_default_is_not_off_screen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=2)

            trace = TraceData(trace_path)
            try:
                renderer = TraceRenderer(trace)
                try:
                    self.assertFalse(bool(renderer.get_plotter().off_screen))
                finally:
                    renderer.close()
            finally:
                trace.close()

    def test_renderer_uses_loader_p95_when_max_force_not_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=5)

            trace = TraceData(trace_path)
            try:
                renderer = TraceRenderer(trace, off_screen=True)
                try:
                    self.assertAlmostEqual(
                        renderer._max_display_force_N,
                        trace.recommended_max_display_force_N,
                        places=6,
                    )
                finally:
                    renderer.close()
            finally:
                trace.close()

    def test_renderer_respects_explicit_max_force_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=5)

            trace = TraceData(trace_path)
            try:
                renderer = TraceRenderer(
                    trace,
                    max_display_force_N=6.0,
                    off_screen=True,
                )
                try:
                    self.assertEqual(renderer._max_display_force_N, 6.0)
                finally:
                    renderer.close()
            finally:
                trace.close()

    def test_renderer_max_force_readout_updates_per_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_mixed_contact_trace(trace_path)

            trace = TraceData(trace_path)
            try:
                renderer = TraceRenderer(trace, off_screen=True)
                try:
                    renderer.render_frame(1)
                    self.assertIn("0.37 N", renderer._max_force_readout_text)
                finally:
                    renderer.close()
            finally:
                trace.close()

    def test_renderer_max_force_readout_zero_contacts_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=2, with_contact=False)

            trace = TraceData(trace_path)
            try:
                renderer = TraceRenderer(trace, off_screen=True)
                try:
                    renderer.render_frame(0)
                    self.assertEqual(
                        renderer._max_force_readout_text,
                        "Max force: 0.00 N",
                    )
                finally:
                    renderer.close()
            finally:
                trace.close()

    def test_renderer_scalar_bar_positioned_bottom_right(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=2)

            trace = TraceData(trace_path)
            try:
                renderer = TraceRenderer(trace, off_screen=True)
                try:
                    self.assertEqual(
                        renderer._scalar_bar_args["position_x"],
                        BOTTOM_RIGHT_SCALAR_BAR_POSITION_X,
                    )
                    self.assertEqual(
                        renderer._scalar_bar_args["position_y"],
                        BOTTOM_RIGHT_SCALAR_BAR_POSITION_Y,
                    )
                    self.assertEqual(
                        renderer._scalar_bar_args["width"],
                        BOTTOM_RIGHT_SCALAR_BAR_WIDTH,
                    )
                    self.assertEqual(
                        renderer._scalar_bar_args["height"],
                        BOTTOM_RIGHT_SCALAR_BAR_HEIGHT,
                    )
                finally:
                    renderer.close()
            finally:
                trace.close()
