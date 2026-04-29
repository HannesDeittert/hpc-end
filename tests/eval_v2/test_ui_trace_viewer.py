from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from steve_recommender.eval_v2.ui_trace_viewer import TraceViewerPanel
from tests.eval_v2.test_force_trace_persistence import _TrialTraceTestHelpers


class TraceViewerPanelTests(_TrialTraceTestHelpers, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication(["-platform", "offscreen"])

    def test_trace_viewer_panel_constructs_without_crash(self) -> None:
        panel = TraceViewerPanel()
        self.assertIsNotNone(panel)

    def test_trace_viewer_panel_renders_first_frame_on_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=3)
            panel = TraceViewerPanel()
            panel.open_trace(trace_path, start_step=0)
            QApplication.processEvents()

            self.assertEqual(panel.current_step, 0)
            self.assertIsNotNone(panel._renderer)
            self.assertIsNotNone(panel._renderer._wire_actor)
            self.assertEqual(panel.step_label.text(), "Step 0 / 2")
            panel.close_trace()

    def test_slider_in_panel_updates_renderer_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=3)
            panel = TraceViewerPanel()
            panel.open_trace(trace_path, start_step=0)
            panel.step_slider.setValue(2)
            QApplication.processEvents()

            self.assertEqual(panel.current_step, 2)
            self.assertEqual(panel._renderer.current_step, 2)
            self.assertEqual(panel.step_label.text(), "Step 2 / 2")
            panel.close_trace()

    def test_panel_close_releases_renderer_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=2)
            panel = TraceViewerPanel()
            panel.open_trace(trace_path)
            panel.close_trace()

            self.assertIsNone(panel._renderer)
            self.assertIsNone(panel._trace_data)
