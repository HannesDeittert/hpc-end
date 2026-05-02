from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from steve_recommender.eval_v2.viewer.qt_replay_widget import QtTraceReplayWidget
from tests.eval_v2.test_force_trace_persistence import _TrialTraceTestHelpers


class QtTraceReplayWidgetTests(_TrialTraceTestHelpers, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication(["-platform", "offscreen"])

    def test_widget_constructs_with_real_trace_without_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=3)
            widget = QtTraceReplayWidget()
            widget.open_trace(trace_path, start_step=0)
            QApplication.processEvents()

            self.assertIsNotNone(widget.renderer)
            self.assertIsNotNone(widget.trace_data)
            widget.close_trace()

    def test_widget_slider_change_updates_renderer_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=3)
            widget = QtTraceReplayWidget()
            widget.open_trace(trace_path, start_step=0)
            widget.step_slider.setValue(2)
            QApplication.processEvents()

            self.assertEqual(widget.current_step, 2)
            self.assertEqual(widget.renderer.current_step, 2)
            widget.close_trace()

    def test_widget_play_button_starts_auto_advance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=4)
            widget = QtTraceReplayWidget()
            widget.open_trace(trace_path, start_step=0)

            widget.play_button.click()
            self.assertTrue(widget._play_timer.isActive())
            widget._play_timer.timeout.emit()
            QApplication.processEvents()

            self.assertEqual(widget.current_step, 1)
            widget.close_trace()

    def test_widget_pause_button_stops_auto_advance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=4)
            widget = QtTraceReplayWidget()
            widget.open_trace(trace_path, start_step=0)

            widget.play_button.click()
            widget.play_button.click()
            self.assertFalse(widget._play_timer.isActive())
            widget._play_timer.timeout.emit()
            QApplication.processEvents()

            self.assertEqual(widget.current_step, 0)
            widget.close_trace()

    def test_widget_step_label_shows_current_and_total(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.h5"
            self._write_trace(trace_path, n_steps=100)
            widget = QtTraceReplayWidget()
            widget.open_trace(trace_path, start_step=47)
            QApplication.processEvents()

            self.assertEqual(widget.step_label.text(), "Step 47 / 99")
            widget.close_trace()
