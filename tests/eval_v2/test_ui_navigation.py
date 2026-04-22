from __future__ import annotations

import os
import threading
import unittest

from PyQt5.QtWidgets import QApplication

from steve_recommender.eval_v2.models import EvaluationReport
from steve_recommender.eval_v2.ui_main import ClinicalMainWindow


class _MainWindowServiceStub:
    def __init__(self) -> None:
        self.report = EvaluationReport(
            job_name="navigation_test",
            generated_at="2026-04-21T00:00:00+00:00",
            summaries=(),
            trials=(),
        )

    def list_anatomies(self, *, registry_path=None):
        _ = registry_path
        return ()

    def list_branches(self, anatomy):
        _ = anatomy
        return ()

    def list_target_modes(self):
        return ()

    def list_execution_wires(self):
        return ()

    def list_startable_wires(self):
        return ()

    def list_registry_policies(self, *, execution_wire=None):
        _ = execution_wire
        return ()

    def list_explicit_policies(self, *, execution_wire=None):
        _ = execution_wire
        return ()

    def list_candidates(self, *, execution_wire, include_cross_wire=True):
        _ = execution_wire, include_cross_wire
        return ()

    def run_evaluation_job(self, job, *, frame_callback=None, progress_callback=None):
        _ = job, frame_callback, progress_callback
        return self.report


class ClinicalMainWindowNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication(["-platform", "offscreen"])

    def test_home_archive_and_setup_screens_are_in_the_stack(self) -> None:
        window = ClinicalMainWindow(service=_MainWindowServiceStub())

        self.assertIsNotNone(window.stack)
        self.assertEqual(window.stack.count(), 3)
        self.assertIs(window.stack.widget(0), window.home_screen)
        self.assertIs(window.stack.widget(1), window.archive_screen)
        self.assertIs(window.stack.widget(2), window.setup_screen)

    def test_home_new_recommendation_switches_to_setup_screen(self) -> None:
        window = ClinicalMainWindow(service=_MainWindowServiceStub())

        window.home_screen.new_setup_requested.emit()

        self.assertIs(window.stack.currentWidget(), window.setup_screen)

    def test_home_archive_button_switches_to_archive_screen(self) -> None:
        window = ClinicalMainWindow(service=_MainWindowServiceStub())

        window.home_screen.archive_requested.emit()

        self.assertIs(window.stack.currentWidget(), window.archive_screen)


if __name__ == "__main__":
    unittest.main()