"""Tests for PipelineRunningPage visualization and ExecutionConfigPage."""

from __future__ import annotations

import os
import threading
import unittest
from pathlib import Path
from dataclasses import replace

import numpy as np
from PyQt5.QtWidgets import QApplication, QPushButton, QSpinBox
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from steve_recommender.eval_v2.models import (
    AorticArchAnatomy,
    BranchEndTarget,
    EvaluationCandidate,
    EvaluationJob,
    EvaluationReport,
    EvaluationScenario,
    ExecutionPlan,
    FluoroscopySpec,
    PolicySpec,
    VisualizationSpec,
    WireRef,
)
from steve_recommender.eval_v2.ui_controller import ClinicalUIController, EvaluationWorker
from steve_recommender.eval_v2.ui_wizard_pages import PipelineRunningPage, ExecutionConfigPage


class MockEvaluationService:
    """Mock service for testing evaluation with frame callbacks."""

    def __init__(self, *, release_event: threading.Event, report: EvaluationReport, frames_to_emit=None) -> None:
        self.release_event = release_event
        self.report = report
        self.jobs: list[EvaluationJob] = []
        self.frames_to_emit = frames_to_emit or []
        self.frame_callback_called = False
        self.progress_callback_called = False

    def list_anatomies(self, *, registry_path=None):
        _ = registry_path
        anatomy = AorticArchAnatomy(arch_type="II", seed=42, record_id="Tree_00")
        return (anatomy,)

    def list_branches(self, anatomy):
        _ = anatomy
        return ()

    def list_target_modes(self):
        return ()

    def list_execution_wires(self):
        execution_wire = WireRef(model="steve_default", wire="standard_j")
        return (execution_wire,)

    def list_startable_wires(self):
        execution_wire = WireRef(model="steve_default", wire="standard_j")
        return (execution_wire,)

    def list_registry_policies(self, *, execution_wire=None):
        _ = execution_wire
        execution_wire = WireRef(model="steve_default", wire="standard_j")
        policy = PolicySpec(
            name="policy_a",
            checkpoint_path=Path("/tmp/policy_a.everl"),
            trained_on_wire=execution_wire,
        )
        return (policy,)

    def list_explicit_policies(self, *, execution_wire=None):
        _ = execution_wire
        return ()

    def build_candidate(self, *, name: str, execution_wire: WireRef, policy: PolicySpec):
        return EvaluationCandidate(name=name, execution_wire=execution_wire, policy=policy)

    def list_candidates(self, *, execution_wire: WireRef, include_cross_wire: bool = True):
        _ = include_cross_wire
        policy = PolicySpec(
            name="policy_a",
            checkpoint_path=Path("/tmp/policy_a.everl"),
            trained_on_wire=execution_wire,
        )
        return (self.build_candidate(name="candidate_a", execution_wire=execution_wire, policy=policy),)

    def run_evaluation_job(
        self,
        job: EvaluationJob,
        *,
        frame_callback=None,
        progress_callback=None,
    ) -> EvaluationReport:
        self.jobs.append(job)
        
        # Emit frames if callback is provided
        if frame_callback is not None:
            self.frame_callback_called = True
            for frame in self.frames_to_emit:
                frame_callback(frame)
        
        # Call progress callback to simulate progress
        if progress_callback is not None:
            self.progress_callback_called = True
            progress_callback("trial_start index=0 seed=0 scenario=test candidate=candidate_a")
            progress_callback("trial_step index=0 step=1")
            progress_callback("trial_end index=0")
        
        if not self.release_event.wait(timeout=2.0):
            raise TimeoutError("test did not release the worker")
        return self.report


def _make_report(job_name: str = "test_job") -> EvaluationReport:
    """Create a minimal evaluation report for testing."""
    from steve_recommender.eval_v2.models import CandidateSummary
    
    execution_wire = WireRef(model="steve_default", wire="standard_j")
    return EvaluationReport(
        job_name=job_name,
        generated_at="2026-04-21T12:00:00+00:00",
        summaries=(
            CandidateSummary(
                scenario_name="scenario_x",
                candidate_name="candidate_a",
                execution_wire=execution_wire,
                trained_on_wire=execution_wire,
                trial_count=1,
                success_rate=1.0,
                score_mean=0.9,
                score_std=0.0,
                steps_total_mean=5.0,
                steps_to_success_mean=3.0,
                tip_speed_max_mean_mm_s=12.5,
                wall_force_max_mean=None,
                wall_force_max_mean_newton=None,
                force_available_rate=0.0,
            ),
        ),
        trials=(),
    )


def _make_job(*, is_visualized: bool = False) -> EvaluationJob:
    """Create an evaluation job for testing."""
    execution_wire = WireRef(model="steve_default", wire="standard_j")
    policy = PolicySpec(
        name="policy_a",
        checkpoint_path=Path("/tmp/policy_a.everl"),
        trained_on_wire=execution_wire,
    )
    candidate = EvaluationCandidate(
        name="candidate_a",
        execution_wire=execution_wire,
        policy=policy,
    )
    anatomy = AorticArchAnatomy(arch_type="II", seed=42, record_id="Tree_00")
    scenario = EvaluationScenario(
        name="scenario_x",
        anatomy=anatomy,
        target=BranchEndTarget(threshold_mm=5.0, branches=("lcca",)),
        fluoroscopy=FluoroscopySpec(),
    )
    
    visualization = VisualizationSpec(enabled=is_visualized, rendered_trials_per_candidate=1)
    execution = ExecutionPlan(
        max_episode_steps=100,
        trials_per_candidate=3,
        base_seed=42,
        visualization=visualization,
    )
    
    return EvaluationJob(
        name="test_job",
        scenarios=(scenario,),
        candidates=(candidate,),
        output_root=Path("/tmp/eval_v2_ui_tests"),
        execution=execution,
    )


class PipelineRunningPageVisualizationTests(unittest.TestCase):
    """Tests for PipelineRunningPage visualization rendering."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication(["-platform", "offscreen"])

    def test_frame_update_with_rgb_array(self) -> None:
        """Test that RGB frames are correctly converted to pixmaps."""
        service = MockEvaluationService(
            release_event=threading.Event(),
            report=_make_report(),
        )
        controller = ClinicalUIController(service=service)
        page = PipelineRunningPage(controller=controller)

        # Create a simple RGB test frame
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        frame[25:75, 25:75, 0] = 255  # Red square in the middle

        page.update_frame(frame)

        # Verify the pixmap was set
        self.assertIsNotNone(page.video_label.pixmap())
        self.assertFalse(page.video_label.pixmap().isNull())

    def test_frame_update_with_grayscale_array(self) -> None:
        """Test that grayscale frames are converted to RGB."""
        service = MockEvaluationService(
            release_event=threading.Event(),
            report=_make_report(),
        )
        controller = ClinicalUIController(service=service)
        page = PipelineRunningPage(controller=controller)

        # Create a grayscale test frame
        frame = np.full((100, 100), 128, dtype=np.uint8)

        page.update_frame(frame)

        # Verify the pixmap was set
        self.assertIsNotNone(page.video_label.pixmap())
        self.assertFalse(page.video_label.pixmap().isNull())

    def test_frame_update_with_single_channel_array(self) -> None:
        """Test that single-channel frames are expanded to RGB."""
        service = MockEvaluationService(
            release_event=threading.Event(),
            report=_make_report(),
        )
        controller = ClinicalUIController(service=service)
        page = PipelineRunningPage(controller=controller)

        # Create a single-channel test frame
        frame = np.full((100, 100, 1), 200, dtype=np.uint8)

        page.update_frame(frame)

        # Verify the pixmap was set
        self.assertIsNotNone(page.video_label.pixmap())
        self.assertFalse(page.video_label.pixmap().isNull())

    def test_frame_update_with_rgba_array(self) -> None:
        """Test that RGBA frames are handled correctly."""
        service = MockEvaluationService(
            release_event=threading.Event(),
            report=_make_report(),
        )
        controller = ClinicalUIController(service=service)
        page = PipelineRunningPage(controller=controller)

        # Create an RGBA test frame
        frame = np.full((100, 100, 4), 200, dtype=np.uint8)
        frame[:, :, 3] = 255  # Set alpha to fully opaque

        page.update_frame(frame)

        # Verify the pixmap was set
        self.assertIsNotNone(page.video_label.pixmap())
        self.assertFalse(page.video_label.pixmap().isNull())

    def test_frame_update_ignores_invalid_arrays(self) -> None:
        """Test that invalid arrays are safely ignored."""
        service = MockEvaluationService(
            release_event=threading.Event(),
            report=_make_report(),
        )
        controller = ClinicalUIController(service=service)
        page = PipelineRunningPage(controller=controller)

        # Test with None
        page.update_frame(None)  # Should not raise

        # Test with empty array
        page.update_frame(np.array([]))  # Should not raise

        # Test with 1D array
        page.update_frame(np.array([1, 2, 3]))  # Should not raise

    def test_visualization_mode_layout_switch(self) -> None:
        """Test that visualization mode correctly switches layouts."""
        service = MockEvaluationService(
            release_event=threading.Event(),
            report=_make_report(),
        )
        controller = ClinicalUIController(service=service)
        page = PipelineRunningPage(controller=controller)
        page.show()  # Show the page to ensure widgets are visible

        # Start in headless mode (both are initially added to layout)
        # The visibility is controlled by _set_mode_layout which is called in __init__
        page._set_mode_layout(is_visualized=False)
        self.assertTrue(page._headless_container.isVisible())
        self.assertFalse(page._visualized_container.isVisible())

        # Switch to visualized mode
        page._set_mode_layout(is_visualized=True)
        self.assertFalse(page._headless_container.isVisible())
        self.assertTrue(page._visualized_container.isVisible())

        # Switch back to headless
        page._set_mode_layout(is_visualized=False)
        self.assertTrue(page._headless_container.isVisible())
        self.assertFalse(page._visualized_container.isVisible())

    def test_progress_update_in_visualized_mode(self) -> None:
        """Test that progress updates are reflected in visualized mode."""
        service = MockEvaluationService(
            release_event=threading.Event(),
            report=_make_report(),
        )
        controller = ClinicalUIController(service=service)
        page = PipelineRunningPage(controller=controller)
        page._visualized_mode = True

        page._set_progress(50)
        self.assertEqual(page.visual_progress_bar.value(), 50)
        self.assertEqual(page.headless_progress_bar.value(), 50)

    def test_progress_update_in_headless_mode(self) -> None:
        """Test that progress updates are reflected in headless mode."""
        service = MockEvaluationService(
            release_event=threading.Event(),
            report=_make_report(),
        )
        controller = ClinicalUIController(service=service)
        page = PipelineRunningPage(controller=controller)
        page._visualized_mode = False

        page._set_progress(75)
        self.assertEqual(page.visual_progress_bar.value(), 75)
        self.assertEqual(page.headless_progress_bar.value(), 75)

    def test_status_label_updates_on_frame_ready(self) -> None:
        """Test that frame ready signal is handled correctly."""
        service = MockEvaluationService(
            release_event=threading.Event(),
            report=_make_report(),
        )
        controller = ClinicalUIController(service=service)
        page = PipelineRunningPage(controller=controller)

        # Create a test frame
        frame = np.full((50, 50, 3), 100, dtype=np.uint8)

        # Simulate frame ready signal
        page._on_frame_ready(frame)

        # Since we're not in running/visualized mode, frame should be ignored
        self.assertIsNone(page.video_label.pixmap())

        # Now simulate running in visualized mode
        page._running = True
        page._visualized_mode = True
        page._on_frame_ready(frame)

        # Frame should be rendered
        self.assertIsNotNone(page.video_label.pixmap())
        self.assertFalse(page.video_label.pixmap().isNull())


class ExecutionConfigPageTests(unittest.TestCase):
    """Tests for ExecutionConfigPage feature cards."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication(["-platform", "offscreen"])

    def setUp(self) -> None:
        service = MockEvaluationService(
            release_event=threading.Event(),
            report=_make_report(),
        )
        self.controller = ClinicalUIController(service=service)
        self.page = ExecutionConfigPage(controller=self.controller)
        self.page.show()  # Show the page to ensure widgets are visible

    def test_deterministic_mode_selection(self) -> None:
        """Test that deterministic mode can be selected."""
        self.assertTrue(self.page.deterministic_card.isChecked())
        self.assertFalse(self.page.stochastic_card.isChecked())

        # Click stochastic
        self.page.stochastic_card.click()
        self.assertFalse(self.page.deterministic_card.isChecked())
        self.assertTrue(self.page.stochastic_card.isChecked())

        # Click deterministic
        self.page.deterministic_card.click()
        self.assertTrue(self.page.deterministic_card.isChecked())
        self.assertFalse(self.page.stochastic_card.isChecked())

    def test_execution_mode_selection(self) -> None:
        """Test that execution mode (headless vs visualized) can be selected."""
        self.assertTrue(self.page.headless_card.isChecked())
        self.assertFalse(self.page.live_card.isChecked())

        # Click live
        self.page.live_card.click()
        self.assertFalse(self.page.headless_card.isChecked())
        self.assertTrue(self.page.live_card.isChecked())

        # Click headless
        self.page.headless_card.click()
        self.assertTrue(self.page.headless_card.isChecked())
        self.assertFalse(self.page.live_card.isChecked())

    def test_runs_row_visibility_with_stochastic(self) -> None:
        """Test that runs_row is visible when stochastic mode is selected."""
        self.page.stochastic_card.click()
        QApplication.processEvents()  # Process signal
        self.assertTrue(self.page.runs_row.isVisible())

        self.page.deterministic_card.click()
        QApplication.processEvents()  # Process signal
        self.assertFalse(self.page.runs_row.isVisible())

    def test_visualized_runs_row_visibility(self) -> None:
        """Test that visualized_runs_row visibility depends on live mode and runs count."""
        # Headless mode: should not be visible
        self.page.headless_card.click()
        QApplication.processEvents()
        self.assertFalse(self.page.visualized_runs_row.isVisible())

        # Deterministic + Live: should not be visible (needs multiple runs)
        self.page.live_card.click()
        QApplication.processEvents()
        self.assertFalse(self.page.visualized_runs_row.isVisible())

        # Stochastic + Live + runs > 1: should be visible
        self.page.stochastic_card.click()
        QApplication.processEvents()
        self.page.runs_spin.setValue(2)
        QApplication.processEvents()
        self.assertTrue(self.page.visualized_runs_row.isVisible())

        # Stochastic + Live + runs = 1: should not be visible
        self.page.runs_spin.setValue(1)
        QApplication.processEvents()
        self.assertFalse(self.page.visualized_runs_row.isVisible())

    def test_runs_spinbox_values(self) -> None:
        """Test that runs spinbox has valid range."""
        self.page.stochastic_card.click()
        
        self.page.runs_spin.setValue(1)
        self.assertEqual(self.page.runs_spin.value(), 1)

        self.page.runs_spin.setValue(50)
        self.assertEqual(self.page.runs_spin.value(), 50)

    def test_visualized_runs_spinbox_values(self) -> None:
        """Test that visualized_runs spinbox has valid range."""
        self.page.live_card.click()
        self.page.stochastic_card.click()
        self.page.runs_spin.setValue(10)
        
        self.page.visualized_runs_spin.setValue(1)
        self.assertEqual(self.page.visualized_runs_spin.value(), 1)

        self.page.visualized_runs_spin.setValue(5)
        self.assertEqual(self.page.visualized_runs_spin.value(), 5)


class VisualizationIntegrationTests(unittest.TestCase):
    """Integration tests for visualization with evaluation worker."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication(["-platform", "offscreen"])

    def test_worker_emits_frames_with_visualization_enabled(self) -> None:
        """Test that worker emits frames when visualization is enabled."""
        release_event = threading.Event()
        report = _make_report()
        
        # Create RGB frames
        frame1 = np.full((50, 50, 3), 100, dtype=np.uint8)
        frame2 = np.full((50, 50, 3), 150, dtype=np.uint8)
        
        service = MockEvaluationService(
            release_event=release_event,
            report=report,
            frames_to_emit=[frame1, frame2],
        )
        
        job = _make_job(is_visualized=True)
        worker = EvaluationWorker(
            service=service,
            job=job,
            frame_supplier=None,
        )

        frame_spy = QSignalSpy(worker.frame_ready)
        finished_spy = QSignalSpy(worker.finished)

        worker.start()
        self.assertTrue(frame_spy.wait(1000))
        
        # Should have received at least one frame
        self.assertGreaterEqual(len(frame_spy), 1)
        
        release_event.set()
        self.assertTrue(finished_spy.wait(1000))
        worker.wait(1000)

    def test_visualization_flag_preserved_in_service_job(self) -> None:
        """Test that visualization settings are preserved in the service job."""
        release_event = threading.Event()
        report = _make_report()
        service = MockEvaluationService(
            release_event=release_event,
            report=report,
        )
        
        job = _make_job(is_visualized=True)
        self.assertTrue(job.execution.visualization.enabled)
        
        worker = EvaluationWorker(
            service=service,
            job=job,
            frame_supplier=None,
        )
        
        finished_spy = QSignalSpy(worker.finished)
        worker.start()
        
        release_event.set()
        self.assertTrue(finished_spy.wait(1000))
        worker.wait(1000)
        
        # Check that service receives the configured visualization settings.
        self.assertEqual(len(service.jobs), 1)
        service_job = service.jobs[0]
        self.assertTrue(service_job.execution.visualization.enabled)


if __name__ == "__main__":
    unittest.main()
