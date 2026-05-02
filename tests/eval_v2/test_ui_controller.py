from __future__ import annotations

from dataclasses import replace
import os
import threading
import unittest
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QSignalSpy

from steve_recommender.eval_v2.models import (
    AorticArchAnatomy,
    AnatomyBranch,
    BranchEndTarget,
    CandidateSummary,
    EvaluationCandidate,
    EvaluationJob,
    EvaluationReport,
    EvaluationScenario,
    FluoroscopySpec,
    PolicySpec,
    TargetModeDescriptor,
    VisualizationSpec,
    WireRef,
)
from steve_recommender.eval_v2.ui_controller import ClinicalUIController, EvaluationWorker


class MockDefaultEvaluationService:
    def __init__(self, *, release_event: threading.Event, report: EvaluationReport) -> None:
        self.release_event = release_event
        self.report = report
        self.jobs: list[EvaluationJob] = []
        self.anatomy = AorticArchAnatomy(arch_type="II", seed=42, record_id="Tree_00")
        self.branch = AnatomyBranch(
            name="lcca",
            centerline_points_vessel_cs=((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
            length_mm=12.5,
        )
        self.target_mode = TargetModeDescriptor(
            kind="branch_end",
            label="Branch End",
            description="Select named branch endpoints.",
            requires_branch_selection=True,
            requires_index_selection=False,
            allows_multi_branch_selection=True,
            requires_manual_points=False,
        )
        self.execution_wire = WireRef(model="steve_default", wire="standard_j")
        self.policy = PolicySpec(
            name="policy_a",
            checkpoint_path=Path("/tmp/policy_a.everl"),
            trained_on_wire=self.execution_wire,
        )

    def list_anatomies(self, *, registry_path=None):
        _ = registry_path
        return (self.anatomy,)

    def get_anatomy(self, *, record_id: str, registry_path=None):
        _ = registry_path
        if record_id != self.anatomy.record_id:
            raise KeyError(record_id)
        return self.anatomy

    def list_branches(self, anatomy: AorticArchAnatomy):
        _ = anatomy
        return (self.branch,)

    def get_branch(self, anatomy: AorticArchAnatomy, *, branch_name: str):
        _ = anatomy
        if branch_name != self.branch.name:
            raise KeyError(branch_name)
        return self.branch

    def list_target_modes(self):
        return (self.target_mode,)

    def list_execution_wires(self):
        return (self.execution_wire,)

    def list_startable_wires(self):
        return (self.execution_wire,)

    def list_registry_policies(self, *, execution_wire=None):
        _ = execution_wire
        return (self.policy,)

    def list_explicit_policies(self, *, execution_wire=None):
        _ = execution_wire
        return ()

    def resolve_policy_from_agent_ref(self, agent_ref):
        raise KeyError(agent_ref)

    def build_candidate(self, *, name: str, execution_wire: WireRef, policy: PolicySpec):
        return EvaluationCandidate(name=name, execution_wire=execution_wire, policy=policy)

    def list_candidates(self, *, execution_wire: WireRef, include_cross_wire: bool = True):
        _ = include_cross_wire
        return (self.build_candidate(name="candidate_a", execution_wire=execution_wire, policy=self.policy),)

    def run_evaluation_job(
        self,
        job: EvaluationJob,
        *,
        frame_callback=None,
        progress_callback=None,
    ) -> EvaluationReport:
        _ = frame_callback, progress_callback
        self.jobs.append(job)
        if not self.release_event.wait(timeout=2.0):
            raise TimeoutError("test did not release the worker")
        return self.report


def _make_report(job_name: str = "job_x") -> EvaluationReport:
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
                force_available_rate=0.0,
            ),
        ),
        trials=(),
    )


def _make_job() -> EvaluationJob:
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
    scenario = EvaluationScenario(
        name="scenario_x",
        anatomy=AorticArchAnatomy(arch_type="II", seed=42, record_id="Tree_00"),
        target=BranchEndTarget(threshold_mm=5.0, branches=("lcca",)),
        fluoroscopy=FluoroscopySpec(),
    )
    return EvaluationJob(
        name="job_x",
        scenarios=(scenario,),
        candidates=(candidate,),
        output_root=Path("/tmp/eval_v2_ui_tests"),
    )


class UiControllerThreadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls._app = QApplication.instance() or QApplication(["-platform", "offscreen"])

    def test_worker_emits_progress_frames_and_finished_without_blocking(self) -> None:
        release_event = threading.Event()
        report = _make_report()
        service = MockDefaultEvaluationService(release_event=release_event, report=report)
        job = _make_job()
        frame = np.arange(9, dtype=np.uint8).reshape(3, 3)

        worker = EvaluationWorker(
            service=service,
            job=job,
            frame_supplier=lambda: (frame,),
        )

        progress_spy = QSignalSpy(worker.progress_updated)
        status_spy = QSignalSpy(worker.status_updated)
        frame_spy = QSignalSpy(worker.frame_ready)
        error_spy = QSignalSpy(worker.error_occurred)
        finished_spy = QSignalSpy(worker.finished)

        worker.start()

        self.assertTrue(frame_spy.wait(1000))
        self.assertTrue(worker.isRunning())
        self.assertGreaterEqual(len(progress_spy), 1)
        self.assertEqual(len(error_spy), 0)

        frame_payload = frame_spy[0][0]
        self.assertIsInstance(frame_payload, np.ndarray)
        np.testing.assert_array_equal(frame_payload, frame)

        release_event.set()
        self.assertTrue(finished_spy.wait(1000))
        worker.wait(1000)

        self.assertGreaterEqual(len(progress_spy), 3)
        self.assertGreaterEqual(len(status_spy), 2)
        self.assertEqual(status_spy[-1][0], "Pipeline complete")
        self.assertEqual(len(error_spy), 0)
        self.assertEqual(len(finished_spy), 1)
        self.assertEqual(finished_spy[0][0], report)

    def test_worker_emits_error_signal_when_service_raises(self) -> None:
        class FailingService(MockDefaultEvaluationService):
            def run_evaluation_job(
                self,
                job: EvaluationJob,
                *,
                frame_callback=None,
                progress_callback=None,
            ) -> EvaluationReport:
                _ = job, frame_callback, progress_callback
                raise RuntimeError("boom")

        release_event = threading.Event()
        service = FailingService(release_event=release_event, report=_make_report())
        job = _make_job()

        worker = EvaluationWorker(service=service, job=job)
        error_spy = QSignalSpy(worker.error_occurred)
        finished_spy = QSignalSpy(worker.finished)

        worker.start()

        self.assertTrue(error_spy.wait(1000))
        worker.wait(1000)

        self.assertEqual(len(error_spy), 1)
        self.assertIsInstance(error_spy[0][0], RuntimeError)
        self.assertEqual(str(error_spy[0][0]), "boom")
        self.assertEqual(len(finished_spy), 0)

    def test_controller_starts_worker_and_forwards_signals(self) -> None:
        release_event = threading.Event()
        report = _make_report("job_controller")
        service = MockDefaultEvaluationService(release_event=release_event, report=report)
        controller = ClinicalUIController(service=service, runtime_preflight=lambda: None)
        job = _make_job()
        frame = np.full((2, 2), 7, dtype=np.uint8)

        controller.set_selected_anatomy(service.anatomy)
        controller.set_selected_target(job.scenarios[0].target)
        controller.set_selected_execution_wire(service.execution_wire)
        controller.set_selected_wires((service.execution_wire,))
        controller.set_selected_candidates(job.candidates)
        controller.set_selected_scenarios(job.scenarios)

        progress_spy = QSignalSpy(controller.progress_updated)
        status_spy = QSignalSpy(controller.status_updated)
        frame_spy = QSignalSpy(controller.frame_ready)
        finished_spy = QSignalSpy(controller.finished)

        worker = controller.start_evaluation(
            job,
            frame_supplier=lambda: (frame,),
        )

        self.assertIs(controller.active_worker, worker)
        self.assertTrue(frame_spy.wait(1000))
        self.assertTrue(worker.isRunning())

        release_event.set()
        self.assertTrue(finished_spy.wait(1000))
        worker.wait(1000)

        self.assertGreaterEqual(len(progress_spy), 3)
        self.assertGreaterEqual(len(status_spy), 2)
        self.assertEqual(status_spy[-1][0], "Pipeline complete")
        self.assertEqual(len(frame_spy), 1)
        self.assertEqual(len(finished_spy), 1)
        self.assertEqual(finished_spy[0][0], report)
        self.assertIsNone(controller.active_worker)

    def test_controller_stores_metric_weights_and_keeps_them_accessible_during_start(self) -> None:
        release_event = threading.Event()
        report = _make_report("job_weights")
        service = MockDefaultEvaluationService(release_event=release_event, report=report)
        controller = ClinicalUIController(service=service, runtime_preflight=lambda: None)
        job = _make_job()

        expected_weights = {
            "success_rate": 1.0,
            "insertion_time": 0.5,
            "tip_force": 0.8,
        }
        for metric_name, value in expected_weights.items():
            controller.set_metric_weight(metric_name, value)

        self.assertEqual(controller.get_metric_weights(), expected_weights)

        finished_spy = QSignalSpy(controller.finished)
        worker = controller.start_evaluation(job)
        self.assertEqual(controller.get_metric_weights(), expected_weights)

        release_event.set()
        self.assertTrue(finished_spy.wait(1000))
        worker.wait(1000)

        self.assertEqual(controller.get_metric_weights(), expected_weights)

    def test_controller_blocks_run_when_runtime_preflight_fails(self) -> None:
        release_event = threading.Event()
        report = _make_report("job_preflight")
        service = MockDefaultEvaluationService(release_event=release_event, report=report)

        def _fail_preflight() -> None:
            raise RuntimeError("missing Sofa in active environment")

        controller = ClinicalUIController(service=service, runtime_preflight=_fail_preflight)
        job = _make_job()

        with self.assertRaises(RuntimeError) as exc:
            controller.start_evaluation(job)

        self.assertIn("missing Sofa", str(exc.exception))
        self.assertEqual(service.jobs, [])
        self.assertIsNone(controller.active_worker)

    def test_controller_preserves_visualization_settings_for_hidden_sofa_streaming(self) -> None:
        release_event = threading.Event()
        report = _make_report("job_visualized")
        service = MockDefaultEvaluationService(release_event=release_event, report=report)
        controller = ClinicalUIController(service=service, runtime_preflight=lambda: None)

        base_job = _make_job()
        visualized_job = replace(
            base_job,
            execution=replace(
                base_job.execution,
                visualization=VisualizationSpec(
                    enabled=True,
                    rendered_trials_per_candidate=1,
                ),
            ),
        )

        finished_spy = QSignalSpy(controller.finished)
        worker = controller.start_evaluation(visualized_job)

        release_event.set()
        self.assertTrue(finished_spy.wait(1000))
        worker.wait(1000)

        self.assertEqual(len(service.jobs), 1)
        self.assertTrue(service.jobs[0].execution.visualization.enabled)
        self.assertTrue(visualized_job.execution.visualization.enabled)


if __name__ == "__main__":
    unittest.main()
