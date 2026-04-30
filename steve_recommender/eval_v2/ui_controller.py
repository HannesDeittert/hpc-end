from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import importlib
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal

from .models import (
    AorticArchAnatomy,
    AnatomyBranch,
    EvaluationCandidate,
    EvaluationJob,
    EvaluationReport,
    EvaluationScenario,
    HistoricalReportSummary,
    ScoringSpec,
    TargetModeDescriptor,
    WireRef,
)
from .service import EvaluationService


FrameSupplier = Callable[[], Iterable[np.ndarray]]
RuntimePreflight = Callable[[], None]


def assert_runtime_dependencies() -> None:
    """Ensure the active Python environment can run the eval runtime.

    We gate run-start on Sofa-related imports so users cannot accidentally start
    a run from an environment that does not contain the simulation stack.
    """

    required_modules = ("Sofa", "SofaRuntime")
    missing: list[str] = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Cannot start evaluation: missing runtime dependencies "
            f"[{joined}] in interpreter {sys.executable}. "
            "Activate the correct environment (with Sofa/stEVE installed) and retry."
        )


@dataclass
class WizardState:
    anatomy: Optional[AorticArchAnatomy] = None
    branch: str = ""
    target_position: Optional[float | str] = None
    selected_wires: list[WireRef] = field(default_factory=list)
    is_deterministic: bool = True
    trials_per_wire: int = 1
    stochastic_environment_mode: str = "random_start"
    is_visualized: bool = False
    visualized_trials_count: int = 1
    worker_count: int = 1


class EvaluationWorker(QThread):
    """Run one eval_v2 job on a background Qt thread.

    The worker is intentionally small: it runs the service job, emits optional
    frame updates from an injected frame supplier, and converts exceptions into
    a signal instead of letting them escape into the Qt event loop.
    """

    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    frame_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(object)
    finished = pyqtSignal(object)

    def __init__(
        self,
        *,
        service: EvaluationService,
        job: EvaluationJob,
        frame_supplier: Optional[FrameSupplier] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._job = job
        self._frame_supplier = frame_supplier
        self._candidate_wire_map = {
            candidate.name: candidate.execution_wire.tool_ref for candidate in self._job.candidates
        }

    @staticmethod
    def _parse_progress_event(message: str) -> Tuple[str, Dict[str, str]]:
        parts = [part for part in str(message).strip().split(" ") if part]
        if not parts:
            return "", {}
        event = parts[0]
        payload: Dict[str, str] = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            payload[key] = value
        return event, payload

    @staticmethod
    def _safe_int(value: Optional[str], *, default: int = 0) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except Exception:
            return default

    def run(self) -> None:
        try:
            total_trials = (
                len(self._job.scenarios) * len(self._job.candidates) * len(self._job.execution.seeds)
            )
            max_episode_steps = max(1, int(self._job.execution.max_episode_steps))
            rendered_trial_limit = max(
                1, int(self._job.execution.visualization.rendered_trials_per_candidate)
            )
            stream_frames = bool(self._job.execution.visualization.enabled)
            trials_started = 0
            trials_completed = 0
            current_step = 0
            active_trial_index = 0
            last_percent = -1

            def _compute_percent(*, completed_trials: int, trial_step: int) -> int:
                if total_trials <= 0:
                    return 100
                bounded_step = max(0.0, min(float(trial_step) / float(max_episode_steps), 1.0))
                progress = (float(completed_trials) + bounded_step) / float(total_trials)
                return max(0, min(100, int(round(progress * 100.0))))

            def _emit_progress(value: int) -> None:
                nonlocal last_percent
                bounded = max(0, min(100, int(value)))
                if bounded == last_percent:
                    return
                last_percent = bounded
                self.progress_updated.emit(bounded)

            _emit_progress(0)
            self.status_updated.emit("Loading Anatomy...")
            _emit_progress(1)

            def _progress_callback(message: str) -> None:
                nonlocal trials_started, trials_completed, current_step, active_trial_index
                event, payload = self._parse_progress_event(message)

                if event == "runtime_prepare":
                    candidate_name = payload.get("candidate", "")
                    wire_name = self._candidate_wire_map.get(candidate_name, candidate_name)
                    if wire_name:
                        self.status_updated.emit(f"Initializing Physics: {wire_name}...")
                    else:
                        self.status_updated.emit("Initializing Physics...")
                    return

                if event == "trial_start":
                    trials_started = min(trials_started + 1, max(total_trials, 1))
                    current_step = 0
                    active_trial_index = max(
                        0,
                        self._safe_int(payload.get("index"), default=0),
                    )
                    candidate_name = payload.get("candidate", "")
                    active_wire = self._candidate_wire_map.get(candidate_name, candidate_name)
                    if total_trials > 0:
                        self.status_updated.emit(
                            f"Running Simulation: {active_wire} - Trial {trials_started} of {total_trials}"
                        )
                    else:
                        self.status_updated.emit(f"Running Simulation: {active_wire}")
                    _emit_progress(_compute_percent(completed_trials=trials_completed, trial_step=0))
                    return

                if event == "trial_step":
                    current_step = max(0, self._safe_int(payload.get("step"), default=0))
                    _emit_progress(
                        _compute_percent(completed_trials=trials_completed, trial_step=current_step)
                    )
                    return

                if event == "trial_end":
                    trials_completed = min(trials_completed + 1, max(total_trials, 0))
                    current_step = 0
                    _emit_progress(_compute_percent(completed_trials=trials_completed, trial_step=0))
                    return

                if event == "parallel_start":
                    workers = self._safe_int(payload.get("workers"), default=1)
                    self.status_updated.emit(
                        f"Running Headless Evaluation: {workers} workers..."
                    )
                    _emit_progress(0)
                    return

                if event == "parallel_trial_done":
                    completed = self._safe_int(payload.get("completed"), default=trials_completed)
                    total = self._safe_int(payload.get("total"), default=total_trials)
                    trials_completed = max(0, min(completed, max(total, total_trials, 1)))
                    self.status_updated.emit(
                        f"Running Headless Evaluation: {trials_completed} of {max(total, total_trials)} trials"
                    )
                    denom = max(1, total, total_trials)
                    _emit_progress(int(round(float(trials_completed) / float(denom) * 100.0)))
                    return

                if event == "parallel_end":
                    trials_completed = max(total_trials, self._safe_int(payload.get("completed"), default=total_trials))
                    _emit_progress(100)
                    return

            def _frame_callback(frame: np.ndarray) -> None:
                if not stream_frames:
                    return
                if active_trial_index >= rendered_trial_limit:
                    return
                self.frame_ready.emit(np.asarray(frame))

            if self._frame_supplier is not None:
                for frame in self._frame_supplier():
                    self.frame_ready.emit(np.asarray(frame))

            report = self._service.run_evaluation_job(
                self._job,
                frame_callback=_frame_callback if stream_frames else None,
                progress_callback=_progress_callback,
            )
            self.status_updated.emit("Pipeline complete")
            _emit_progress(100)
            self.finished.emit(report)
        except Exception as exc:  # pragma: no cover - defensive guard
            self.error_occurred.emit(exc)
            self.status_updated.emit(f"Error: {exc}")
            self.progress_updated.emit(100)


class ClinicalUIController(QObject):
    """View-model for the future clinical UI.

    The controller deliberately stays inside eval_v2 and only exposes data/query
    methods plus the thread launcher used by the UI layer.
    """

    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    frame_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(object)
    finished = pyqtSignal(object)

    def __init__(
        self,
        *,
        service: EvaluationService,
        runtime_preflight: Optional[RuntimePreflight] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._runtime_preflight: RuntimePreflight = runtime_preflight or assert_runtime_dependencies
        self._active_worker: Optional[EvaluationWorker] = None

        self.selected_anatomy: Optional[AorticArchAnatomy] = None
        self.selected_target = None
        self.selected_execution_wire: Optional[WireRef] = None
        self.selected_wires: Tuple[WireRef, ...] = ()
        self.selected_candidates: Tuple[EvaluationCandidate, ...] = ()
        self.selected_scenarios: Tuple[EvaluationScenario, ...] = ()
        self._metric_weights: Dict[str, float] = {
            "success_rate": 1.0,
            "insertion_time": 0.5,
            "tip_force": 0.8,
        }
        self._default_results_controls: Dict[str, float] = {}
        self._results_controls: Dict[str, float] = {}
        self._results_controls_schema: Dict[str, Dict[str, Any]] = {}
        self.configure_results_scoring(ScoringSpec())
        self._wizard_state = WizardState()

    @property
    def active_worker(self) -> Optional[EvaluationWorker]:
        return self._active_worker

    def get_anatomies(self, *, registry_path=None):
        return self._service.list_anatomies(registry_path=registry_path)

    def get_execution_wires(self):
        return self._service.list_execution_wires()

    def get_startable_wires(self):
        return self._service.list_startable_wires()

    def get_branches(self, anatomy: AorticArchAnatomy) -> Tuple[AnatomyBranch, ...]:
        return self._service.list_branches(anatomy)

    def get_branch(self, anatomy: AorticArchAnatomy, *, branch_name: str) -> AnatomyBranch:
        return self._service.get_branch(anatomy, branch_name=branch_name)

    def get_target_modes(self) -> Tuple[TargetModeDescriptor, ...]:
        return self._service.list_target_modes()

    def get_candidates(
        self,
        *,
        execution_wire: WireRef,
        include_cross_wire: bool = True,
    ) -> Tuple[EvaluationCandidate, ...]:
        return self._service.list_candidates(
            execution_wire=execution_wire,
            include_cross_wire=include_cross_wire,
        )

    def list_historical_reports(self) -> Tuple[HistoricalReportSummary, ...]:
        return self._service.list_historical_reports()

    def load_manifest_from_disk(self, manifest_json_path: Path) -> EvaluationReport:
        return self._service.load_manifest_from_disk(manifest_json_path)

    def save_clinical_feedback(
        self,
        report_id: str,
        feedback_data: Dict[str, object],
    ) -> Path:
        return self._service.save_clinical_feedback(
            report_id=report_id,
            feedback_data=dict(feedback_data),
        )

    def set_selected_anatomy(self, anatomy: Optional[AorticArchAnatomy]) -> None:
        self.selected_anatomy = anatomy

    def set_selected_target(self, target) -> None:
        self.selected_target = target

    def set_selected_execution_wire(self, wire: Optional[WireRef]) -> None:
        self.selected_execution_wire = wire

    def set_selected_wires(self, wires: Sequence[WireRef]) -> None:
        self.selected_wires = tuple(wires)

    def set_selected_candidates(self, candidates: Sequence[EvaluationCandidate]) -> None:
        self.selected_candidates = tuple(candidates)

    def set_selected_scenarios(self, scenarios: Sequence[EvaluationScenario]) -> None:
        self.selected_scenarios = tuple(scenarios)

    def set_metric_weight(self, metric_name: str, value: float) -> None:
        self._metric_weights[str(metric_name)] = float(value)

    def get_metric_weights(self) -> Dict[str, float]:
        return dict(self._metric_weights)

    def configure_results_scoring(self, scoring_spec: Optional[ScoringSpec]) -> None:
        scoring = scoring_spec or ScoringSpec()
        default_weights = dict(scoring.candidate_score.default_weights)
        self._default_results_controls = {
            "score_weight_safety": float(default_weights.get("score_safety", 0.5)),
            "score_weight_efficiency": float(default_weights.get("score_efficiency", 0.5)),
            "score_lambda": float(scoring.candidate_score.lambda_),
            "score_beta": float(scoring.candidate_score.beta),
            "force_max_N": float(scoring.force.force_max_N),
            "force_score_c": float(scoring.safety_score.c),
            "force_score_p": float(scoring.safety_score.p),
            "force_score_k": float(scoring.safety_score.k),
            "force_score_F50_N": float(scoring.safety_score.F50_N),
        }
        self._results_controls = dict(self._default_results_controls)
        self._results_controls_schema = {
            "Candidate Score": {
                "controls": {
                    "Safety Weight": {
                        "key": "score_weight_safety",
                        "default": self._default_results_controls["score_weight_safety"],
                        "min": 0.0,
                        "max": 1.0,
                        "resolution": 100,
                        "format": "{value:.2f}",
                    },
                    "Efficiency Weight": {
                        "key": "score_weight_efficiency",
                        "default": self._default_results_controls["score_weight_efficiency"],
                        "min": 0.0,
                        "max": 1.0,
                        "resolution": 100,
                        "format": "{value:.2f}",
                    },
                    "Lambda": {
                        "key": "score_lambda",
                        "default": self._default_results_controls["score_lambda"],
                        "min": 0.0,
                        "max": 3.0,
                        "resolution": 100,
                        "format": "{value:.2f}",
                    },
                    "Beta": {
                        "key": "score_beta",
                        "default": self._default_results_controls["score_beta"],
                        "min": 0.0,
                        "max": 2.0,
                        "resolution": 100,
                        "format": "{value:.2f}",
                    },
                },
            },
            "Safety Model": {
                "controls": {
                    "Force Max (N)": {
                        "key": "force_max_N",
                        "default": self._default_results_controls["force_max_N"],
                        "min": 0.1,
                        "max": 5.0,
                        "resolution": 100,
                        "format": "{value:.2f}",
                    },
                    "Force Score c": {
                        "key": "force_score_c",
                        "default": self._default_results_controls["force_score_c"],
                        "min": 0.0,
                        "max": 2.0,
                        "resolution": 100,
                        "format": "{value:.2f}",
                    },
                    "Force Score p": {
                        "key": "force_score_p",
                        "default": self._default_results_controls["force_score_p"],
                        "min": 0.1,
                        "max": 5.0,
                        "resolution": 100,
                        "format": "{value:.2f}",
                    },
                    "Force Score k": {
                        "key": "force_score_k",
                        "default": self._default_results_controls["force_score_k"],
                        "min": 0.1,
                        "max": 20.0,
                        "resolution": 100,
                        "format": "{value:.2f}",
                    },
                    "Force Score F50 (N)": {
                        "key": "force_score_F50_N",
                        "default": self._default_results_controls["force_score_F50_N"],
                        "min": 0.1,
                        "max": 5.0,
                        "resolution": 100,
                        "format": "{value:.2f}",
                    },
                },
            },
        }

    def get_default_results_controls(self) -> Dict[str, float]:
        return dict(self._default_results_controls)

    def set_results_control_value(self, control_name: str, value: float) -> None:
        self._results_controls[str(control_name)] = float(value)

    def get_results_control_values(self) -> Dict[str, float]:
        return dict(self._results_controls)

    def reset_results_control_values(self) -> Dict[str, float]:
        self._results_controls = dict(self._default_results_controls)
        return self.get_results_control_values()

    def get_results_control_schema(self) -> Dict[str, Dict[str, Any]]:
        return deepcopy(self._results_controls_schema)

    def reset_wizard_state(self) -> None:
        self._wizard_state = WizardState()

    def get_wizard_state(self) -> WizardState:
        return self._wizard_state

    def set_wizard_anatomy(self, anatomy: Optional[AorticArchAnatomy]) -> None:
        self._wizard_state.anatomy = anatomy

    def set_wizard_branch(self, branch: str) -> None:
        self._wizard_state.branch = str(branch).strip()

    def set_wizard_target_position(self, target_position: Optional[float | str]) -> None:
        self._wizard_state.target_position = target_position

    def set_wizard_selected_wires(self, wires: Sequence[WireRef]) -> None:
        self._wizard_state.selected_wires = list(wires)

    def set_wizard_execution_config(
        self,
        *,
        is_deterministic: bool,
        trials_per_wire: int,
        stochastic_environment_mode: str,
        is_visualized: bool,
        visualized_trials_count: int,
        worker_count: int,
    ) -> None:
        self._wizard_state.is_deterministic = bool(is_deterministic)
        self._wizard_state.trials_per_wire = max(int(trials_per_wire), 1)
        self._wizard_state.stochastic_environment_mode = str(stochastic_environment_mode)
        self._wizard_state.is_visualized = bool(is_visualized)
        self._wizard_state.visualized_trials_count = max(int(visualized_trials_count), 1)
        self._wizard_state.worker_count = max(int(worker_count), 1)

    def can_forward_from_step(self, step_index: int) -> bool:
        state = self._wizard_state
        if step_index == 0:
            return state.anatomy is not None
        if step_index == 1:
            return bool(state.branch)
        if step_index == 2:
            return state.target_position is not None
        if step_index == 3:
            return len(state.selected_wires) >= 1
        if step_index == 4:
            if state.trials_per_wire < 1:
                return False
            if state.stochastic_environment_mode not in {"fixed_start", "random_start"}:
                return False
            if state.is_visualized and state.visualized_trials_count > state.trials_per_wire:
                return False
            return True
        return True

    def start_evaluation(
        self,
        job: EvaluationJob,
        *,
        frame_supplier: Optional[FrameSupplier] = None,
    ) -> EvaluationWorker:
        if self._active_worker is not None and self._active_worker.isRunning():
            raise RuntimeError("An evaluation is already running")

        self._runtime_preflight()

        worker = EvaluationWorker(
            service=self._service,
            job=job,
            frame_supplier=frame_supplier,
            parent=self,
        )
        worker.progress_updated.connect(self.progress_updated.emit)
        worker.status_updated.connect(self.status_updated.emit)
        worker.frame_ready.connect(self.frame_ready.emit)
        worker.error_occurred.connect(self.error_occurred.emit)
        worker.finished.connect(self._handle_worker_finished)
        worker.error_occurred.connect(self._handle_worker_error)
        worker.finished.connect(self.finished.emit)
        worker.finished.connect(worker.deleteLater)
        worker.error_occurred.connect(worker.deleteLater)
        self._active_worker = worker
        worker.start()
        return worker

    def _handle_worker_finished(self, report: EvaluationReport) -> None:
        _ = report
        self._active_worker = None

    def _handle_worker_error(self, error: Exception) -> None:
        _ = error
        self._active_worker = None


__all__ = [
    "assert_runtime_dependencies",
    "ClinicalUIController",
    "EvaluationWorker",
    "FrameSupplier",
    "RuntimePreflight",
    "WizardState",
]
