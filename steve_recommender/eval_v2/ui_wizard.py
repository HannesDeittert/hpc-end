from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from .models import (
    BranchEndTarget,
    BranchIndexTarget,
    EvaluationCandidate,
    EvaluationJob,
    EvaluationScenario,
    ExecutionPlan,
    FluoroscopySpec,
    VisualizationSpec,
)
from .ui_controller import ClinicalUIController
from .ui_wizard_pages import (
    AnatomySelectionPage,
    BranchSelectionPage,
    ExecutionConfigPage,
    PipelineRunningPage,
    ResultsPage,
    TargetSelectionPage,
    WireSelectionPage,
    WizardPage,
)


class WizardShell(QWidget):
    home_requested = pyqtSignal()

    def __init__(self, *, controller: ClinicalUIController, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setObjectName("wizardShell")

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        logo_path = Path(__file__).resolve().parent / "assets" / "logo.svg"
        if logo_path.exists():
            logo = QSvgWidget(str(logo_path))
            logo.setFixedSize(252, 80)
            header.addWidget(logo, stretch=0)
        else:
            header.addWidget(QLabel("Chair Logo"), stretch=0)
        header.addStretch(1)
        self.home_button = QPushButton("Home")
        self.home_button.setProperty("buttonRole", "ghost")
        self.home_button.clicked.connect(self._on_home_clicked)
        header.addWidget(self.home_button, stretch=0)
        root.addLayout(header)

        self.page_stack = QStackedWidget()
        self.page_anatomy = AnatomySelectionPage(controller=self.controller)
        self.page_branch = BranchSelectionPage(controller=self.controller)
        self.page_target = TargetSelectionPage(controller=self.controller)
        self.page_wires = WireSelectionPage(controller=self.controller)
        self.page_execution = ExecutionConfigPage(controller=self.controller)
        self.page_pipeline = PipelineRunningPage(controller=self.controller)
        self.page_results = ResultsPage(controller=self.controller)

        self._pages = (
            self.page_anatomy,
            self.page_branch,
            self.page_target,
            self.page_wires,
            self.page_execution,
            self.page_pipeline,
            self.page_results,
        )
        for page in self._pages:
            self.page_stack.addWidget(page)

        root.addWidget(self.page_stack, stretch=1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        self.back_button = QPushButton("Back")
        self.forward_button = QPushButton("Forward")
        self.back_button.setProperty("buttonRole", "ghost")
        self.forward_button.setProperty("buttonRole", "primary")
        self.back_button.clicked.connect(self._go_back)
        self.forward_button.clicked.connect(self._go_forward)
        self.forward_button.setEnabled(False)
        footer.addWidget(self.back_button, stretch=0)
        footer.addStretch(1)
        footer.addWidget(self.forward_button, stretch=0)
        root.addLayout(footer)

        self.page_stack.currentChanged.connect(self._on_step_changed)
        self.page_pipeline.run_finished.connect(self._on_pipeline_finished)

        self._bound_page: Optional[WizardPage] = None
        self._historical_results_mode = False
        self._on_step_changed(0)

    def _on_home_clicked(self) -> None:
        self._historical_results_mode = False
        self.controller.reset_wizard_state()
        self.page_stack.setCurrentIndex(0)
        self.home_requested.emit()

    def _on_step_changed(self, index: int) -> None:
        page = self._pages[index]
        if self._bound_page is not None:
            try:
                self._bound_page.validation_passed.disconnect(self._on_validation_changed)
            except TypeError:
                pass
        self._bound_page = page
        self._bound_page.validation_passed.connect(self._on_validation_changed)

        page.on_activated()
        self.back_button.setEnabled(index > 0)

        if index == 4:
            self.forward_button.setText("Run Simulation")
        else:
            self.forward_button.setText("Forward")

        running_step = index == 5
        self.back_button.setVisible(not running_step)
        self.forward_button.setVisible(not running_step)

        if index == 6 or self._historical_results_mode:
            self.forward_button.setEnabled(False)
            self.back_button.setEnabled(False)
            self.back_button.setVisible(False)
        else:
            self.forward_button.setEnabled(page.is_valid() and self.controller.can_forward_from_step(index))

    def _on_validation_changed(self, valid: bool) -> None:
        index = self.page_stack.currentIndex()
        if index in (5, 6):
            self.forward_button.setEnabled(False)
            return
        self.forward_button.setEnabled(bool(valid) and self.controller.can_forward_from_step(index))

    def _go_back(self) -> None:
        index = self.page_stack.currentIndex()
        if index <= 0:
            return
        self.page_stack.setCurrentIndex(index - 1)

    def _go_forward(self) -> None:
        index = self.page_stack.currentIndex()
        if not self.controller.can_forward_from_step(index):
            return

        if index < 4:
            self.page_stack.setCurrentIndex(index + 1)
            return

        if index == 4:
            job = self._build_job_from_state()
            self._historical_results_mode = False
            self.page_stack.setCurrentIndex(5)
            self.page_pipeline.start_run(job)

    def _on_pipeline_finished(self, report) -> None:
        self.page_results.set_report(report, is_from_archive=False)
        self.controller.reset_wizard_state()
        self._historical_results_mode = False
        self.page_stack.setCurrentIndex(6)

    def show_historical_report(self, report) -> None:
        self._historical_results_mode = True
        self.page_results.set_report(report, is_from_archive=True)
        self.page_stack.setCurrentIndex(6)

    def _build_job_from_state(self) -> EvaluationJob:
        state = self.controller.get_wizard_state()
        if state.anatomy is None:
            raise ValueError("Wizard state incomplete: anatomy")
        if not state.branch:
            raise ValueError("Wizard state incomplete: branch")
        if state.target_position is None:
            raise ValueError("Wizard state incomplete: target_position")
        if not state.selected_wires:
            raise ValueError("Wizard state incomplete: selected_wires")

        branch = self.controller.get_branch(state.anatomy, branch_name=state.branch)
        if state.target_position == "terminal_end":
            target = BranchEndTarget(threshold_mm=5.0, branches=(state.branch,))
        else:
            ratio = float(state.target_position)
            index = int(round(max(0.0, min(1.0, ratio)) * max(branch.point_count - 1, 0)))
            target = BranchIndexTarget(branch=state.branch, index=index, threshold_mm=5.0)

        scenario = EvaluationScenario(
            name="wizard_scenario",
            anatomy=state.anatomy,
            target=target,
            fluoroscopy=FluoroscopySpec(),
        )

        candidates: list[EvaluationCandidate] = []
        for wire in state.selected_wires:
            wire_candidates = self.controller.get_candidates(execution_wire=wire, include_cross_wire=False)
            if wire_candidates:
                candidates.append(wire_candidates[0])

        if not candidates:
            raise ValueError("No candidates resolved for selected wires")

        trials = 1 if state.is_deterministic else max(1, int(state.trials_per_wire))
        execution = ExecutionPlan(
            trials_per_candidate=trials,
            policy_mode="deterministic" if state.is_deterministic else "stochastic",
            visualization=VisualizationSpec(
                enabled=state.is_visualized,
                rendered_trials_per_candidate=(
                    max(1, int(state.visualized_trials_count)) if state.is_visualized else 1
                ),
            ),
        )

        return EvaluationJob(
            name=f"wizard_job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            scenarios=(scenario,),
            candidates=tuple(candidates),
            execution=execution,
        )


__all__ = ["WizardShell"]
