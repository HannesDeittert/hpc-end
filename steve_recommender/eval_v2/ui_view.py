from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .models import (
    AorticArchAnatomy,
    BranchEndTarget,
    EvaluationJob,
    EvaluationScenario,
    ExecutionPlan,
)
from .ui_controller import ClinicalUIController


class DoctorDashboard(QWidget):
    """Clinical dashboard shell for eval_v2 with live frame display."""

    def __init__(self, *, controller: ClinicalUIController, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.controller = controller

        self._anatomy_by_id: Dict[str, AorticArchAnatomy] = {}

        self.setWindowTitle("Clinical Decision Support")

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        controls_box = QGroupBox("Clinical Setup")
        controls_layout = QFormLayout(controls_box)

        self.anatomy_combo = QComboBox()
        self.branch_combo = QComboBox()
        self.wire_combo = QComboBox()
        self.run_button = QPushButton("Run Simulation")

        controls_layout.addRow("Anatomy", self.anatomy_combo)
        controls_layout.addRow("Branch", self.branch_combo)
        controls_layout.addRow("Wire", self.wire_combo)

        priorities_box = QGroupBox("Clinical Priorities")
        priorities_layout = QFormLayout(priorities_box)

        self.success_rate_slider = QSlider(Qt.Horizontal)
        self.insertion_speed_slider = QSlider(Qt.Horizontal)
        self.wall_force_slider = QSlider(Qt.Horizontal)
        self.success_rate_value_label = QLabel()
        self.insertion_speed_value_label = QLabel()
        self.wall_force_value_label = QLabel()

        self._setup_weight_slider(
            slider=self.success_rate_slider,
            value_label=self.success_rate_value_label,
            metric_name="success_rate",
            initial=1.0,
        )
        self._setup_weight_slider(
            slider=self.insertion_speed_slider,
            value_label=self.insertion_speed_value_label,
            metric_name="insertion_time",
            initial=0.5,
        )
        self._setup_weight_slider(
            slider=self.wall_force_slider,
            value_label=self.wall_force_value_label,
            metric_name="tip_force",
            initial=0.8,
        )

        priorities_layout.addRow(
            "Success Rate",
            self._build_slider_row(self.success_rate_slider, self.success_rate_value_label),
        )
        priorities_layout.addRow(
            "Insertion Speed",
            self._build_slider_row(self.insertion_speed_slider, self.insertion_speed_value_label),
        )
        priorities_layout.addRow(
            "Safety (Wall Force)",
            self._build_slider_row(self.wall_force_slider, self.wall_force_value_label),
        )

        controls_layout.addRow(priorities_box)
        controls_layout.addRow(self.run_button)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)
        self.video_label = QLabel("Live fluoroscopy stream")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background: #111; color: #ddd; border: 1px solid #444;")
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)

        right_panel.addWidget(self.video_label, stretch=1)
        right_panel.addWidget(self.status_label)

        root_layout.addWidget(controls_box, stretch=0)
        root_layout.addLayout(right_panel, stretch=1)

        self.anatomy_combo.currentIndexChanged.connect(self._on_anatomy_changed)
        self.run_button.clicked.connect(self._on_run_clicked)

        self.controller.frame_ready.connect(self._on_frame_ready)
        self.controller.progress_updated.connect(self._on_progress)
        self.controller.status_updated.connect(self._on_status)
        self.controller.error_occurred.connect(self._on_error)
        self.controller.finished.connect(self._on_finished)

        self._populate_anatomies()
        self._populate_wires()

    def _populate_anatomies(self) -> None:
        anatomies = self.controller.get_anatomies()
        self._anatomy_by_id.clear()
        self.anatomy_combo.clear()
        for anatomy in anatomies:
            record_id = anatomy.record_id or f"{anatomy.arch_type}:{anatomy.seed}"
            self._anatomy_by_id[record_id] = anatomy
            self.anatomy_combo.addItem(record_id)
        if self.anatomy_combo.count() > 0:
            self._on_anatomy_changed(0)

    def _populate_wires(self) -> None:
        wires = self.controller.get_execution_wires()
        self.wire_combo.clear()
        for wire in wires:
            self.wire_combo.addItem(wire.tool_ref, userData=wire)
        if self.wire_combo.count() > 0:
            self.controller.set_selected_execution_wire(self.wire_combo.currentData())

    def _on_anatomy_changed(self, index: int) -> None:
        if index < 0:
            return
        record_id = self.anatomy_combo.itemText(index)
        anatomy = self._anatomy_by_id.get(record_id)
        if anatomy is None:
            return
        self.controller.set_selected_anatomy(anatomy)

        branches = self.controller.get_branches(anatomy)
        self.branch_combo.clear()
        for branch in branches:
            self.branch_combo.addItem(branch.name)

    def _on_run_clicked(self) -> None:
        anatomy_record = self.anatomy_combo.currentText().strip()
        anatomy = self._anatomy_by_id.get(anatomy_record)
        wire = self.wire_combo.currentData()
        branch = self.branch_combo.currentText().strip()
        if anatomy is None or wire is None or not branch:
            self.status_label.setText("Please select anatomy, branch, and wire before running.")
            return

        self.controller.set_selected_anatomy(anatomy)
        self.controller.set_selected_execution_wire(wire)

        candidates = self.controller.get_candidates(execution_wire=wire, include_cross_wire=False)
        if not candidates:
            self.status_label.setText("No candidates available for selected wire.")
            return

        target = BranchEndTarget(threshold_mm=5.0, branches=(branch,))
        scenario = EvaluationScenario(
            name="clinical_dashboard",
            anatomy=anatomy,
            target=target,
        )
        job = EvaluationJob(
            name="clinical_dashboard_job",
            scenarios=(scenario,),
            candidates=(candidates[0],),
            execution=ExecutionPlan(
                trials_per_candidate=1,
                base_seed=123,
                policy_device="cpu",
            ),
        )

        self.run_button.setEnabled(False)
        self.status_label.setText("Launching simulation...")
        self.controller.start_evaluation(job)

    def _on_frame_ready(self, frame: np.ndarray) -> None:
        image = self._numpy_to_qimage(np.asarray(frame))
        if image is None:
            self.status_label.setText("Received unsupported frame format.")
            return
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def _on_progress(self, percent: int) -> None:
        self.status_label.setText(f"{max(0, min(100, int(percent)))}%")

    def _on_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _on_error(self, error: Exception) -> None:
        self.run_button.setEnabled(True)
        self.status_label.setText(f"Error: {error}")

    def _on_finished(self, report) -> None:
        self.run_button.setEnabled(True)
        self.status_label.setText(
            f"Completed job={report.job_name} summaries={len(report.summaries)} trials={len(report.trials)}"
        )

    def _setup_weight_slider(
        self,
        *,
        slider: QSlider,
        value_label: QLabel,
        metric_name: str,
        initial: float,
    ) -> None:
        slider.setRange(0, 100)
        slider.setSingleStep(1)
        slider.setPageStep(5)

        def _on_value_changed(value: int) -> None:
            normalized = value / 100.0
            value_label.setText(f"{normalized:.2f}")
            self.controller.set_metric_weight(metric_name, normalized)

        slider.valueChanged.connect(_on_value_changed)
        slider.setValue(int(round(initial * 100)))

    @staticmethod
    def _build_slider_row(slider: QSlider, value_label: QLabel) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        value_label.setMinimumWidth(40)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row_layout.addWidget(slider, stretch=1)
        row_layout.addWidget(value_label)
        return row

    @staticmethod
    def _numpy_to_qimage(frame: np.ndarray) -> Optional[QImage]:
        if frame.ndim == 2:
            height, width = frame.shape
            bytes_per_line = frame.strides[0]
            return QImage(
                frame.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_Grayscale8,
            ).copy()

        if frame.ndim == 3 and frame.shape[2] == 3:
            height, width, _ = frame.shape
            bytes_per_line = frame.strides[0]
            return QImage(
                frame.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888,
            ).copy()

        if frame.ndim == 3 and frame.shape[2] == 4:
            height, width, _ = frame.shape
            bytes_per_line = frame.strides[0]
            return QImage(
                frame.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGBA8888,
            ).copy()

        return None


__all__ = ["DoctorDashboard"]
