from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGraphicsColorizeEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QStackedWidget,
)

from .models import (
    AnatomyBranch,
    AorticArchAnatomy,
    EvaluationJob,
    EvaluationReport,
    ScoringSpec,
    TrialResult,
    WireRef,
)
from .scoring import (
    force_within_safety_threshold,
    score_trial,
    soft_score_total,
    valid_for_ranking,
)
from .ui_controller import ClinicalUIController
from .ui_trace_viewer import TraceViewerPanel


class WizardPage(QWidget):
    validation_passed = pyqtSignal(bool)

    def __init__(
        self, *, controller: ClinicalUIController, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self._is_valid = False

    def on_activated(self) -> None:
        self.set_valid(False)

    def is_valid(self) -> bool:
        return self._is_valid

    def set_valid(self, valid: bool) -> None:
        self._is_valid = bool(valid)
        self.validation_passed.emit(self._is_valid)


class Anatomy3DWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mesh_actor = None
        self._branch_actor = None
        self._branch_actors: Dict[str, object] = {}
        self._selected_branch_name: Optional[str] = None
        self._target_actor = None
        self._pv = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plotter = None
        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor

            off_screen = os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen"
            self._pv = pv
            self._plotter = QtInteractor(self, off_screen=off_screen)
            self._plotter.set_background("#2A2A2A")
            layout.addWidget(self._plotter.interactor)
        except Exception:  # pragma: no cover - environment dependent
            layout.addWidget(QLabel("pyvistaqt is not available in this environment."))

    def clear_scene(self) -> None:
        if self._plotter is None:
            return
        self._plotter.clear()
        self._mesh_actor = None
        self._branch_actor = None
        self._branch_actors = {}
        self._selected_branch_name = None
        self._target_actor = None

    def load_anatomy(self, anatomy: AorticArchAnatomy) -> None:
        if self._plotter is None:
            return
        self.clear_scene()

        mesh_path = anatomy.visualization_mesh_path or anatomy.simulation_mesh_path
        if mesh_path is None:
            return
        if not mesh_path.exists():
            return

        if self._pv is None:
            return
        mesh = self._pv.read(str(mesh_path))
        self._mesh_actor = self._plotter.add_mesh(
            mesh,
            color="#8B949E",
            opacity=0.35,
            smooth_shading=True,
        )
        self._plotter.reset_camera()

    def show_branches(self, branches: Sequence[AnatomyBranch]) -> None:
        if self._plotter is None or self._pv is None:
            return

        for actor in self._branch_actors.values():
            self._plotter.remove_actor(actor)
        self._branch_actors = {}
        self._selected_branch_name = None

        for branch in branches:
            points = np.asarray(branch.centerline_points_vessel_cs, dtype=float)
            if len(points) < 2:
                continue
            spline = self._pv.Spline(points, len(points))
            actor = self._plotter.add_mesh(
                spline, color="#5c7080", opacity=0.45, line_width=3
            )
            self._branch_actors[branch.name] = actor

        self._plotter.render()

    def highlight_branch(self, branch: AnatomyBranch) -> None:
        if self._plotter is None:
            return

        if branch.name in self._branch_actors:
            if self._selected_branch_name is not None:
                previous = self._branch_actors.get(self._selected_branch_name)
                if previous is not None:
                    self._style_actor(
                        previous, color="#5c7080", opacity=0.45, line_width=3
                    )

            current = self._branch_actors[branch.name]
            self._style_actor(current, color="#F2AC32", opacity=1.0, line_width=7)
            self._selected_branch_name = branch.name
            self._plotter.render()
            return

        points = np.asarray(branch.centerline_points_vessel_cs, dtype=float)
        if self._pv is None or len(points) < 2:
            return
        if self._branch_actor is not None:
            self._plotter.remove_actor(self._branch_actor)
        spline = self._pv.Spline(points, len(points))
        self._branch_actor = self._plotter.add_mesh(
            spline, color="#F2AC32", opacity=1.0, line_width=7
        )
        self._plotter.render()

    @staticmethod
    def _style_actor(
        actor: object, *, color: str, opacity: float, line_width: float
    ) -> None:
        prop = None
        try:
            prop = actor.prop
        except Exception:
            try:
                prop = actor.GetProperty()
            except Exception:
                prop = None
        if prop is None:
            return
        try:
            prop.color = color
        except Exception:
            pass
        try:
            prop.opacity = float(opacity)
        except Exception:
            pass
        try:
            prop.line_width = float(line_width)
        except Exception:
            pass

    def mark_target(self, branch: AnatomyBranch, ratio: float) -> None:
        if self._plotter is None:
            return
        if self._target_actor is not None:
            self._plotter.remove_actor(self._target_actor)

        points = np.asarray(branch.centerline_points_vessel_cs, dtype=float)
        if len(points) == 1:
            point = points[0]
        else:
            scaled = max(0.0, min(1.0, ratio)) * (len(points) - 1)
            lower = int(np.floor(scaled))
            upper = min(lower + 1, len(points) - 1)
            alpha = scaled - lower
            point = (1.0 - alpha) * points[lower] + alpha * points[upper]

        if self._pv is None:
            return
        sphere = self._pv.Sphere(radius=1.0, center=tuple(float(v) for v in point))
        self._target_actor = self._plotter.add_mesh(sphere, color="#90be6d")


class AnatomyCardWidget(QWidget):
    clicked = pyqtSignal()

    def __init__(
        self,
        *,
        title: str,
        metadata: str,
        thumbnail: Optional[QPixmap] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("anatomyCard")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("selected", "false")
        self.setCursor(Qt.PointingHandCursor)
        self._base_thumbnail = thumbnail or self._build_placeholder_thumbnail(size=140)

        self.setStyleSheet(
            """
AnatomyCardWidget {
    background-color: #2A2A2A;
    border: 2px solid #30363D;
    border-radius: 12px;
    padding: 16px;
}

AnatomyCardWidget:hover {
    background-color: #333333;
    border: 2px solid #555555;
}

AnatomyCardWidget QLabel#anatomyTitle {
    font-size: 16px;
    font-weight: bold;
    color: #E6EDF3;
    margin-top: 12px;
}

AnatomyCardWidget QLabel#anatomyMeta {
    font-size: 13px;
    color: #8B949E;
    margin-top: 4px;
}

AnatomyCardWidget[selected="true"] {
    background-color: rgba(242, 172, 50, 0.05);
    border: 2px solid #F2AC32;
}

AnatomyCardWidget[selected="true"] QLabel#anatomyTitle {
    color: #F2AC32;
}
"""
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(4)

        self.thumbnail_label = QLabel()
        self.thumbnail_label.setObjectName("anatomyThumbnail")
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setFixedSize(150, 150)
        self.thumbnail_label.setPixmap(self._base_thumbnail)

        self._colorize_effect = QGraphicsColorizeEffect(self.thumbnail_label)
        self.thumbnail_label.setGraphicsEffect(self._colorize_effect)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("anatomyTitle")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.meta_label = QLabel(metadata)
        self.meta_label.setObjectName("anatomyMeta")
        self.meta_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.thumbnail_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.title_label)
        layout.addWidget(self.meta_label)

        self.set_selected(False)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", "true" if selected else "false")
        if selected:
            self._colorize_effect.setColor(QColor("#F2AC32"))
            self._colorize_effect.setStrength(1.0)
        else:
            self._colorize_effect.setColor(QColor("#8B949E"))
            self._colorize_effect.setStrength(0.85)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    @staticmethod
    def _build_placeholder_thumbnail(*, size: int) -> QPixmap:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor("#E6EDF3"))
        pen.setWidth(6)
        painter.setPen(pen)

        center_x = size // 2
        top = int(size * 0.18)
        bottom = int(size * 0.82)

        painter.drawLine(center_x, top, center_x, bottom)
        painter.drawLine(center_x, int(size * 0.45), int(size * 0.24), int(size * 0.30))
        painter.drawLine(center_x, int(size * 0.52), int(size * 0.78), int(size * 0.36))
        painter.drawLine(center_x, int(size * 0.64), int(size * 0.30), int(size * 0.76))

        painter.end()
        return pixmap


class AnatomySelectionPage(WizardPage):
    def __init__(
        self, *, controller: ClinicalUIController, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(controller=controller, parent=parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        self._title = QLabel("Step 1: Select Anatomy")
        self._title.setStyleSheet("font-size: 20px; font-weight: 600;")
        root.addWidget(self._title)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(24)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._scroll.setWidget(self._container)
        root.addWidget(self._scroll, stretch=1)

        self._cards: List[AnatomyCardWidget] = []
        self._anatomies: List[AorticArchAnatomy] = []
        self._thumbnail_cache: Dict[str, QPixmap] = {}

    def on_activated(self) -> None:
        self._render_anatomies()

    def _render_anatomies(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._cards.clear()
        self._anatomies = list(self.controller.get_anatomies())
        self.set_valid(False)

        for idx, anatomy in enumerate(self._anatomies):
            label = anatomy.record_id or f"{anatomy.arch_type}:{anatomy.seed}"
            metadata = f"arch={anatomy.arch_type} seed={anatomy.seed}"
            thumbnail = self._thumbnail_for_anatomy(anatomy)
            card = AnatomyCardWidget(
                title=label, metadata=metadata, thumbnail=thumbnail
            )
            card.setMinimumSize(220, 240)
            card.clicked.connect(lambda i=idx: self._on_selected(i))
            row = idx // 3
            col = idx % 3
            self._grid.addWidget(card, row, col)
            self._cards.append(card)

    def _thumbnail_for_anatomy(self, anatomy: AorticArchAnatomy) -> Optional[QPixmap]:
        mesh_path = anatomy.visualization_mesh_path or anatomy.simulation_mesh_path
        if mesh_path is None or not mesh_path.exists():
            return None

        cache_key = str(mesh_path.resolve())
        cached = self._thumbnail_cache.get(cache_key)
        if cached is not None:
            return cached

        pixmap = self._render_mesh_thumbnail(mesh_path)
        if pixmap is not None:
            self._thumbnail_cache[cache_key] = pixmap
        return pixmap

    @staticmethod
    def _render_mesh_thumbnail(mesh_path, *, size: int = 150) -> Optional[QPixmap]:
        try:
            import pyvista as pv
        except Exception:
            return None

        try:
            plotter = pv.Plotter(off_screen=True, window_size=(size, size))
            plotter.set_background("#1f1f1f")
            mesh = pv.read(str(mesh_path))
            plotter.add_mesh(
                mesh,
                color="#d9dee6",
                smooth_shading=True,
                ambient=0.25,
                diffuse=0.7,
                specular=0.2,
            )
            plotter.view_isometric()
            plotter.camera.zoom(1.2)
            image = np.asarray(
                plotter.screenshot(return_img=True, window_size=(size, size))
            )
            plotter.close()
        except Exception:
            return None

        if image.ndim != 3:
            return None

        image = np.ascontiguousarray(image)
        h, w = image.shape[:2]
        if image.shape[2] == 4:
            qimg = QImage(
                image.data, w, h, image.strides[0], QImage.Format_RGBA8888
            ).copy()
            return QPixmap.fromImage(qimg)
        if image.shape[2] == 3:
            qimg = QImage(
                image.data, w, h, image.strides[0], QImage.Format_RGB888
            ).copy()
            return QPixmap.fromImage(qimg)
        return None

    def _on_selected(self, index: int) -> None:
        for idx, card in enumerate(self._cards):
            card.set_selected(idx == index)
        anatomy = self._anatomies[index]
        self.controller.set_wizard_anatomy(anatomy)
        self.controller.set_wizard_branch("")
        self.controller.set_wizard_target_position(None)
        self.controller.set_wizard_selected_wires(())
        self.set_valid(True)


class BranchSelectionPage(WizardPage):
    def __init__(
        self, *, controller: ClinicalUIController, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(controller=controller, parent=parent)

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        self._view_container = QWidget()
        self._view_layout = QVBoxLayout(self._view_container)
        self._view_layout.setContentsMargins(0, 0, 0, 0)
        self.view: Optional[Anatomy3DWidget] = None
        root.addWidget(self._view_container, stretch=3)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)
        self._title = QLabel("Step 2: Select Branch")
        self._title.setStyleSheet("font-size: 18px; font-weight: 600;")
        right_panel.addWidget(self._title)

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._radio_layout = QVBoxLayout()
        self._radio_layout.setSpacing(12)

        self._branch_buttons_container = QWidget()
        self._branch_buttons_container.setStyleSheet(
            """
QPushButton#branchSelectorButton {
    background-color: #2A2A2A;
    border: 1px solid #30363D;
    border-radius: 8px;
    color: #E6EDF3;
    font-size: 16px;
    font-weight: 500;
    padding: 16px 24px;
    text-align: left;
}

QPushButton#branchSelectorButton:hover {
    background-color: #333333;
    border: 1px solid #555555;
}

QPushButton#branchSelectorButton:checked {
    background-color: rgba(242, 172, 50, 0.1);
    border: 2px solid #F2AC32;
    color: #F2AC32;
    font-weight: bold;
}
"""
        )
        self._branch_buttons_container.setLayout(self._radio_layout)
        self._branch_buttons_container.setMaximumWidth(300)
        right_panel.addWidget(self._branch_buttons_container)
        right_panel.addStretch(1)

        root.addLayout(right_panel, stretch=2)
        self._current_branch: Optional[AnatomyBranch] = None

    def _ensure_view(self) -> Anatomy3DWidget:
        if self.view is None:
            self.view = Anatomy3DWidget()
            self._view_layout.addWidget(self.view)
        return self.view

    def on_activated(self) -> None:
        state = self.controller.get_wizard_state()
        anatomy = state.anatomy
        self.set_valid(False)
        if anatomy is None:
            return

        self._ensure_view().load_anatomy(anatomy)
        self._populate_branches(anatomy)

    def _populate_branches(self, anatomy: AorticArchAnatomy) -> None:
        while self._radio_layout.count():
            item = self._radio_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        branches = self.controller.get_branches(anatomy)
        self._ensure_view().show_branches(branches)
        for branch in branches:
            button = QPushButton(branch.name)
            button.setObjectName("branchSelectorButton")
            button.setCheckable(True)
            button.toggled.connect(
                lambda checked, b=branch: self._on_branch_toggled(b, checked)
            )
            self._radio_layout.addWidget(button)
            self._button_group.addButton(button)

    def _on_branch_toggled(self, branch: AnatomyBranch, checked: bool) -> None:
        if not checked:
            return
        self._current_branch = branch
        self.controller.set_wizard_branch(branch.name)
        self.controller.set_wizard_target_position(None)
        self._ensure_view().highlight_branch(branch)
        self.set_valid(True)


class TargetSelectionPage(WizardPage):
    def __init__(
        self, *, controller: ClinicalUIController, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(controller=controller, parent=parent)

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        self._view_container = QWidget()
        self._view_layout = QVBoxLayout(self._view_container)
        self._view_layout.setContentsMargins(0, 0, 0, 0)
        self.view: Optional[Anatomy3DWidget] = None
        root.addWidget(self._view_container, stretch=3)

        right = QVBoxLayout()
        self._title = QLabel("Step 3: Target Along Branch")
        self._title.setStyleSheet("font-size: 18px; font-weight: 600;")
        right.addWidget(self._title)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider_value = QLabel("0%")
        right.addWidget(self.slider)
        right.addWidget(self.slider_value)

        self.terminal_button = QPushButton("Terminal End")
        self.terminal_button.clicked.connect(self._on_terminal_end)
        right.addWidget(self.terminal_button)
        right.addStretch(1)

        root.addLayout(right, stretch=2)

        self._branch: Optional[AnatomyBranch] = None

    def _ensure_view(self) -> Anatomy3DWidget:
        if self.view is None:
            self.view = Anatomy3DWidget()
            self._view_layout.addWidget(self.view)
        return self.view

    def on_activated(self) -> None:
        state = self.controller.get_wizard_state()
        anatomy = state.anatomy
        if anatomy is None or not state.branch:
            self.set_valid(False)
            return

        self._branch = self.controller.get_branch(anatomy, branch_name=state.branch)
        self._ensure_view().load_anatomy(anatomy)
        self._ensure_view().highlight_branch(self._branch)
        self.set_valid(state.target_position is not None)

    def _on_slider_changed(self, value: int) -> None:
        if self._branch is None:
            return
        ratio = float(value) / 100.0
        self.slider_value.setText(f"{value}%")
        self.controller.set_wizard_target_position(ratio)
        self._ensure_view().mark_target(self._branch, ratio)
        self.set_valid(True)

    def _on_terminal_end(self) -> None:
        if self._branch is None:
            return
        self.slider.setValue(100)
        self.controller.set_wizard_target_position("terminal_end")
        self._ensure_view().mark_target(self._branch, 1.0)
        self.set_valid(True)


class WireAccordionCard(QWidget):
    toggled = pyqtSignal()

    def __init__(
        self,
        *,
        model_name: str,
        subtitle: str,
        variant_count: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("wireAccordionCard")

        self.setStyleSheet(
            """
WireAccordionCard {
    background-color: #2A2A2A;
    border: 1px solid #30363D;
    border-radius: 12px;
    margin-bottom: 12px;
}

WireAccordionCard QLabel {
    background-color: transparent;
    border: none;
}

WireAccordionCard QLabel#wireModelTitle {
    font-size: 18px;
    font-weight: bold;
    color: #E6EDF3;
}

WireAccordionCard QLabel#wireCompany {
    font-size: 13px;
    color: #8B949E;
}

WireAccordionCard QPushButton#expandButton {
    background-color: transparent;
    color: #8B949E;
    font-size: 16px;
    font-weight: bold;
    border: none;
    padding: 8px;
}

WireAccordionCard QPushButton#expandButton:hover {
    color: #E6EDF3;
}

WireAccordionCard QPushButton#candidateToggleButton {
    background-color: #444444;
    border: 1px solid #555555;
    border-radius: 8px;
    color: #E6EDF3;
    font-size: 14px;
    padding: 12px 16px;
    text-align: center;
}

WireAccordionCard QPushButton#candidateToggleButton:hover {
    background-color: #4d4d4d;
    border: 1px solid #7a7a7a;
}

WireAccordionCard QPushButton#candidateToggleButton:checked {
    background-color: #F2AC32;
    border: 2px solid #F2AC32;
    color: #111418;
    font-weight: bold;
}

WireAccordionCard QWidget#candidateBody {
    background-color: #444444;
    border: none;
    border-radius: 8px;
}
"""
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(2)
        self.model_title = QLabel(model_name)
        self.model_title.setObjectName("wireModelTitle")
        self.company_label = QLabel(subtitle)
        self.company_label.setObjectName("wireCompany")
        left.addWidget(self.model_title)
        left.addWidget(self.company_label)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(2)
        self.count_label = QLabel(f"{variant_count} Checkpoints")
        self.count_label.setObjectName("wireCompany")
        self.expand_button = QPushButton("▼")
        self.expand_button.setObjectName("expandButton")
        self.expand_button.clicked.connect(self._toggle_body)
        right.addWidget(self.count_label, alignment=Qt.AlignRight)
        right.addWidget(self.expand_button, alignment=Qt.AlignRight)

        header.addLayout(left, stretch=1)
        header.addLayout(right, stretch=0)
        root.addLayout(header)

        self.body = QWidget()
        self.body.setAttribute(Qt.WA_StyledBackground, True)
        self.body.setObjectName("candidateBody")
        self.body.setVisible(False)
        self.body_layout = QGridLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setHorizontalSpacing(10)
        self.body_layout.setVerticalSpacing(10)
        root.addWidget(self.body)

        self._candidate_buttons: List[QPushButton] = []

    def add_candidate_button(self, button: QPushButton) -> None:
        index = len(self._candidate_buttons)
        columns = 3
        row = index // columns
        col = index % columns
        self.body_layout.addWidget(button, row, col)
        self._candidate_buttons.append(button)

    def candidate_buttons(self) -> Tuple[QPushButton, ...]:
        return tuple(self._candidate_buttons)

    def _toggle_body(self) -> None:
        visible = not self.body.isVisible()
        self.body.setVisible(visible)
        self.expand_button.setText("▲" if visible else "▼")


class WireSelectionPage(WizardPage):
    def __init__(
        self, *, controller: ClinicalUIController, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(controller=controller, parent=parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        top_row = QHBoxLayout()
        self.select_all_button = QPushButton("Select All")
        self.clear_button = QPushButton("Clear")
        self.counter_label = QLabel("Selected: 0")
        top_row.addWidget(self.select_all_button)
        top_row.addWidget(self.clear_button)
        top_row.addStretch(1)
        top_row.addWidget(self.counter_label)
        root.addLayout(top_row)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_content.setAttribute(Qt.WA_StyledBackground, True)
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(16)
        self.scroll.setWidget(self.scroll_content)
        root.addWidget(self.scroll, stretch=1)

        self._candidate_buttons: List[QPushButton] = []
        self._button_to_wire: Dict[QPushButton, WireRef] = {}
        self._accordion_cards: List[WireAccordionCard] = []

        self.select_all_button.clicked.connect(self._select_all)
        self.clear_button.clicked.connect(self._clear_all)

    def on_activated(self) -> None:
        self._populate_tree()

    def _populate_tree(self) -> None:
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._candidate_buttons = []
        self._button_to_wire = {}
        self._accordion_cards = []

        grouped = self._load_grouped_candidates()
        for model in sorted(grouped.keys()):
            entries = grouped[model]
            card = WireAccordionCard(
                model_name=self._pretty_wire_model(model),
                subtitle=f"Model: {model}",
                variant_count=len(entries),
            )

            for wire, candidate in entries:
                label = self._candidate_label(wire=wire, candidate_name=candidate.name)
                button = QPushButton(label)
                button.setCheckable(True)
                button.setObjectName("candidateToggleButton")
                button.toggled.connect(self._sync_selection)
                card.add_candidate_button(button)
                self._candidate_buttons.append(button)
                self._button_to_wire[button] = wire

            self.scroll_layout.addWidget(card)
            self._accordion_cards.append(card)

        self.scroll_layout.addStretch(1)
        self._sync_selection()

    def _select_all(self) -> None:
        for button in self._candidate_buttons:
            button.setChecked(True)
        self._sync_selection()

    def _clear_all(self) -> None:
        for button in self._candidate_buttons:
            button.setChecked(False)
        self._sync_selection()

    def _sync_selection(self, *_args) -> None:
        selected_by_ref: Dict[str, WireRef] = {}
        for button in self._candidate_buttons:
            if not button.isChecked():
                continue
            wire = self._button_to_wire.get(button)
            if wire is None:
                continue
            selected_by_ref[wire.tool_ref] = wire

        selected = tuple(selected_by_ref.values())

        self.controller.set_wizard_selected_wires(selected)
        self.counter_label.setText(f"Selected: {len(selected)}")
        self.set_valid(len(selected) >= 1)

    def _load_grouped_candidates(self) -> Dict[str, List[Tuple[WireRef, object]]]:
        startable = {wire.tool_ref for wire in self.controller.get_startable_wires()}
        grouped: Dict[str, List[Tuple[WireRef, object]]] = {}

        for wire in self.controller.get_execution_wires():
            if wire.tool_ref not in startable:
                continue
            candidates = self.controller.get_candidates(
                execution_wire=wire, include_cross_wire=False
            )
            valid_candidates = [c for c in candidates if self._is_candidate_valid(c)]
            if not valid_candidates:
                continue
            grouped.setdefault(wire.model, []).extend(
                (wire, c) for c in valid_candidates
            )

        return grouped

    @staticmethod
    def _is_candidate_valid(candidate: object) -> bool:
        policy = getattr(candidate, "policy", None)
        if policy is None:
            return False
        checkpoint = getattr(policy, "checkpoint_path", None)
        if checkpoint is None:
            return False
        path = Path(checkpoint)
        return path.exists()

    @staticmethod
    def _candidate_label(*, wire: WireRef, candidate_name: str) -> str:
        return f"{wire.wire} • {candidate_name}"

    @staticmethod
    def _pretty_wire_model(model: str) -> str:
        return str(model).replace("_", " ").strip().title()


class ExecutionConfigPage(WizardPage):
    def __init__(
        self, *, controller: ClinicalUIController, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(controller=controller, parent=parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(0)

        self.setStyleSheet(
            """
QPushButton.FeatureCard,
QPushButton[class="FeatureCard"] {
    background-color: #2A2A2A;
    border: 1px solid #30363D;
    border-radius: 12px;
    color: #E6EDF3;
    padding: 20px;
    font-size: 15px;
    text-align: center;
    min-height: 100px;
}

QPushButton.FeatureCard:hover,
QPushButton[class="FeatureCard"]:hover {
    background-color: #333333;
    border: 1px solid #555555;
}

QPushButton.FeatureCard:checked,
QPushButton[class="FeatureCard"]:checked {
    background-color: rgba(242, 172, 50, 0.08);
    border: 2px solid #F2AC32;
    color: #F2AC32;
    font-weight: bold;
}

QSpinBox {
    background-color: #1C2128;
    border: 1px solid #30363D;
    border-radius: 6px;
    color: #E6EDF3;
    padding: 8px 12px;
    font-size: 16px;
    min-width: 100px;
}

QSpinBox::up-button,
QSpinBox::down-button {
    width: 24px;
    background-color: #2A2A2A;
    border-left: 1px solid #30363D;
}

QSpinBox::up-button:hover,
QSpinBox::down-button:hover {
    background-color: #333333;
}
"""
        )

        root.addStretch(1)

        container = QWidget()
        container.setMaximumWidth(800)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(14)

        behavior_title = QLabel("1. Select Simulation Behavior")
        behavior_title.setStyleSheet(
            "font-size: 18px; font-weight: 700; color: #E6EDF3; margin-bottom: 8px;"
        )
        container_layout.addWidget(behavior_title)

        behavior_row = QHBoxLayout()
        behavior_row.setSpacing(12)
        self.deterministic_card = QPushButton(
            "Deterministic\nStandard baseline evaluation."
        )
        self.stochastic_card = QPushButton(
            "Stochastic (Random)\nMultiple runs to analyze variance."
        )
        self._mark_feature_card(self.deterministic_card)
        self._mark_feature_card(self.stochastic_card)
        self.deterministic_card.setCheckable(True)
        self.stochastic_card.setCheckable(True)
        self.deterministic_card.setChecked(True)

        self.behavior_group = QButtonGroup(self)
        self.behavior_group.setExclusive(True)
        self.behavior_group.addButton(self.deterministic_card)
        self.behavior_group.addButton(self.stochastic_card)

        behavior_row.addWidget(self.deterministic_card)
        behavior_row.addWidget(self.stochastic_card)
        container_layout.addLayout(behavior_row)

        self.runs_row = QWidget()
        runs_layout = QHBoxLayout(self.runs_row)
        runs_layout.setContentsMargins(0, 0, 0, 0)
        runs_layout.setSpacing(10)
        self.runs_label = QLabel("Runs per wire:")
        self.runs_spin = QSpinBox()
        self.runs_spin.setRange(1, 100)
        self.runs_spin.setValue(1)
        runs_layout.addWidget(self.runs_label)
        runs_layout.addWidget(self.runs_spin)
        runs_layout.addStretch(1)
        container_layout.addWidget(self.runs_row)

        self.stochastic_env_mode_row = QWidget()
        stochastic_env_mode_layout = QHBoxLayout(self.stochastic_env_mode_row)
        stochastic_env_mode_layout.setContentsMargins(0, 0, 0, 0)
        stochastic_env_mode_layout.setSpacing(10)
        self.stochastic_env_mode_label = QLabel("Stochastic environment:")
        self.stochastic_env_mode_combo = QComboBox()
        self.stochastic_env_mode_combo.addItem(
            "Random Start, Randomized Policy", "random_start"
        )
        self.stochastic_env_mode_combo.addItem(
            "Fixed Start, Randomized Policy", "fixed_start"
        )
        stochastic_env_mode_layout.addWidget(self.stochastic_env_mode_label)
        stochastic_env_mode_layout.addWidget(self.stochastic_env_mode_combo)
        stochastic_env_mode_layout.addStretch(1)
        container_layout.addWidget(self.stochastic_env_mode_row)

        execution_title = QLabel("2. Execution Mode")
        execution_title.setStyleSheet(
            "font-size: 18px; font-weight: 700; color: #E6EDF3; margin-top: 32px; margin-bottom: 8px;"
        )
        container_layout.addWidget(execution_title)

        execution_row = QHBoxLayout()
        execution_row.setSpacing(12)
        self.headless_card = QPushButton(
            "Headless Mode\nMaximum performance. Background calculation."
        )
        self.live_card = QPushButton(
            "Live Visualization\nWatch the fluoroscopy stream in real-time."
        )
        self._mark_feature_card(self.headless_card)
        self._mark_feature_card(self.live_card)
        self.headless_card.setCheckable(True)
        self.live_card.setCheckable(True)
        self.headless_card.setChecked(True)

        self.execution_group = QButtonGroup(self)
        self.execution_group.setExclusive(True)
        self.execution_group.addButton(self.headless_card)
        self.execution_group.addButton(self.live_card)

        execution_row.addWidget(self.headless_card)
        execution_row.addWidget(self.live_card)
        container_layout.addLayout(execution_row)

        self.visualized_runs_row = QWidget()
        visualized_layout = QHBoxLayout(self.visualized_runs_row)
        visualized_layout.setContentsMargins(0, 0, 0, 0)
        visualized_layout.setSpacing(10)
        self.visualized_runs_label = QLabel("Visualized runs count:")
        self.visualized_runs_spin = QSpinBox()
        self.visualized_runs_spin.setRange(1, 100)
        self.visualized_runs_spin.setValue(1)
        visualized_layout.addWidget(self.visualized_runs_label)
        visualized_layout.addWidget(self.visualized_runs_spin)
        visualized_layout.addStretch(1)
        container_layout.addWidget(self.visualized_runs_row)

        root_row = QHBoxLayout()
        root_row.setContentsMargins(0, 0, 0, 0)
        root_row.addStretch(1)
        root_row.addWidget(container)
        root_row.addStretch(1)
        root.addLayout(root_row)
        root.addStretch(1)

        for widget in (
            self.deterministic_card,
            self.stochastic_card,
            self.headless_card,
            self.live_card,
        ):
            widget.toggled.connect(self._sync_state)
        self.runs_spin.valueChanged.connect(self._sync_state)
        self.stochastic_env_mode_combo.currentIndexChanged.connect(self._sync_state)
        self.visualized_runs_spin.valueChanged.connect(self._sync_state)

        self._sync_state()

    def on_activated(self) -> None:
        self._sync_state()

    def _sync_state(self) -> None:
        is_deterministic = self.deterministic_card.isChecked()
        is_visualized = self.live_card.isChecked()
        self.runs_row.setVisible(True)

        show_stochastic_env_mode = (not is_deterministic) and self.runs_spin.value() > 1
        self.stochastic_env_mode_row.setVisible(show_stochastic_env_mode)

        show_visualized_count = is_visualized and self.runs_spin.value() > 1
        self.visualized_runs_row.setVisible(show_visualized_count)

        trials = self.runs_spin.value()
        visualized_count = (
            1 if not show_visualized_count else self.visualized_runs_spin.value()
        )
        state = self.controller.get_wizard_state()
        total_trials = max(1, int(trials) * max(1, len(state.selected_wires)))
        cpu_budget = max(1, int(os.cpu_count() or 1) - 5)
        worker_count = 1 if is_visualized else max(1, min(total_trials, cpu_budget))

        self.controller.set_wizard_execution_config(
            is_deterministic=is_deterministic,
            trials_per_wire=trials,
            stochastic_environment_mode=str(
                self.stochastic_env_mode_combo.currentData()
            ),
            is_visualized=is_visualized,
            visualized_trials_count=visualized_count,
            worker_count=worker_count,
        )
        self.set_valid(self.controller.can_forward_from_step(4))

    def _mark_feature_card(self, button: QPushButton) -> None:
        button.setProperty("class", "FeatureCard")
        self.style().polish(button)


class PipelineRunningPage(WizardPage):
    run_finished = pyqtSignal(object)

    def __init__(
        self, *, controller: ClinicalUIController, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(controller=controller, parent=parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(0)

        self._visualized_mode = False
        self._running = False
        self._headless_phase = "loading"
        self._last_progress = 0
        self._last_frame_array: Optional[np.ndarray] = None

        self._visualized_container = QWidget()
        visualized_layout = QVBoxLayout(self._visualized_container)
        visualized_layout.setContentsMargins(0, 0, 0, 0)
        visualized_layout.setSpacing(12)

        self.visual_status_label = QLabel("Waiting to start...")
        self.visual_status_label.setAlignment(Qt.AlignCenter)
        self.visual_status_label.setStyleSheet(
            "font-size: 20px; font-weight: 600; color: #E6EDF3;"
        )

        video_row = QHBoxLayout()
        video_row.setContentsMargins(0, 0, 0, 0)
        video_row.addStretch(1)

        self.video_label = QLabel("Live fluoroscopy stream")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 800)
        self.video_label.setStyleSheet(
            "background: #101215; color: #E6EDF3; border: 2px solid #30363D; border-radius: 10px;"
        )
        video_row.addWidget(self.video_label)
        video_row.addStretch(1)

        self.visual_progress_bar = QProgressBar()
        self.visual_progress_bar.setRange(0, 100)
        self.visual_progress_bar.setValue(0)
        self.visual_progress_bar.setFixedHeight(24)
        self.visual_progress_bar.setProperty("progressVariant", "footer")

        visualized_layout.addWidget(self.visual_status_label)
        visualized_layout.addLayout(video_row, stretch=1)
        visualized_layout.addWidget(self.visual_progress_bar)

        self._headless_container = QWidget()
        headless_outer = QVBoxLayout(self._headless_container)
        headless_outer.setContentsMargins(0, 0, 0, 0)
        headless_outer.setSpacing(0)
        headless_outer.addStretch(1)

        centered_row = QHBoxLayout()
        centered_row.setContentsMargins(0, 0, 0, 0)
        centered_row.addStretch(1)

        dashboard = QWidget()
        dashboard.setMaximumWidth(900)
        dashboard.setMinimumWidth(800)
        dashboard_layout = QVBoxLayout(dashboard)
        dashboard_layout.setContentsMargins(20, 20, 20, 20)
        dashboard_layout.setSpacing(14)

        self.headless_progress_bar = QProgressBar()
        self.headless_progress_bar.setRange(0, 100)
        self.headless_progress_bar.setValue(0)
        self.headless_progress_bar.setFixedHeight(30)
        self.headless_progress_bar.setProperty("progressVariant", "hero")

        self.headless_status_label = QLabel("Simulating...")
        self.headless_status_label.setWordWrap(True)
        self.headless_status_label.setAlignment(Qt.AlignCenter)
        self.headless_status_label.setSizePolicy(
            QSizePolicy.MinimumExpanding,
            QSizePolicy.Minimum,
        )
        self.headless_status_label.setMinimumWidth(700)
        self.headless_status_label.setStyleSheet(
            "font-size: 24px; font-weight: 700; color: #F2AC32;"
        )

        self.headless_step_loading_label = QLabel()
        self.headless_step_physics_label = QLabel()
        self.headless_step_trials_label = QLabel()
        for step_label in (
            self.headless_step_loading_label,
            self.headless_step_physics_label,
            self.headless_step_trials_label,
        ):
            step_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            step_label.setStyleSheet("font-size: 14px; color: #8B949E;")

        dashboard_layout.addWidget(self.headless_progress_bar)
        dashboard_layout.addWidget(self.headless_status_label)
        dashboard_layout.addWidget(self.headless_step_loading_label)
        dashboard_layout.addWidget(self.headless_step_physics_label)
        dashboard_layout.addWidget(self.headless_step_trials_label)

        centered_row.addWidget(dashboard)
        centered_row.addStretch(1)

        headless_outer.addLayout(centered_row)
        headless_outer.addStretch(1)

        root.addWidget(self._visualized_container, stretch=1)
        root.addWidget(self._headless_container, stretch=1)
        self._set_mode_layout(is_visualized=False)
        self._refresh_headless_steps()

        self.controller.progress_updated.connect(self._on_progress)
        self.controller.status_updated.connect(self._on_status)
        self.controller.frame_ready.connect(self._on_frame_ready)
        self.controller.error_occurred.connect(self._on_error)
        self.controller.finished.connect(self._on_finished)

    def start_run(self, job: EvaluationJob) -> None:
        if self._running:
            return

        self._visualized_mode = bool(job.execution.visualization.enabled)
        self._set_mode_layout(is_visualized=self._visualized_mode)
        self._last_progress = 0
        self._set_progress(0)
        self._last_frame_array = None
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("Waiting for native frame stream...")
        self._headless_phase = "loading"
        self._refresh_headless_steps()
        self.visual_status_label.setText("Launching evaluation pipeline...")
        self.headless_status_label.setText("Preparing Simulation")
        self._running = True

        try:
            self.controller.start_evaluation(job)
        except Exception as exc:
            self._running = False
            self._on_status(f"Error: {exc}")
            return

    def _set_mode_layout(self, *, is_visualized: bool) -> None:
        self._visualized_container.setVisible(is_visualized)
        self._headless_container.setVisible(not is_visualized)

    def _set_progress(self, percent: int) -> None:
        bounded = max(0, min(100, int(percent)))
        self.visual_progress_bar.setValue(bounded)
        self.headless_progress_bar.setValue(bounded)
        self._last_progress = bounded

    def _on_progress(self, percent: int) -> None:
        if not self._running:
            return
        self._set_progress(percent)
        if self._headless_phase in ("running", "completed", "failed"):
            self._refresh_headless_steps()

    @staticmethod
    def _extract_wire_name(status_text: str) -> Optional[str]:
        prefix = "Running Simulation:"
        if not status_text.startswith(prefix):
            return None
        payload = status_text[len(prefix) :].strip()
        if not payload:
            return None
        if " - Trial " in payload:
            return payload.split(" - Trial ", 1)[0].strip() or None
        return payload

    def _on_status(self, status_text: str) -> None:
        if (
            not self._running
            and status_text != "Pipeline complete"
            and not status_text.startswith("Error:")
        ):
            return

        self.visual_status_label.setText(status_text)

        if status_text.startswith("Loading Anatomy"):
            self._headless_phase = "loading"
            self.headless_status_label.setText("Loading Anatomy")
        elif status_text.startswith("Initializing Physics"):
            self._headless_phase = "physics"
            self.headless_status_label.setText("Initializing Physics")
        elif status_text.startswith("Running Simulation:"):
            self._headless_phase = "running"
            wire_name = self._extract_wire_name(status_text) or "wire"
            self.headless_status_label.setText(f"Simulating {wire_name}")
        elif status_text == "Pipeline complete":
            self._headless_phase = "completed"
            self.headless_status_label.setText("Simulation Complete.")
        elif status_text.startswith("Error:"):
            self._headless_phase = "failed"
            self.headless_status_label.setText(status_text)

        self._refresh_headless_steps()

    def _on_frame_ready(self, frame: np.ndarray) -> None:
        if not self._running or not self._visualized_mode:
            return
        self.update_frame(np.asarray(frame))

    def update_frame(self, frame_array: np.ndarray) -> None:
        if (
            frame_array is None
            or not isinstance(frame_array, np.ndarray)
            or frame_array.size == 0
        ):
            return

        if frame_array.ndim == 2:
            frame_array = np.repeat(frame_array[:, :, np.newaxis], 3, axis=2)
        elif frame_array.ndim == 3 and frame_array.shape[2] == 1:
            frame_array = np.repeat(frame_array, 3, axis=2)

        if frame_array.ndim != 3:
            return

        frame_array = np.require(frame_array, np.uint8, "C")

        height, width, channels = frame_array.shape
        if channels == 3:
            qformat = QImage.Format_RGB888
        elif channels == 4:
            qformat = QImage.Format_RGBA8888
        else:
            return

        bytes_per_line = channels * width
        # Create QImage with copied data to avoid memory ownership issues
        # The QImage constructor needs a proper copy to ensure the data is not garbage collected
        frame_copy = np.ascontiguousarray(frame_array)
        q_image = QImage(
            frame_copy.tobytes(),
            width,
            height,
            bytes_per_line,
            qformat,
        )

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setText("")
        self._last_frame_array = frame_copy

    def _on_error(self, error: Exception) -> None:
        if not self._running:
            return
        self._running = False
        self._on_status(f"Error: {error}")

    def _on_finished(self, report: EvaluationReport) -> None:
        if not self._running:
            return
        self._running = False
        self._set_progress(100)
        self._on_status("Pipeline complete")
        self.run_finished.emit(report)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._visualized_mode and self._last_frame_array is not None:
            self.update_frame(self._last_frame_array)

    def _refresh_headless_steps(self) -> None:
        if self._headless_phase == "loading":
            self.headless_step_loading_label.setText("⏳ Loading Anatomy...")
            self.headless_step_physics_label.setText("• Initializing Physics...")
            self.headless_step_trials_label.setText("• Running Trials (0%)...")
            return
        if self._headless_phase == "physics":
            self.headless_step_loading_label.setText("✓ Loading Anatomy...")
            self.headless_step_physics_label.setText("⏳ Initializing Physics...")
            self.headless_step_trials_label.setText("• Running Trials (0%)...")
            return
        if self._headless_phase == "running":
            self.headless_step_loading_label.setText("✓ Loading Anatomy...")
            self.headless_step_physics_label.setText("✓ Initializing Physics...")
            self.headless_step_trials_label.setText(
                f"⏳ Running Trials ({max(0, min(100, self._last_progress))}%)..."
            )
            return
        if self._headless_phase == "completed":
            self.headless_step_loading_label.setText("✓ Loading Anatomy...")
            self.headless_step_physics_label.setText("✓ Initializing Physics...")
            self.headless_step_trials_label.setText("✓ Running Trials (100%)...")
            return
        self.headless_step_loading_label.setText("✓ Loading Anatomy...")
        self.headless_step_physics_label.setText("✓ Initializing Physics...")
        self.headless_step_trials_label.setText(
            f"⚠ Running Trials halted ({max(0, min(100, self._last_progress))}%)"
        )


class WireAttemptRowWidget(QWidget):
    remove_requested = pyqtSignal(object)

    def __init__(
        self,
        *,
        attempt_index: int,
        wire_options: Sequence[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setObjectName("AttemptRowWidget")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.attempt_label = QLabel()
        self.attempt_label.setProperty("textRole", "small")
        self.attempt_label.setFixedWidth(70)

        self.wire_combo = QComboBox()
        self.wire_combo.setProperty("class", "FeedbackCombo")
        self.wire_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.outcome_combo = QComboBox()
        self.outcome_combo.setProperty("class", "FeedbackCombo")
        self.outcome_combo.addItems(("Success", "Failure", "Complication"))
        self.outcome_combo.setFixedWidth(140)

        self.delete_button = QPushButton("X")
        self.delete_button.setProperty("class", "DeleteRowBtn")
        self.delete_button.setFixedWidth(24)
        self.delete_button.clicked.connect(
            lambda _checked=False: self.remove_requested.emit(self)
        )

        layout.addWidget(self.attempt_label, stretch=0)
        layout.addWidget(self.wire_combo, stretch=1)
        layout.addWidget(self.outcome_combo, stretch=0)
        layout.addWidget(self.delete_button, stretch=0)

        self.set_attempt_index(attempt_index)
        self.set_wire_options(wire_options)

    def set_attempt_index(self, attempt_index: int) -> None:
        self.attempt_label.setText(f"Attempt {max(1, int(attempt_index))}:")

    def set_wire_options(self, wire_options: Sequence[str]) -> None:
        current_wire = self.current_wire()
        self.wire_combo.blockSignals(True)
        self.wire_combo.clear()
        self.wire_combo.addItems([str(wire_name) for wire_name in wire_options])
        if current_wire:
            current_index = self.wire_combo.findText(current_wire)
            self.wire_combo.setCurrentIndex(current_index if current_index >= 0 else -1)
        else:
            self.wire_combo.setCurrentIndex(-1)
        self.wire_combo.blockSignals(False)

    def current_wire(self) -> str:
        return self.wire_combo.currentText().strip()

    def current_outcome(self) -> str:
        return self.outcome_combo.currentText().strip()


class ResultsPage(WizardPage):
    def __init__(
        self,
        *,
        controller: ClinicalUIController,
        parent: Optional[QWidget] = None,
        is_from_archive: bool = False,
    ) -> None:
        super().__init__(controller=controller, parent=parent)

        self._is_from_archive = bool(is_from_archive)
        self._report: Optional[EvaluationReport] = None
        self._report_id: Optional[str] = None
        self._leaderboard_rows: list[dict[str, float | str]] = []
        self._schema_defaults = self.controller.get_results_control_schema()
        self._control_sliders: Dict[str, QSlider] = {}
        self._control_value_labels: Dict[str, QLabel] = {}
        self._control_meta: Dict[str, Dict[str, Any]] = {}
        self._is_resetting_weights = False
        self._feedback_wire_options: list[str] = []
        self._feedback_attempt_rows: list[WireAttemptRowWidget] = []
        self._current_wire_trials: list[TrialResult] = []

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        left_panel = QWidget()
        left_panel.setObjectName("LeftWeightPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)
        left_panel.setMaximumWidth(420)
        left_panel.setMinimumWidth(320)

        weights_title = QLabel("Evaluation Weights")
        weights_title.setProperty("textRole", "h2")
        left_layout.addWidget(weights_title)

        self._weights_scroll = QScrollArea()
        self._weights_scroll.setWidgetResizable(True)
        self._weights_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._weights_scroll.setObjectName("LeftWeightsScroll")
        self._weights_content = QWidget()
        self._weights_content.setObjectName("LeftWeightsContent")
        self._weights_layout = QVBoxLayout(self._weights_content)
        self._weights_layout.setContentsMargins(0, 0, 0, 0)
        self._weights_layout.setSpacing(10)
        self._weights_scroll.setWidget(self._weights_content)
        left_layout.addWidget(self._weights_scroll, stretch=1)

        self.reset_weights_button = QPushButton("Reset to Defaults")
        self.reset_weights_button.clicked.connect(self._reset_to_default_weights)
        left_layout.addWidget(self.reset_weights_button)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        kpi_row_widget = QWidget()
        kpi_row_layout = QHBoxLayout(kpi_row_widget)
        kpi_row_layout.setContentsMargins(0, 0, 0, 0)
        kpi_row_layout.setSpacing(12)

        top_card, self.top_recommendation_value = self._create_kpi_card(
            "Top Recommendation"
        )
        safety_card, self.highest_safety_value = self._create_kpi_card(
            "Highest Safety Score"
        )
        speed_card, self.fastest_insertion_value = self._create_kpi_card(
            "Highest Efficiency"
        )
        kpi_row_layout.addWidget(top_card)
        kpi_row_layout.addWidget(safety_card)
        kpi_row_layout.addWidget(speed_card)

        chart_card = QWidget()
        chart_card.setProperty("class", "BentoCard")
        chart_card.setMaximumHeight(180)
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(12, 12, 12, 12)
        chart_layout.setSpacing(8)
        chart_title = QLabel("Top Wires: Final Score")
        chart_title.setProperty("textRole", "h2")
        chart_layout.addWidget(chart_title)
        self._chart_rows_container = QWidget()
        self._chart_rows_container.setObjectName("ChartContainerWidget")
        self._chart_rows_layout = QVBoxLayout(self._chart_rows_container)
        self._chart_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._chart_rows_layout.setSpacing(8)
        chart_layout.addWidget(self._chart_rows_container, stretch=1)

        self.leaderboard_table = QTableWidget(0, 6)
        self.leaderboard_table.setHorizontalHeaderLabels(
            (
                "Rank",
                "Wire Name",
                "Final Score",
                "Valid Rate",
                "Mean Soft Score",
                "Soft Score Std",
            )
        )
        self.leaderboard_table.horizontalHeader().setStretchLastSection(True)
        self.leaderboard_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.leaderboard_table.setSelectionMode(QTableWidget.SingleSelection)

        self.deep_dive_table = QTableWidget(0, 8)
        self.deep_dive_table.setHorizontalHeaderLabels(
            (
                "Trial",
                "Success",
                "Valid",
                "Time (steps)",
                "End Reason",
                "Max Force (N)",
                "Trial Score",
                "Replay",
            )
        )
        self.deep_dive_table.horizontalHeader().setStretchLastSection(True)

        tables_splitter = QSplitter(Qt.Horizontal)
        leaderboard_container = QWidget()
        leaderboard_layout = QVBoxLayout(leaderboard_container)
        leaderboard_layout.setContentsMargins(0, 0, 0, 0)
        leaderboard_layout.setSpacing(8)
        leaderboard_title = QLabel("Leaderboard")
        leaderboard_title.setProperty("textRole", "h2")
        leaderboard_layout.addWidget(leaderboard_title)
        leaderboard_layout.addWidget(self.leaderboard_table, stretch=1)

        details_container = QWidget()
        details_layout = QVBoxLayout(details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(8)
        details_title = QLabel("Individual Trials")
        details_title.setProperty("textRole", "h2")
        details_layout.addWidget(details_title)
        self._details_stack = QStackedWidget()
        details_list_page = QWidget()
        details_list_layout = QVBoxLayout(details_list_page)
        details_list_layout.setContentsMargins(0, 0, 0, 0)
        details_list_layout.setSpacing(8)
        details_list_layout.addWidget(self.deep_dive_table, stretch=1)

        self.feedback_group = QGroupBox("Clinical Feedback (Ground Truth)")
        self.feedback_group.setObjectName("ClinicalFeedbackGroup")
        feedback_layout = QVBoxLayout(self.feedback_group)
        feedback_layout.setContentsMargins(12, 12, 12, 12)
        feedback_layout.setSpacing(8)

        feedback_subtitle = QLabel("Log Clinical Attempts")
        feedback_subtitle.setProperty("textRole", "small")
        feedback_layout.addWidget(feedback_subtitle)

        self.feedback_attempts_scroll = QScrollArea()
        self.feedback_attempts_scroll.setObjectName("FeedbackScrollArea")
        self.feedback_attempts_scroll.setWidgetResizable(True)
        self.feedback_attempts_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self.feedback_attempts_container = QWidget()
        self.feedback_attempts_layout = QVBoxLayout(self.feedback_attempts_container)
        self.feedback_attempts_layout.setContentsMargins(0, 0, 0, 0)
        self.feedback_attempts_layout.setSpacing(6)
        self.feedback_attempts_layout.addStretch(1)
        self.feedback_attempts_scroll.setWidget(self.feedback_attempts_container)
        feedback_layout.addWidget(self.feedback_attempts_scroll, stretch=1)

        self.feedback_add_attempt_button = QPushButton("+ Add Wire Attempt")
        self.feedback_add_attempt_button.setObjectName("AddAttemptBtn")
        self.feedback_add_attempt_button.setProperty("buttonRole", "ghost")
        feedback_layout.addWidget(self.feedback_add_attempt_button, stretch=0)

        feedback_notes_label = QLabel("Clinician Notes (Optional)")
        feedback_notes_label.setProperty("textRole", "small")
        feedback_layout.addWidget(feedback_notes_label)

        self.feedback_notes_edit = QTextEdit()
        self.feedback_notes_edit.setPlaceholderText("Optional clinical notes")
        self.feedback_notes_edit.setFixedHeight(90)
        feedback_layout.addWidget(self.feedback_notes_edit)

        self.feedback_save_button = QPushButton("Save Feedback")
        self.feedback_save_button.setProperty("buttonRole", "primary")
        self.feedback_status_label = QLabel("")
        self.feedback_status_label.setProperty("textRole", "small")
        feedback_layout.addWidget(self.feedback_save_button)
        feedback_layout.addWidget(self.feedback_status_label)
        self.feedback_group.setVisible(self._is_from_archive)
        details_list_layout.addWidget(self.feedback_group, stretch=0)

        self.trace_viewer_panel = TraceViewerPanel()
        self.trace_viewer_panel.back_requested.connect(self._show_trial_list_panel)

        self._details_stack.addWidget(details_list_page)
        self._details_stack.addWidget(self.trace_viewer_panel)
        details_layout.addWidget(self._details_stack, stretch=1)

        tables_splitter.addWidget(leaderboard_container)
        tables_splitter.addWidget(details_container)
        tables_splitter.setStretchFactor(0, 3)
        tables_splitter.setStretchFactor(1, 3)

        right_layout.addWidget(kpi_row_widget, stretch=0)
        right_layout.addWidget(chart_card, stretch=1)
        right_layout.addWidget(tables_splitter, stretch=5)

        root.addWidget(left_panel, stretch=1)
        root.addWidget(right_panel, stretch=4)

        self.feedback_add_attempt_button.clicked.connect(self._add_feedback_attempt_row)
        self.feedback_save_button.clicked.connect(self._save_feedback)
        self.leaderboard_table.itemSelectionChanged.connect(self._show_wire_deep_dive)

        self._build_dynamic_weight_controls(self._schema_defaults)
        self._populate_feedback_wire_options()
        self._reset_feedback_attempt_rows()
        self._reset_to_default_weights()
        self._clear_kpis()
        self._update_score_chart([])

    def set_report(
        self, report: EvaluationReport, *, is_from_archive: Optional[bool] = None
    ) -> None:
        if is_from_archive is not None:
            self._is_from_archive = bool(is_from_archive)
        self.feedback_group.setVisible(self._is_from_archive)
        self._show_trial_list_panel()

        self._report = report
        self._report_id = None
        if report.artifacts is not None:
            self._report_id = str(report.artifacts.output_dir)
        self.controller.configure_results_scoring(report.scoring_spec or ScoringSpec())
        self._schema_defaults = self.controller.get_results_control_schema()
        self._build_dynamic_weight_controls(self._schema_defaults)
        self._reset_to_default_weights()
        self._populate_feedback_wire_options()
        self._reset_feedback_attempt_rows()
        self.feedback_status_label.setText("")
        self._recalculate_leaderboard()

    def _build_dynamic_weight_controls(
        self, schema: Dict[str, Dict[str, object]]
    ) -> None:
        self._control_sliders.clear()
        self._control_value_labels.clear()
        self._control_meta.clear()

        while self._weights_layout.count():
            item = self._weights_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for category_name, category_config in schema.items():
            category_card = QGroupBox(str(category_name))
            category_card.setProperty("class", "CategoryBox")
            category_layout = QVBoxLayout(category_card)
            category_layout.setContentsMargins(10, 10, 10, 10)
            category_layout.setSpacing(8)

            controls = category_config.get("controls", {})
            if isinstance(controls, dict):
                for control_label, control_meta in controls.items():
                    if not isinstance(control_meta, dict):
                        continue
                    key = str(control_meta.get("key", control_label))
                    minimum = float(control_meta.get("min", 0.0))
                    maximum = float(control_meta.get("max", 1.0))
                    resolution = max(int(control_meta.get("resolution", 100)), 1)
                    default_value = float(control_meta.get("default", minimum))

                    row = QWidget()
                    row_layout = QHBoxLayout(row)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.setSpacing(8)
                    row_label = QLabel(str(control_label))
                    slider = QSlider(Qt.Horizontal)
                    slider.setRange(0, resolution)
                    slider_value = max(
                        0,
                        min(
                            resolution,
                            int(
                                round(
                                    ((default_value - minimum) / max(maximum - minimum, 1e-9))
                                    * resolution
                                )
                            ),
                        ),
                    )
                    slider.setValue(slider_value)
                    value_label = QLabel()
                    value_label.setMinimumWidth(52)
                    value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    row_layout.addWidget(row_label)
                    row_layout.addWidget(slider, stretch=1)
                    row_layout.addWidget(value_label, stretch=0)

                    self._control_sliders[key] = slider
                    self._control_value_labels[key] = value_label
                    self._control_meta[key] = {
                        "min": minimum,
                        "max": maximum,
                        "resolution": resolution,
                        "format": str(control_meta.get("format", "{value:.2f}")),
                    }
                    slider.valueChanged.connect(self._on_weight_changed)
                    category_layout.addWidget(row)

            self._weights_layout.addWidget(category_card)

        self._weights_layout.addStretch(1)
        self._refresh_weight_labels()

    def _refresh_weight_labels(self) -> None:
        for key, slider in self._control_sliders.items():
            label = self._control_value_labels.get(key)
            meta = self._control_meta.get(key, {})
            if label is not None:
                value = self._slider_value_to_float(key, slider.value())
                label.setText(str(meta.get("format", "{value:.2f}")).format(value=value))

    def _slider_value_to_float(self, key: str, slider_value: int) -> float:
        meta = self._control_meta.get(key, {})
        minimum = float(meta.get("min", 0.0))
        maximum = float(meta.get("max", 1.0))
        resolution = max(int(meta.get("resolution", 100)), 1)
        ratio = float(slider_value) / float(resolution)
        return minimum + ratio * (maximum - minimum)

    def _current_control_values(self) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for key, slider in self._control_sliders.items():
            values[key] = self._slider_value_to_float(key, slider.value())
        return values

    def _on_weight_changed(self) -> None:
        self._refresh_weight_labels()
        if self._is_resetting_weights:
            return
        for key, value in self._current_control_values().items():
            self.controller.set_results_control_value(key, value)
        self._recalculate_leaderboard()

    def _reset_to_default_weights(self) -> None:
        self._is_resetting_weights = True
        defaults = self.controller.get_default_results_controls()

        for key, slider in self._control_sliders.items():
            meta = self._control_meta.get(key, {})
            minimum = float(meta.get("min", 0.0))
            maximum = float(meta.get("max", 1.0))
            resolution = max(int(meta.get("resolution", 100)), 1)
            default_value = float(defaults.get(key, minimum))
            slider_value = max(
                0,
                min(
                    resolution,
                    int(
                        round(
                            ((default_value - minimum) / max(maximum - minimum, 1e-9))
                            * resolution
                        )
                    ),
                ),
            )
            slider.setValue(slider_value)

        self._is_resetting_weights = False
        self._on_weight_changed()

    def _create_kpi_card(self, title: str) -> Tuple[QWidget, QLabel]:
        card = QWidget()
        card.setProperty("class", "BentoCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setProperty("class", "KPITitle")
        value_label = QLabel("n/a")
        value_label.setProperty("class", "KPIValue")
        value_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(value_label, stretch=1)
        return card, value_label

    def _clear_kpis(self) -> None:
        self.top_recommendation_value.setText("n/a")
        self.highest_safety_value.setText("n/a")
        self.fastest_insertion_value.setText("n/a")

    def _populate_feedback_wire_options(self) -> None:
        self._feedback_wire_options = sorted(
            {wire.tool_ref for wire in self.controller.get_execution_wires()}
        )
        for row in self._feedback_attempt_rows:
            row.set_wire_options(self._feedback_wire_options)

    def _clear_feedback_attempt_rows(self) -> None:
        while self._feedback_attempt_rows:
            row = self._feedback_attempt_rows.pop()
            self.feedback_attempts_layout.removeWidget(row)
            row.deleteLater()

    def _reset_feedback_attempt_rows(self) -> None:
        self._clear_feedback_attempt_rows()
        self._add_feedback_attempt_row()

    def _add_feedback_attempt_row(self, _checked: bool = False) -> None:
        row = WireAttemptRowWidget(
            attempt_index=len(self._feedback_attempt_rows) + 1,
            wire_options=self._feedback_wire_options,
        )
        row.remove_requested.connect(self._remove_feedback_attempt_row)
        insert_index = max(0, self.feedback_attempts_layout.count() - 1)
        self.feedback_attempts_layout.insertWidget(insert_index, row)
        self._feedback_attempt_rows.append(row)
        self._refresh_feedback_attempt_labels()

    def _remove_feedback_attempt_row(self, row: WireAttemptRowWidget) -> None:
        if row not in self._feedback_attempt_rows:
            return
        self._feedback_attempt_rows.remove(row)
        self.feedback_attempts_layout.removeWidget(row)
        row.deleteLater()
        if not self._feedback_attempt_rows:
            self._add_feedback_attempt_row()
            return
        self._refresh_feedback_attempt_labels()

    def _refresh_feedback_attempt_labels(self) -> None:
        for index, row in enumerate(self._feedback_attempt_rows):
            row.set_attempt_index(index + 1)

    def _active_scoring_spec(self) -> ScoringSpec:
        base_scoring = self._report.scoring_spec if self._report is not None and self._report.scoring_spec is not None else ScoringSpec()
        values = self.controller.get_results_control_values()
        default_weights = {
            "score_safety": float(values.get("score_weight_safety", 0.5)),
            "score_efficiency": float(values.get("score_weight_efficiency", 0.5)),
        }
        return replace(
            base_scoring,
            force=replace(
                base_scoring.force,
                force_max_N=float(values.get("force_max_N", base_scoring.force.force_max_N)),
            ),
            safety_score=replace(
                base_scoring.safety_score,
                c=float(values.get("force_score_c", base_scoring.safety_score.c)),
                p=float(values.get("force_score_p", base_scoring.safety_score.p)),
                k=float(values.get("force_score_k", base_scoring.safety_score.k)),
                F50_N=float(
                    values.get("force_score_F50_N", base_scoring.safety_score.F50_N)
                ),
                F_max_N=float(values.get("force_max_N", base_scoring.force.force_max_N)),
            ),
            candidate_score=replace(
                base_scoring.candidate_score,
                lambda_=float(values.get("score_lambda", base_scoring.candidate_score.lambda_)),
                beta=float(values.get("score_beta", base_scoring.candidate_score.beta)),
                default_weights=default_weights,
                active_components=tuple(
                    name
                    for name, weight in default_weights.items()
                    if float(weight) > 0.0
                )
                or ("score_safety", "score_efficiency"),
            ),
        )

    def _wire_rows_from_trials(self) -> list[dict[str, float | str]]:
        if self._report is None:
            return []
        scoring = self._active_scoring_spec()
        max_episode_steps = (
            self._report.execution_plan.max_episode_steps
            if self._report.execution_plan is not None
            else 1000
        )
        grouped: dict[str, list[dict[str, float | str | TrialResult | bool | None]]] = {}
        for trial in self._report.trials:
            breakdown = score_trial(
                telemetry=trial.telemetry,
                max_episode_steps=max_episode_steps,
                scoring=scoring,
            )
            is_valid = valid_for_ranking(
                telemetry=trial.telemetry,
                max_episode_steps=max_episode_steps,
                scoring=scoring,
            )
            within_threshold = force_within_safety_threshold(
                telemetry=trial.telemetry,
                scoring=scoring,
            )
            soft_total = soft_score_total(breakdown=breakdown, scoring=scoring)
            wire_name = trial.execution_wire.tool_ref
            grouped.setdefault(wire_name, []).append(
                {
                    "trial": trial,
                    "valid": is_valid,
                    "within_threshold": within_threshold,
                    "score_total": soft_total,
                    "score_success": float(breakdown.success),
                    "score_efficiency": float(breakdown.efficiency),
                    "score_safety": float(breakdown.safety),
                    "score_smoothness": (
                        None
                        if breakdown.smoothness is None
                        else float(breakdown.smoothness)
                    ),
                }
            )

        rows: list[dict[str, float | str]] = []
        for wire_name, entries in grouped.items():
            n_trials = len(entries)
            valid_scores = [
                float(entry["score_total"]) for entry in entries if bool(entry["valid"])
            ]
            valid_rate = (
                float(sum(1 for entry in entries if bool(entry["valid"]))) / float(n_trials)
                if n_trials > 0
                else 0.0
            )
            soft_mean = float(sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0
            if len(valid_scores) > 1:
                variance = sum((score - soft_mean) ** 2 for score in valid_scores) / float(
                    len(valid_scores) - 1
                )
                soft_std = math.sqrt(max(variance, 0.0))
            else:
                soft_std = 0.0
            final_score = (
                (valid_rate ** float(scoring.candidate_score.lambda_))
                * max(0.0, soft_mean - float(scoring.candidate_score.beta) * soft_std)
            ) if valid_scores else 0.0

            safety_mean = float(
                sum(float(entry["score_safety"]) for entry in entries) / float(n_trials)
            ) if n_trials > 0 else 0.0
            efficiency_mean = float(
                sum(float(entry["score_efficiency"]) for entry in entries) / float(n_trials)
            ) if n_trials > 0 else 0.0
            times = [
                float(
                    trial.telemetry.steps_to_success
                    if trial.telemetry.steps_to_success is not None
                    else trial.telemetry.steps_total
                )
                for trial in (entry["trial"] for entry in entries)
            ]
            rows.append(
                {
                    "wire": wire_name,
                    "overall": final_score,
                    "valid_rate": valid_rate,
                    "soft_mean": soft_mean,
                    "soft_std": soft_std,
                    "safety": safety_mean,
                    "efficiency": efficiency_mean,
                    "time_steps": (sum(times) / float(len(times))) if times else float("inf"),
                }
            )
        rows.sort(key=lambda row: (float(row["overall"]), float(row["valid_rate"])), reverse=True)
        return rows

    def _recalculate_leaderboard(self) -> None:
        if self._report is None or not self._report.summaries:
            self._leaderboard_rows = []
            self.leaderboard_table.setRowCount(0)
            self.deep_dive_table.setRowCount(0)
            self._clear_kpis()
            self._update_score_chart([])
            return

        rows = self._wire_rows_from_trials()
        self._leaderboard_rows = rows
        self._update_kpi_cards(rows)
        self._update_score_chart(rows)

        previously_selected_wire = None
        current_row = self.leaderboard_table.currentRow()
        if 0 <= current_row < len(rows):
            previously_selected_wire = str(rows[current_row]["wire"])

        self.leaderboard_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            self.leaderboard_table.setItem(
                row_index, 0, QTableWidgetItem(str(row_index + 1))
            )
            self.leaderboard_table.setItem(
                row_index, 1, QTableWidgetItem(str(row["wire"]))
            )
            self.leaderboard_table.setItem(
                row_index, 2, QTableWidgetItem(f"{float(row['overall']):.4f}")
            )
            self.leaderboard_table.setItem(
                row_index, 3, QTableWidgetItem(f"{float(row['valid_rate']):.4f}")
            )
            self.leaderboard_table.setItem(
                row_index, 4, QTableWidgetItem(f"{float(row['soft_mean']):.4f}")
            )
            self.leaderboard_table.setItem(
                row_index, 5, QTableWidgetItem(f"{float(row['soft_std']):.4f}")
            )

        target_row = 0
        if previously_selected_wire is not None:
            for index, row in enumerate(rows):
                if str(row["wire"]) == previously_selected_wire:
                    target_row = index
                    break

        if rows:
            self.leaderboard_table.selectRow(target_row)
            self._show_wire_deep_dive()
        else:
            self.deep_dive_table.setRowCount(0)

    def _update_kpi_cards(self, rows: Sequence[dict[str, float | str]]) -> None:
        if not rows:
            self._clear_kpis()
            return

        top = rows[0]
        safest = max(rows, key=lambda row: float(row["safety"]))
        fastest = max(rows, key=lambda row: float(row["efficiency"]))

        self.top_recommendation_value.setText(
            f"{top['wire']}\n{float(top['overall']):.3f}"
        )
        self.highest_safety_value.setText(
            f"{safest['wire']}\n{float(safest['safety']):.3f}"
        )
        self.fastest_insertion_value.setText(
            f"{fastest['wire']}\n{float(fastest['efficiency']):.3f}"
        )

    def _update_score_chart(self, rows: Sequence[dict[str, float | str]]) -> None:
        while self._chart_rows_layout.count():
            item = self._chart_rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        top_rows = list(rows[:3])
        if not top_rows:
            empty_label = QLabel("No score data available.")
            empty_label.setProperty("textRole", "small")
            self._chart_rows_layout.addWidget(empty_label)
            self._chart_rows_layout.addStretch(1)
            return

        for index, row in enumerate(top_rows):
            wire_name = str(row["wire"])
            score = max(0.0, min(1.0, float(row["overall"])))

            row_widget = QWidget()
            row_widget.setObjectName("ChartRowWidget")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(10)

            wire_label = QLabel()
            wire_label.setProperty("class", "ChartWireLabel")
            wire_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            wire_label.setFixedWidth(250)
            wire_label.setWordWrap(False)
            wire_label.setText(
                wire_label.fontMetrics().elidedText(wire_name, Qt.ElideMiddle, 250)
            )
            wire_label.setToolTip(wire_name)

            bar = QProgressBar()
            bar.setProperty("class", "ChartBar")
            bar.setProperty("chartRank", "winner" if index == 0 else "runner")
            bar.setTextVisible(False)
            bar.setRange(0, 100)
            bar.setValue(int(round(score * 100.0)))
            bar.setMinimumWidth(220)
            bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            score_label = QLabel(f"{score:.2f}")
            score_label.setProperty("class", "ChartScoreLabel")
            score_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            score_label.setMinimumWidth(52)

            row_layout.addWidget(wire_label, stretch=0)
            row_layout.addWidget(bar, stretch=1)
            row_layout.addWidget(score_label, stretch=0)

            self._chart_rows_layout.addWidget(row_widget)

        self._chart_rows_layout.addStretch(1)

    def _show_wire_deep_dive(self) -> None:
        if self._report is None:
            self.deep_dive_table.setRowCount(0)
            return

        row = self.leaderboard_table.currentRow()
        if row < 0 or row >= len(self._leaderboard_rows):
            self.deep_dive_table.setRowCount(0)
            return

        wire_name = str(self._leaderboard_rows[row]["wire"])
        trials = [
            trial
            for trial in self._report.trials
            if trial.execution_wire.tool_ref == wire_name
        ]
        scoring = self._active_scoring_spec()
        max_episode_steps = (
            self._report.execution_plan.max_episode_steps
            if self._report.execution_plan is not None
            else 1000
        )
        self._current_wire_trials = list(trials)
        self._show_trial_list_panel()

        self.deep_dive_table.setRowCount(len(trials))
        for idx, trial in enumerate(trials):
            breakdown = score_trial(
                telemetry=trial.telemetry,
                max_episode_steps=max_episode_steps,
                scoring=scoring,
            )
            is_valid = valid_for_ranking(
                telemetry=trial.telemetry,
                max_episode_steps=max_episode_steps,
                scoring=scoring,
            )
            force_value = None
            if trial.telemetry.forces is not None:
                force_value = trial.telemetry.forces.wire_force_normal_trial_max_N
            time_value = trial.telemetry.steps_to_success
            if time_value is None:
                time_value = trial.telemetry.steps_total

            self.deep_dive_table.setItem(
                idx, 0, QTableWidgetItem(str(trial.trial_index + 1))
            )
            self.deep_dive_table.setItem(
                idx, 1, QTableWidgetItem(str(bool(trial.telemetry.success)))
            )
            self.deep_dive_table.setItem(idx, 2, QTableWidgetItem(str(bool(is_valid))))
            self.deep_dive_table.setItem(
                idx,
                3,
                QTableWidgetItem(str(time_value)),
            )
            self.deep_dive_table.setItem(
                idx,
                4,
                QTableWidgetItem(str(trial.telemetry.end_reason)),
            )
            self.deep_dive_table.setItem(
                idx,
                5,
                QTableWidgetItem(
                    "n/a" if force_value is None else f"{float(force_value):.4f}"
                ),
            )
            self.deep_dive_table.setItem(
                idx,
                6,
                QTableWidgetItem(
                    f"{float(soft_score_total(breakdown=breakdown, scoring=scoring)):.4f}"
                ),
            )
            view_button = QPushButton("View")
            trace_path = getattr(trial.artifacts, "trace_h5_path", None)
            view_button.setEnabled(trace_path is not None)
            view_button.clicked.connect(
                lambda _checked=False, trial_index=idx: self._open_trial_trace_by_index(
                    trial_index
                )
            )
            self.deep_dive_table.setCellWidget(idx, 7, view_button)

    def _open_trial_trace_by_index(self, trial_index: int) -> None:
        if trial_index < 0 or trial_index >= len(self._current_wire_trials):
            return
        trial = self._current_wire_trials[trial_index]
        trace_path = getattr(trial.artifacts, "trace_h5_path", None)
        if trace_path is None:
            return
        self.trace_viewer_panel.open_trace(Path(trace_path), start_step=0)
        self._details_stack.setCurrentWidget(self.trace_viewer_panel)

    def _show_trial_list_panel(self) -> None:
        self.trace_viewer_panel.close_trace()
        self._details_stack.setCurrentIndex(0)

    def _save_feedback(self) -> None:
        if self._report_id is None or self._report is None:
            self.feedback_status_label.setText(
                "No report context available for feedback."
            )
            return

        attempts: list[dict[str, object]] = []
        for index, row in enumerate(self._feedback_attempt_rows):
            wire_name = row.current_wire()
            if not wire_name:
                continue
            attempts.append(
                {
                    "attempt_index": index + 1,
                    "wire": wire_name,
                    "outcome": row.current_outcome(),
                }
            )

        if not attempts:
            self.feedback_status_label.setText(
                "Select at least one wire attempt before saving."
            )
            return

        payload = {
            "job_name": self._report.job_name,
            "wire_actually_used": str(attempts[0]["wire"]),
            "clinical_outcome": str(attempts[0]["outcome"]),
            "wire_attempts": attempts,
            "clinician_notes": self.feedback_notes_edit.toPlainText().strip(),
        }
        feedback_path = self.controller.save_clinical_feedback(self._report_id, payload)
        self.feedback_status_label.setText(f"Saved feedback to {feedback_path.name}")


__all__ = [
    "WizardPage",
    "AnatomySelectionPage",
    "BranchSelectionPage",
    "TargetSelectionPage",
    "WireSelectionPage",
    "ExecutionConfigPage",
    "PipelineRunningPage",
    "ResultsPage",
]
