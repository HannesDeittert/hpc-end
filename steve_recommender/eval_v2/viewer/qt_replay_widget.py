"""Shared Qt replay widget hosting the Phase F trace renderer and controls."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pyvista as pv
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from .renderer import TraceRenderer
from .trace_data import TraceData


DEFAULT_PLAYBACK_FPS = 30
PLAY_BUTTON_ICON_PLAY = QStyle.SP_MediaPlay
PLAY_BUTTON_ICON_PAUSE = QStyle.SP_MediaPause


class QtTraceReplayWidget(QWidget):
    """Qt host for one trace replay with shared controls and embedded 3D view."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._trace_data: Optional[TraceData] = None
        self._renderer: Optional[TraceRenderer] = None
        self._plotter = None
        self._plotter_widget: Optional[QWidget] = None
        self._plotter_owned = False
        self._step_count = 0
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(int(round(1000.0 / DEFAULT_PLAYBACK_FPS)))
        self._play_timer.timeout.connect(self._advance_playback)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self._plotter_container = QWidget()
        self._plotter_layout = QVBoxLayout(self._plotter_container)
        self._plotter_layout.setContentsMargins(0, 0, 0, 0)
        self._plotter_layout.setSpacing(0)
        root.addWidget(self._plotter_container, stretch=1)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(PLAY_BUTTON_ICON_PLAY))
        self.play_button.setToolTip("Play")
        self.play_button.clicked.connect(self._toggle_playback)
        controls.addWidget(self.play_button, stretch=0)

        self.step_slider = QSlider(Qt.Horizontal)
        self.step_slider.setRange(0, 0)
        self.step_slider.valueChanged.connect(self._on_slider_changed)
        controls.addWidget(self.step_slider, stretch=1)

        self.step_label = QLabel("Step 0 / 0")
        controls.addWidget(self.step_label, stretch=0)

        self.time_label = QLabel("t = 0.000 s")
        controls.addWidget(self.time_label, stretch=0)

        root.addLayout(controls)

        self._init_plotter_widget()

    @property
    def current_step(self) -> int:
        """Return the currently displayed simulation step index."""

        return int(self.step_slider.value())

    @property
    def renderer(self) -> Optional[TraceRenderer]:
        """Return the active renderer instance when a trace is open."""

        return self._renderer

    @property
    def trace_data(self) -> Optional[TraceData]:
        """Return the active trace loader when a trace is open."""

        return self._trace_data

    def open_trace(
        self,
        trace_path: Path,
        *,
        start_step: int = 0,
        max_force_n: Optional[float] = None,
    ) -> None:
        """Load a persisted trace and display the requested start step."""

        self.close_trace()
        if self._plotter is None:
            self._init_plotter_widget()
        self._trace_data = TraceData(Path(trace_path))
        if self._plotter is None:
            raise RuntimeError("Replay plotter is unavailable in this environment.")
        self._renderer = TraceRenderer(
            self._trace_data,
            max_display_force_N=max_force_n,
            plotter=self._plotter,
        )
        self._step_count = int(self._trace_data.n_steps)
        self.step_slider.blockSignals(True)
        self.step_slider.setRange(0, max(self._step_count - 1, 0))
        self.step_slider.setValue(max(0, min(int(start_step), self._step_count - 1)))
        self.step_slider.blockSignals(False)
        self._set_play_state(False)
        self._render_step(self.current_step)

    def close_trace(self) -> None:
        """Release the renderer, plotter state, and trace reader."""

        self._play_timer.stop()
        self._set_play_state(False)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._trace_data is not None:
            self._trace_data.close()
            self._trace_data = None
        if self._plotter is not None:
            self._plotter.clear()
            if self._plotter_owned:
                self._plotter.close()
                self._plotter = None
                self._plotter_owned = False
        self._step_count = 0
        self.step_slider.blockSignals(True)
        self.step_slider.setRange(0, 0)
        self.step_slider.setValue(0)
        self.step_slider.blockSignals(False)
        self.step_label.setText("Step 0 / 0")
        self.time_label.setText("t = 0.000 s")

    def _init_plotter_widget(self) -> None:
        while self._plotter_layout.count():
            item = self._plotter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        offscreen_mode = os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen"
        if offscreen_mode:
            self._plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
            self._plotter_owned = True
            placeholder = QLabel("Off-screen replay renderer active.")
            self._plotter_widget = placeholder
            self._plotter_layout.addWidget(placeholder)
            return

        from pyvistaqt import QtInteractor

        self._plotter = QtInteractor(
            self._plotter_container,
            off_screen=offscreen_mode,
        )
        self._plotter_widget = self._plotter.interactor
        self._plotter_layout.addWidget(self._plotter_widget)

    def _on_slider_changed(self, value: int) -> None:
        if self._renderer is None:
            return
        self._render_step(int(value))

    def _render_step(self, step: int) -> None:
        if self._renderer is None:
            return
        clamped_step = max(0, min(int(step), max(self._step_count - 1, 0)))
        self._renderer.render_frame(clamped_step)
        current = int(self._renderer.current_step)
        self.step_label.setText(f"Step {current} / {max(self._step_count - 1, 0)}")
        sim_time_s = 0.0
        if self._renderer.current_frame is not None:
            sim_time_s = float(self._renderer.current_frame.sim_time_s)
        self.time_label.setText(f"t = {sim_time_s:.3f} s")
        if self.step_slider.value() != current:
            self.step_slider.blockSignals(True)
            self.step_slider.setValue(current)
            self.step_slider.blockSignals(False)

    def _toggle_playback(self) -> None:
        if self._renderer is None or self._step_count <= 1:
            return
        if self._play_timer.isActive():
            self._play_timer.stop()
            self._set_play_state(False)
            return
        self._play_timer.start()
        self._set_play_state(True)

    def _set_play_state(self, playing: bool) -> None:
        icon = PLAY_BUTTON_ICON_PAUSE if playing else PLAY_BUTTON_ICON_PLAY
        tooltip = "Pause" if playing else "Play"
        self.play_button.setIcon(self.style().standardIcon(icon))
        self.play_button.setToolTip(tooltip)

    def _advance_playback(self) -> None:
        if not self._play_timer.isActive() or self._renderer is None:
            return
        next_step = self.current_step + 1
        if next_step >= self._step_count:
            self._play_timer.stop()
            self._set_play_state(False)
            return
        self.step_slider.setValue(next_step)
