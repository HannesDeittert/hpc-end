"""Embedded Qt replay panel for eval_v2 trial traces."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .viewer.qt_replay_widget import QtTraceReplayWidget
from .viewer.renderer import TraceRenderer
from .viewer.trace_data import TraceData


class TraceViewerPanel(QWidget):
    """Inline Qt host for the shared trace replay widget."""

    back_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.back_button = QPushButton("Back to Trials")
        self.back_button.clicked.connect(self.back_requested.emit)
        header.addWidget(self.back_button)
        header.addStretch(1)
        root.addLayout(header)

        self.replay_widget = QtTraceReplayWidget()
        root.addWidget(self.replay_widget, stretch=1)

    @property
    def current_step(self) -> int:
        """Return the currently displayed simulation step index."""

        return self.replay_widget.current_step

    @property
    def _renderer(self) -> Optional[TraceRenderer]:
        return self.replay_widget.renderer

    @property
    def _trace_data(self) -> Optional[TraceData]:
        return self.replay_widget.trace_data

    @property
    def step_slider(self):
        return self.replay_widget.step_slider

    @property
    def play_button(self):
        return self.replay_widget.play_button

    @property
    def step_label(self):
        return self.replay_widget.step_label

    def open_trace(
        self,
        trace_path: Path,
        *,
        start_step: int = 0,
        max_force_n: Optional[float] = None,
    ) -> None:
        """Load a persisted trace into the embedded replay widget."""

        self.replay_widget.open_trace(
            trace_path,
            start_step=start_step,
            max_force_n=max_force_n,
        )

    def close_trace(self) -> None:
        """Release the embedded replay widget resources."""

        self.replay_widget.close_trace()
