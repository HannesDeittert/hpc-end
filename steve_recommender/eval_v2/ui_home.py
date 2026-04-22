from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class HomeScreen(QWidget):
    archive_requested = pyqtSignal()
    new_setup_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("homeScreen")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(24)

        self._setup_logo_area(root_layout)
        self._setup_central_controls(root_layout)

    def _setup_logo_area(self, root_layout: QVBoxLayout) -> None:
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)


        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_path = Path(__file__).resolve().parent / "assets" / "logo.svg"
        pixmap = QPixmap(str(logo_path)) if logo_path.exists() else QPixmap()
        if not pixmap.isNull():
            scaled = pixmap.scaled(250, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled)
            logo_label.setFixedHeight(90)
        else:
            logo_label.setText("Chair Logo")

        top_row.addStretch(1)
        top_row.addWidget(logo_label, stretch=0)
        top_row.addStretch(1)
        root_layout.addLayout(top_row)

    def _setup_central_controls(self, root_layout: QVBoxLayout) -> None:
        root_layout.addStretch(1)

        center_row = QHBoxLayout()
        center_row.setSpacing(28)

        self.archive_button = QPushButton("View Archive")
        self.archive_button.setProperty("buttonRole", "secondary")
        self.archive_button.setProperty("buttonSize", "hero")
        self.new_recommendation_button = QPushButton("Start New Recommendation")
        self.new_recommendation_button.setProperty("buttonRole", "primary")
        self.new_recommendation_button.setProperty("buttonSize", "hero")

        self.archive_button.clicked.connect(self.archive_requested.emit)
        self.new_recommendation_button.clicked.connect(self.new_setup_requested.emit)

        center_row.addStretch(1)
        center_row.addWidget(self.archive_button)
        center_row.addSpacing(20)
        center_row.addWidget(self.new_recommendation_button)
        center_row.addStretch(1)
        root_layout.addLayout(center_row)
        root_layout.addStretch(2)


__all__ = ["HomeScreen"]