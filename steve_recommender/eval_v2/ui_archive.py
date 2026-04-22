from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .models import EvaluationReport, HistoricalReportSummary
from .ui_controller import ClinicalUIController


class ArchiveCardWidget(QWidget):
    view_details_requested = pyqtSignal(object)

    def __init__(
        self,
        *,
        summary: HistoricalReportSummary,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.summary = summary
        self.setObjectName("archiveCard")
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.setStyleSheet(
            """
/* 1. The Main Card Surface */
ArchiveCardWidget {
    background-color: #2A2A2A;
    border: 1px solid #30363D;
    border-radius: 12px;
}

/* 2. FIX: Force all inner layout containers to be transparent */
ArchiveCardWidget > QWidget {
    background-color: transparent;
}

/* 3. FIX: Strip the gray background off all text labels */
ArchiveCardWidget QLabel {
    background-color: transparent;
    border: none;
    padding: 0px;
}

/* 4. Specific Typography Overrides */
ArchiveCardWidget QLabel#jobNameLabel {
    font-size: 18px;
    font-weight: 700;
    color: #E6EDF3;
}

ArchiveCardWidget QLabel#wiresLabel {
    font-size: 14px;
    color: #E6EDF3;
    margin-top: 4px;
}

ArchiveCardWidget QLabel#metadataLabel {
    font-size: 13px;
    font-weight: 500;
    color: #8B949E;
    margin-top: 8px;
}

/* 5. Ghost Button Styling for 'View Details' */
ArchiveCardWidget QPushButton {
    background-color: transparent;
    color: #E6EDF3;
    font-size: 14px;
    font-weight: 700;
    border: 1px solid #555555;
    border-radius: 8px;
    padding: 8px 16px;
}

ArchiveCardWidget QPushButton:hover {
    background-color: rgba(242, 172, 50, 0.1);
    color: #F2AC32;
    border: 1px solid #F2AC32;
}
"""
        )

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(20)

        metadata_column = QVBoxLayout()
        metadata_column.setSpacing(4)

        title = QLabel(summary.job_name)
        title.setObjectName("jobNameLabel")
        wires = QLabel(f"Wires Tested: {', '.join(summary.tested_wires)}")
        wires.setObjectName("wiresLabel")

        metadata_row = QHBoxLayout()
        metadata_row.setContentsMargins(0, 0, 0, 0)
        metadata_row.setSpacing(10)

        anatomy = QLabel(f"Anatomy: {summary.anatomy}")
        anatomy.setObjectName("metadataLabel")
        separator = QLabel("•")
        separator.setObjectName("metadataLabel")
        generated = QLabel(f"Date: {summary.generated_at}")
        generated.setObjectName("metadataLabel")

        metadata_row.addWidget(anatomy)
        metadata_row.addWidget(separator)
        metadata_row.addWidget(generated)
        metadata_row.addStretch(1)

        metadata_column.addWidget(title)
        metadata_column.addWidget(wires)
        metadata_column.addLayout(metadata_row)

        root_layout.addLayout(metadata_column, stretch=1)

        action_column = QVBoxLayout()
        action_column.setContentsMargins(0, 0, 0, 0)
        action_column.setSpacing(0)
        action_column.addStretch(1)

        self.view_details_button = QPushButton("View Details →")
        self.view_details_button.clicked.connect(self._emit_view_requested)
        action_column.addWidget(self.view_details_button)
        action_column.addStretch(1)

        root_layout.addLayout(action_column, stretch=0)

    def _emit_view_requested(self) -> None:
        self.view_details_requested.emit(self.summary)


class ArchiveScreen(QWidget):
    home_requested = pyqtSignal()
    report_selected = pyqtSignal(object)

    def __init__(
        self,
        *,
        controller: ClinicalUIController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setObjectName("archiveScreen")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(18)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        logo_path = Path(__file__).resolve().parent / "assets" / "logo.svg"
        if logo_path.exists():
            logo = QSvgWidget(str(logo_path))
            logo.setFixedSize(252, 80)
            header_row.addWidget(logo, stretch=0)
        else:
            header_row.addWidget(QLabel("Chair Logo"), stretch=0)
        header_row.addStretch(1)
        self.home_button = QPushButton("Home")
        self.home_button.setProperty("buttonRole", "ghost")
        self.home_button.clicked.connect(self.home_requested.emit)
        header_row.addWidget(self.home_button, stretch=0)
        root_layout.addLayout(header_row)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(16)

        scroll_area.setWidget(self._content)
        root_layout.addWidget(scroll_area, stretch=1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh_reports()

    def refresh_reports(self) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        summaries = self.controller.list_historical_reports()
        if not summaries:
            self._content_layout.addWidget(QLabel("No historical reports found."))
            self._content_layout.addStretch(1)
            return

        for summary in summaries:
            card = ArchiveCardWidget(summary=summary)
            card.view_details_requested.connect(self._on_view_details_requested)
            self._content_layout.addWidget(card)

        self._content_layout.addStretch(1)

    def _on_view_details_requested(self, summary: HistoricalReportSummary) -> None:
        report = self.controller.load_report_from_disk(summary.report_json_path)
        self.report_selected.emit(report)


__all__ = ["ArchiveCardWidget", "ArchiveScreen"]