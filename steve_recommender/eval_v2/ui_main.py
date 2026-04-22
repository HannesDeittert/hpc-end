from __future__ import annotations

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QStackedWidget, QVBoxLayout, QWidget

from .service import DefaultEvaluationService
from .ui_archive import ArchiveScreen
from .ui_controller import ClinicalUIController
from .ui_home import HomeScreen
from .ui_wizard import WizardShell


def _build_global_stylesheet() -> str:
    return """
ClinicalMainWindow,
QStackedWidget,
HomeScreen,
ArchiveScreen,
WizardShell {
    background-color: #444444;
}

QWidget {
    color: #E6EDF3;
    background-color: #444444;
    font-size: 14px;
}

QLabel {
    color: #E6EDF3;
    font-size: 14px;
}

QPushButton {
    background-color: #2A2A2A;
    border: 1px solid #1C1C1C;
    border-radius: 8px;
    padding: 10px 16px;
    color: #E6EDF3;
    font-size: 14px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #383838;
    border: 1px solid #555555;
}

QPushButton:pressed {
    background-color: #1A1A1A;
}

QPushButton[buttonRole="primary"] {
    background-color: #F2AC32;
    color: #111418;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
}

QPushButton[buttonRole="primary"]:hover {
    background-color: #F7C369;
    border: none;
}

QPushButton[buttonRole="primary"]:pressed {
    background-color: #e6a12d;
    border: none;
}

QPushButton[buttonRole="primary"]:disabled {
    background-color: #555555;
    color: #888888;
}

QPushButton[buttonRole="secondary"] {
    background-color: #2A2A2A;
    color: #E6EDF3;
    border: 1px solid #1C1C1C;
    border-radius: 8px;
}

QPushButton[buttonRole="secondary"]:hover {
    background-color: #383838;
    border: 1px solid #555555;
}

QPushButton[buttonRole="secondary"]:pressed {
    background-color: #1A1A1A;
}

QPushButton[buttonRole="ghost"] {
    background-color: transparent;
    color: #E6EDF3;
    border: 1px solid #8B949E;
    border-radius: 8px;
}

QPushButton[buttonRole="ghost"]:hover {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid #E6EDF3;
}

QPushButton[buttonRole="ghost"]:pressed {
    background-color: rgba(255, 255, 255, 0.1);
}

QPushButton[buttonSize="hero"] {
    min-width: 200px;
    min-height: 150px;
    border-radius: 12px;
    font-size: 18px;
    padding: 24px 32px;
}

QGroupBox {
    background-color: #2A2A2A;
    border: 1px solid #30363D;
    border-radius: 12px;
    margin-top: 10px;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #F2AC32;
}

ArchiveCardWidget {
    background-color: #2A2A2A;
    border: 1px solid #30363D;
    border-radius: 14px;
}

QLabel[textRole="h1"] {
    font-size: 24px;
    font-weight: 700;
    color: #E6EDF3;
}

QLabel[textRole="h2"] {
    font-size: 18px;
    font-weight: 600;
    color: #E6EDF3;
}

QLabel[textRole="body"] {
    font-size: 14px;
    font-weight: 400;
    color: #E6EDF3;
}

QLabel[textRole="small"] {
    font-size: 12px;
    font-weight: 400;
    color: #8B949E;
}

QProgressBar {
    background-color: #1C2128;
    border: 1px solid #30363D;
    border-radius: 8px;
    text-align: center;
    color: #E6EDF3;
    font-weight: bold;
    height: 24px;
}

QProgressBar::chunk {
    background-color: #F2AC32;
    border-radius: 6px;
}

QProgressBar[progressVariant="hero"] {
    border-radius: 14px;
    font-size: 16px;
}

QProgressBar[progressVariant="hero"]::chunk {
    border-radius: 12px;
}

QWidget[class~="BentoCard"] {
    background-color: #2A2A2A;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 16px;
}

QWidget[class~="BentoCard"] QLabel,
QGroupBox QLabel {
    background-color: transparent;
    border: none;
}

QWidget#LeftWeightPanel {
    background-color: #2A2A2A;
    border-radius: 12px;
    border: 1px solid #30363D;
}

QScrollArea#LeftWeightsScroll {
    background-color: transparent;
    border: none;
}

QScrollArea#LeftWeightsScroll QWidget#qt_scrollarea_viewport {
    background-color: transparent;
    border: none;
}

QWidget#LeftWeightsContent {
    background-color: transparent;
    border: none;
}

QGroupBox[class~="CategoryBox"],
QGroupBox.CategoryBox {
    background-color: #444444;
    border: 1px solid #1C1C1C;
    border-radius: 8px;
    margin-top: 24px;
    padding-top: 16px;
}

QGroupBox[class~="CategoryBox"]::title,
QGroupBox.CategoryBox::title {
    color: #F2AC32;
    background-color: transparent;
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0px 4px;
    font-size: 14px;
    font-weight: bold;
}

QWidget#LeftWeightPanel QLabel {
    background-color: transparent;
    border: none;
    color: #E6EDF3;
}

QWidget#LeftWeightPanel QPushButton {
    background-color: #444444;
    border: 1px solid #30363D;
    border-radius: 8px;
    color: #E6EDF3;
    padding: 10px;
    font-weight: bold;
}

QWidget#LeftWeightPanel QPushButton:hover {
    background-color: #555555;
}

QGroupBox#ClinicalFeedbackGroup QComboBox {
    background-color: #1C2128;
    border: 1px solid #30363D;
    border-radius: 4px;
    padding: 4px 8px;
    color: #E6EDF3;
    min-height: 24px;
}

QGroupBox#ClinicalFeedbackGroup QComboBox::drop-down {
    border: none;
}

QGroupBox#ClinicalFeedbackGroup QComboBox::down-arrow {
    image: none;
}

QWidget#AttemptRowWidget {
    background-color: #2A2A2A !important;
    border: none !important;
}

QWidget#AttemptRowWidget > QLabel {
    background-color: #2A2A2A !important;
    color: #8B949E;
    border: none !important;
}

QScrollArea#FeedbackScrollArea {
    background-color: #2A2A2A !important;
    border: none !important;
}

QScrollArea#FeedbackScrollArea > QWidget#qt_scrollarea_viewport {
    background-color: #2A2A2A !important;
}

QScrollArea#FeedbackScrollArea > QWidget#qt_scrollarea_viewport > QWidget {
    background-color: #2A2A2A !important;
}

QPushButton#AddAttemptBtn {
    background-color: transparent;
    border: 1px dashed #8B949E;
    color: #8B949E;
    border-radius: 6px;
    padding: 8px;
    margin-bottom: 12px;
}

QPushButton#AddAttemptBtn:hover {
    border: 1px dashed #E6EDF3;
    color: #E6EDF3;
    background-color: rgba(255, 255, 255, 0.05);
}

QPushButton[class~="DeleteRowBtn"],
QPushButton.DeleteRowBtn {
    background-color: transparent;
    color: #DA3633;
    font-weight: bold;
    border: none;
    min-width: 24px;
    max-width: 24px;
    padding: 0;
}

QPushButton[class~="DeleteRowBtn"]:hover,
QPushButton.DeleteRowBtn:hover {
    color: #FF5555;
}

QLabel[class~="KPITitle"] {
    color: #8B949E;
    font-size: 13px;
    font-weight: bold;
}

QLabel[class~="KPIValue"] {
    color: #F2AC32;
    font-size: 22px;
    font-weight: bold;
    margin-top: 8px;
}

QSlider::groove:horizontal {
    border: 1px solid #1C1C1C;
    height: 6px;
    background: #1C2128;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #F2AC32;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #E6EDF3;
    border: 1px solid #555555;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}

QProgressBar[class~="ChartBar"] {
    background-color: #1C2128;
    border: none;
    border-radius: 4px;
    height: 16px;
    min-height: 16px;
    max-height: 16px;
}

QProgressBar[class~="ChartBar"]::chunk {
    background-color: #5c7080;
    border-radius: 4px;
}

QProgressBar[class~="ChartBar"][chartRank="winner"]::chunk {
    background-color: #F2AC32;
}

QWidget#ChartContainerWidget,
QFrame#ChartContainerWidget {
    background-color: #2A2A2A;
    border: none;
}

QWidget#ChartRowWidget {
    background-color: #2A2A2A;
    border: none;
}

QWidget#ChartRowWidget QLabel {
    background-color: #2A2A2A;
    border: none;
}

QLabel[class~="ChartWireLabel"] {
    color: #E6EDF3;
    font-size: 13px;
    font-weight: 500;
}

QLabel[class~="ChartScoreLabel"] {
    color: #F2AC32;
    font-size: 13px;
    font-weight: bold;
}

QTableWidget {
    background-color: #2A2A2A;
    border: 1px solid #30363D;
    border-radius: 8px;
    color: #E6EDF3;
    gridline-color: #30363D;
    selection-background-color: rgba(242, 172, 50, 0.15);
    selection-color: #F2AC32;
}

QHeaderView::section {
    background-color: #1C2128;
    color: #8B949E;
    border: none;
    border-bottom: 1px solid #30363D;
    padding: 6px;
    font-weight: bold;
}
"""


class ClinicalMainWindow(QWidget):
    def __init__(self, *, service: DefaultEvaluationService | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("clinicalMainWindow")

        app = QApplication.instance()
        if app is not None:
            app.setFont(QFont("Segoe UI", 10))
            app.setStyleSheet(_build_global_stylesheet())

        self.service = service or DefaultEvaluationService()
        self.controller = ClinicalUIController(service=self.service)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.stack = QStackedWidget()
        self.stack.setObjectName("mainStack")
        self.home_screen = HomeScreen()
        self.archive_screen = ArchiveScreen(controller=self.controller)
        self.setup_screen = WizardShell(controller=self.controller)

        self.stack.addWidget(self.home_screen)
        self.stack.addWidget(self.archive_screen)
        self.stack.addWidget(self.setup_screen)

        root_layout.addWidget(self.stack)

        self.home_screen.new_setup_requested.connect(self.show_setup_screen)
        self.home_screen.archive_requested.connect(self.show_archive_screen)
        self.archive_screen.home_requested.connect(self.show_home_screen)
        self.archive_screen.report_selected.connect(self.show_historical_report)
        self.setup_screen.home_requested.connect(self.show_home_screen)

        self.stack.setCurrentWidget(self.home_screen)
        self.setWindowTitle("Clinical Decision Support System")

    def show_home_screen(self) -> None:
        self.controller.reset_wizard_state()
        self.stack.setCurrentWidget(self.home_screen)

    def show_archive_screen(self) -> None:
        self.stack.setCurrentWidget(self.archive_screen)

    def show_setup_screen(self) -> None:
        self.stack.setCurrentWidget(self.setup_screen)

    def show_historical_report(self, report) -> None:
        self.setup_screen.show_historical_report(report)
        self.stack.setCurrentWidget(self.setup_screen)


__all__ = ["ClinicalMainWindow"]
