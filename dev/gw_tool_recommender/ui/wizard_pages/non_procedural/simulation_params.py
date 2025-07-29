# ui/wizard_pages/non_procedural/simulation_params.py

from PyQt5.QtWidgets import (
    QWizardPage, QLabel, QDoubleSpinBox,
    QFormLayout, QVBoxLayout, QWidget, QHBoxLayout
)
from PyQt5.QtCore import pyqtSignal

class SimulationParamsPage(QWizardPage):
    """
    Page to set simulation parameters: visual edges per millimeter,
    plus separate translational and rotational speed limits.
    """
    completeChanged = pyqtSignal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setTitle("Simulation Parameters")
        self.setSubTitle("Specify detail level and speed limits.")

        # Visual edges per mm
        self.edges_spin = QDoubleSpinBox()
        self.edges_spin.setRange(0.0, 1e3)
        self.edges_spin.setDecimals(2)
        self.edges_spin.setSuffix(" edges/mm")
        self.registerField("visualEdges*", self.edges_spin)
        self.edges_spin.valueChanged.connect(self.completeChanged)

        # Translational speed limit (e.g. mm/s)
        self.trans_spin = QDoubleSpinBox()
        self.trans_spin.setRange(0.0, 1e6)
        self.trans_spin.setDecimals(2)
        self.trans_spin.setSuffix(" mm/s")
        self.registerField("transSpeed*", self.trans_spin)
        self.trans_spin.valueChanged.connect(self.completeChanged)

        # Rotational speed limit (e.g. deg/s)
        self.rot_spin = QDoubleSpinBox()
        self.rot_spin.setRange(0.0, 1e6)
        self.rot_spin.setDecimals(2)
        self.rot_spin.setSuffix(" °/s")
        self.registerField("rotSpeed*", self.rot_spin)
        self.rot_spin.valueChanged.connect(self.completeChanged)

        # Layout
        form = QFormLayout()
        form.addRow(QLabel("Visual Edges per mm:"), self.edges_spin)

        # Group the two speed limits horizontally
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Translational:"))
        hbox.addWidget(self.trans_spin)
        hbox.addSpacing(20)
        hbox.addWidget(QLabel("Rotational:"))
        hbox.addWidget(self.rot_spin)
        form.addRow(QLabel("Speed Limits:"), hbox)

        main_layout = QVBoxLayout()
        main_layout.addLayout(form)
        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def initializePage(self):
        # sensible defaults each time the page is shown
        self.edges_spin.setValue(0.5)
        self.trans_spin.setValue(100.0)
        self.rot_spin.setValue(30.0)

    def isComplete(self) -> bool:
        # require edges > 0, and both speeds > 0
        return (
            self.edges_spin.value() > 0.0
            and self.trans_spin.value() > 0.0
            and self.rot_spin.value() > 0.0
        )

