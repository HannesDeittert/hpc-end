# ui/wizard_pages/non_procedural/diameter_params.py

from PyQt5.QtWidgets import (
    QWizardPage, QLabel, QDoubleSpinBox,
    QFormLayout, QVBoxLayout, QWidget
)
from PyQt5.QtCore import pyqtSignal

class DiameterParamsPage(QWizardPage):
    # notify wizard that completeness might have changed
    completeChanged = pyqtSignal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setTitle("Diameter Parameters")
        self.setSubTitle("Specify the outer and inner diameters (in mm).")

        # Outer Diameter
        self.outer_spin = QDoubleSpinBox()
        self.outer_spin.setRange(0.0, 1e6)
        self.outer_spin.setSuffix(" mm")
        self.outer_spin.setDecimals(2)
        self.registerField("outerDiameter*", self.outer_spin)
        self.outer_spin.valueChanged.connect(self.completeChanged)

        # Inner Diameter
        self.inner_spin = QDoubleSpinBox()
        self.inner_spin.setRange(0.0, 1e6)
        self.inner_spin.setSuffix(" mm")
        self.inner_spin.setDecimals(2)
        self.registerField("innerDiameter*", self.inner_spin)
        self.inner_spin.valueChanged.connect(self.completeChanged)

        # Layout
        form = QFormLayout()
        form.addRow(QLabel("Outer Diameter:"), self.outer_spin)
        form.addRow(QLabel("Inner Diameter:"), self.inner_spin)

        main_layout = QVBoxLayout()
        main_layout.addLayout(form)
        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def initializePage(self):
        # Reset defaults when shown
        self.outer_spin.setValue(0.0)
        self.inner_spin.setValue(0.0)

    def isComplete(self) -> bool:
        outer = self.outer_spin.value()
        inner = self.inner_spin.value()
        # Outer must be > 0; inner can be 0 up to less than outer
        return (outer > 0.0) and (0.0 <= inner < outer)
