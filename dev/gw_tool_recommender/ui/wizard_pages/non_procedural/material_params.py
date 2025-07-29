# ui/wizard_pages/non_procedural/material_params.py

from PyQt5.QtWidgets import (
    QWizardPage, QLabel, QDoubleSpinBox,
    QFormLayout, QVBoxLayout, QWidget
)
from PyQt5.QtCore import pyqtSignal

class MaterialParamsPage(QWizardPage):
    # Signal to notify wizard that page completeness might have changed
    completeChanged = pyqtSignal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setTitle("Material Parameters")
        self.setSubTitle("Specify material properties with units.")

        # Poisson's ratio (dimensionless, 0–0.5)
        self.poisson_spin = QDoubleSpinBox()
        self.poisson_spin.setRange(0.0, 0.5)
        self.poisson_spin.setDecimals(3)
        self.registerField("poissonRatio*", self.poisson_spin)
        self.poisson_spin.valueChanged.connect(self.completeChanged)

        # Young's modulus in GPa
        self.young_spin = QDoubleSpinBox()
        self.young_spin.setRange(0.0, 10000.0)
        self.young_spin.setSuffix(" MPa")
        self.young_spin.setDecimals(2)
        self.registerField("youngModulus*", self.young_spin)
        self.young_spin.valueChanged.connect(self.completeChanged)

        # Mass density in kg/m³
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(0.0, 20000.0)
        self.density_spin.setSuffix(" g/mm³")
        self.density_spin.setDecimals(1)
        self.registerField("massDensity*", self.density_spin)
        self.density_spin.valueChanged.connect(self.completeChanged)

        # Layout
        form = QFormLayout()
        form.addRow("Poisson Ratio:", self.poisson_spin)
        form.addRow("Young's Modulus:", self.young_spin)
        form.addRow("Mass Density:", self.density_spin)

        main_layout = QVBoxLayout()
        main_layout.addLayout(form)
        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def initializePage(self):
        # Reset to typical defaults when shown
        self.poisson_spin.setValue(0.3)
        self.young_spin.setValue(210.0)
        self.density_spin.setValue(7850.0)

    def isComplete(self) -> bool:
        # All material properties must be > 0 (except poisson can be 0)
        # But we treat every spin as required: poissonRatio > 0, young >0, density >0
        return (
            self.poisson_spin.value() >= 0.0 and
            self.young_spin.value() > 0.0 and
            self.density_spin.value() > 0.0
        )

