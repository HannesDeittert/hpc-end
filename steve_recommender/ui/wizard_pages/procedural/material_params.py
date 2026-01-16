from PyQt5.QtWidgets import (
    QWizardPage, QLabel, QDoubleSpinBox,
    QFormLayout, QVBoxLayout, QWidget
)
from PyQt5.QtCore import pyqtSignal

class MaterialParamsPage(QWizardPage):
    """
    Page to set material parameters for procedural shape:
      - Poisson's ratio
      - Young's modulus for straight section
      - Young's modulus for tip/spire
      - Mass density for straight section
      - Mass density for tip/spire
    """
    completeChanged = pyqtSignal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setTitle("Material Parameters")
        self.setSubTitle("Specify mechanical properties for your procedural shape.")

        # Poisson's ratio
        self.poisson_spin = QDoubleSpinBox()
        self.poisson_spin.setRange(0.0, 0.5)
        self.poisson_spin.setDecimals(3)
        self.registerField("proc_poissonRatio*", self.poisson_spin)
        self.poisson_spin.valueChanged.connect(self.completeChanged)

        # Young's modulus - straight
        self.young_str_spin = QDoubleSpinBox()
        self.young_str_spin.setRange(0.0, 1e9)
        self.young_str_spin.setDecimals(2)
        self.young_str_spin.setSuffix(" Pa")
        self.registerField("youngStraight*", self.young_str_spin)
        self.young_str_spin.valueChanged.connect(self.completeChanged)

        # Young's modulus - tip/spire
        self.young_tip_spin = QDoubleSpinBox()
        self.young_tip_spin.setRange(0.0, 1e9)
        self.young_tip_spin.setDecimals(2)
        self.young_tip_spin.setSuffix(" Pa")
        self.registerField("youngTip*", self.young_tip_spin)
        self.young_tip_spin.valueChanged.connect(self.completeChanged)

        # Mass density - straight
        self.density_str_spin = QDoubleSpinBox()
        self.density_str_spin.setRange(0.0, 1.0)
        self.density_str_spin.setDecimals(6)
        self.density_str_spin.setSuffix(" kg/mm^3")
        self.registerField("densityStraight*", self.density_str_spin)
        self.density_str_spin.valueChanged.connect(self.completeChanged)

        # Mass density - tip/spire
        self.density_tip_spin = QDoubleSpinBox()
        self.density_tip_spin.setRange(0.0, 1.0)
        self.density_tip_spin.setDecimals(6)
        self.density_tip_spin.setSuffix(" kg/mm^3")
        self.registerField("densityTip*", self.density_tip_spin)
        self.density_tip_spin.valueChanged.connect(self.completeChanged)

        # Layout
        form = QFormLayout()
        form.addRow(QLabel("Poisson's Ratio:"), self.poisson_spin)
        form.addRow(QLabel("Young's Modulus (Straight):"), self.young_str_spin)
        form.addRow(QLabel("Young's Modulus (Tip):"), self.young_tip_spin)
        form.addRow(QLabel("Mass Density (Straight):"), self.density_str_spin)
        form.addRow(QLabel("Mass Density (Tip):"), self.density_tip_spin)

        main_layout = QVBoxLayout()
        main_layout.addLayout(form)
        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def initializePage(self):
        # default values
        self.poisson_spin.setValue(0.49)
        self.young_str_spin.setValue(80e3)
        self.young_tip_spin.setValue(17e3)
        self.density_str_spin.setValue(0.000021)
        self.density_tip_spin.setValue(0.000021)

    def isComplete(self) -> bool:
        # check values are positive and Poisson <=0.5
        return (
            0.0 <= self.poisson_spin.value() <= 0.5 and
            self.young_str_spin.value() > 0.0 and
            self.young_tip_spin.value() > 0.0 and
            self.density_str_spin.value() > 0.0 and
            self.density_tip_spin.value() > 0.0
        )
