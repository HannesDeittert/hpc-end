import math
from PyQt5.QtWidgets import (
    QWizardPage, QLabel, QDoubleSpinBox,
    QFormLayout, QVBoxLayout, QWidget
)
from PyQt5.QtCore import pyqtSignal

class ShapeParamsPage(QWizardPage):
    """
    Wizard page to collect procedural shape parameters:
      - overall length
      - tip radius
      - tip outer diameter
      - tip inner diameter
      - tip angle
      - spire height
      - straight outer diameter
      - straight inner diameter
    """
    completeChanged = pyqtSignal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setTitle("Shape Parameters")
        self.setSubTitle("Specify the procedural shape dimensions.")

        # Overall length
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.0, 1e6)
        self.length_spin.setDecimals(2)
        self.length_spin.setSuffix(" mm")
        self.registerField("overallLength*", self.length_spin)
        self.length_spin.valueChanged.connect(self.completeChanged)

        # Tip radius
        self.tip_radius_spin = QDoubleSpinBox()
        self.tip_radius_spin.setRange(0.0, 1e6)
        self.tip_radius_spin.setDecimals(2)
        self.tip_radius_spin.setSuffix(" mm")
        self.registerField("tipRadius*", self.tip_radius_spin)
        self.tip_radius_spin.valueChanged.connect(self.completeChanged)

        # Tip outer diameter
        self.tip_outer_dia_spin = QDoubleSpinBox()
        self.tip_outer_dia_spin.setRange(0.0, 1e6)
        self.tip_outer_dia_spin.setDecimals(2)
        self.tip_outer_dia_spin.setSuffix(" mm")
        self.registerField("tipOuterDia*", self.tip_outer_dia_spin)
        self.tip_outer_dia_spin.valueChanged.connect(self.completeChanged)

        # Tip inner diameter
        self.tip_inner_dia_spin = QDoubleSpinBox()
        self.tip_inner_dia_spin.setRange(0.0, 1e6)
        self.tip_inner_dia_spin.setDecimals(2)
        self.tip_inner_dia_spin.setSuffix(" mm")
        self.registerField("tipInnerDia*", self.tip_inner_dia_spin)
        self.tip_inner_dia_spin.valueChanged.connect(self.completeChanged)

        # Tip angle
        self.tip_angle_spin = QDoubleSpinBox()
        self.tip_angle_spin.setRange(0.0, 2 * math.pi)
        self.tip_angle_spin.setDecimals(3)
        self.tip_angle_spin.setSuffix(" rad")
        self.registerField("tipAngle*", self.tip_angle_spin)
        self.tip_angle_spin.valueChanged.connect(self.completeChanged)

        # Spire height
        self.spire_height_spin = QDoubleSpinBox()
        self.spire_height_spin.setRange(0.0, 1e6)
        self.spire_height_spin.setDecimals(2)
        self.spire_height_spin.setSuffix(" mm")
        self.registerField("spireHeight*", self.spire_height_spin)
        self.spire_height_spin.valueChanged.connect(self.completeChanged)

        # Straight outer diameter
        self.outer_dia_spin = QDoubleSpinBox()
        self.outer_dia_spin.setRange(0.0, 1e6)
        self.outer_dia_spin.setDecimals(2)
        self.outer_dia_spin.setSuffix(" mm")
        self.registerField("straightOuterDia*", self.outer_dia_spin)
        self.outer_dia_spin.valueChanged.connect(self.completeChanged)

        # Straight inner diameter
        self.inner_dia_spin = QDoubleSpinBox()
        self.inner_dia_spin.setRange(0.0, 1e6)
        self.inner_dia_spin.setDecimals(2)
        self.inner_dia_spin.setSuffix(" mm")
        self.registerField("straightInnerDia*", self.inner_dia_spin)
        self.inner_dia_spin.valueChanged.connect(self.completeChanged)

        # Layout
        form = QFormLayout()
        form.addRow(QLabel("Overall Length (mm):"), self.length_spin)
        form.addRow(QLabel("Tip Radius (mm):"), self.tip_radius_spin)
        form.addRow(QLabel("Tip Outer Dia (mm):"), self.tip_outer_dia_spin)
        form.addRow(QLabel("Tip Inner Dia (mm):"), self.tip_inner_dia_spin)
        form.addRow(QLabel("Tip Angle (rad):"), self.tip_angle_spin)
        form.addRow(QLabel("Spire Height (mm):"), self.spire_height_spin)
        form.addRow(QLabel("Straight Outer Dia (mm):"), self.outer_dia_spin)
        form.addRow(QLabel("Straight Inner Dia (mm):"), self.inner_dia_spin)

        main_layout = QVBoxLayout()
        main_layout.addLayout(form)
        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def initializePage(self):
        # sensible defaults
        self.length_spin.setValue(100.0)
        self.tip_radius_spin.setValue(5.0)
        self.tip_outer_dia_spin.setValue(10.0)
        self.tip_inner_dia_spin.setValue(2.0)
        self.tip_angle_spin.setValue(0.4 * math.pi)
        self.spire_height_spin.setValue(20.0)
        self.outer_dia_spin.setValue(2.0)
        self.inner_dia_spin.setValue(0.5)

    def isComplete(self) -> bool:
        length = self.length_spin.value()
        tip_r = self.tip_radius_spin.value()
        tip_o = self.tip_outer_dia_spin.value()
        tip_i = self.tip_inner_dia_spin.value()
        tip_a = self.tip_angle_spin.value()
        spire_h = self.spire_height_spin.value()
        out_d = self.outer_dia_spin.value()
        in_d  = self.inner_dia_spin.value()

        return (
            length > 0.0 and
            tip_r >= 0.0 and
            tip_o > 0.0 and
            0.0 <= tip_i < tip_o and
            tip_a > 0.0 and
            spire_h > 0.0 and
            out_d > 0.0 and
            0.0 <= in_d < out_d
        )
