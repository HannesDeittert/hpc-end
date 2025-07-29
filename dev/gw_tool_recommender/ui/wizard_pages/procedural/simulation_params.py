from typing import Tuple
from PyQt5.QtWidgets import (
    QWizardPage, QLabel, QDoubleSpinBox,
    QFormLayout, QVBoxLayout, QWidget, QHBoxLayout
)
from PyQt5.QtCore import pyqtSignal

class SimulationParamsPage(QWizardPage):
    """
    Page for procedural shape simulation parameters:
      - visual edges per mm
      - collision edges per mm (tip)
      - collision edges per mm (straight)
      - beams per mm (tip)
      - beams per mm (straight)
      - velocity limit (translational, rotational)
    """
    completeChanged = pyqtSignal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setTitle("Simulation Parameters")
        self.setSubTitle("Specify mesh densities and velocity limits.")

        # Visual edges per mm
        self.visu_edges_spin = QDoubleSpinBox()
        self.visu_edges_spin.setRange(0.0, 1e3)
        self.visu_edges_spin.setDecimals(2)
        self.visu_edges_spin.setSuffix(" edges/mm")
        self.registerField("visuEdges*", self.visu_edges_spin)
        self.visu_edges_spin.valueChanged.connect(self.completeChanged)

        # Collision edges per mm (tip)
        self.collis_tip_spin = QDoubleSpinBox()
        self.collis_tip_spin.setRange(0.0, 1e3)
        self.collis_tip_spin.setDecimals(2)
        self.collis_tip_spin.setSuffix(" edges/mm")
        self.registerField("collisTip*", self.collis_tip_spin)
        self.collis_tip_spin.valueChanged.connect(self.completeChanged)

        # Collision edges per mm (straight)
        self.collis_str_spin = QDoubleSpinBox()
        self.collis_str_spin.setRange(0.0, 1e3)
        self.collis_str_spin.setDecimals(2)
        self.collis_str_spin.setSuffix(" edges/mm")
        self.registerField("collisStraight*", self.collis_str_spin)
        self.collis_str_spin.valueChanged.connect(self.completeChanged)

        # Beams per mm (tip)
        self.beams_tip_spin = QDoubleSpinBox()
        self.beams_tip_spin.setRange(0.0, 1e3)
        self.beams_tip_spin.setDecimals(2)
        self.beams_tip_spin.setSuffix(" beams/mm")
        self.registerField("beamsTip*", self.beams_tip_spin)
        self.beams_tip_spin.valueChanged.connect(self.completeChanged)

        # Beams per mm (straight)
        self.beams_str_spin = QDoubleSpinBox()
        self.beams_str_spin.setRange(0.0, 1e3)
        self.beams_str_spin.setDecimals(2)
        self.beams_str_spin.setSuffix(" beams/mm")
        self.registerField("beamsStraight*", self.beams_str_spin)
        self.beams_str_spin.valueChanged.connect(self.completeChanged)

        # Velocity limit (translational, rotational)
        self.trans_speed_spin = QDoubleSpinBox()
        self.trans_speed_spin.setRange(0.0, 1e4)
        self.trans_speed_spin.setDecimals(2)
        self.trans_speed_spin.setSuffix(" mm/s")
        self.registerField("proc_transSpeed*", self.trans_speed_spin)
        self.trans_speed_spin.valueChanged.connect(self.completeChanged)

        self.rot_speed_spin = QDoubleSpinBox()
        self.rot_speed_spin.setRange(0.0, 1e4)
        self.rot_speed_spin.setDecimals(2)
        self.rot_speed_spin.setSuffix(" rad/s")
        self.registerField("proc_rotSpeed*", self.rot_speed_spin)
        self.rot_speed_spin.valueChanged.connect(self.completeChanged)

        # Layout
        form = QFormLayout()
        form.addRow(QLabel("Visual Edges per mm:"), self.visu_edges_spin)
        form.addRow(QLabel("Collision Edges Tip (edges/mm):"), self.collis_tip_spin)
        form.addRow(QLabel("Collision Edges Straight (edges/mm):"), self.collis_str_spin)
        form.addRow(QLabel("Beams per mm Tip:"), self.beams_tip_spin)
        form.addRow(QLabel("Beams per mm Straight:"), self.beams_str_spin)

        # Velocity as horizontal group
        hvel = QHBoxLayout()
        hvel.addWidget(QLabel("Trans:") )
        hvel.addWidget(self.trans_speed_spin)
        hvel.addSpacing(20)
        hvel.addWidget(QLabel("Rot:"))
        hvel.addWidget(self.rot_speed_spin)
        form.addRow(QLabel("Velocity Limit:"), hvel)

        main_layout = QVBoxLayout()
        main_layout.addLayout(form)
        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def initializePage(self):
        # sensible defaults
        self.visu_edges_spin.setValue(0.5)
        self.collis_tip_spin.setValue(2.0)
        self.collis_str_spin.setValue(0.1)
        self.beams_tip_spin.setValue(1.4)
        self.beams_str_spin.setValue(0.09)
        self.trans_speed_spin.setValue(50.0)
        self.rot_speed_spin.setValue(3.14)

    def isComplete(self) -> bool:
        # all densities and speeds must be > 0
        return (
            self.visu_edges_spin.value() > 0 and
            self.collis_tip_spin.value() > 0 and
            self.collis_str_spin.value() > 0 and
            self.beams_tip_spin.value() > 0 and
            self.beams_str_spin.value() > 0 and
            self.trans_speed_spin.value() > 0 and
            self.rot_speed_spin.value() > 0
        )