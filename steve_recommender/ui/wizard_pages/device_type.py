# ui/wizard_pages/device_type.py

from PyQt5.QtWidgets import (
    QWizardPage, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QSizePolicy
)
from PyQt5.QtCore import Qt

class DeviceTypePage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Device Type")
        self.setSubTitle("Choose the device type.")

        # Instruction
        lbl = QLabel("Please choose the device type:")
        lbl.setAlignment(Qt.AlignCenter)

        # Three equally expanding buttons
        self.proc_btn = QPushButton("Procedural")
        self.nonproc_btn = QPushButton("Non‑Procedural")
        for btn in (self.proc_btn, self.nonproc_btn):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            btn.clicked.connect(self._on_choose)

        # Layout them side by side
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        btn_layout.addWidget(self.proc_btn)
        btn_layout.addWidget(self.nonproc_btn)

        # Main layout
        layout = QVBoxLayout()
        layout.setSpacing(30)
        layout.addWidget(lbl)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        for btn in (self.proc_btn, self.nonproc_btn):
            btn.setAutoDefault(False)
            btn.setDefault(False)

    def initializePage(self):
        # Hide the default Next/Back buttons
        self.wizard().button(self.wizard().NextButton).hide()
        self.wizard().button(self.wizard().BackButton).hide()

    def _on_choose(self):
        sender = self.sender()
        wiz = self.wizard()
        if sender is self.proc_btn:
            wiz.device_type = "procedural"
            wiz.isProcedural = True
        elif sender is self.nonproc_btn:
            wiz.device_type = "non_procedural"
            wiz.isProcedural = False
        wiz.next()
