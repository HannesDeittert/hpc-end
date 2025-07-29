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
        self.setSubTitle("Is this device procedural or non‑procedural?")

        # Instruction
        lbl = QLabel("Please choose the device type:")
        lbl.setAlignment(Qt.AlignCenter)

        # Two equally expanding buttons
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

    def initializePage(self):
        # Hide the default Next/Back buttons
        self.wizard().button(self.wizard().NextButton).hide()
        self.wizard().button(self.wizard().BackButton).hide()

    def _on_choose(self):
        sender = self.sender()
        is_proc = (sender is self.proc_btn)
        wiz = self.wizard()
        wiz.isProcedural = is_proc
        wiz.next()
