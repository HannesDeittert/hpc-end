# ui/wizard_pages/non_procedural/general_params.py

from typing import Tuple
from PyQt5.QtWidgets import (
    QWizardPage, QLabel, QLineEdit, QPushButton,
    QColorDialog, QFormLayout, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QColor

class GeneralParamsPage(QWizardPage):
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setTitle("General Parameters")
        self.setSubTitle("Enter basic information for your non‑procedural device.")

        # Internal storage for color as 0–255 ints
        self.color: Tuple[int, int, int] = (0, 0, 0)

        # Widgets
        self.name_edit = QLineEdit()
        self.registerField("nonName*", self.name_edit)

        self.color_button = QPushButton("Choose Color…")
        self.color_display = QLabel("(0, 0, 0)")
        self.color_button.clicked.connect(self.choose_color)

        self.desc_edit = QLineEdit()
        self.registerField("nonDesc", self.desc_edit)

        # Layout
        form = QFormLayout()
        form.addRow("Name:", self.name_edit)
        form.addRow("Color:", self.color_button)
        form.addRow("Current RGB:", self.color_display)
        form.addRow("Description:", self.desc_edit)

        main_layout = QVBoxLayout()
        main_layout.addLayout(form)
        self.setLayout(main_layout)

    def choose_color(self):
        # Show dialog initialized to current color
        start = QColor(*self.color)
        qcolor: QColor = QColorDialog.getColor(start, parent=self)
        if qcolor.isValid():
            # Store raw 0–255 tuple
            r, g, b, _ = qcolor.getRgb()
            self.color = (r, g, b)
            # Update display
            self.color_display.setText(f"({r}, {g}, {b})")
