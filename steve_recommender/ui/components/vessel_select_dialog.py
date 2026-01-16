import os
import json
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QDialog,
    QScrollArea,
    QFrame,
    QLabel,
)
from PyQt5.QtCore import Qt, QSize
import importlib.util

class VesselSelectionDialog(QDialog):
    """Popup dialog that displays available VesselTrees as selectable cards."""
    def __init__(self, parent, vessel_dir):
        super().__init__(parent)
        self.setWindowTitle("Select VesselTree")
        self.setMinimumSize(QSize(400, 300))
        self.selected_folder = None

        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QFrame()
        container_layout = QVBoxLayout(container)

        # For each subfolder, create a card
        for name in sorted(os.listdir(vessel_dir)):
            folder = os.path.join(vessel_dir, name)
            if not os.path.isdir(folder):
                continue
            # Load metadata
            desc_path = os.path.join(folder, 'description.json')
            meta = {}
            if os.path.isfile(desc_path):
                try:
                    with open(desc_path, 'r') as f:
                        meta = json.load(f)
                except:
                    pass
            # Card frame
            card = QFrame()
            card.setFrameShape(QFrame.StyledPanel)
            card.setLineWidth(1)
            clayout = QVBoxLayout(card)
            title = QLabel(f"<b>{name}</b>")
            clayout.addWidget(title)
            # show key metadata fields
            for key in ('arch_type','seed'):
                if key in meta:
                    clayout.addWidget(QLabel(f"{key}: {meta[key]}"))
            # clickable
            card.mouseReleaseEvent = lambda ev, d=folder: self._on_card_clicked(d)
            container_layout.addWidget(card)

        scroll.setWidget(container)
        layout.addWidget(scroll)

    def _on_card_clicked(self, folder):
        self.selected_folder = folder
        self.accept()