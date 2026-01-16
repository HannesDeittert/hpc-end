import os
import json
from pathlib import Path

from PyQt5.QtWidgets import QInputDialog

from PyQt5.QtCore import QFileSystemWatcher
from PyQt5.QtWidgets import QWidget, QListWidget, QPushButton, QVBoxLayout, QWizard, QListWidgetItem, QHBoxLayout, \
    QFrame

from .components.tool_card import ToolCard
from .components.model_detail import ModelDetailDialog

from steve_recommender.storage import data_root, ensure_model, list_models, read_json, list_wires


class ManageToolsWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.data_dir = str(data_root())

        # —— List Widget ——
        self.list = QListWidget()
        self.list.setFrameShape(QFrame.NoFrame)
        self.list.setSpacing(8)

        # —— Buttons ——
        self.btn_create_model = QPushButton("Create New Model")
        self.btn_create_model.clicked.connect(self._create_model)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._load_tools)

        # —— Layout ——
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_create_model)
        btn_row.addWidget(self.btn_refresh)
        btn_row.addStretch()

        main = QVBoxLayout(self)
        main.addWidget(self.list)
        main.addLayout(btn_row)
        self.setLayout(main)

        # —— Auto‑reload on FS changes ——
        if os.path.isdir(self.data_dir):
            self.watcher = QFileSystemWatcher([self.data_dir])
            self.watcher.directoryChanged.connect(self._load_tools)

        # initial load
        self._load_tools()

    def _create_model(self):
        name, ok = QInputDialog.getText(self, "Create Model", "Model name:")
        if not ok:
            return
        model_name = (name or "").strip()
        if not model_name:
            return

        desc, ok = QInputDialog.getText(self, "Create Model", "Model description (optional):")
        if not ok:
            return

        ensure_model(model_name, description=desc or "")
        self._load_tools()

    def _load_tools(self):
        """Clears and repopulates the model list from disk."""
        self.list.clear()
        if not os.path.isdir(self.data_dir):
            self.list.addItem("No models found")
            return

        found = False
        for model_name in list_models():
            model_path = os.path.join(self.data_dir, model_name)
            def_file = os.path.join(model_path, "model_definition.json")
            try:
                definition = read_json(Path(def_file))
            except Exception:
                definition = {"name": model_name, "description": ""}

            wire_count = len(list_wires(model_name))

            card = ToolCard(model_name, definition, wire_count, count_label="wire")
            card.clicked.connect(self._on_card_clicked)

            item = QListWidgetItem()
            item.setSizeHint(card.sizeHint())

            self.list.addItem(item)
            self.list.setItemWidget(item, card)
            found = True

        if not found:
            self.list.addItem("No models found")

    def _on_card_clicked(self, model_name):
        dlg = ModelDetailDialog(model_name, parent=self)
        dlg.exec_()
