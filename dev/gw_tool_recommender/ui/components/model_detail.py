from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)

from dev.gw_tool_recommender.storage import (
    list_wires,
    model_definition_path,
    wire_agents_dir,
    wire_definition_path,
)
from dev.gw_tool_recommender.ui.components.tool_detail import ToolDetailDialog
from dev.gw_tool_recommender.ui.wizard import ToolWizard


class ModelDetailDialog(QDialog):
    def __init__(self, model_name: str, parent=None):
        super().__init__(parent)
        self.model_name = model_name

        definition = {}
        def_path = model_definition_path(model_name)
        if def_path.exists():
            from dev.gw_tool_recommender.storage import read_json

            definition = read_json(def_path)

        name = definition.get("name", model_name)
        desc = definition.get("description", "")

        self.setWindowTitle(name)
        self.resize(560, 420)

        main = QVBoxLayout(self)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(8)

        title = QLabel(name)
        font = title.font()
        font.setPointSize(14)
        font.setBold(True)
        title.setFont(font)
        main.addWidget(title)

        if desc.strip():
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            main.addWidget(desc_lbl)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        main.addWidget(divider)

        header = QHBoxLayout()
        header.addWidget(QLabel("Wires"))
        header.addStretch()
        create_btn = QPushButton("Create Wire")
        create_btn.clicked.connect(self._create_wire)
        header.addWidget(create_btn)
        main.addLayout(header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.list_container = QFrame()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(6)
        self.scroll.setWidget(self.list_container)
        main.addWidget(self.scroll, 1)

        self._populate()

    def _populate(self) -> None:
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        wires = list_wires(self.model_name)
        if not wires:
            self.list_layout.addWidget(QLabel("No wires found."))
            self.list_layout.addStretch()
            return

        for wire in wires:
            row = QHBoxLayout()
            name = QLabel(wire)
            row.addWidget(name)
            row.addStretch()

            agents_path = wire_agents_dir(self.model_name, wire)
            agent_count = 0
            if agents_path.exists():
                agent_count = sum(1 for p in agents_path.iterdir() if p.is_dir())
            row.addWidget(QLabel(f"{agent_count} agent{'s' if agent_count != 1 else ''}"))

            open_btn = QPushButton("Open")
            open_btn.clicked.connect(lambda _, w=wire: self._open_wire(w))
            row.addWidget(open_btn)

            self.list_layout.addLayout(row)

        self.list_layout.addStretch()

    def _create_wire(self) -> None:
        wiz = ToolWizard(parent=self, model_name=self.model_name)
        wiz.setOption(wiz.HaveNextButtonOnLastPage, False)
        wiz.finished.connect(lambda _: self._populate())
        wiz.exec_()

    def _open_wire(self, wire_name: str) -> None:
        dlg = ToolDetailDialog(self.model_name, wire_name, parent=self)
        dlg.exec_()
        self._populate()

