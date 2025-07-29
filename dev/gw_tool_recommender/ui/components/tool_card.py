# in manage_tools_widget.py (or a separate widgets.py)

import os, json
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame, QLabel, QHBoxLayout, QVBoxLayout
)

class ToolCard(QFrame):
    clicked = pyqtSignal(str)   # will emit the tool’s directory name

    def __init__(self, tool_dir: str, definition: dict, agent_count: int):
        super().__init__()
        self.tool_dir = tool_dir

        # 1) give this frame a unique name
        self.setObjectName("ToolCardFrame")
        # 2) we’ll let CSS draw our border, so turn off the default panel
        self.setFrameShape(QFrame.NoFrame)

        # 3) style only our card
        self.setStyleSheet("""
        QFrame#ToolCardFrame {
          border: 1px solid #ccc;
          border-radius: 6px;
          padding: 8px;
          background-color: white;
        }
        QLabel#nameLabel {
          font-size: 14pt;
          font-weight: bold;
        }
        QLabel#descLabel {
          font-size: 9pt;
          color: #666;
        }
        QLabel#countLabel {
          font-size: 9pt;
          color: #333;
        }
        """)

        name = definition.get("name", tool_dir)
        desc = definition.get("description", "")

        # left side: name + optional description
        name_lbl = QLabel(name)
        name_lbl.setObjectName("nameLabel")

        desc_lbl = QLabel(desc)
        desc_lbl.setObjectName("descLabel")
        desc_lbl.setWordWrap(True)
        desc_lbl.setVisible(bool(desc.strip()))

        left_vbox = QVBoxLayout()
        left_vbox.addWidget(name_lbl)
        if desc.strip():
            left_vbox.addWidget(desc_lbl)
        left_vbox.addStretch()

        # right side: agent count
        count_lbl = QLabel(f"{agent_count} agent{'s' if agent_count != 1 else ''}")
        count_lbl.setObjectName("countLabel")
        count_lbl.setAlignment(Qt.AlignRight | Qt.AlignTop)

        # assemble
        h = QHBoxLayout(self)
        h.addLayout(left_vbox)
        h.addWidget(count_lbl)

    def mousePressEvent(self, event):
        # emit the tool directory name so you can look it up again later
        self.clicked.emit(self.tool_dir)
        super().mousePressEvent(event)

