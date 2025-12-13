import os
import json
import shutil
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QMessageBox,
    QInputDialog,
    QSizePolicy
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / ".git").exists():
            return parent
    return here.parents[-1]


class ToolDetailDialog(QDialog):
    def __init__(self, tool_dir: str, parent=None):
        super().__init__(parent)
        self.tool_dir = tool_dir

        # Determine base data path and tool-specific paths
        self.data_root = str(_repo_root() / "data")
        tool_path = os.path.join(self.data_root, tool_dir)
        def_path = os.path.join(tool_path, "tool_definition.json")

        # Load tool definition
        with open(def_path, 'r') as f:
            definition = json.load(f)
        name = definition.get('name', tool_dir)
        desc = definition.get('description', '')

        # Dialog settings
        self.setWindowTitle(name)
        self.resize(500, 400)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(6)
        main_layout.setAlignment(Qt.AlignTop)

        # --- Header: Name + Info Button ---
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        title_lbl = QLabel(name)
        title_font = title_lbl.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_lbl.setFont(title_font)
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        info_btn = QPushButton('i')
        info_btn.setFixedSize(24, 24)
        info_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        info_btn.clicked.connect(self._show_tool_info)
        header_layout.addWidget(info_btn)
        main_layout.addLayout(header_layout)

        # --- Description ---
        if desc.strip():
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_font = desc_lbl.font()
            desc_font.setPointSize(9)
            desc_lbl.setFont(desc_font)
            main_layout.addWidget(desc_lbl)

        # --- Divider ---
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(divider)

        # --- Agents List ---
        agents_path = os.path.join(tool_path, 'agents')
        if os.path.isdir(agents_path) and os.listdir(agents_path):
            for agent in sorted(os.listdir(agents_path)):
                agent_dir = os.path.join(agents_path, agent)
                if not os.path.isdir(agent_dir):
                    continue
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(4)
                agent_lbl = QLabel(agent)
                row.addWidget(agent_lbl)
                ainfo_btn = QPushButton('i')
                ainfo_btn.setFixedSize(20, 20)
                ainfo_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                ainfo_btn.clicked.connect(lambda _, a=agent: self._show_agent_info(a))
                row.addWidget(ainfo_btn)
                row.addStretch()
                delete_btn = QPushButton('Delete')
                delete_btn.clicked.connect(lambda _, a=agent: self._confirm_delete(a))
                row.addWidget(delete_btn)
                main_layout.addLayout(row)
        else:
            none_lbl = QLabel("No agents found.")
            main_layout.addWidget(none_lbl)

        # Push everything up by consuming leftover space
        main_layout.addStretch()

        # --- Footer Buttons: Train & Optimize ---
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(8)
        footer_layout.addStretch()

        self.train_btn = QPushButton("Train")
        self.train_btn.setFixedHeight(28)
        self.train_btn.clicked.connect(self._open_train_wizard)
        footer_layout.addWidget(self.train_btn)

        self.optimize_btn = QPushButton("Optimize")
        self.optimize_btn.setFixedHeight(28)
        self.optimize_btn.clicked.connect(self._open_optimize_wizard)
        footer_layout.addWidget(self.optimize_btn)

        main_layout.addLayout(footer_layout)

    def _show_tool_info(self):
        QMessageBox.information(
            self,
            "Tool Information",
            "Detailed tool information will appear here."
        )

    def _show_agent_info(self, agent_name: str):
        QMessageBox.information(
            self,
            f"Agent: {agent_name}",
            f"Information about agent '{agent_name}' goes here."
        )

    def _confirm_delete(self, agent_name: str):
        text, ok = QInputDialog.getText(
            self,
            "Confirm Deletion",
            f"Type 'delete agent' to confirm deletion of agent '{agent_name}':"
        )
        if ok and text == 'delete agent':
            agent_path = os.path.join(
                self.data_root,
                self.tool_dir,
                'agents',
                agent_name
            )
            try:
                shutil.rmtree(agent_path)
                QMessageBox.information(
                    self,
                    "Deleted",
                    f"Agent '{agent_name}' has been deleted."
                )
                self.accept()
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to delete agent: {e}"
                )
        elif ok:
            QMessageBox.warning(
                self,
                "Aborted",
                "Deletion aborted: incorrect confirmation text."
            )

    def _open_train_wizard(self):
        # TODO: import and launch your Train wizard here
        print("Opening train wizard...")
        #from .wizard import TrainWizard
       # wiz = TrainWizard(self.tool_dir, parent=self)
        #wiz.exec_()

    def _open_optimize_wizard(self):
        # TODO: import and launch your Optimize wizard here
        print("Opening opti wizard")
        #from .wizard import OptimizeWizard
        #wiz = OptimizeWizard(self.tool_dir, parent=self)
        #wiz.exec_()
