import os, json

from PyQt5.QtCore import QFileSystemWatcher
from PyQt5.QtWidgets import QWidget, QListWidget, QPushButton, QVBoxLayout, QWizard, QListWidgetItem, QHBoxLayout, \
    QFrame

from .components.tool_card import ToolCard
from .components.tool_detail import ToolDetailDialog
from .wizard import ToolWizard

class ManageToolsWidget(QWidget):
    def __init__(self):
        super().__init__()

        # where our on‑disk tools live
        self.data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data")
        )

        # —— List Widget ——
        self.list = QListWidget()
        self.list.setFrameShape(QFrame.NoFrame)
        self.list.setSpacing(8)

        # —— Buttons ——
        self.btn_create = QPushButton("Create New Tool")
        self.btn_create.clicked.connect(self._open_wizard)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._load_tools)

        # —— Layout ——
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_create)
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

    def _open_wizard(self):
        wiz = ToolWizard(self)
        wiz.setOption(wiz.HaveNextButtonOnLastPage, False)
        # when the wizard closes (regardless of success), reload
        wiz.finished.connect(lambda _: self._load_tools())
        wiz.exec_()

    def _load_tools(self):
        """ Clears and repopulates the list from disk. """
        self.list.clear()
        if not os.path.isdir(self.data_dir):
            self.list.addItem("No tools found")
            return

        found = False
        for tool_dir in sorted(os.listdir(self.data_dir)):
            tool_path = os.path.join(self.data_dir, tool_dir)
            if not os.path.isdir(tool_path):
                continue

            # load definition
            def_file = os.path.join(tool_path, "tool_definition.json")
            try:
                with open(def_file, "r") as f:
                    definition = json.load(f)
            except Exception:
                continue

            # count agents
            agents_path = os.path.join(tool_path, "agents")
            count = (
                sum(
                    1 for d in os.listdir(agents_path)
                    if os.path.isdir(os.path.join(agents_path, d))
                )
                if os.path.isdir(agents_path) else 0
            )

            # create card + list‑item
            card = ToolCard(tool_dir, definition, count)
            card.clicked.connect(self._on_card_clicked)

            item = QListWidgetItem()
            item.setSizeHint(card.sizeHint())

            self.list.addItem(item)
            self.list.setItemWidget(item, card)
            found = True

        if not found:
            self.list.addItem("No tools found")

    def _on_card_clicked(self, tool_dir):
        dlg = ToolDetailDialog(tool_dir, parent=self)
        dlg.exec_()
