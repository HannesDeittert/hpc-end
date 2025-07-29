import os
from PyQt5.QtWidgets import QWidget, QListWidget, QPushButton, QVBoxLayout
from ui.wizard import ToolWizard

class ManageToolsWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.list = QListWidget()
        self._load_tools()

        self.btn_create = QPushButton("Create New Tool")
        self.btn_create.clicked.connect(self._open_wizard)

        layout = QVBoxLayout()
        layout.addWidget(self.list)
        layout.addWidget(self.btn_create)
        self.setLayout(layout)

    def _load_tools(self):
        base = "data_base"
        if os.path.isdir(base):
            tools = os.listdir(base)
            if tools:
                self.list.addItems(tools)
                return
        self.list.addItem("No tools found")

    def _open_wizard(self):
        wiz = ToolWizard(self)
        wiz.exec_()
