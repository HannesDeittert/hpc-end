# ui/home.py
from PyQt5.QtWidgets import (
    QWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QPushButton,
    QTextEdit,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QHeaderView,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt

class HomeWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.selected_agents = set()

        # ←—— Left pane: tools / agents tree
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Tool / Agent", "Action"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.setSelectionMode(QAbstractItemView.MultiSelection)
        self._populate_tree()
        self.tree.itemClicked.connect(self._on_item_clicked)

        # ←—— Right pane: detail + buttons
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setPlaceholderText("Select an agent to see details…")

        # the two buttons under the detail view
        self.select_vessel_btn = QPushButton("Select Vessel")
        self.compare_btn = QPushButton("Compare")

        # layout for those two buttons
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.select_vessel_btn)
        btn_layout.addWidget(self.compare_btn)
        btn_layout.addStretch(1)

        # stack detail + buttons vertically
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.detail, 1)
        right_layout.addLayout(btn_layout)

        # ─── combine left and right in one row
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.tree, 1)
        main_layout.addLayout(right_layout, 3)

        self.setLayout(main_layout)

    def _populate_tree(self):
        tools = {
            "Tool A": ["Agent A1", "Agent A2"],
            "Tool B": ["Agent B1", "Agent B2", "Agent B3"],
            "Tool C": [],
        }

        for tool_name, agents in tools.items():
            tool_item = QTreeWidgetItem([tool_name, ""])
            tool_item.setFlags(tool_item.flags() | Qt.ItemIsSelectable)
            self.tree.addTopLevelItem(tool_item)

            if agents:
                for agent_name in agents:
                    agent_item = QTreeWidgetItem([agent_name, ""])
                    agent_item.setFlags(agent_item.flags() | Qt.ItemIsSelectable)
                    tool_item.addChild(agent_item)

                    btn = QPushButton("Select")
                    btn.setCheckable(True)
                    btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                    btn.toggled.connect(
                        lambda checked, a=agent_name, it=agent_item:
                            self._toggle_agent(a, it, checked)
                    )
                    self.tree.setItemWidget(agent_item, 1, btn)
            else:
                none_item = QTreeWidgetItem(["(no agents)", ""])
                none_item.setDisabled(True)
                tool_item.addChild(none_item)

        self.tree.expandAll()

    def _toggle_agent(self, agent_name, item, selected):
        btn = self.tree.itemWidget(item, 1)
        font = item.font(0)

        if selected:
            self.selected_agents.add(agent_name)
            btn.setText("Unselect")
            font.setBold(True)
            item.setSelected(True)
        else:
            self.selected_agents.discard(agent_name)
            btn.setText("Select")
            font.setBold(False)
            item.setSelected(False)

        item.setFont(0, font)

    def _on_item_clicked(self, item, column):
        name = item.text(0)
        if name in self.selected_agents:
            self.detail.setHtml(f"<b>✅ {name}</b> is selected.<br><br>More info…")
        elif item.childCount() == 0 and "(no agents)" not in name:
            self.detail.setHtml(f"ℹ️ <b>{name}</b> (not selected).<br>Click “Select” to pick it.")
        else:
            self.detail.clear()



