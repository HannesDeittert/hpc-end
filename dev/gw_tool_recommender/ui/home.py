import os, json
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
        # Determine the data directory (one level up from ui folder)
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data')
        )
        self.tree.clear()

        if not os.path.isdir(base_dir):
            return

        for tool_dir in sorted(os.listdir(base_dir)):
            tool_path = os.path.join(base_dir, tool_dir)
            if not os.path.isdir(tool_path):
                continue

            # Read tool_definition.json to get display name
            def_file = os.path.join(tool_path, 'tool_definition.json')
            try:
                with open(def_file, 'r') as f:
                    definition = json.load(f)
                tool_name = definition.get('name', tool_dir)
            except Exception:
                tool_name = tool_dir

            # Create tree item for tool
            tool_item = QTreeWidgetItem([tool_name, ''])
            tool_item.setFlags(tool_item.flags() | Qt.ItemIsSelectable)
            self.tree.addTopLevelItem(tool_item)

            # Populate agents under this tool
            agents_path = os.path.join(tool_path, 'agents')
            if os.path.isdir(agents_path) and os.listdir(agents_path):
                for agent_dir in sorted(os.listdir(agents_path)):
                    full_agent_path = os.path.join(agents_path, agent_dir)
                    if not os.path.isdir(full_agent_path):
                        continue

                    agent_item = QTreeWidgetItem([agent_dir, ''])
                    agent_item.setFlags(agent_item.flags() | Qt.ItemIsSelectable)
                    tool_item.addChild(agent_item)

                    btn = QPushButton("Select")
                    btn.setCheckable(True)
                    btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                    btn.toggled.connect(
                        lambda checked, a=agent_dir, it=agent_item:
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




