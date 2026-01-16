import os, json, importlib.util, traceback
import pyvista as pv
from pyvistaqt import QtInteractor
import numpy as np
from PyQt5.QtWidgets import (
    QWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QPushButton,
    QTextEdit,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QHeaderView,
    QAbstractItemView, QDialog,
)
from PyQt5.QtCore import Qt

from steve_recommender.ui.components.anatomy_select_dialog import AnatomySelectionDialog
from steve_recommender.storage import data_root, list_models, list_wires, wire_agents_dir


class HomeWidget(QWidget):
    """Home page: shows tools / agents and lets user pick a *.py* vessel‑generator."""

    def __init__(self):
        super().__init__()
        self.selected_agents = set()
        self.selected_anatomy = None  # AorticArchRecord-like dict
        self.selected_anatomy_centerline = None  # path to centerline.npz

        # ─── Left pane: tree ─────────────────────────────────────────────
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Tool / Agent", "Action"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.setSelectionMode(QAbstractItemView.MultiSelection)
        self._populate_tree()
        self.tree.itemClicked.connect(self._on_item_clicked)

        # ─── Right pane: detail / 3‑D viewer ────────────────────────────
        self.detail = QTextEdit(readOnly=True)
        self.detail.setPlaceholderText("Select an agent to see details…")

        self.plotter = QtInteractor(self)
        self.plotter.setVisible(False)

        # Buttons
        self.select_vessel_btn = QPushButton("Select Anatomy")
        self.select_vessel_btn.clicked.connect(self._toggle_anatomy_selection)
        self.compare_btn = QPushButton("Compare")

        # Layouts
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.select_vessel_btn)
        btn_layout.addWidget(self.compare_btn)
        btn_layout.addStretch()

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.detail, 1)
        right_layout.addWidget(self.plotter, 1)
        right_layout.addLayout(btn_layout)

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.tree, 1)
        main_layout.addLayout(right_layout, 3)

    # ===================================================================
    # Populate tree from <project>/data
    # ===================================================================
    def _populate_tree(self):
        base_dir = str(data_root())
        self.tree.clear()
        if not os.path.isdir(base_dir):
            return

        for model in list_models():
            model_item = QTreeWidgetItem([model, ""])
            model_item.setFlags(model_item.flags() | Qt.ItemIsSelectable)
            self.tree.addTopLevelItem(model_item)

            wires = list_wires(model)
            if not wires:
                none_item = QTreeWidgetItem(["(no wires)", ""])
                none_item.setDisabled(True)
                model_item.addChild(none_item)
                continue

            for wire in wires:
                wire_item = QTreeWidgetItem([wire, ""])
                wire_item.setFlags(wire_item.flags() | Qt.ItemIsSelectable)
                model_item.addChild(wire_item)

                agents_path = wire_agents_dir(model, wire)
                if agents_path.exists():
                    for agent_dir in sorted(p.name for p in agents_path.iterdir() if p.is_dir()):
                        agent_id = f"{model}/{wire}:{agent_dir}"
                        agent_item = QTreeWidgetItem([agent_dir, ""])
                        agent_item.setData(0, Qt.UserRole, agent_id)
                        agent_item.setFlags(agent_item.flags() | Qt.ItemIsSelectable)
                        wire_item.addChild(agent_item)

                        btn = QPushButton("Select")
                        btn.setCheckable(True)
                        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                        btn.setFixedSize(60, 22)
                        font = btn.font()
                        font.setPointSize(8)
                        btn.setFont(font)
                        btn.toggled.connect(
                            lambda checked, a=agent_id, it=agent_item: self._toggle_agent(a, it, checked)
                        )

                        self.tree.setItemWidget(agent_item, 1, btn)
                else:
                    none_item = QTreeWidgetItem(["(no agents)", ""])
                    none_item.setDisabled(True)
                    wire_item.addChild(none_item)

        self.tree.expandAll()

    # -------------------------------------------------------------------
    # Agent selection helpers
    # -------------------------------------------------------------------
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
        agent_id = item.data(0, Qt.UserRole)
        if agent_id and agent_id in self.selected_agents:
            self.detail.setHtml(f"<b>{name}</b> is selected.<br><br>More info…")
        elif item.childCount() == 0 and "(no agents)" not in name:
            self.detail.setHtml(f"ℹ️ <b>{name}</b> (not selected).<br>Click 'Select' to pick it.")
        else:
            self.detail.clear()

    # ===================================================================
    # Vessel‑folder selection and mesh rendering
    # ===================================================================
    def _toggle_anatomy_selection(self):
        if self.selected_anatomy_centerline:
            self._clear_anatomy()
            return

        dlg = AnatomySelectionDialog(self)
        if dlg.exec_() != QDialog.Accepted or dlg.selected_record is None:
            return

        self.selected_anatomy = dlg.selected_record

        centerline_path = dlg.resolve_centerline_path(dlg.selected_record)
        if centerline_path is None:
            self.detail.append("<p style='color:red;'>Selected anatomy has no centerline data.</p>")
            return

        self.selected_anatomy_centerline = str(centerline_path)

        self._show_centerlines(self.selected_anatomy_centerline)
        self.select_vessel_btn.setText("Unselect Anatomy")

    def _show_centerlines(self, npz_path: str):
        self.detail.setVisible(False)
        self.plotter.setVisible(True)
        self.plotter.clear()
        data = np.load(npz_path, allow_pickle=True)
        branch_names = data.get("branch_names", [])
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        for i, name in enumerate(branch_names.tolist() if hasattr(branch_names, "tolist") else []):
            coords_key = f"branch_{name}_coords"
            if coords_key not in data:
                continue
            pts = np.asarray(data[coords_key], dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            lines = np.hstack([[pts.shape[0]], np.arange(pts.shape[0])]).astype(np.int64)
            poly = pv.PolyData(pts, lines)
            self.plotter.add_mesh(poly, color=colors[i % len(colors)], line_width=4)
        self.plotter.reset_camera()

    def _clear_anatomy(self):
        self.selected_anatomy = None
        self.selected_anatomy_centerline = None
        self.plotter.clear()
        self.plotter.setVisible(False)
        self.detail.setVisible(True)
        self.select_vessel_btn.setText("Select Anatomy")
