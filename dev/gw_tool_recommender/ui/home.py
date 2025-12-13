import os, json, importlib.util, traceback
import pyvista as pv
from pyvistaqt import QtInteractor
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

from dev.gw_tool_recommender.ui.components.vessel_select_dialog import VesselSelectionDialog
from dev.gw_tool_recommender.storage import data_root, list_models, list_wires, wire_agents_dir


class HomeWidget(QWidget):
    """Home page: shows tools / agents and lets user pick a *.py* vessel‑generator."""

    # Folder that contains generated static VesselTree subfolders
    VESSEL_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'aorticarch_dir/vesseltrees')
    )

    def __init__(self):
        super().__init__()
        self.selected_agents = set()
        self.selected_vessel_mesh = None  # path to mesh produced by selected .py

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
        self.select_vessel_btn = QPushButton("Select Vesseltree")
        self.select_vessel_btn.clicked.connect(self._toggle_vessel_selection)
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
    def _toggle_vessel_selection(self):
        if self.selected_vessel_mesh:
            # Already showing: clear selection
            self._clear_vessel()
            return

        # Open dialog of cards
        dlg = VesselSelectionDialog(self, self.VESSEL_DIR)
        if dlg.exec_() != QDialog.Accepted or not dlg.selected_folder:
            return
        folder = dlg.selected_folder
        # Find first supported mesh file in mesh/ directory
        mesh_path = None
        mesh_dir = os.path.join(folder, 'mesh')
        if os.path.isdir(mesh_dir):
            for fname in os.listdir(mesh_dir):
                ext = Path(fname).suffix.lower()
                if ext in ('.obj', '.ply', '.vtp', '.vtk', '.stl'):
                    mesh_path = os.path.join(mesh_dir, fname)
                    break
        if not mesh_path or not os.path.isfile(mesh_path):
            self.detail.append(f'<p style="color:red;">No mesh file found in {mesh_dir}.</p>')
            return

        # Load mesh: OBJ (with faces) can be read directly, others fallback
        mesh = None
        ext = Path(mesh_path).suffix.lower()
        if ext == '.obj':
            try:
                mesh = pv.read(mesh_path)
            except Exception:
                mesh = None
        else:
            # For STL or others, sniff header for 'solid'
            try:
                with open(mesh_path, 'r', errors='ignore') as fh:
                    first = fh.readline().strip().lower()
            except Exception:
                first = ''
            if ext == '.stl' and first.startswith('solid'):
                try:
                    mesh = pv.read(mesh_path)
                except Exception:
                    mesh = None
        if mesh is None:
            # Fallback: use meshio to parse triangles
            try:
                import meshio
                m = meshio.read(mesh_path)
                pts = m.points
                tris = None
                if hasattr(m, 'cells_dict'):
                    tris = m.cells_dict.get('triangle')
                else:
                    for cell in m.cells:
                        if cell.type == 'triangle':
                            tris = cell.data
                            break
                if tris is None:
                    raise RuntimeError('No triangle data in mesh')
                mesh = pv.PolyData(pts, tris)
            except Exception as e:
                tb = traceback.format_exc(limit=3)
                self.detail.append(f'<p style="color:red;">Failed to load mesh:<pre>{tb}</pre></p>')
                return

        self.selected_vessel = folder
        self.selected_vessel_mesh = mesh_path
        self._show_mesh(mesh)
        self.select_vessel_btn.setText("Unselect Vesseltree")

    def _show_mesh(self, mesh):
        self.detail.setVisible(False)
        self.plotter.setVisible(True)
        self.plotter.clear()
        # Add mesh with smooth shading and edge contours for better visualization
        self.plotter.add_mesh(
            mesh,
            color='#8B0000',  # Blood red
            smooth_shading=True,
            show_edges=True,
            edge_color='black',
            lighting=True
        )
        # Enable shadows and enhanced lighting
        try:
            self.plotter.enable_shadows()
        except AttributeError:
            # Older PyVista versions may not support shadows
            pass
        # Optionally enable eye-dome lighting for depth enhancement
        try:
            self.plotter.enable_eye_dome_lighting()
        except AttributeError:
            pass
        self.plotter.reset_camera()

    def _clear_vessel(self):
        self.selected_vessel = None
        self.selected_vessel_mesh = None
        self.plotter.clear()
        self.plotter.setVisible(False)
        self.detail.setVisible(True)
        self.select_vessel_btn.setText("Select Vesseltree")


