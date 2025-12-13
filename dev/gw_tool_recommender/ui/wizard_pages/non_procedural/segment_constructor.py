import os
from typing import List, Dict, Any, Any as AnyType
from PyQt5.QtWidgets import (
    QWizardPage, QWidget, QPushButton, QListWidget, QListWidgetItem,
    QStackedWidget, QSplitter, QVBoxLayout, QHBoxLayout, QFormLayout,
    QDoubleSpinBox, QLabel, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
import sys  # for float range limits

PartType = Dict[str, Any]

class SegmentConstructorPage(QWizardPage):
    """
    Wizard page that allows adding, reordering, editing, and deleting Straight or Arc elements,
    and shows a live mesh preview on the right when all elements are valid.
    Elements are stored internally as dicts for UI binding, but converted to real
    Arc/StraightPart objects when building the mesh.
    """
    completeChanged = pyqtSignal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setTitle("Segment Constructor")
        self.setSubTitle("Add, reorder, delete, and configure parts to build your segment.")

        # Internal list of element dicts
        self.elements: List[PartType] = []

        # Add & remove buttons
        self.btn_add_straight = QPushButton("Add Straight")
        self.btn_add_arc = QPushButton("Add Arc")
        self.btn_remove = QPushButton("Remove Selected")
        for btn in (self.btn_add_straight, self.btn_add_arc):
            btn.clicked.connect(self._on_add)
        self.btn_remove.clicked.connect(self._on_remove)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_add_straight)
        btn_layout.addWidget(self.btn_add_arc)
        btn_layout.addWidget(self.btn_remove)

        # Parts list (drag & drop)
        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QListWidget.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.MoveAction)
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)
        self.list_widget.model().rowsMoved.connect(self._on_reorder)

        # Editor stack
        self.editor_stack = QStackedWidget()
        self._build_editors()

        # Preview stack: placeholder text and 3D view
        self.preview_stack = QStackedWidget()
        self.placeholder = QLabel("Mesh preview will appear here.")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.preview_stack.addWidget(self.placeholder)
        from pyvistaqt import QtInteractor  # lazy import
        self.plotter = QtInteractor(self)
        self.preview_stack.addWidget(self.plotter)

        # Layout assembly
        left_layout = QVBoxLayout()
        left_layout.addLayout(btn_layout)
        left_layout.addWidget(self.list_widget)
        left_layout.addWidget(self.editor_stack)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self.preview_stack)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _build_editors(self):
        # Straight editor
        straight_widget = QWidget()
        f1 = QFormLayout()
        self.len_spin = QDoubleSpinBox(); self.len_spin.setSuffix(" mm"); self.len_spin.setDecimals(2)
        self.collis_spin = QDoubleSpinBox(); self.collis_spin.setDecimals(2)
        self.beams_spin = QDoubleSpinBox(); self.beams_spin.setDecimals(2)
        for spin in (self.len_spin, self.collis_spin, self.beams_spin):
            spin.valueChanged.connect(self._on_property_changed)
        f1.addRow("Length (mm):", self.len_spin)
        f1.addRow("Collis edges/mm:", self.collis_spin)
        f1.addRow("Beams/mm:", self.beams_spin)
        straight_widget.setLayout(f1)

        # Arc editor
        arc_widget = QWidget()
        f2 = QFormLayout()
        self.radius_spin = QDoubleSpinBox(); self.radius_spin.setSuffix(" mm"); self.radius_spin.setDecimals(2)
        self.angle_spin      = QDoubleSpinBox(); self.angle_spin.setSuffix("°"); self.angle_spin.setDecimals(2); self.angle_spin.setRange(-sys.float_info.max, sys.float_info.max)
        self.outspin         = QDoubleSpinBox(); self.outspin.setSuffix("°"); self.outspin.setDecimals(2); self.outspin.setRange(-sys.float_info.max, sys.float_info.max)
        self.arc_collis_spin = QDoubleSpinBox(); self.arc_collis_spin.setDecimals(2)
        self.arc_beams_spin = QDoubleSpinBox(); self.arc_beams_spin.setDecimals(2)
        for spin in (self.radius_spin, self.angle_spin, self.outspin,
                     self.arc_collis_spin, self.arc_beams_spin):
            spin.valueChanged.connect(self._on_property_changed)
        f2.addRow("Radius (mm):", self.radius_spin)
        f2.addRow("Angle in-plane (°):", self.angle_spin)
        f2.addRow("Angle out-of-plane (°):", self.outspin)
        f2.addRow("Collis edges/mm:", self.arc_collis_spin)
        f2.addRow("Beams/mm:", self.arc_beams_spin)
        arc_widget.setLayout(f2)

        self.editor_stack.addWidget(straight_widget)
        self.editor_stack.addWidget(arc_widget)


    def _on_add(self):
        """Called by Add‑Straight and Add‑Arc buttons."""
        sender = self.sender()
        if sender is self.btn_add_straight:
            part = {
                'type': 'straight',
                'length': 1.0,
                'collis_edges_per_mm': 1.0,
                'beams_per_mm': 1.0
            }
            label = f"Straight: L={part['length']:.2f}"
        else:
            part = {
                'type': 'arc',
                'radius': 1.0,
                'angle_in_plane_deg': 90.0,
                'angle_out_of_plane_deg': 0.0,
                'collis_edges_per_mm': 1.0,
                'beams_per_mm': 1.0
            }
            label = f"Arc: R={part['radius']:.2f}"

        # 1) Keep in our model list
        self.elements.append(part)

        # 2) Create a QListWidgetItem, attach the dict, and show it
        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, part)
        self.list_widget.addItem(item)
        self.list_widget.setCurrentItem(item)

        # 3) Refresh preview & wizard‐completeness
        self._update_preview()
        self.completeChanged.emit()

    def _on_remove(self):
        idx = self.list_widget.currentRow()
        if idx < 0:
            return
        self.elements.pop(idx)
        self.list_widget.takeItem(idx)
        self._update_preview()
        self.completeChanged.emit()

    def _on_selection_changed(self, index: int):
        if index < 0 or index >= len(self.elements):
            return
        e = self.elements[index]
        if e['type'] == 'straight':
            self.editor_stack.setCurrentIndex(0)
            self.len_spin.setValue(e['length'])
            self.collis_spin.setValue(e['collis_edges_per_mm'])
            self.beams_spin.setValue(e['beams_per_mm'])
        else:
            self.editor_stack.setCurrentIndex(1)
            self.radius_spin.setValue(e['radius'])
            self.angle_spin.setValue(e['angle_in_plane_deg'])
            self.outspin.setValue(e['angle_out_of_plane_deg'])
            self.arc_collis_spin.setValue(e['collis_edges_per_mm'])
            self.arc_beams_spin.setValue(e['beams_per_mm'])

    def _on_reorder(self, *args):
        """Called whenever the user drag‑&‑drops items in the QListWidget."""
        # 1) Rebuild self.elements in the visual order
        new_list: List[PartType] = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            part = item.data(Qt.UserRole)
            new_list.append(part)
        self.elements = new_list

        # 2) Update preview & completeness
        self._update_preview()
        self.completeChanged.emit()

        # 3) Refresh property‐editor for the current selection
        self._on_selection_changed(self.list_widget.currentRow())

    def _on_property_changed(self):
        idx = self.list_widget.currentRow()
        if idx < 0:
            return
        e = self.elements[idx]
        if e['type'] == 'straight':
            e['length'] = self.len_spin.value()
            e['collis_edges_per_mm'] = self.collis_spin.value()
            e['beams_per_mm'] = self.beams_spin.value()
            self.list_widget.currentItem().setText(f"Straight: L={e['length']:.2f}")
        else:
            e['radius'] = self.radius_spin.value()
            e['angle_in_plane_deg'] = self.angle_spin.value()
            e['angle_out_of_plane_deg'] = self.outspin.value()
            e['collis_edges_per_mm'] = self.arc_collis_spin.value()
            e['beams_per_mm'] = self.arc_beams_spin.value()
            self.list_widget.currentItem().setText(f"Arc: R={e['radius']:.2f}")
        self._update_preview()
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._all_valid()

    def _all_valid(self) -> bool:
        if not self.elements:
            return False
        for e in self.elements:
            if e['type'] == 'straight':
                if e['length'] <= 0 \
                   or e['collis_edges_per_mm'] <= 0 or e['beams_per_mm'] <= 0:
                    return False
            else:
                if e['radius'] <= 0 \
                   or (e['angle_in_plane_deg'] == 0 and e['angle_out_of_plane_deg'] == 0) \
                   or e['collis_edges_per_mm'] <= 0 \
                   or e['beams_per_mm'] <= 0:
                    return False
        return True

    def _update_preview(self):
        if not self._all_valid():
            self.preview_stack.setCurrentIndex(0)
            self.placeholder.setText("Please fill in all fields to see a live preview.")
            return
        self.build_mesh()
        self.preview_stack.setCurrentIndex(1)

    def build_mesh(self):
        from eve.intervention.device.device import Arc, StraightPart, MeshDevice
        import pyvista as pv

        wiz = self.wizard()
        outer = wiz.page_nonproc_diameter.outer_spin.value()
        inner = wiz.page_nonproc_diameter.inner_spin.value()
        mat_pg = wiz.page_nonproc_material
        gen_pg = wiz.page_nonproc_start
        sim_pg = wiz.page_nonproc_simulation

        poisson = mat_pg.poisson_spin.value()
        young = mat_pg.young_spin.value()
        density = mat_pg.density_spin.value()
        color = gen_pg.color
        visu_edges_per_mm = sim_pg.edges_spin.value()
        velocity_limit = [sim_pg.trans_spin.value(), sim_pg.rot_spin.value()]

        model_elems: List[AnyType] = []
        for e in self.elements:
            if e['type'] == 'straight':
                model_elems.append(
                    StraightPart(
                        length=e['length'],
                        visu_edges_per_mm=visu_edges_per_mm,
                        collis_edges_per_mm=e['collis_edges_per_mm'],
                        beams_per_mm=e['beams_per_mm'],
                    )
                )
            else:
                model_elems.append(
                    Arc(
                        radius=e['radius'],
                        angle_in_plane_deg=e['angle_in_plane_deg'],
                        angle_out_of_plane_deg=e['angle_out_of_plane_deg'],
                        visu_edges_per_mm=visu_edges_per_mm,
                        collis_edges_per_mm=e['collis_edges_per_mm'],
                        beams_per_mm=e['beams_per_mm'],
                    )
                )

        self.model_elements = model_elems

        mesh_dev = MeshDevice(
            elements=model_elems,
            outer_diameter=outer,
            inner_diameter=inner,
            poisson_ratio=poisson,
            young_modulus=young,
            mass_density=density,
            color=color,
        )

        self.plotter.clear()
        mesh = pv.read(mesh_dev.sofa_device.mesh_path)
        self.plotter.add_mesh(mesh, show_edges=True)
        self.plotter.reset_camera()

    def initializePage(self):
        # Called when this page is shown: swap Finish text to Save and toggle enabled
        wiz = self.wizard()
        wiz.setButtonText(wiz.FinishButton, "Save")
        finish_btn = wiz.button(wiz.FinishButton)
        finish_btn.setEnabled(self._all_valid())
        # re-enable/disable Save as completeness changes
        self.completeChanged.connect(lambda: finish_btn.setEnabled(self._all_valid()))

    def validatePage(self) -> bool:
        # Invoked when the Save button is pressed
        if not self._all_valid():
            return False
        self.save_tool()
        return True

    def save_tool(self):
        import os, json
        from pathlib import Path
        from eve.intervention.device.device import Arc, StraightPart, MeshDevice

        wiz = self.wizard()
        tool_name = wiz.field('nonName') or "MyTool"

        # 1) Create project folder
        here = Path(__file__).resolve()
        repo_root = next((p for p in (here.parent, *here.parents) if (p / ".git").exists()), here.parents[-1])
        base_dir = str(repo_root / "data" / tool_name)
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'agents'), exist_ok=True)

        # 2) Write tool_definition.json
        with open(os.path.join(base_dir, 'tool_definition.json'), 'w') as f:
            json.dump(
                {'name': tool_name, 'description': wiz.field('nonDesc') or "", 'type': "non procedural"},
                f, indent=2
            )

        # 3) Generate Python source for the elements list
        elem_lines = []
        for part in self.model_elements:
            if isinstance(part, StraightPart):
                params = {
                    'length': part.length,
                    'visu_edges_per_mm': part.visu_edges_per_mm,
                    'collis_edges_per_mm': part.collis_edges_per_mm,
                    'beams_per_mm': part.beams_per_mm,
                }
                args = ", ".join(f"{k}={v!r}" for k, v in params.items())
                elem_lines.append(f"        StraightPart({args})")
            else:  # Arc
                params = {
                    'radius': part.radius,
                    'angle_in_plane_deg': part.angle_in_plane_deg,
                    'angle_out_of_plane_deg': part.angle_out_of_plane_deg,
                    'visu_edges_per_mm': part.visu_edges_per_mm,
                    'collis_edges_per_mm': part.collis_edges_per_mm,
                    'beams_per_mm': part.beams_per_mm,
                }
                args = ", ".join(f"{k}={v!r}" for k, v in params.items())
                elem_lines.append(f"        Arc({args})")

        elements_block = "[\n" + ",\n".join(elem_lines) + "\n    ]"

        # 4) Write tool.py
        tool_py = os.path.join(base_dir, 'tool.py')
        with open(tool_py, 'w') as f:
            f.write(f"""
from dataclasses import dataclass
from typing import Tuple
from eve.intervention.device.device import MeshDevice, StraightPart, Arc

@dataclass
class {tool_name}(MeshDevice):
    name: str = {tool_name!r}
    velocity_limit: Tuple[float, float] = ({wiz.page_nonproc_simulation.trans_spin.value()}, {wiz.page_nonproc_simulation.rot_spin.value()})
    outer_diameter: float = {wiz.page_nonproc_diameter.outer_spin.value()}
    inner_diameter: float = {wiz.page_nonproc_diameter.inner_spin.value()}
    poisson_ratio: float = {wiz.page_nonproc_material.poisson_spin.value()}
    young_modulus: float = {wiz.page_nonproc_material.young_spin.value()}
    mass_density: float = {wiz.page_nonproc_material.density_spin.value()}
    visu_edges_per_mm: float = {wiz.page_nonproc_simulation.edges_spin.value()}
    color: Tuple[int, int, int] = {wiz.page_nonproc_start.color!r}

    elements = {elements_block}

    @property
    def length(self) -> float:
        return self.sofa_device.length

    def __post_init__(self):
        super().__init__(
            self.elements,
            self.outer_diameter,
            self.inner_diameter,
            self.poisson_ratio,
            self.young_modulus,
            self.mass_density,
            self.color,
        )
""")

        # create agents folder
        os.makedirs(os.path.join(base_dir, 'agents'), exist_ok=True)

        # optional feedback
        print(f"Tool '{tool_name}' saved to {base_dir}")
