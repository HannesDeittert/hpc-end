import os, json
from pathlib import Path
from PyQt5.QtWidgets import QWizardPage, QLabel, QVBoxLayout, QWizard
from PyQt5.QtCore import pyqtSignal

from steve_recommender.storage import ensure_model, wire_agents_dir, wire_dir


class ReviewPage(QWizardPage):
    """
    Final review page for procedural shapes: shows a confirmation label,
    swaps 'Next' to 'Save', and on Save generates tool.py, tool_definition.json,
    and creates agents/ folder under data/<tool_name>/
    """
    completeChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFinalPage(True)
        self.setTitle("Review & Save")
        self.setSubTitle("Review parameters and save your procedural tool.")

        self.confirm_label = QLabel("All set! Click Save to generate your tool.")
        self.confirm_label.setWordWrap(True)
        layout = QVBoxLayout()
        layout.addWidget(self.confirm_label)
        self.setLayout(layout)

    def initializePage(self):
        wiz = self.wizard()
        # Rename the Finish button to “Save”
        wiz.setButtonText(wiz.FinishButton, "Save")
        # Always enabled
        wiz.button(wiz.FinishButton).setEnabled(True)
        # Hide the Next button entirely
        wiz.setOption(QWizard.HaveNextButtonOnLastPage, False)

    def nextId(self):
        # No “next” page, so Qt will hide Next and show Finish
        return -1

    def validatePage(self):
        """
        Called when user clicks Finish/Save.
        Run your _save_procedural_tool() here, then return True
        so that wizard.accept() is called (i.e. the dialog closes).
        """
        self._save_procedural_tool()
        return True

    def _save_procedural_tool(self):
        wiz = self.wizard()
        model_name = getattr(wiz, "model_name", None) or "DefaultModel"

        # --- General params (from ProceduralGeneralParamsPage) ---
        gen_page = wiz.page_proc_start
        tool_name = gen_page.name_edit.text().strip() or "ProceduralTool"
        tool_desc = gen_page.desc_edit.text().strip() or ""
        color = gen_page.color

        # --- Shape params (from ProceduralShapeParamsPage) ---
        shape_pg = wiz.page_proc_shape
        length = shape_pg.length_spin.value()
        tip_r = shape_pg.tip_radius_spin.value()
        tip_o = shape_pg.tip_outer_dia_spin.value()
        tip_i = shape_pg.tip_inner_dia_spin.value()
        tip_a = shape_pg.tip_angle_spin.value()
        spire_h = shape_pg.spire_height_spin.value()
        out_d = shape_pg.outer_dia_spin.value()
        in_d = shape_pg.inner_dia_spin.value()

        # --- Material params (from ProceduralMaterialParamsPage) ---
        mat_pg = wiz.page_proc_material
        poisson = mat_pg.poisson_spin.value()
        young_str = mat_pg.young_str_spin.value()
        young_tip = mat_pg.young_tip_spin.value()
        dens_str = mat_pg.density_str_spin.value()
        dens_tip = mat_pg.density_tip_spin.value()

        # --- Simulation params (from ProceduralSimulationParamsPage) ---
        sim_pg = getattr(wiz, "page_proc_simulaiton", None) or wiz.page_proc_simulation
        visu = sim_pg.visu_edges_spin.value()
        collis_tip = sim_pg.collis_tip_spin.value()
        collis_str = sim_pg.collis_str_spin.value()
        beams_tip = sim_pg.beams_tip_spin.value()
        beams_str = sim_pg.beams_str_spin.value()
        trans_spd = sim_pg.trans_speed_spin.value()
        rot_spd = sim_pg.rot_speed_spin.value()

        # --- prepare output dirs ---
        ensure_model(model_name)
        base = str(wire_dir(model_name, tool_name))
        os.makedirs(base, exist_ok=True)
        os.makedirs(str(wire_agents_dir(model_name, tool_name)), exist_ok=True)
        # Make tools importable for multiprocessing ("spawn") runs.
        Path(os.path.join(base, "__init__.py")).touch(exist_ok=True)

        # --- write JSON definition ---
        with open(os.path.join(base, 'tool_definition.json'), 'w') as f:
            json.dump({
                'name': tool_name,
                'description': tool_desc,
                'type': 'procedural',
                'model': model_name,
            }, f, indent=2)

        # --- generate Python source ---
        # build the key_points tuple
        key_points = (0.0, length - shape_pg.length_spin.value(), length)
        # build the class
        tool_py = os.path.join(base, 'tool.py')
        with open(tool_py, 'w') as f:
            f.write(f"""
from dataclasses import dataclass, field
from typing import Tuple, Union, List
from steve_recommender.adapters import eve

Device = eve.intervention.device.device.Device

@dataclass
class {tool_name}(Device):
    name: str = {tool_name!r}
    length: float = {length}
    straight_length: float = {length - (length - shape_pg.tip_radius_spin.value())}
    spire_diameter: float = {tip_o}
    spire_height: float = {spire_h}
    poisson_ratio: float = {poisson}
    young_modulus: float = {young_str}
    young_modulus_extremity: float = {young_tip}
    radius: float = {out_d / 2}
    radius_extremity: float = {tip_o / 2}
    inner_radius: float = {in_d / 2}
    inner_radius_extremity: float = {tip_i / 2}
    mass_density: float = {dens_str}
    mass_density_extremity: float = {dens_tip}
    num_edges: float = {visu}
    num_edges_collis: Union[float, Tuple[float,...]] = {collis_str!r}
    density_of_beams: Union[float, Tuple[float,...]] = {beams_str!r}
    key_points: Tuple[float, ...] = {key_points!r}
    color: Tuple[int,int,int] = {color!r}
    velocity_limit: Tuple[float,float] = ({trans_spd}, {rot_spd})
    is_a_procedural_shape: bool = field(init=False, default=True, repr=False)
    mesh_path: str = field(init=False, default=None, repr=False)

    def __post_init__(self):
        super().__init__(
            self.length,
            self.straight_length,
            self.spire_diameter,
            self.spire_height,
            self.poisson_ratio,
            self.young_modulus,
            self.young_modulus_extremity,
            self.radius,
            self.radius_extremity,
            self.inner_radius,
            self.inner_radius_extremity,
            self.mass_density,
            self.mass_density_extremity,
            self.num_edges,
            self.num_edges_collis,
            self.density_of_beams,
            self.key_points,
            self.color,
        )
""")

        print(f"Procedural tool '{tool_name}' saved to {base}")
