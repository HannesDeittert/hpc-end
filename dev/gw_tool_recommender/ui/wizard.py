# ui/wizard.py

from PyQt5.QtWidgets import QWizard
from .wizard_pages.device_type import DeviceTypePage
from .wizard_pages.procedural.general_params import GeneralParamsPage as ProceduralGeneralParamsPage
from .wizard_pages.non_procedural.general_params import GeneralParamsPage as NonProceduralGeneralParamsPage

from .wizard_pages.non_procedural.diameter_params import DiameterParamsPage as NonProceduralDiameterParamsPage
from .wizard_pages.non_procedural.material_params import MaterialParamsPage as NonProceduralMaterialParamsPage
from .wizard_pages.non_procedural.simulation_params import SimulationParamsPage as NonProceduralSimulationParamsPage
from .wizard_pages.non_procedural.segment_constructor import SegmentConstructorPage as NonProceduralSegmentParamsPage
class ToolWizard(QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Tool Wizard")

        # — Core branching pages —
        dev_page = DeviceTypePage(self)
        self.page_device_type = dev_page
        self.page_device_type_id = self.addPage(dev_page)

        # Procedural branch: start at general params
        proc_page = ProceduralGeneralParamsPage(self)
        self.page_proc_start = proc_page
        self.page_proc_start_id = self.addPage(proc_page)

        # Non‑procedural branch: start at general params
        np_gen = NonProceduralGeneralParamsPage(self)
        self.page_nonproc_start = np_gen
        self.page_nonproc_start_id = self.addPage(np_gen)
        # Next: diameter parameters for non‑procedural device
        np_diam = NonProceduralDiameterParamsPage(self)
        self.page_nonproc_diameter = np_diam
        self.page_nonproc_diameter_id = self.addPage(np_diam)
        # Next: material parameters for non‑procedural device
        np_mat = NonProceduralMaterialParamsPage(self)
        self.page_nonproc_material = np_mat
        self.page_nonproc_material_id= self.addPage(np_mat)
        # Next: simulation parameters for non‑procedural device
        np_sim = NonProceduralSimulationParamsPage(self)
        self.page_nonproc_simulation = np_sim
        self.page_nonproc_simulation_id = self.addPage(np_sim)
        # Next: segment constructor parameters for non‑procedural device
        np_seg = NonProceduralSegmentParamsPage(self)
        self.page_nonproc_segment = np_seg
        self.page_nonproc_segment_id = self.addPage(np_seg)
        # (Then additional non‑procedural pages...)

        # Manage Next/Back visibility on the first page
        self.currentIdChanged.connect(self._on_page_changed)

    def nextId(self):
        # if we’re on the device‑type page…
        if self.currentId() == self.page_device_type_id:
            # choose which branch’s *start* page ID
            return (
                self.page_proc_start_id
                if getattr(self, "isProcedural", False)
                else self.page_nonproc_start_id
            )
        # otherwise fall back to normal sequencing
        return super().nextId()

    def _on_page_changed(self, page_id: int):
        btn_next = self.button(QWizard.NextButton)
        btn_back = self.button(QWizard.BackButton)
        if page_id == self.page_device_type:
            btn_next.hide()
            btn_back.hide()
        else:
            btn_next.show()
            btn_back.show()





