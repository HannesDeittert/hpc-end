# ui/wizard.py

from PyQt5.QtWidgets import QWizard
from .wizard_pages.device_type import DeviceTypePage
from .wizard_pages.model_select import ModelSelectPage
from .wizard_pages.procedural.general_params import GeneralParamsPage as ProceduralGeneralParamsPage
from .wizard_pages.procedural.shape_params import ShapeParamsPage as ProceduralShapeParamsPage
from .wizard_pages.procedural.material_params import MaterialParamsPage as ProceduralMaterialParamsPage
from .wizard_pages.procedural.simulation_params import SimulationParamsPage as ProceduralSimulationParamsPage
from .wizard_pages.procedural.review import ReviewPage as ProceduralReviewPage

from .wizard_pages.non_procedural.general_params import GeneralParamsPage as NonProceduralGeneralParamsPage
from .wizard_pages.non_procedural.diameter_params import DiameterParamsPage as NonProceduralDiameterParamsPage
from .wizard_pages.non_procedural.material_params import MaterialParamsPage as NonProceduralMaterialParamsPage
from .wizard_pages.non_procedural.simulation_params import SimulationParamsPage as NonProceduralSimulationParamsPage
from .wizard_pages.non_procedural.segment_constructor import SegmentConstructorPage as NonProceduralSegmentParamsPage
class ToolWizard(QWizard):
    def __init__(self, parent=None, model_name=None):
        super().__init__(parent)
        self.setWindowTitle("New Tool Wizard")
        self.model_name = model_name

        # — Core branching pages —
        dev_page = DeviceTypePage(self)
        self.page_device_type = dev_page
        self.page_device_type_id = self.addPage(dev_page)

        model_page = None
        if self.model_name is None:
            model_page = ModelSelectPage(self)
            self.page_model_select = model_page
            self.page_model_select_id = self.addPage(model_page)

        # Procedural branch: start at general params
        proc_page = ProceduralGeneralParamsPage(self)
        self.page_proc_start = proc_page
        self.page_proc_start_id = self.addPage(proc_page)
        # Next: shape parameters for procedural device
        p_shape = ProceduralShapeParamsPage(self)
        self.page_proc_shape = p_shape
        self.page_proc_shape_id = self.addPage(p_shape)
        # Next: material parameters for procedural device
        p_material = ProceduralMaterialParamsPage(self)
        self.page_proc_material = p_material
        self.page_proc_material_id = self.addPage(p_material)
        # Next: material parameters for procedural device
        p_simulation = ProceduralSimulationParamsPage(self)
        self.page_proc_simulaiton = p_simulation
        self.page_proc_simulaiton_id = self.addPage(p_simulation)
        # Next: material parameters for procedural device
        p_review = ProceduralReviewPage(self)
        self.page_proc_review = p_review
        self.page_proc_review_id = self.addPage(p_review)

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
            if getattr(self, "model_name", None) is None:
                return self.page_model_select_id
            return self.page_proc_start_id if getattr(self, "isProcedural", False) else self.page_nonproc_start_id
        if getattr(self, "model_name", None) is None and self.currentId() == getattr(self, "page_model_select_id", -1):
            return self.page_proc_start_id if getattr(self, "isProcedural", False) else self.page_nonproc_start_id
        # otherwise fall back to normal sequencing
        return super().nextId()

    def _on_page_changed(self, page_id: int):
        btn_next = self.button(QWizard.NextButton)
        btn_back = self.button(QWizard.BackButton)
        if page_id == self.page_device_type_id:
            btn_next.hide()
            btn_back.hide()
        else:
            btn_next.show()
            btn_back.show()




