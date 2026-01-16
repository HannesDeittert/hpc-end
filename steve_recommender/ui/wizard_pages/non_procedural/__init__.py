# ui/wizard_pages/non_procedural/__init__.py

from .general_params          import GeneralParamsPage
from .diameter_params         import DiameterParamsPage
from .material_params         import MaterialParamsPage
from .simulation_params       import SimulationParamsPage
from .segment_constructor     import SegmentConstructorPage


__all__ = [
    "GeneralParamsPage",
    "DiameterParamsPage",
    "MaterialParamsPage",
    "SimulationParamsPage",
    "SegmentConstructorPage",
]