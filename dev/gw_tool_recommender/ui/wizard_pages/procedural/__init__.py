# ui/wizard_pages/procedural/__init__.py

from .general_params import GeneralParamsPage
from .shape_params import ShapeParamsPage
from .material_params import MaterialParamsPage
from .simulation_params import SimulationParamsPage
from .review import ReviewPage


__all__ = [
    "GeneralParamsPage",
    "ShapeParamsPage",
    "MaterialParamsPage",
    "SimulationParamsPage",
    "ReviewPage",
]