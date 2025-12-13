
from dataclasses import dataclass
from typing import Tuple
from eve.intervention.device.device import MeshDevice, StraightPart, Arc

@dataclass
class Test(MeshDevice):
    name: str = 'Test'
    velocity_limit: Tuple[float, float] = (100.0, 30.0)
    outer_diameter: float = 1.0
    inner_diameter: float = 0.0
    poisson_ratio: float = 0.3
    young_modulus: float = 210.0
    mass_density: float = 7850.0
    visu_edges_per_mm: float = 0.5
    color: Tuple[int, int, int] = (85, 170, 127)

    elements = [
        StraightPart(length=10.0, visu_edges_per_mm=0.5, collis_edges_per_mm=4.0, beams_per_mm=3.0)
    ]

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
