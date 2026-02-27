from dataclasses import dataclass, field
from typing import List, Tuple
from steve_recommender.devices.multi_section import (
    BendSection,
    MultiSectionDevice,
    SectionMaterial,
    StraightSection,
)

@dataclass
class MultiSectionDemo(MultiSectionDevice):
    name: str = 'MultiSectionDemo'
    sections: List = field(default_factory=lambda: [
        StraightSection(
            length=120.0,
            material=SectionMaterial(**{'young_modulus': 80000.0, 'poisson_ratio': 0.49, 'mass_density': 2.1e-05, 'radius': 0.45, 'inner_radius': 0.0, 'visu_edges_per_mm': 0.5, 'collis_edges_per_mm': 2.0, 'beams_per_mm': 1.4}),
        ),
        BendSection(
            bend_radius=5.0,
            bend_angle_rad=1.5707963267948966,
            spire_height=0.0,
            material=SectionMaterial(**{'young_modulus': 20000.0, 'poisson_ratio': 0.49, 'mass_density': 2.1e-05, 'radius': 0.35, 'inner_radius': 0.0, 'visu_edges_per_mm': 0.5, 'collis_edges_per_mm': 2.0, 'beams_per_mm': 1.4}),
        ),
        StraightSection(
            length=20.0,
            material=SectionMaterial(**{'young_modulus': 20000.0, 'poisson_ratio': 0.49, 'mass_density': 2.1e-05, 'radius': 0.35, 'inner_radius': 0.0, 'visu_edges_per_mm': 0.5, 'collis_edges_per_mm': 2.0, 'beams_per_mm': 1.4}),
        ),
    ])
    velocity_limit: Tuple[float, float] = (50.0, 3.14)
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
