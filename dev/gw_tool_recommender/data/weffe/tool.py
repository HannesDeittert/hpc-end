
from dataclasses import dataclass, field
from typing import Tuple, Union, List
from eve.intervention.device.device import Device

@dataclass
class weffe(Device):
    name: str = 'weffe'
    length: float = 100.0
    straight_length: float = 5.0
    spire_diameter: float = 10.0
    spire_height: float = 20.0
    poisson_ratio: float = 0.49
    young_modulus: float = 80000.0
    young_modulus_extremity: float = 17000.0
    radius: float = 1.0
    radius_extremity: float = 5.0
    inner_radius: float = 0.25
    inner_radius_extremity: float = 1.0
    mass_density: float = 2.1e-05
    mass_density_extremity: float = 2.1e-05
    num_edges: float = 0.5
    num_edges_collis: Union[float, Tuple[float,...]] = 0.1
    density_of_beams: Union[float, Tuple[float,...]] = 0.09
    key_points: Tuple[float, ...] = (0.0, 0.0, 100.0)
    color: Tuple[int,int,int] = (0, 0, 0)
    velocity_limit: Tuple[float,float] = (50.0, 3.14)
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
