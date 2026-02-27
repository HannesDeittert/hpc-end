from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math

from eve.intervention.device.device import Device
from eve.intervention.device.sofadevice import ProceduralShape


@dataclass
class UniversalII_Abbott(Device):
    name: str = "UniversalII_Abbott"

    # Kinematic limits (translation mm/s, rotation rad/s)
    velocity_limit: Tuple[float, float] = (20.0, 6.0)

    # Geometry (mm)
    length: float = 450.0
    tip_length: float = 45.0

    tip_outer_diameter: float = 0.3556
    tip_inner_diameter: float = 0.0
    straight_outer_diameter: float = 0.3556
    straight_inner_diameter: float = 0.0

    # Material (effective segment-wise parameters)
    poisson_ratio: float = 0.33
    young_modulus_tip: float = 1.27e9
    young_modulus_straight: float = 8.93e9
    mass_density_tip: float = 4000.0
    mass_density_straight: float = 2500.0

    # Discretisation (start values; adjust for speed/stability)
    visu_edges_per_mm: float = 0.5
    collis_edges_per_mm_tip: float = 2.0
    collis_edges_per_mm_straight: float = 0.1
    beams_per_mm_tip: float = 1.4
    beams_per_mm_straight: float = 0.5

    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        spire_height = 0.0
        straight_length = float(self.length - self.tip_length)
        spire_diameter = 0.0

        num_edges = int(math.ceil(self.visu_edges_per_mm * self.length))
        num_edges_collis_tip = int(math.ceil(self.collis_edges_per_mm_tip * self.tip_length))
        num_edges_collis_straight = int(
            math.ceil(self.collis_edges_per_mm_straight * straight_length)
        )

        beams_tip = int(math.ceil(self.tip_length * self.beams_per_mm_tip))
        beams_straight = int(math.ceil(straight_length * self.beams_per_mm_straight))

        self.sofa_device = ProceduralShape(
            length=self.length,
            straight_length=straight_length,
            spire_diameter=spire_diameter,
            spire_height=spire_height,
            poisson_ratio=self.poisson_ratio,
            young_modulus=self.young_modulus_straight,
            young_modulus_extremity=self.young_modulus_tip,
            radius=self.straight_outer_diameter / 2.0,
            radius_extremity=self.tip_outer_diameter / 2.0,
            inner_radius=self.straight_inner_diameter / 2.0,
            inner_radius_extremity=self.tip_inner_diameter / 2.0,
            mass_density=self.mass_density_straight,
            mass_density_extremity=self.mass_density_tip,
            num_edges=num_edges,
            num_edges_collis=(num_edges_collis_straight, num_edges_collis_tip),
            density_of_beams=(beams_straight, beams_tip),
            key_points=(0.0, straight_length, self.length),
            color=self.color,
        )
