from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math

from eve.intervention.device.device import Device
from eve.intervention.device.sofadevice import ProceduralShape


@dataclass
class StandardJ035_PTFE(Device):
    name: str = "StandardJ035_PTFE"

    # Kinematic limits (translation mm/s, rotation rad/s)
    velocity_limit: Tuple[float, float] = (20.0, 6.0)

    # Geometry (mm)
    length: float = 1500.0
    tip_radius: float = 3.0
    tip_angle: float = math.pi  # 180°

    tip_outer_diameter: float = 0.89
    tip_inner_diameter: float = 0.0
    straight_outer_diameter: float = 0.89
    straight_inner_diameter: float = 0.0

    # Material (effective segment-wise parameters)
    poisson_ratio: float = 0.38
    young_modulus_tip: float = 6.0e10
    young_modulus_straight: float = 1.8e11
    mass_density_tip: float = 7500.0
    mass_density_straight: float = 7800.0

    # Discretisation (start values; adjust for speed/stability)
    visu_edges_per_mm: float = 0.5
    collis_edges_per_mm_tip: float = 2.0
    collis_edges_per_mm_straight: float = 0.1
    beams_per_mm_tip: float = 1.4
    beams_per_mm_straight: float = 0.09

    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        spire_height = 0.0
        tip_length = float(self.tip_radius * self.tip_angle)
        straight_length = float(self.length - tip_length)
        spire_diameter = float(self.tip_radius * 2.0)

        num_edges = int(math.ceil(self.visu_edges_per_mm * self.length))
        num_edges_collis_tip = int(math.ceil(self.collis_edges_per_mm_tip * tip_length))
        num_edges_collis_straight = int(
            math.ceil(self.collis_edges_per_mm_straight * straight_length)
        )

        beams_tip = int(math.ceil(tip_length * self.beams_per_mm_tip))
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

