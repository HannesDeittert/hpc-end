from dataclasses import dataclass
from typing import Tuple
import math

from eve.intervention.device import Device
from eve.intervention.device.sofadevice import ProceduralShape


@dataclass
class StraightEps(Device):
    """Straight device with a tiny non-zero spire diameter to avoid degeneracy."""

    name: str = "guidewire"
    velocity_limit: Tuple[float, float] = (50, 3.14)
    length: float = 450.0
    tip_length: float = 15.2
    tip_outer_diameter: float = 0.7
    tip_inner_diameter: float = 0.0
    straight_outer_diameter: float = 0.89
    straight_inner_diameter: float = 0.0
    poisson_ratio: float = 0.49
    young_modulus_tip: float = 17e3
    young_modulus_straight: float = 80e3
    mass_density_tip: float = 0.000021
    mass_density_straight: float = 0.000021
    visu_edges_per_mm: float = 0.5
    collis_edges_per_mm_tip: float = 2.0
    collis_edges_per_mm_straight: float = 0.1
    beams_per_mm_tip: float = 1.4
    beams_per_mm_straight: float = 0.09
    # Tiny non-zero spire diameter (mm)
    spire_diameter: float = 0.45
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        spire_height = 0.0
        straight_length = self.length - self.tip_length
        num_edges = math.ceil(self.visu_edges_per_mm * self.length)
        num_edges_collis_tip = math.ceil(self.collis_edges_per_mm_tip * self.tip_length)
        num_edges_collis_straight = math.ceil(
            self.collis_edges_per_mm_straight * straight_length
        )
        beams_tip = math.ceil(self.tip_length * self.beams_per_mm_tip)
        beams_straight = math.ceil(straight_length * self.beams_per_mm_straight)

        self.sofa_device = ProceduralShape(
            length=self.length,
            straight_length=straight_length,
            spire_diameter=self.spire_diameter,
            spire_height=spire_height,
            poisson_ratio=self.poisson_ratio,
            young_modulus=self.young_modulus_straight,
            young_modulus_extremity=self.young_modulus_tip,
            radius=self.straight_outer_diameter / 2,
            radius_extremity=self.tip_outer_diameter / 2,
            inner_radius=self.straight_inner_diameter / 2,
            inner_radius_extremity=self.tip_inner_diameter / 2,
            mass_density=self.mass_density_straight,
            mass_density_extremity=self.mass_density_tip,
            num_edges=num_edges,
            num_edges_collis=(num_edges_collis_straight, num_edges_collis_tip),
            density_of_beams=(beams_straight, beams_tip),
            key_points=(0.0, straight_length, self.length),
            color=self.color,
        )
