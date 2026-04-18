from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math

from eve.intervention.device.device import Device
from eve.intervention.device.sofadevice import ProceduralShape

BASELINE_TIP_LENGTH_MM = 15.205308443374598
BASELINE_TIP_RADIUS_MM = 12.1

@dataclass
class JShaped_UniversalII_StandardJ(Device):
    name: str = 'abbott_universal_II_jshaped'
    velocity_limit: Tuple[float, float] = (35.0, 3.14)
    length: float = 450.0
    tip_radius: float = 12.1
    tip_angle: float = 1.2566370614359172
    tip_outer_diameter: float = 0.36
    tip_inner_diameter: float = 0.0
    straight_outer_diameter: float = 0.36
    straight_inner_diameter: float = 0.0
    poisson_ratio: float = 0.33
    young_modulus_tip: float = 1800.0
    young_modulus_straight: float = 8500.0
    mass_density_tip: float = 2.1e-05
    mass_density_straight: float = 2.1e-05
    visu_edges_per_mm: float = 0.5
    collis_edges_per_mm_tip: float = 2.0
    collis_edges_per_mm_straight: float = 0.1
    beams_per_mm_tip: float = 1.4
    beams_per_mm_straight: float = 0.5
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        # Keep baseline distal arc length invariant for this family.
        self.tip_angle = BASELINE_TIP_LENGTH_MM / float(self.tip_radius)
        spire_height = 0.0
        tip_length = self.tip_radius * self.tip_angle
        straight_length = self.length - tip_length
        spire_diameter = self.tip_radius * 2.0
        num_edges = math.ceil(self.visu_edges_per_mm * self.length)
        num_edges_collis_tip = math.ceil(self.collis_edges_per_mm_tip * tip_length)
        num_edges_collis_straight = math.ceil(
            self.collis_edges_per_mm_straight * straight_length
        )
        beams_tip = math.ceil(tip_length * self.beams_per_mm_tip)
        beams_straight = math.ceil(straight_length * self.beams_per_mm_straight)

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
