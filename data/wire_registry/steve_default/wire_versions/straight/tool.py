from dataclasses import dataclass

from data.wire_registry.steve_default.wire_versions.standard_j.tool import (
    BASELINE_TIP_LENGTH_MM,
    JShaped_Default_StandardJ,
)

STRAIGHT_TIP_RADIUS_MM = 1000.0

@dataclass
class JShaped_Default_Straight(JShaped_Default_StandardJ):
    name: str = 'guidewire_straight'
    tip_radius: float = STRAIGHT_TIP_RADIUS_MM
    tip_angle: float = BASELINE_TIP_LENGTH_MM / STRAIGHT_TIP_RADIUS_MM

    def __post_init__(self) -> None:
        # Keep distal arc length equal to the family baseline.
        self.tip_angle = BASELINE_TIP_LENGTH_MM / float(self.tip_radius)
        super().__post_init__()
