from dataclasses import dataclass

from data.wire_registry.amplatz_super_stiff.wire_versions.standard_j.tool import (
    BASELINE_TIP_LENGTH_MM,
    JShaped_AmplatzSuperStiff_StandardJ,
)

GENTLE_TIP_RADIUS_MM = 25.0

@dataclass
class JShaped_AmplatzSuperStiff_Gentle(JShaped_AmplatzSuperStiff_StandardJ):
    name: str = 'amplatz_super_stiff_jshaped_gentle'
    tip_radius: float = GENTLE_TIP_RADIUS_MM
    tip_angle: float = BASELINE_TIP_LENGTH_MM / GENTLE_TIP_RADIUS_MM

    def __post_init__(self) -> None:
        # Keep distal arc length equal to the family baseline.
        self.tip_angle = BASELINE_TIP_LENGTH_MM / float(self.tip_radius)
        super().__post_init__()
