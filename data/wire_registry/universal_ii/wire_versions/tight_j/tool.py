from dataclasses import dataclass

from data.wire_registry.universal_ii.wire_versions.standard_j.tool import (
    BASELINE_TIP_LENGTH_MM,
    JShaped_UniversalII_StandardJ,
)

TIGHT_J_TIP_RADIUS_MM = 8.0

@dataclass
class JShaped_UniversalII_TightJ(JShaped_UniversalII_StandardJ):
    name: str = 'abbott_universal_II_jshaped_tight_j'
    tip_radius: float = TIGHT_J_TIP_RADIUS_MM
    tip_angle: float = BASELINE_TIP_LENGTH_MM / TIGHT_J_TIP_RADIUS_MM

    def __post_init__(self) -> None:
        # Keep distal arc length equal to the family baseline.
        self.tip_angle = BASELINE_TIP_LENGTH_MM / float(self.tip_radius)
        super().__post_init__()
