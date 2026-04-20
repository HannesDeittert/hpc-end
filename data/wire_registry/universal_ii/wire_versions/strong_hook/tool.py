from dataclasses import dataclass

from data.wire_registry.universal_ii.wire_versions.standard_j.tool import (
    BASELINE_TIP_LENGTH_MM,
    JShaped_UniversalII_StandardJ,
)

STRONG_HOOK_TIP_RADIUS_MM = 6.0

@dataclass
class JShaped_UniversalII_StrongHook(JShaped_UniversalII_StandardJ):
    name: str = 'abbott_universal_II_jshaped_strong_hook'
    tip_radius: float = STRONG_HOOK_TIP_RADIUS_MM
    tip_angle: float = BASELINE_TIP_LENGTH_MM / STRONG_HOOK_TIP_RADIUS_MM

    def __post_init__(self) -> None:
        # Keep distal arc length equal to the family baseline.
        self.tip_angle = BASELINE_TIP_LENGTH_MM / float(self.tip_radius)
        super().__post_init__()
