from dataclasses import dataclass
import math

from data.wire_registry.steve_default.wire_versions.default.tool import JShaped_Default


DEFAULT_TIP_LENGTH_MM = 12.1 * 0.4 * math.pi
OVERSTRAIGHT_TIP_RADIUS_MM = 1000.0


@dataclass
class JShaped_Default_StraightTip(JShaped_Default):
    # Large radius + adjusted angle => nearly straight tip with same arc length.
    tip_radius: float = OVERSTRAIGHT_TIP_RADIUS_MM
    tip_angle: float = DEFAULT_TIP_LENGTH_MM / OVERSTRAIGHT_TIP_RADIUS_MM

    def __post_init__(self):
        # Keep tip arc length equal to default J-shaped tip for any radius value.
        self.tip_angle = DEFAULT_TIP_LENGTH_MM / float(self.tip_radius)
        super().__post_init__()
