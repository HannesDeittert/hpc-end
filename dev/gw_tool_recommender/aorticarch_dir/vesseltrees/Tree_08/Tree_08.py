
from eve.intervention.vesseltree import VesselTree
from eve.intervention.vesseltree.aorticarch import AorticArch, ArchType

class Tree_08(VesselTree):
    """
    Deterministic VesselTree alias of AorticArch with fixed parameters.
    """
    def __init__(self) -> None:
        super().__init__(
            arch_type=ArchType.IV,
            seed=1008,
            rotation_yzx_deg=(0.0, 0.0, 0.0),
            scaling_xyzd=(1.0, 1.0, 1.0, 1.0),
            omit_axis=None,
        )
        # Build branches and coordinate space once
        self.reset()
