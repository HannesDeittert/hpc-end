
from steve_recommender.adapters import eve

VesselTree = eve.intervention.vesseltree.VesselTree
AorticArch = eve.intervention.vesseltree.aorticarch.AorticArch
ArchType = eve.intervention.vesseltree.aorticarch.ArchType

class Tree_11(VesselTree):
    """
    Deterministic VesselTree alias of AorticArch with fixed parameters.
    """
    def __init__(self) -> None:
        super().__init__(
            arch_type=ArchType.VII,
            seed=1011,
            rotation_yzx_deg=(0.0, 0.0, 0.0),
            scaling_xyzd=(1.0, 1.0, 1.0, 1.0),
            omit_axis=None,
        )
        # Build branches and coordinate space once
        self.reset()
