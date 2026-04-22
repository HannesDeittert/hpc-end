from __future__ import annotations

import numpy as np

from third_party.stEVE.eve.intervention.fluoroscopy import (
    Fluoroscopy,
    TrackingOnly,
)
from third_party.stEVE.eve.intervention.simulation import Simulation
from third_party.stEVE.eve.intervention.target import (
    BranchEnd,
    BranchIndex,
    Manual,
    Target,
)
from third_party.stEVE.eve.intervention.vesseltree import (
    AorticArch,
    ArchType,
    VesselTree,
)

from .models import (
    AorticArchAnatomy,
    BranchEndTarget,
    BranchIndexTarget,
    FluoroscopySpec,
    ManualTarget,
    TargetSpec,
)


def build_aortic_arch(anatomy: AorticArchAnatomy) -> AorticArch:
    """Build a real stEVE `AorticArch` from one eval_v2 anatomy model."""

    vessel_tree = AorticArch(
        arch_type=ArchType(anatomy.arch_type),
        seed=anatomy.seed,
        rotation_yzx_deg=anatomy.rotation_yzx_deg,
        scaling_xyzd=anatomy.scaling_xyzd,
        omit_axis=anatomy.omit_axis,
    )
    if anatomy.visualization_mesh_path is not None:
        vessel_tree.visu_mesh_path = str(anatomy.visualization_mesh_path)
    return vessel_tree


def build_fluoroscopy(
    *,
    spec: FluoroscopySpec,
    vessel_tree: VesselTree,
    simulation: Simulation,
) -> TrackingOnly:
    """Build a real stEVE fluoroscopy object."""

    return TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=spec.image_frequency_hz,
        image_rot_zx=spec.image_rot_zx_deg,
    )


def build_target(
    target: TargetSpec,
    *,
    vessel_tree: VesselTree,
    fluoroscopy: Fluoroscopy,
) -> Target:
    """Build one real stEVE target instance from an eval_v2 target spec."""

    if isinstance(target, BranchEndTarget):
        return BranchEnd(
            vessel_tree=vessel_tree,
            fluoroscopy=fluoroscopy,
            threshold=target.threshold_mm,
            branches=list(target.branches),
        )
    if isinstance(target, BranchIndexTarget):
        return BranchIndex(
            vessel_tree=vessel_tree,
            fluoroscopy=fluoroscopy,
            threshold=target.threshold_mm,
            branch=target.branch,
            idx=target.index,
        )
    if isinstance(target, ManualTarget):
        return Manual(
            targets_vessel_cs=[
                np.asarray(point, dtype=float) for point in target.targets_vessel_cs
            ],
            threshold=target.threshold_mm,
            fluoroscopy=fluoroscopy,
        )
    raise TypeError(f"Unsupported target spec: {type(target)!r}")


__all__ = ["build_aortic_arch", "build_fluoroscopy", "build_target"]
