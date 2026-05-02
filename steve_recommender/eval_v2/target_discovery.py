from __future__ import annotations

from typing import Tuple

from .builders import build_aortic_arch
from .models import AorticArchAnatomy, AnatomyBranch, TargetModeDescriptor


TARGET_MODE_DESCRIPTORS: Tuple[TargetModeDescriptor, ...] = (
    TargetModeDescriptor(
        kind="branch_end",
        label="Branch End",
        description="Select one or more named branches and target a terminal endpoint.",
        requires_branch_selection=True,
        requires_index_selection=False,
        allows_multi_branch_selection=True,
        requires_manual_points=False,
    ),
    TargetModeDescriptor(
        kind="branch_index",
        label="Branch Index",
        description="Select one branch and one exact centerline index for a fixed target.",
        requires_branch_selection=True,
        requires_index_selection=True,
        allows_multi_branch_selection=False,
        requires_manual_points=False,
    ),
    TargetModeDescriptor(
        kind="centerline_random",
        label="Centerline Random",
        description="Sample one random centerline point from the allowed branches and keep it fixed via seed.",
        requires_branch_selection=False,
        requires_index_selection=False,
        allows_multi_branch_selection=True,
        requires_manual_points=False,
    ),
    TargetModeDescriptor(
        kind="manual",
        label="Manual Coordinates",
        description="Provide explicit vessel-space coordinates instead of a named branch target.",
        requires_branch_selection=False,
        requires_index_selection=False,
        allows_multi_branch_selection=False,
        requires_manual_points=True,
    ),
)


def _to_branch_descriptor(branch: object) -> AnatomyBranch:
    name = str(getattr(branch, "name"))
    coordinates = tuple(
        tuple(float(value) for value in point)
        for point in getattr(branch, "coordinates").tolist()
    )
    length_mm = float(getattr(branch, "length"))
    return AnatomyBranch(
        name=name,
        centerline_points_vessel_cs=coordinates,
        length_mm=length_mm,
    )


class AnatomyTargetDiscovery:
    """Discover branch-based target options from one selected anatomy."""

    def list_branches(self, anatomy: AorticArchAnatomy) -> Tuple[AnatomyBranch, ...]:
        vessel_tree = build_aortic_arch(anatomy)
        vessel_tree.reset()
        return tuple(
            _to_branch_descriptor(branch)
            for branch in sorted(vessel_tree.branches, key=lambda item: item.name)
        )

    def get_branch(
        self,
        anatomy: AorticArchAnatomy,
        *,
        branch_name: str,
    ) -> AnatomyBranch:
        for branch in self.list_branches(anatomy):
            if branch.name == branch_name:
                return branch
        raise KeyError(f"Unknown branch name: {branch_name}")

    def list_target_modes(self) -> Tuple[TargetModeDescriptor, ...]:
        return TARGET_MODE_DESCRIPTORS


__all__ = ["AnatomyTargetDiscovery", "TARGET_MODE_DESCRIPTORS"]
