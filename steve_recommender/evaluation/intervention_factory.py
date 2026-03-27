from __future__ import annotations

import os
from typing import Tuple

from steve_recommender.devices import make_device
from steve_recommender.adapters import eve


class _SofaBeamAdapterSimulatedFixedWall(
    eve.intervention.simulation.sofabeamadapter.SofaBeamAdapter
):
    """Experiment-only variant: wall is simulated but collision-geometry-fixed.

    We intentionally keep the vessel wall geometry static via ``moving=False``,
    but set collision models to ``simulated=True`` so wall-side reactions can be
    exposed through simulated collision mechanics.
    """

    def _add_vessel_tree(self, mesh_path):
        vessel_object = self.root.addChild("vesselTree")
        vessel_object.addObject(
            "MeshObjLoader",
            filename=mesh_path,
            flipNormals=False,
            name="meshLoader",
        )
        vessel_object.addObject(
            "MeshTopology",
            position="@meshLoader.position",
            triangles="@meshLoader.triangles",
        )
        vessel_object.addObject("MechanicalObject", name="dofs", src="@meshLoader")
        vessel_object.addObject("TriangleCollisionModel", moving=False, simulated=True)
        vessel_object.addObject("LineCollisionModel", moving=False, simulated=True)
        self._vessel_object = vessel_object


def build_aortic_arch_intervention(*, tool_ref: str, anatomy) -> Tuple[object, float]:
    """Build a MonoPlaneStatic intervention for an aortic arch with a fixed endpoint."""

    arch_type = eve.intervention.vesseltree.ArchType(anatomy.arch_type)
    vessel_tree = eve.intervention.vesseltree.AorticArch(
        arch_type=arch_type,
        seed=anatomy.seed,
        rotation_yzx_deg=tuple(anatomy.rotation_yzx_deg) if anatomy.rotation_yzx_deg else None,
        scaling_xyzd=tuple(anatomy.scaling_xyzd) if anatomy.scaling_xyzd else None,
        omit_axis=anatomy.omit_axis,
    )
    device = make_device(tool_ref)

    if os.environ.get("STEVE_EXPERIMENT_WALL_SIMULATED_FIXED", "").strip() == "1":
        simulation = _SofaBeamAdapterSimulatedFixedWall(friction=anatomy.friction)
    else:
        simulation = eve.intervention.simulation.sofabeamadapter.SofaBeamAdapter(
            friction=anatomy.friction
        )
    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=anatomy.image_frequency_hz,
        image_rot_zx=list(anatomy.image_rot_zx_deg),
    )

    if anatomy.target_mode != "branch_end":
        raise ValueError(f"Unsupported target_mode: {anatomy.target_mode}")

    target = eve.intervention.target.BranchEnd(
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
        threshold=anatomy.target_threshold_mm,
        branches=list(anatomy.target_branches),
    )

    intervention = eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
        stop_device_at_tree_end=True,
        normalize_action=True,
    )

    # dt used to compute action-step velocities.
    action_dt_s = 1.0 / float(anatomy.image_frequency_hz)
    return intervention, action_dt_s
