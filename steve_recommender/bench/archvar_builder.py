from __future__ import annotations

from typing import Any

import eve


def build_archvar_intervention(
    device: Any,
    episodes_between_arch_change: int = 1,
    stop_device_at_tree_end: bool = True,
    normalize_action: bool = False,
):
    vessel_tree = eve.intervention.vesseltree.AorticArchRandom(
        episodes_between_change=episodes_between_arch_change,
        scale_diameter_array=[0.85],
        arch_types_filter=[eve.intervention.vesseltree.ArchType.I],
    )

    simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.1)

    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=7.5,
        image_rot_zx=[25, 0],
        image_center=[0, 0, 0],
        field_of_view=None,
    )

    target = eve.intervention.target.CenterlineRandom(
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
        threshold=5,
        branches=["lcca", "rcca", "lsa", "rsa", "bct", "co"],
    )

    return eve.intervention.MonoPlaneStatic(
        vessel_tree,
        [device],
        simulation,
        fluoroscopy,
        target,
        stop_device_at_tree_end,
        normalize_action,
    )
