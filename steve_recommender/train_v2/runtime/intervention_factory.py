"""Build real interventions for train_v2 without legacy training helpers."""

from __future__ import annotations

from steve_recommender.bench import build_archvar_intervention, resolve_device
from eve.intervention import MonoPlaneStatic
from eve.intervention.simulation import SofaBeamAdapter

from steve_recommender.eval_v2.builders import (
    build_aortic_arch,
    build_fluoroscopy,
    build_target,
)
from steve_recommender.eval_v2.discovery import (
    FileBasedAnatomyDiscovery,
    FileBasedWireRegistryDiscovery,
)
from steve_recommender.eval_v2.models import (
    BranchEndTarget,
    FluoroscopySpec,
)
from steve_recommender.eval_v2.runtime import build_device

from ..config import RuntimeSpec


def build_intervention(*, runtime_spec: RuntimeSpec) -> MonoPlaneStatic:
    """Build a runnable intervention from anatomy and wire registry ids."""

    if runtime_spec.anatomy_id is None:
        device, _ = resolve_device(
            runtime_spec.tool_ref,
            runtime_spec.tool_module,
            runtime_spec.tool_class,
        )
        return build_archvar_intervention(device=device)

    anatomy = FileBasedAnatomyDiscovery().get_anatomy(record_id=runtime_spec.anatomy_id)
    wire_registry = FileBasedWireRegistryDiscovery()
    execution_wire = next(
        wire
        for wire in wire_registry.list_execution_wires()
        if wire.tool_ref == runtime_spec.tool_ref
    )
    device = build_device(execution_wire)
    vessel_tree = build_aortic_arch(anatomy)
    simulation = SofaBeamAdapter(friction=float(runtime_spec.friction_mu))
    fluoroscopy = build_fluoroscopy(
        spec=FluoroscopySpec(
            image_frequency_hz=runtime_spec.fluoroscopy_frequency_hz,
            image_rot_zx_deg=runtime_spec.fluoroscopy_rot_zx_deg,
        ),
        vessel_tree=vessel_tree,
        simulation=simulation,
    )
    target = build_target(
        BranchEndTarget(branches=runtime_spec.target_branches),
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
    )
    return MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
        stop_device_at_tree_end=True,
        normalize_action=False,
    )
