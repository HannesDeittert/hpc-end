"""Inspect scene structure and force data availability for debugging monitor initialization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import Sofa  # type: ignore
    import SofaRuntime  # type: ignore
except ImportError:
    print("ERROR: SOFA runtime not available", file=sys.stderr)
    sys.exit(1)

import eve.intervention.simulation
import eve.intervention.target
import eve.intervention.vesseltree


def _find_object_by_path(root: Any, path: str) -> Optional[Any]:
    """Navigate SOFA scene tree using @ notation path."""
    path_clean = str(path).lstrip("@").strip()
    if not path_clean:
        return None
    parts = path_clean.split("/")
    current = root
    for part in parts:
        try:
            current = getattr(current, part, None)
            if current is None:
                return None
        except Exception:
            return None
    return current


def _inspect_mechanical_object(obj: Any, name: str) -> dict:
    """Extract info from MechanicalObject."""
    result = {
        "type": "MechanicalObject",
        "name": name,
        "exists": True,
        "template": "unknown",
        "size": 0,
        "positions": "unavailable",
        "velocities": "unavailable",
        "forces": "unavailable",
        "externalForces": "unavailable",
        "constraints": "unavailable",
    }
    if obj is None:
        result["exists"] = False
        return result
    try:
        result["template"] = str(getattr(obj, "template", "?"))
    except Exception:
        pass
    try:
        pos = getattr(obj, "position", None)
        if pos is not None:
            size = len(pos)
            result["size"] = size
            if size > 0:
                result["positions"] = f"available ({size} DOFs)"
    except Exception:
        pass
    try:
        vel = getattr(obj, "velocity", None)
        if vel is not None and len(vel) > 0:
            result["velocities"] = f"available ({len(vel)} values)"
    except Exception:
        pass
    try:
        forces = getattr(obj, "force", None)
        if forces is not None and len(forces) > 0:
            norm = float(sum(f * f for f in forces) ** 0.5) if forces else 0.0
            result["forces"] = f"available ({len(forces)} values, norm={norm:.6e})"
    except Exception:
        pass
    try:
        ext_forces = getattr(obj, "externalForce", None)
        if ext_forces is not None and len(ext_forces) > 0:
            norm = float(sum(f * f for f in ext_forces) ** 0.5) if ext_forces else 0.0
            result["externalForces"] = f"available ({len(ext_forces)} values, norm={norm:.6e})"
    except Exception:
        pass
    try:
        constraints = getattr(obj, "constraint", None)
        if constraints is not None and len(constraints) > 0:
            result["constraints"] = f"available ({len(constraints)} values)"
    except Exception:
        pass
    return result


def inspect_scene(intervention: Any) -> dict:
    """Inspect SOFA scene structure for force data availability."""
    sim = intervention.simulation
    root = sim.root

    result = {
        "scene_ready": False,
        "error": None,
        "objects": {},
    }

    try:
        # Check InstrumentCombined
        instruments = getattr(root, "InstrumentCombined", None)
        result["objects"]["InstrumentCombined"] = {
            "exists": instruments is not None,
            "children": [],
        }
        if instruments is not None:
            for child_name in ["DOFs", "CollisionModel"]:
                child = getattr(instruments, child_name, None)
                if child is not None:
                    if child_name == "DOFs":
                        result["objects"]["InstrumentCombined/DOFs"] = _inspect_mechanical_object(
                            child, "DOFs"
                        )
                    elif child_name == "CollisionModel":
                        collision_model = child
                        result["objects"]["InstrumentCombined/CollisionModel"] = {
                            "exists": True,
                            "children": [],
                        }
                        collision_dofs = getattr(collision_model, "CollisionDOFs", None)
                        if collision_dofs is not None:
                            result["objects"][
                                "InstrumentCombined/CollisionModel/CollisionDOFs"
                            ] = _inspect_mechanical_object(collision_dofs, "CollisionDOFs")

        # Check vesselTree
        vessel = getattr(root, "vesselTree", None)
        result["objects"]["vesselTree"] = {"exists": vessel is not None, "children": []}
        if vessel is not None:
            dofs = getattr(vessel, "dofs", None)
            if dofs is not None:
                result["objects"]["vesselTree/dofs"] = _inspect_mechanical_object(dofs, "dofs")
            topo = getattr(vessel, "MeshTopology", None)
            result["objects"]["vesselTree/MeshTopology"] = {"exists": topo is not None}

        # Check LCP
        lcp = getattr(root, "LCP", None)
        result["objects"]["LCP"] = {"exists": lcp is not None}
        if lcp is not None:
            try:
                compute_forces = getattr(lcp, "computeConstraintForces", None)
                if compute_forces is not None:
                    result["objects"]["LCP"]["computeConstraintForces"] = bool(compute_forces)
            except Exception:
                pass
            try:
                forces = getattr(lcp, "constraintForces", None)
                if forces is not None:
                    size = len(forces) if hasattr(forces, "__len__") else 0
                    norm = float(sum(f * f for f in forces) ** 0.5) if size > 0 else 0.0
                    result["objects"]["LCP"]["constraintForces"] = (
                        f"available ({size} values, norm={norm:.6e})"
                    )
            except Exception:
                pass

        # Check if passive monitor exists
        monitor = getattr(root, "wire_wall_force_monitor", None)
        result["objects"]["wire_wall_force_monitor"] = {
            "exists": monitor is not None,
        }
        if monitor is not None:
            try:
                available = getattr(monitor, "available", None)
                result["objects"]["wire_wall_force_monitor"]["available"] = bool(available)
            except Exception:
                pass
            try:
                total_force = getattr(monitor, "totalForceNorm", None)
                result["objects"]["wire_wall_force_monitor"]["totalForceNorm"] = (
                    float(total_force) if total_force is not None else None
                )
            except Exception:
                pass

        result["scene_ready"] = True
    except Exception as exc:
        result["error"] = str(exc)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SOFA scene structure and force data")
    parser.add_argument(
        "--arch-type", default="aortic_arch_j", help="Vessel architecture type"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tool-ref", default="amplatz_superstiff/standard_j", help="Wire reference")
    parser.add_argument(
        "--output", type=Path, default=Path("results/debug/scene_structure.json"), help="Output JSON"
    )
    args = parser.parse_args()

    print(f"[inspector] Building intervention with arch_type={args.arch_type} seed={args.seed}")
    try:
        arch_type = getattr(eve.intervention.vesseltree.ArchType, str(args.arch_type))
    except AttributeError:
        print(f"ERROR: Unknown arch_type {args.arch_type}", file=sys.stderr)
        sys.exit(1)

    vessel_tree = eve.intervention.vesseltree.AorticArch(
        arch_type=arch_type,
        seed=int(args.seed),
    )

    from debug.force_playground.single_jwire_basic_scene import make_device

    try:
        device = make_device(str(args.tool_ref))
    except Exception as exc:
        print(f"ERROR: Failed to load device {args.tool_ref}: {exc}", file=sys.stderr)
        sys.exit(1)

    simulation = eve.intervention.simulation.sofabeamadapter.SofaBeamAdapter(friction=0.2)
    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=30.0,
        image_rot_zx=[0.0, 0.0],
    )
    target = eve.intervention.target.BranchEnd(
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
        threshold=3.0,
        branches=["lcca"],
    )
    intervention = eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
    )
    intervention.make_non_mp()

    print(f"[inspector] Resetting intervention...")
    intervention.reset(seed=int(args.seed))

    print(f"[inspector] Inspecting scene structure...")
    report = inspect_scene(intervention)

    print(json.dumps(report, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[inspector] Report written to {args.output}")


if __name__ == "__main__":
    main()
