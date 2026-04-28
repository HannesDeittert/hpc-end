"""Detailed monitor state inspection during wall-press contact simulation."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)

try:
    import Sofa  # type: ignore
except ImportError:
    print("ERROR: SOFA runtime not available", file=sys.stderr)
    sys.exit(1)

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import eve.intervention.simulation
import eve.intervention.target
import eve.intervention.vesseltree
from debug.force_playground.single_jwire_basic_scene import make_device
from steve_recommender.eval_v2.force_telemetry import EvalV2ForceTelemetryCollector
from steve_recommender.eval_v2.models import ForceTelemetrySpec, ForceUnits


def _read_data_field(obj: any, name: str, default: any = None) -> any:
    try:
        data = getattr(obj, name)
        if hasattr(data, "value"):
            return data.value
        return data
    except Exception:
        pass
    try:
        data = obj.findData(name)
        return data.value
    except Exception:
        return default


def _vec_array_norm(value: any) -> float:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return float("nan")
    if arr.size == 0:
        return 0.0
    if arr.ndim == 1:
        arr = arr.reshape((-1,))
    if not np.all(np.isfinite(arr)):
        arr = np.where(np.isfinite(arr), arr, 0.0)
    return float(np.linalg.norm(arr))


def _lcp_force_like_max_newton(sim: any, force_scale_to_newton: float) -> float:
    root = getattr(sim, "root", None)
    if root is None:
        return float("nan")
    lcp = getattr(root, "LCP", None)
    if lcp is None:
        return float("nan")
    raw = _read_data_field(lcp, "constraintForces", None)
    if raw is None:
        return float("nan")
    lam = np.asarray(raw, dtype=np.float64).reshape(-1)
    if lam.size == 0:
        return 0.0
    dt_s = float(_read_data_field(root, "dt", 0.0) or 0.0)
    if dt_s <= 0.0:
        return float("nan")
    return float(np.max(np.abs(lam) / dt_s) * force_scale_to_newton)


def inspect_monitor_state(sim: any, step: int) -> dict:
    """Inspect detailed state of the passive monitor."""
    root = sim.root
    monitor = getattr(root, "wire_wall_force_monitor", None)
    
    result = {
        "step": step,
        "monitor_exists": monitor is not None,
        "data": {}
    }
    
    if monitor is None:
        return result
    
    try:
        available = bool(_read_data_field(monitor, "available", False))
        result["data"]["available"] = available
    except Exception as e:
        result["data"]["available_error"] = str(e)
    
    for field_name in [
        "totalForceNorm",
        "contactCount",
        "wallSegmentCount", 
        "source",
        "status",
    ]:
        try:
            value = _read_data_field(monitor, field_name, None)
            result["data"][field_name] = value
        except Exception as e:
            result["data"][f"{field_name}_error"] = str(e)
    
    # Try to read force vectors
    try:
        seg_forces = _read_data_field(monitor, "segmentForceVectors", None)
        if seg_forces is not None:
            seg_list = list(seg_forces) if hasattr(seg_forces, '__iter__') else []
            result["data"]["segmentForceVectors_count"] = len(seg_list)
            if seg_list:
                norms = [float((x**2 + y**2 + z**2)**0.5) for x,y,z in seg_list[:5]]
                result["data"]["segmentForceVectors_sample_norms"] = norms
    except Exception as e:
        result["data"]["segmentForceVectors_error"] = str(e)
    
    # Check collision state
    try:
        collision_dofs = getattr(sim._instruments_combined.CollisionModel, "CollisionDOFs", None)  # noqa
        if collision_dofs is not None:
            positions = list(_read_data_field(collision_dofs, "position", []))
            velocities = list(_read_data_field(collision_dofs, "velocity", []))
            forces = list(_read_data_field(collision_dofs, "force", []))
            result["data"]["collision_dofs"] = {
                "positions_count": len(positions),
                "velocities_count": len(velocities),
                "forces_count": len(forces),
            }
    except Exception as e:
        result["data"]["collision_dofs_error"] = str(e)
    
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Inspect monitor state during contact")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--insert-action", type=float, default=0.40)
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"Building intervention...")
    arch_type = eve.intervention.vesseltree.ArchType.I
    vessel_tree = eve.intervention.vesseltree.AorticArch(
        arch_type=arch_type,
        seed=int(args.seed),
    )
    device = make_device("steve_default/standard_j")
    simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.2)
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
    
    logger.info(f"Resetting intervention...")
    intervention.reset(seed=int(args.seed))

    # Use eval_v2 collector path so runtime setup matches production semantics.
    units = ForceUnits(length_unit="mm", mass_unit="kg", time_unit="s")
    spec = ForceTelemetrySpec(
        mode="constraint_projected_si_validated",
        required=False,
        contact_epsilon=1e-7,
        units=units,
    )
    collector = EvalV2ForceTelemetryCollector(spec=spec, action_dt_s=1.0 / 7.5)
    status = collector.ensure_runtime(intervention=intervention)
    force_scale_to_newton = 1e-3
    logger.info(
        "Force collector runtime: configured=%s source=%s error=%s",
        status.configured,
        status.source,
        status.error,
    )
    
    logger.info(f"Running {args.steps} steps with insert_action={args.insert_action}...")
    monitor_states = []
    per_step_trace = []
    
    for step in range(args.steps):
        # Apply constant insertion motion for wall contact
        action = np.asarray([args.insert_action, 0.0], dtype=np.float32)
        intervention.step(action)
        collector.capture_step(intervention=intervention, step_index=step + 1)
        
        # Inspect monitor state
        state = inspect_monitor_state(intervention.simulation, step)
        monitor_states.append(state)

        sim = intervention.simulation
        root = sim.root
        wire_dofs = getattr(sim._instruments_combined, "DOFs", None)  # noqa: SLF001
        coll_dofs = getattr(sim._instruments_combined.CollisionModel, "CollisionDOFs", None)  # noqa: SLF001

        monitor_total_n = float(state.get("data", {}).get("totalForceNorm", 0.0) or 0.0)
        monitor_contact_count = int(state.get("data", {}).get("contactCount", 0) or 0)

        wire_force_norm_scene = _vec_array_norm(_read_data_field(wire_dofs, "force", []))
        collision_force_norm_scene = _vec_array_norm(_read_data_field(coll_dofs, "force", []))
        wire_force_norm_n = wire_force_norm_scene * force_scale_to_newton
        collision_force_norm_n = collision_force_norm_scene * force_scale_to_newton
        lcp_force_like_max_n = _lcp_force_like_max_newton(sim, force_scale_to_newton)

        per_step_trace.append(
            {
                "step": int(step),
                "monitor_total_force_norm_n": float(monitor_total_n),
                "monitor_contact_count": int(monitor_contact_count),
                "wire_force_norm_n": float(wire_force_norm_n),
                "collision_force_norm_n": float(collision_force_norm_n),
                "lcp_force_like_max_n": float(lcp_force_like_max_n),
                "dt_s": float(_read_data_field(root, "dt", 0.0) or 0.0),
            }
        )
        
        if step % 10 == 0:
            logger.info(f"Step {step}: available={state['data'].get('available', '?')} "
                        f"totalForceNorm={state['data'].get('totalForceNorm', '?')} "
                        f"contactCount={state['data'].get('contactCount', '?')}")
    
    # Summary
    available_steps = sum(1 for s in monitor_states if s['data'].get('available', False))
    max_force = max((float(s['data'].get('totalForceNorm', 0.0) or 0.0) for s in monitor_states), default=0.0)
    max_contacts = max((int(s['data'].get('contactCount', 0) or 0) for s in monitor_states), default=0)
    max_wire_force_n = max((float(x["wire_force_norm_n"]) for x in per_step_trace), default=0.0)
    max_collision_force_n = max((float(x["collision_force_norm_n"]) for x in per_step_trace), default=0.0)
    max_lcp_force_like_n = max((float(x["lcp_force_like_max_n"]) for x in per_step_trace), default=0.0)
    lcp_nonzero_steps = int(sum(1 for x in per_step_trace if float(x["lcp_force_like_max_n"]) > 0.0))
    monitor_nonzero_steps = int(sum(1 for x in per_step_trace if float(x["monitor_total_force_norm_n"]) > 0.0))
    
    report = {
        "summary": {
            "total_steps": args.steps,
            "steps_with_available_data": available_steps,
            "max_total_force_norm": max_force,
            "max_contact_count": max_contacts,
            "insert_action": args.insert_action,
            "collector_source": status.source,
            "collector_configured": bool(status.configured),
            "max_wire_force_norm_n": float(max_wire_force_n),
            "max_collision_force_norm_n": float(max_collision_force_n),
            "max_lcp_force_like_n": float(max_lcp_force_like_n),
            "lcp_nonzero_steps": int(lcp_nonzero_steps),
            "monitor_nonzero_steps": int(monitor_nonzero_steps),
        },
        "per_step_trace": per_step_trace,
        "final_states": monitor_states[-5:],
    }
    
    logger.info("\n" + json.dumps(report, indent=2))
    
    output_path = Path("results/debug/monitor_wall_press_inspection.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nFull report written to {output_path}")


if __name__ == "__main__":
    main()
