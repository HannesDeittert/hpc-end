"""
Diagnostic runner: execute one eval_v2 trial and dump per-step force/contact fields.

Usage:
    source scripts/sofa_env.sh && conda activate master-project && \\
    python debug/force_playground/eval_v2_runner_force_trace.py \\
      --anatomy Tree_00 --branch lcca \\
      --candidate archvar_original_best \\
      --wire-model steve_default --wire-name standard_j \\
      --steps 400

Output: JSON report in results/debug/<timestamp>_eval_v2_runner_force_trace/

Per-step fields recorded:
  - step index, action_dt_s, simulation root.dt
  - monitor: availability, totalForceNorm, contactCount, status, source
  - wire DOF: force norm, externalForce norm (should be ~0 for constraint contacts)
  - collision DOF: force norm, externalForce norm
  - LCP: constraintForces count, max |lambda|, sum |lambda|, force-like max (N)
  - contact export: available, contact count, explicit coverage, status
  - object path validity checks

Architectural note:
  In SOFA FreeMotionAnimationLoop + LCPConstraintSolver, contact forces live as
  impulses in constraintForces (= dt * lambda).  MechanicalObject.force and
  .externalForce are NOT populated for constraint-based contacts.  The C++ monitor
  (WireWallForceMonitor) reads force/externalForce and therefore reads zero in real runs.
  The WireWallContactExport component can map LCP rows -> wall triangles but is not yet
  wired into the telemetry collector.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from steve_recommender.eval_v2.force_telemetry import EvalV2ForceTelemetryCollector
from steve_recommender.eval_v2.discovery import DEFAULT_WIRE_REGISTRY_PATH
from steve_recommender.eval_v2.models import (
    BranchEndTarget,
    EvaluationScenario,
    ExecutionPlan,
    FluoroscopySpec,
    WireRef,
)
from steve_recommender.eval_v2.runner import (
    _flatten_observation,
    _reset_play_policy,
    _reset_single_trial_env,
    _select_action,
    _to_env_action,
    build_single_trial_env,
)
from steve_recommender.eval_v2.runtime import prepare_evaluation_runtime
from steve_recommender.eval_v2.service import DefaultEvaluationService


def _read_data_field(obj: Any, name: str, default: Any = None) -> Any:
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


def _vec_array_norm(value: Any) -> float:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return float("nan")
    if arr.size == 0:
        return 0.0
    flat = arr.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return 0.0
    return float(np.linalg.norm(finite))


def _lcp_stats(root: Any, force_scale_to_newton: float) -> dict[str, Any]:
    """Return per-step LCP diagnostic dict."""
    lcp = getattr(root, "LCP", None)
    if lcp is None:
        return {"lcp_available": False}
    raw = _read_data_field(lcp, "constraintForces", None)
    if raw is None:
        return {"lcp_available": True, "lcp_count": 0}
    lam = np.asarray(raw, dtype=np.float64).reshape(-1)
    finite_lam = lam[np.isfinite(lam)]
    count = int(lam.size)
    if finite_lam.size == 0:
        return {"lcp_available": True, "lcp_count": count, "lcp_max_abs": 0.0, "lcp_sum_abs": 0.0}
    dt_s = float(_read_data_field(root, "dt", 0.0) or 0.0)
    abs_lam = np.abs(finite_lam)
    lcp_max = float(np.max(abs_lam))
    lcp_sum = float(np.sum(abs_lam))
    force_like_max_n = (lcp_max / dt_s * force_scale_to_newton) if dt_s > 0 else float("nan")
    force_like_sum_n = (lcp_sum / dt_s * force_scale_to_newton) if dt_s > 0 else float("nan")
    return {
        "lcp_available": True,
        "lcp_count": count,
        "lcp_finite_count": int(finite_lam.size),
        "lcp_max_abs": lcp_max,
        "lcp_sum_abs": lcp_sum,
        "lcp_force_like_max_n": float(force_like_max_n),
        "lcp_force_like_sum_n": float(force_like_sum_n),
        "dt_s": dt_s,
    }


def _mo_force_stats(mo: Any, force_scale_to_newton: float) -> dict[str, Any]:
    """Read force + externalForce norms from a MechanicalObject."""
    if mo is None:
        return {"mo_available": False}
    force_norm_n = _vec_array_norm(_read_data_field(mo, "force", [])) * force_scale_to_newton
    ext_force_norm_n = _vec_array_norm(_read_data_field(mo, "externalForce", [])) * force_scale_to_newton
    return {
        "mo_available": True,
        "force_norm_n": float(force_norm_n),
        "ext_force_norm_n": float(ext_force_norm_n),
    }


def _monitor_stats(monitor: Any, force_scale_to_newton: float) -> dict[str, Any]:
    """Read all diagnostic fields from the passive WireWallForceMonitor."""
    if monitor is None:
        return {"monitor_attached": False}
    available = bool(_read_data_field(monitor, "available", False))
    total_norm = float(_read_data_field(monitor, "totalForceNorm", 0.0) or 0.0)
    contact_count = int(_read_data_field(monitor, "contactCount", 0) or 0)
    wall_segment_count = int(_read_data_field(monitor, "wallSegmentCount", 0) or 0)
    source = str(_read_data_field(monitor, "source", ""))
    status = str(_read_data_field(monitor, "status", ""))
    seg_forces = _read_data_field(monitor, "segmentForceVectors", [])
    seg_count = 0
    seg_max_norm_n = 0.0
    try:
        arr = np.asarray(seg_forces, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            seg_count = int(arr.shape[0])
            norms = np.linalg.norm(arr[:, :3], axis=1)
            if norms.size > 0:
                seg_max_norm_n = float(np.nanmax(norms)) * force_scale_to_newton
    except Exception:
        pass
    return {
        "monitor_attached": True,
        "available": available,
        "total_force_norm_n": total_norm * force_scale_to_newton,
        "contact_count": contact_count,
        "wall_segment_count": wall_segment_count,
        "segment_force_count": seg_count,
        "segment_max_force_n": seg_max_norm_n,
        "source": source,
        "status": status,
    }


def _contact_export_stats(root: Any) -> dict[str, Any]:
    """Read WireWallContactExport fields if present."""
    exporter = getattr(root, "wire_wall_contact_export", None)
    if exporter is None:
        return {"contact_export_attached": False}
    available = bool(_read_data_field(exporter, "available", False))
    contact_count = int(_read_data_field(exporter, "contactCount", 0) or 0)
    explicit_coverage = float(_read_data_field(exporter, "explicitCoverage", 0.0) or 0.0)
    ordering_stable = bool(_read_data_field(exporter, "orderingStable", False))
    status = str(_read_data_field(exporter, "status", ""))
    source = str(_read_data_field(exporter, "source", ""))
    tri_ids = _read_data_field(exporter, "wallTriangleIds", [])
    constraint_rows = _read_data_field(exporter, "constraintRowIndices", [])
    try:
        tri_valid_flags = list(_read_data_field(exporter, "triangleIdValidFlags", []))
        n_valid_tri = int(sum(1 for f in tri_valid_flags if f))
    except Exception:
        n_valid_tri = -1
    try:
        constraint_valid_flags = list(_read_data_field(exporter, "constraintRowValidFlags", []))
        n_valid_rows = int(sum(1 for f in constraint_valid_flags if f))
    except Exception:
        n_valid_rows = -1
    return {
        "contact_export_attached": True,
        "available": available,
        "contact_count": contact_count,
        "explicit_coverage": explicit_coverage,
        "ordering_stable": ordering_stable,
        "n_valid_triangle_ids": n_valid_tri,
        "n_valid_constraint_rows": n_valid_rows,
        "status": status,
        "source": source,
    }


def _link_validity(sim: Any) -> dict[str, Any]:
    """Check that key object paths resolve correctly."""
    result: dict[str, Any] = {}
    root = getattr(sim, "root", None)
    if root is None:
        return {"root_available": False}
    result["root_available"] = True
    result["root_dt"] = float(_read_data_field(root, "dt", float("nan")) or float("nan"))

    # LCP solver
    lcp = getattr(root, "LCP", None)
    result["lcp_available"] = lcp is not None
    if lcp is not None:
        result["lcp_computeConstraintForces"] = bool(
            _read_data_field(lcp, "computeConstraintForces", False)
        )
        result["lcp_build_lcp"] = bool(_read_data_field(lcp, "build_lcp", False))

    # vessel tree
    vessel = getattr(root, "vesselTree", None)
    result["vessel_tree_available"] = vessel is not None
    if vessel is not None:
        vessel_dofs = getattr(vessel, "dofs", None)
        vessel_topo = getattr(vessel, "MeshTopology", None)
        result["vessel_dofs_available"] = vessel_dofs is not None
        result["vessel_topology_available"] = vessel_topo is not None
        if vessel_topo is not None:
            tri_count = 0
            try:
                tris = _read_data_field(vessel_topo, "triangles", [])
                tri_count = len(list(tris)) if tris is not None else 0
            except Exception:
                pass
            result["vessel_triangle_count"] = tri_count

    # instruments
    instruments = getattr(root, "InstrumentCombined", None)
    result["instruments_available"] = instruments is not None
    if instruments is not None:
        wire_dofs = getattr(instruments, "DOFs", None)
        result["wire_dofs_available"] = wire_dofs is not None
        if wire_dofs is not None:
            try:
                n_wire = int(wire_dofs.getSize()) if hasattr(wire_dofs, "getSize") else -1
            except Exception:
                n_wire = -1
            result["wire_dof_count"] = n_wire
        coll_model = getattr(instruments, "CollisionModel", None)
        result["collision_model_available"] = coll_model is not None
        if coll_model is not None:
            coll_dofs = getattr(coll_model, "CollisionDOFs", None)
            result["collision_dofs_available"] = coll_dofs is not None
            if coll_dofs is not None:
                try:
                    n_coll = int(coll_dofs.getSize()) if hasattr(coll_dofs, "getSize") else -1
                except Exception:
                    n_coll = -1
                result["collision_dof_count"] = n_coll

    # monitors
    result["passive_monitor_attached"] = getattr(root, "wire_wall_force_monitor", None) is not None
    result["contact_export_attached"] = getattr(root, "wire_wall_contact_export", None) is not None

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one eval_v2 trial through runner path and write force diagnostics"
    )
    p.add_argument("--anatomy", default="Tree_00")
    p.add_argument("--branch", default="lcca")
    p.add_argument("--candidate", default="archvar_original_best")
    p.add_argument("--wire-model", default="steve_default")
    p.add_argument("--wire-name", default="standard_j")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--output-root", default="results/debug")
    p.add_argument("--run-name", default="eval_v2_runner_force_trace")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    service = DefaultEvaluationService()
    anatomy = service.get_anatomy(record_id=args.anatomy)
    execution_wire = WireRef(model=args.wire_model, wire=args.wire_name)

    candidates = service.list_candidates(execution_wire=execution_wire, include_cross_wire=False)
    candidate = next((c for c in candidates if c.name == args.candidate), None)
    if candidate is None:
        available = ", ".join(c.name for c in candidates[:10])
        raise ValueError(f"Candidate not found: {args.candidate}. Sample available: {available}")

    scenario = EvaluationScenario(
        name="runner_trace_scenario",
        anatomy=anatomy,
        target=BranchEndTarget(threshold_mm=5.0, branches=(args.branch,)),
        fluoroscopy=FluoroscopySpec(),
    )
    execution = ExecutionPlan(
        trials_per_candidate=1,
        base_seed=int(args.seed),
        max_episode_steps=int(args.steps),
        policy_device="cpu",
        policy_mode="deterministic",
        worker_count=1,
    )

    runtime = prepare_evaluation_runtime(
        candidate=candidate,
        scenario=scenario,
        registry_path=DEFAULT_WIRE_REGISTRY_PATH,
        policy_device=execution.policy_device,
    )

    env = build_single_trial_env(
        runtime,
        max_episode_steps=execution.max_episode_steps,
        visualisation=None,
    )
    observation, _ = _reset_single_trial_env(env, seed=int(args.seed))
    _reset_play_policy(runtime.play_policy)

    collector = EvalV2ForceTelemetryCollector(
        spec=runtime.scenario.force_telemetry,
        action_dt_s=runtime.scenario.action_dt_s,
    )
    status = collector.ensure_runtime(intervention=runtime.intervention)

    # mm, kg, s → 1e-3 N
    force_scale_to_newton = 1e-3

    # Capture link validity once after ensure_runtime
    sim = runtime.intervention.simulation
    links = _link_validity(sim)
    print("[runner-force-trace] Link validity check:")
    for k, v in links.items():
        print(f"  {k}: {v}")

    trace: list[dict[str, Any]] = []

    for step in range(1, int(args.steps) + 1):
        flat_state = _flatten_observation(observation)
        action = _select_action(runtime, flat_state=flat_state, execution=execution)
        env_action = _to_env_action(
            action,
            env=env,
            normalize_action=runtime.scenario.normalize_action,
        )
        observation, _, terminated, truncated, _ = env.step(env_action)
        collector.capture_step(intervention=runtime.intervention, step_index=step)

        root = sim.root
        instruments = getattr(root, "InstrumentCombined", None)
        wire_dofs = getattr(instruments, "DOFs", None) if instruments else None
        coll_model = getattr(instruments, "CollisionModel", None) if instruments else None
        coll_dofs = getattr(coll_model, "CollisionDOFs", None) if coll_model else None

        trace.append({
            "step": int(step),
            "action_dt_s": float(runtime.scenario.action_dt_s),
            "monitor": _monitor_stats(getattr(root, "wire_wall_force_monitor", None), force_scale_to_newton),
            "wire_dofs": _mo_force_stats(wire_dofs, force_scale_to_newton),
            "collision_dofs": _mo_force_stats(coll_dofs, force_scale_to_newton),
            "lcp": _lcp_stats(root, force_scale_to_newton),
            "contact_export": _contact_export_stats(root),
        })

        if terminated or truncated:
            break

    summary = collector.build_summary()

    # Aggregate summary
    n = len(trace)
    monitor_nonzero = sum(1 for r in trace if r["monitor"].get("total_force_norm_n", 0.0) > 0.0)
    lcp_nonzero = sum(1 for r in trace if r["lcp"].get("lcp_max_abs", 0.0) > 0.0)
    max_monitor_n = max((r["monitor"].get("total_force_norm_n", 0.0) for r in trace), default=0.0)
    max_lcp_n = max((r["lcp"].get("lcp_force_like_max_n", 0.0) or 0.0 for r in trace), default=0.0)
    max_wire_n = max((r["wire_dofs"].get("force_norm_n", 0.0) for r in trace), default=0.0)
    max_coll_n = max((r["collision_dofs"].get("force_norm_n", 0.0) for r in trace), default=0.0)

    report = {
        "meta": {
            "anatomy": args.anatomy,
            "branch": args.branch,
            "candidate": args.candidate,
            "execution_wire": f"{args.wire_model}/{args.wire_name}",
            "seed": int(args.seed),
            "configured": bool(status.configured),
            "collector_source": status.source,
            "collector_error": status.error or "",
            "force_channel": str(summary.channel),
            "force_summary_source": str(summary.source),
            "force_available_for_score": bool(summary.available_for_score),
            "force_validation_status": str(summary.validation_status),
            "force_quality_tier": str(summary.quality_tier),
            "tip_force_available": bool(summary.tip_force_available),
            "tip_force_validation_status": str(summary.tip_force_validation_status),
        },
        "links": links,
        "summary": {
            "steps_total": n,
            "monitor_nonzero_steps": monitor_nonzero,
            "lcp_nonzero_steps": lcp_nonzero,
            "max_monitor_force_n": float(max_monitor_n),
            "max_lcp_force_like_n": float(max_lcp_n),
            "max_wire_force_norm_n": float(max_wire_n),
            "max_collision_force_norm_n": float(max_coll_n),
            "contact_count_max": int(summary.contact_count_max),
            "contact_detected_any": bool(summary.contact_detected_any),
            "lcp_max_abs_max_raw": summary.lcp_max_abs_max,
            "lcp_sum_abs_mean_raw": summary.lcp_sum_abs_mean,
            "total_force_norm_max_n": summary.total_force_norm_max,
            "force_unit_scale_to_newton": float(collector._force_scale_to_newton),
            "mapping": getattr(collector, "_last_constraint_projection", None),
        },
        "per_step_trace": trace,
        "triangle_force_records": getattr(collector, "_triangle_force_records", []),
        "wire_force_records": getattr(collector, "_wire_force_records", []),
    }

    run_tag = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(args.output_root) / f"{run_tag}_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "runner_force_trace.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[runner-force-trace] output={out_path}")
    print(json.dumps(report["meta"], indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
