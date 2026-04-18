from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from steve_recommender.adapters import eve
from steve_recommender.evaluation import load_config
from steve_recommender.evaluation.force_si import unit_scale_to_si_newton
from steve_recommender.evaluation.info_collectors import SofaWallForceInfo
from steve_recommender.evaluation.intervention_factory import (
    build_aortic_arch_intervention,
)


def _read_data(obj: Any, name: str) -> Any:
    try:
        value = getattr(obj, name)
    except Exception:
        return None
    if hasattr(value, "value"):
        try:
            return value.value
        except Exception:
            return None
    return value


def _set_data(obj: Any, name: str, value: Any) -> bool:
    try:
        data = getattr(obj, name)
        if hasattr(data, "value"):
            data.value = value
        else:
            setattr(obj, name, value)
        return True
    except Exception:
        pass
    try:
        data = obj.findData(name)
        data.value = value
        return True
    except Exception:
        return False


def _to_array(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=dtype)
    try:
        return np.asarray(value, dtype=dtype)
    except Exception:
        return np.zeros((0,), dtype=dtype)


def _to_vec3(value: Any) -> np.ndarray:
    arr = _to_array(value, dtype=np.float64).reshape((-1,))
    if arr.size < 3:
        return np.zeros((3,), dtype=np.float64)
    out = arr[:3]
    if not np.all(np.isfinite(out)):
        return np.zeros((3,), dtype=np.float64)
    return out.astype(np.float64)


def _parse_constraint_rows(raw: Any) -> List[Tuple[int, int, np.ndarray]]:
    text = str(raw or "").strip()
    if not text:
        return []
    entries: List[Tuple[int, int, np.ndarray]] = []
    for line in text.splitlines():
        toks = line.split()
        if len(toks) < 4:
            continue
        try:
            row_idx = int(float(toks[0]))
            n_blocks = int(float(toks[1]))
        except Exception:
            continue
        if n_blocks <= 0:
            continue
        payload = toks[2:]
        if not payload or len(payload) % n_blocks != 0:
            continue
        block_width = len(payload) // n_blocks
        if block_width < 4:
            continue
        for bi in range(n_blocks):
            base = bi * block_width
            block = payload[base : base + block_width]
            try:
                dof_idx = int(float(block[0]))
                coeff = np.asarray(
                    [float(block[1]), float(block[2]), float(block[3])],
                    dtype=np.float64,
                )
            except Exception:
                continue
            if not np.all(np.isfinite(coeff)):
                continue
            entries.append((row_idx, dof_idx, coeff))
    return entries


def _project_lambda_dt(
    *,
    lambda_dt: np.ndarray,
    row_entries: List[Tuple[int, int, np.ndarray]],
    n_dofs: int,
) -> np.ndarray:
    if int(n_dofs) <= 0 or lambda_dt.size == 0 or not row_entries:
        return np.zeros((3,), dtype=np.float64)
    per_dof = np.zeros((int(n_dofs), 3), dtype=np.float64)
    for row_idx, dof_idx, coeff in row_entries:
        if row_idx < 0 or row_idx >= lambda_dt.shape[0]:
            continue
        if dof_idx < 0 or dof_idx >= per_dof.shape[0]:
            continue
        per_dof[dof_idx] += coeff * float(lambda_dt[row_idx])
    return np.sum(per_dof, axis=0, dtype=np.float64).reshape((3,))


def _preview(arr: np.ndarray, *, rows: int, cols: int) -> str:
    if arr.size == 0:
        return "[]"
    if arr.ndim == 0:
        return str(arr.item())
    if arr.ndim == 1:
        clip = arr[:cols]
        tail = " ..." if arr.shape[0] > cols else ""
        return np.array2string(clip, precision=4, suppress_small=True) + tail
    r = min(rows, arr.shape[0])
    c = min(cols, arr.shape[1]) if arr.ndim >= 2 else cols
    clip = arr[:r, :c] if arr.ndim >= 2 else arr[:r]
    tail = " ..." if arr.shape[0] > r or (arr.ndim >= 2 and arr.shape[1] > c) else ""
    return np.array2string(clip, precision=4, suppress_small=True) + tail


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _read_exporter(sim: Any) -> Dict[str, Any]:
    exporter = getattr(getattr(sim, "root", None), "wire_wall_contact_export", None)
    if exporter is None:
        return {"exists": False}
    out: Dict[str, Any] = {"exists": True}
    for key in (
        "available",
        "status",
        "source",
        "explicitCoverage",
        "orderingStable",
        "wallPoints",
        "wallTriangleIds",
        "constraintRowIndices",
        "collisionDofIndices",
        "triangleIdValidFlags",
        "constraintRowValidFlags",
        "collisionDofValidFlags",
        "inRangeFlags",
        "mappingCompleteFlags",
        "orderingStableFlags",
        "contactKinds",
    ):
        out[key] = _read_data(exporter, key)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Step-through debugger for the real evaluation scene (real wire + real vascular tree). "
            "Prints and stores per-step raw SOFA arrays, LCP/contact telemetry and wall-force info."
        )
    )
    p.add_argument("--config", required=True, help="Evaluation YAML config path")
    p.add_argument("--agent-index", type=int, default=0)
    p.add_argument("--tool-ref", default=None, help="Override tool_ref from config")
    p.add_argument("--seed", type=int, default=None, help="Override reset seed (default from config)")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--insert-action", type=float, default=1.0)
    p.add_argument("--rotate-action", type=float, default=0.0)
    p.add_argument("--contact-distance", type=float, default=None)
    p.add_argument("--alarm-distance", type=float, default=None)
    p.add_argument("--contact-epsilon", type=float, default=1e-7)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--preview-rows", type=int, default=4)
    p.add_argument("--preview-cols", type=int, default=6)
    p.add_argument("--interactive", dest="interactive", action="store_true")
    p.add_argument("--non-interactive", dest="interactive", action="store_false")
    p.set_defaults(interactive=True)
    p.add_argument("--show-sofa", dest="show_sofa", action="store_true")
    p.add_argument("--no-show-sofa", dest="show_sofa", action="store_false")
    p.set_defaults(show_sofa=False)
    p.add_argument(
        "--sofa-keyboard-step",
        dest="sofa_keyboard_step",
        action="store_true",
        help=(
            "When --show-sofa is enabled: control stepping/actions from the SOFA window via keyboard "
            "(Space/Enter step, arrows steer, c run/pause, q quit)."
        ),
    )
    p.add_argument("--no-sofa-keyboard-step", dest="sofa_keyboard_step", action="store_false")
    p.set_defaults(sofa_keyboard_step=True)
    p.add_argument(
        "--action-step-delta",
        type=float,
        default=0.05,
        help="Delta applied to insert-action on Up/Down key in SOFA keyboard mode.",
    )
    p.add_argument(
        "--rotate-step-delta",
        type=float,
        default=0.02,
        help="Delta applied to rotate-action on Left/Right key in SOFA keyboard mode.",
    )
    p.add_argument(
        "--start-continuous",
        action="store_true",
        help="Start unpaused in SOFA keyboard mode (toggle with key 'c').",
    )
    p.add_argument("--output-root", default="results/force_playground")
    p.add_argument("--run-name", default="eval_scene_step_debug")
    return p


def _setup_sofa_viewer(intervention: Any) -> Tuple[Any, Any]:
    viewer = eve.visualisation.SofaPygame(intervention=intervention)
    viewer.reset(episode_nr=0)
    # Force first draw so window appears and holds current frame.
    viewer.render()
    pygame_mod = getattr(viewer, "_pygame", None)
    if pygame_mod is None:
        raise RuntimeError("pygame module unavailable from SofaPygame viewer")
    return viewer, pygame_mod


def _poll_sofa_keyboard(
    *,
    pygame_mod: Any,
    insert_action: float,
    rotate_action: float,
    action_step_delta: float,
    rotate_step_delta: float,
    pending_steps: int,
    run_continuous: bool,
) -> Tuple[float, float, int, bool, bool]:
    should_quit = False
    for event in pygame_mod.event.get():
        etype = int(getattr(event, "type", -1))
        if etype == int(pygame_mod.QUIT):
            should_quit = True
            continue
        if etype != int(pygame_mod.KEYDOWN):
            continue
        key = int(getattr(event, "key", -1))
        if key in {int(pygame_mod.K_ESCAPE), int(pygame_mod.K_q)}:
            should_quit = True
            continue
        if key in {int(pygame_mod.K_SPACE), int(pygame_mod.K_RETURN), int(pygame_mod.K_n)}:
            pending_steps += 1
            print(f"[eval-step-debug] step queued ({pending_steps})")
            continue
        if key == int(pygame_mod.K_c):
            run_continuous = not bool(run_continuous)
            print(
                "[eval-step-debug] continuous={mode}".format(
                    mode="on" if run_continuous else "off",
                )
            )
            continue
        if key == int(pygame_mod.K_UP):
            insert_action += float(action_step_delta)
            print(f"[eval-step-debug] insert_action={insert_action:.6g}")
            continue
        if key == int(pygame_mod.K_DOWN):
            insert_action -= float(action_step_delta)
            print(f"[eval-step-debug] insert_action={insert_action:.6g}")
            continue
        if key == int(pygame_mod.K_LEFT):
            rotate_action -= float(rotate_step_delta)
            print(f"[eval-step-debug] rotate_action={rotate_action:.6g}")
            continue
        if key == int(pygame_mod.K_RIGHT):
            rotate_action += float(rotate_step_delta)
            print(f"[eval-step-debug] rotate_action={rotate_action:.6g}")
            continue
    return insert_action, rotate_action, pending_steps, run_continuous, should_quit


def main() -> None:
    args = _build_parser().parse_args()

    cfg = load_config(args.config)
    if not cfg.agents:
        raise ValueError("Config has no agents")
    if int(args.agent_index) < 0 or int(args.agent_index) >= len(cfg.agents):
        raise IndexError(
            f"agent-index out of range: {args.agent_index} (available: 0..{len(cfg.agents)-1})"
        )
    agent = cfg.agents[int(args.agent_index)]
    tool_ref = str(args.tool_ref) if args.tool_ref else str(agent.tool)
    seed = int(args.seed) if args.seed is not None else int(cfg.base_seed)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{ts}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    intervention, dt_s = build_aortic_arch_intervention(
        tool_ref=tool_ref,
        anatomy=cfg.anatomy,
        force_extraction=cfg.force_extraction,
    )
    intervention.make_non_mp()
    intervention.reset(seed=seed)
    sim = intervention.simulation

    lmd = getattr(getattr(sim, "root", None), "localmindistance", None)
    if lmd is not None:
        if args.contact_distance is not None:
            _set_data(lmd, "contactDistance", float(args.contact_distance))
        if args.alarm_distance is not None:
            _set_data(lmd, "alarmDistance", float(args.alarm_distance))

    force_info = SofaWallForceInfo(
        intervention,
        mode=cfg.force_extraction.mode,
        required=cfg.force_extraction.required,
        contact_epsilon=float(args.contact_epsilon),
        plugin_path=cfg.force_extraction.plugin_path,
        units=cfg.force_extraction.units,
        constraint_dt_s=float(dt_s),
    )
    force_info.step()

    lcp = sim.root.LCP
    wire_dofs = sim._instruments_combined.DOFs
    collision_dofs = sim._instruments_combined.CollisionModel.CollisionDOFs
    wire_mesh = getattr(sim._instruments_combined, "MeshLines", None)
    collision_edge_set = getattr(sim._instruments_combined.CollisionModel, "collisEdgeSet", None)

    scale_to_newton = (
        float(unit_scale_to_si_newton(cfg.force_extraction.units))
        if cfg.force_extraction.units is not None
        else 1.0
    )

    summary_csv = run_dir / "steps_summary.csv"
    snapshots_jsonl = run_dir / "steps_snapshots.jsonl"
    use_sofa_keyboard = bool(args.show_sofa) and bool(args.sofa_keyboard_step) and bool(args.interactive)
    viewer = None
    pygame_mod = None
    if bool(args.show_sofa):
        try:
            viewer, pygame_mod = _setup_sofa_viewer(intervention)
            print("[eval-step-debug] sofa_window=enabled")
            if use_sofa_keyboard:
                print(
                    "[eval-step-debug] sofa_keys: Space/Enter/n=step | c=run/pause | "
                    "Up/Down=insert | Left/Right=rotate | q/Esc=quit"
                )
        except Exception as exc:
            print(f"[eval-step-debug] sofa_window=disabled reason={exc}")
            viewer = None
            pygame_mod = None
            use_sofa_keyboard = False

    run_cfg = {
        "args": vars(args),
        "config_path": str(args.config),
        "agent": asdict(agent),
        "tool_ref": tool_ref,
        "seed": seed,
        "dt_s": float(dt_s),
        "force_mode": cfg.force_extraction.mode,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"[eval-step-debug] run_dir={run_dir}")
    print(
        "[eval-step-debug] tool={tool} force_mode={mode} dt_s={dt:.6g} seed={seed}".format(
            tool=tool_ref,
            mode=cfg.force_extraction.mode,
            dt=float(dt_s),
            seed=seed,
        )
    )
    print(
        "[eval-step-debug] contactDistance={cd} alarmDistance={ad} computeConstraintForces={ccf}".format(
            cd=_read_data(lmd, "contactDistance") if lmd is not None else None,
            ad=_read_data(lmd, "alarmDistance") if lmd is not None else None,
            ccf=_read_data(lcp, "computeConstraintForces"),
        )
    )
    print(f"[eval-step-debug] summary_csv={summary_csv}")
    print(f"[eval-step-debug] snapshots_jsonl={snapshots_jsonl}")

    summary_fields = [
        "step",
        "time_s",
        "insert_action",
        "rotate_action",
        "n_lcp",
        "n_constraint_rows",
        "lcp_active_rows",
        "lcp_max_abs",
        "lcp_abs_sum",
        "wall_contact_detected",
        "wall_contact_count",
        "wall_total_force_norm_n",
        "wall_lcp_max_abs",
        "quality_tier",
        "association_method",
        "association_coverage",
        "association_explicit_force_coverage",
        "active_constraint_step",
        "projection_gap_n",
        "wire_nodes",
        "wire_edges",
        "wire_node_force_max",
        "wire_edge_force_max",
        "collision_nodes",
        "collision_edges",
        "collision_force_max",
        "native_export_available",
        "native_export_status",
    ]

    insert_action = float(args.insert_action)
    rotate_action = float(args.rotate_action)
    pending_steps = 0
    run_continuous = bool(args.start_continuous)
    step_idx = 0

    with summary_csv.open("w", newline="", encoding="utf-8") as csv_f, snapshots_jsonl.open(
        "w", encoding="utf-8"
    ) as jsonl_f:
        writer = csv.DictWriter(csv_f, fieldnames=summary_fields, delimiter=";")
        writer.writeheader()
        try:
            while step_idx < int(args.steps):
                if bool(args.interactive):
                    if use_sofa_keyboard and pygame_mod is not None:
                        while True:
                            (
                                insert_action,
                                rotate_action,
                                pending_steps,
                                run_continuous,
                                should_quit,
                            ) = _poll_sofa_keyboard(
                                pygame_mod=pygame_mod,
                                insert_action=insert_action,
                                rotate_action=rotate_action,
                                action_step_delta=float(args.action_step_delta),
                                rotate_step_delta=float(args.rotate_step_delta),
                                pending_steps=int(pending_steps),
                                run_continuous=bool(run_continuous),
                            )
                            if bool(should_quit):
                                print("[eval-step-debug] stop requested by user")
                                return
                            if bool(run_continuous):
                                break
                            if int(pending_steps) > 0:
                                pending_steps -= 1
                                break
                            time.sleep(0.02)
                    else:
                        while True:
                            prompt = (
                                "[eval-step-debug] step={s:03d} action=({ia:.4f},{ra:.4f}) "
                                "Enter=step | a <insert> <rotate> | q=quit > "
                            ).format(
                                s=step_idx,
                                ia=insert_action,
                                ra=rotate_action,
                            )
                            cmd = input(prompt).strip()
                            if cmd == "":
                                break
                            if cmd.lower() in {"q", "quit", "exit"}:
                                print("[eval-step-debug] stop requested by user")
                                return
                            if cmd.lower().startswith("a "):
                                parts = cmd.split()
                                if len(parts) != 3:
                                    print("[eval-step-debug] expected: a <insert> <rotate>")
                                    continue
                                try:
                                    insert_action = float(parts[1])
                                    rotate_action = float(parts[2])
                                except Exception:
                                    print("[eval-step-debug] invalid float values")
                                    continue
                                print(
                                    "[eval-step-debug] updated action to insert={ia:.6g} rotate={ra:.6g}".format(
                                        ia=insert_action,
                                        ra=rotate_action,
                                    )
                                )
                                continue
                            print("[eval-step-debug] unknown command")

                action = np.asarray([[insert_action, rotate_action]], dtype=np.float32)
                intervention.step(action)
                force_info.step()
                info = force_info.info

                lcp_vals = _to_array(_read_data(lcp, "constraintForces"), dtype=np.float64).reshape((-1,))
                lcp_active_rows = int(np.count_nonzero(np.abs(lcp_vals) > float(args.contact_epsilon)))
                lcp_max_abs = float(np.max(np.abs(lcp_vals))) if lcp_vals.size else 0.0
                lcp_abs_sum = float(np.sum(np.abs(lcp_vals))) if lcp_vals.size else 0.0

                collision_constraint_raw = _read_data(collision_dofs, "constraint")
                row_entries = _parse_constraint_rows(collision_constraint_raw)
                row_count = len({int(r) for r, _d, _c in row_entries})
                lambda_dt = (
                    lcp_vals / float(dt_s)
                    if lcp_vals.size and float(dt_s) > 0.0
                    else np.zeros((0,), dtype=np.float64)
                )

                wire_pos = _to_array(_read_data(wire_dofs, "position"), dtype=np.float64)
                wire_force = _to_array(_read_data(wire_dofs, "force"), dtype=np.float64)
                wire_external_force = _to_array(_read_data(wire_dofs, "externalForce"), dtype=np.float64)
                wire_constraint_raw = _read_data(wire_dofs, "constraint")

                collision_pos = _to_array(_read_data(collision_dofs, "position"), dtype=np.float64)
                collision_force = _to_array(_read_data(collision_dofs, "force"), dtype=np.float64)
                collision_external_force = _to_array(
                    _read_data(collision_dofs, "externalForce"), dtype=np.float64
                )

                wire_edges = _to_array(_read_data(wire_mesh, "edges"), dtype=np.int64).reshape((-1, 2))
                collision_edges = _to_array(
                    _read_data(collision_edge_set, "edges"), dtype=np.int64
                ).reshape((-1, 2))

                collision_n_dofs = int(collision_pos.shape[0]) if collision_pos.ndim >= 2 else 0
                projected_scene = _project_lambda_dt(
                    lambda_dt=lambda_dt,
                    row_entries=row_entries,
                    n_dofs=collision_n_dofs,
                )
                projected_n = -projected_scene * float(scale_to_newton)
                wall_force_n = _to_vec3(info.get("wall_total_force_vector_N", [0.0, 0.0, 0.0]))
                projection_gap_n = float(np.linalg.norm(projected_n - wall_force_n))

                wire_node_force_xyz = (
                    wire_force[:, :3].astype(np.float64)
                    if wire_force.ndim >= 2 and wire_force.shape[1] >= 3
                    else np.zeros((0, 3), dtype=np.float64)
                )
                wire_node_force_norms = (
                    np.linalg.norm(wire_node_force_xyz, axis=1)
                    if wire_node_force_xyz.size
                    else np.zeros((0,), dtype=np.float64)
                )
                wire_node_force_max = float(np.max(wire_node_force_norms)) if wire_node_force_norms.size else 0.0

                wire_edge_force_xyz = np.zeros((0, 3), dtype=np.float64)
                if wire_edges.size and wire_node_force_xyz.size:
                    valid = (
                        (wire_edges[:, 0] >= 0)
                        & (wire_edges[:, 1] >= 0)
                        & (wire_edges[:, 0] < wire_node_force_xyz.shape[0])
                        & (wire_edges[:, 1] < wire_node_force_xyz.shape[0])
                    )
                    if np.any(valid):
                        ve = wire_edges[valid]
                        wire_edge_force_xyz = 0.5 * (
                            wire_node_force_xyz[ve[:, 0]] + wire_node_force_xyz[ve[:, 1]]
                        )
                wire_edge_force_norms = (
                    np.linalg.norm(wire_edge_force_xyz, axis=1)
                    if wire_edge_force_xyz.size
                    else np.zeros((0,), dtype=np.float64)
                )
                wire_edge_force_max = float(np.max(wire_edge_force_norms)) if wire_edge_force_norms.size else 0.0

                collision_force_xyz = (
                    collision_force[:, :3].astype(np.float64)
                    if collision_force.ndim >= 2 and collision_force.shape[1] >= 3
                    else np.zeros((0, 3), dtype=np.float64)
                )
                collision_force_norms = (
                    np.linalg.norm(collision_force_xyz, axis=1)
                    if collision_force_xyz.size
                    else np.zeros((0,), dtype=np.float64)
                )
                collision_force_max = (
                    float(np.max(collision_force_norms)) if collision_force_norms.size else 0.0
                )

                export_info = _read_exporter(sim)
                export_available = (
                    int(bool(export_info.get("available", False))) if export_info.get("exists") else 0
                )
                export_status = (
                    str(export_info.get("status", "missing")) if export_info.get("exists") else "missing"
                )

                row: Dict[str, Any] = {
                    "step": int(step_idx),
                    "time_s": float(step_idx) * float(dt_s),
                    "insert_action": float(insert_action),
                    "rotate_action": float(rotate_action),
                    "n_lcp": int(lcp_vals.size),
                    "n_constraint_rows": int(row_count),
                    "lcp_active_rows": int(lcp_active_rows),
                    "lcp_max_abs": float(lcp_max_abs),
                    "lcp_abs_sum": float(lcp_abs_sum),
                    "wall_contact_detected": int(bool(info.get("wall_contact_detected", False))),
                    "wall_contact_count": int(info.get("wall_contact_count", 0)),
                    "wall_total_force_norm_n": float(
                        _to_array(info.get("wall_total_force_norm_N", 0.0), dtype=np.float64).reshape((-1,))[0]
                        if np.asarray(info.get("wall_total_force_norm_N", 0.0)).size
                        else 0.0
                    ),
                    "wall_lcp_max_abs": float(info.get("wall_lcp_max_abs", 0.0)),
                    "quality_tier": str(info.get("wall_force_quality_tier", "unavailable")),
                    "association_method": str(info.get("wall_force_association_method", "none")),
                    "association_coverage": float(info.get("wall_force_association_coverage", float("nan"))),
                    "association_explicit_force_coverage": float(
                        info.get("wall_force_association_explicit_force_coverage", float("nan"))
                    ),
                    "active_constraint_step": int(bool(info.get("wall_force_active_constraint_step", False))),
                    "projection_gap_n": float(projection_gap_n),
                    "wire_nodes": int(wire_pos.shape[0]) if wire_pos.ndim >= 2 else 0,
                    "wire_edges": int(wire_edges.shape[0]) if wire_edges.ndim == 2 else 0,
                    "wire_node_force_max": float(wire_node_force_max),
                    "wire_edge_force_max": float(wire_edge_force_max),
                    "collision_nodes": int(collision_pos.shape[0]) if collision_pos.ndim >= 2 else 0,
                    "collision_edges": int(collision_edges.shape[0]) if collision_edges.ndim == 2 else 0,
                    "collision_force_max": float(collision_force_max),
                    "native_export_available": int(export_available),
                    "native_export_status": export_status,
                }
                writer.writerow(row)
                csv_f.flush()

                snapshot = {
                    "summary": row,
                    "lcp": {
                        "constraintForces": lcp_vals,
                        "constraintForces_dt": lambda_dt,
                        "constraintRowsParsedCount": int(row_count),
                        "projection_world_scene": projected_scene,
                        "projection_world_n": projected_n,
                        "wall_total_force_vector_n": wall_force_n,
                        "projection_gap_n": float(projection_gap_n),
                    },
                    "wire": {
                        "position": wire_pos,
                        "force": wire_force,
                        "externalForce": wire_external_force,
                        "constraint_raw": str(wire_constraint_raw or ""),
                        "edges": wire_edges,
                        "node_force_xyz": wire_node_force_xyz,
                        "edge_force_xyz": wire_edge_force_xyz,
                    },
                    "collision": {
                        "position": collision_pos,
                        "force": collision_force,
                        "externalForce": collision_external_force,
                        "constraint_raw": str(collision_constraint_raw or ""),
                        "edges": collision_edges,
                    },
                    "force_info": info,
                    "native_export": export_info,
                }
                jsonl_f.write(json.dumps(_to_jsonable(snapshot), sort_keys=True) + "\n")
                jsonl_f.flush()

                if int(args.print_every) > 0 and (step_idx % int(args.print_every) == 0):
                    print(
                        "[eval-step-debug] step={s:03d} lcp_active={la:3d}/{nl:3d} max|lambda|={lm:.6g} "
                        "contact={cd}:{cc} quality={qt} assoc={am} gapN={gap:.6g}".format(
                            s=step_idx,
                            la=int(row["lcp_active_rows"]),
                            nl=int(row["n_lcp"]),
                            lm=float(row["lcp_max_abs"]),
                            cd=int(row["wall_contact_detected"]),
                            cc=int(row["wall_contact_count"]),
                            qt=str(row["quality_tier"]),
                            am=str(row["association_method"]),
                            gap=float(row["projection_gap_n"]),
                        )
                    )
                    print(
                        "[eval-step-debug] wire pos{wps} force{wfs} edges={we} | coll pos{cps} force{cfs} edges={ce}".format(
                            wps=tuple(wire_pos.shape),
                            wfs=tuple(wire_force.shape),
                            we=int(row["wire_edges"]),
                            cps=tuple(collision_pos.shape),
                            cfs=tuple(collision_force.shape),
                            ce=int(row["collision_edges"]),
                        )
                    )
                    print(
                        "[eval-step-debug] preview lcp={} wire_force_xyz={} coll_force_xyz={}".format(
                            _preview(lcp_vals, rows=int(args.preview_rows), cols=int(args.preview_cols)),
                            _preview(wire_node_force_xyz, rows=int(args.preview_rows), cols=3),
                            _preview(collision_force_xyz, rows=int(args.preview_rows), cols=3),
                        )
                    )

                if viewer is not None:
                    try:
                        viewer.render()
                    except Exception as exc:
                        print(f"[eval-step-debug] viewer_render_error={exc}")
                        viewer = None
                        pygame_mod = None
                        use_sofa_keyboard = False

                step_idx += 1
        finally:
            if viewer is not None:
                try:
                    viewer.close()
                except Exception:
                    pass

    print(f"[eval-step-debug] finished steps={step_idx}")
    print(f"[eval-step-debug] wrote {summary_csv}")
    print(f"[eval-step-debug] wrote {snapshots_jsonl}")


if __name__ == "__main__":
    main()
