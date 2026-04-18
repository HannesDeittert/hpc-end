from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

from steve_recommender.evaluation.force_si import unit_scale_to_si_newton

from .config import ControlConfig, ForcePlaygroundConfig
from .controllers import ForceApplicator, build_controller
from .scene_factory import build_scene


def parse_constraint_rows(raw: str) -> List[Tuple[int, List[Tuple[int, np.ndarray]]]]:
    """Parse SOFA MechanicalObject.constraint text rows.

    Supports rows with variable block width:
      row_id n_blocks [point_id dirx diry dirz (...optional extra...)] * n_blocks
    """
    lines = [ln.strip() for ln in str(raw or "").splitlines() if ln.strip()]
    out: List[Tuple[int, List[Tuple[int, np.ndarray]]]] = []
    for line in lines:
        toks = line.split()
        if len(toks) < 4:
            continue
        try:
            row_id = int(float(toks[0]))
            n_blocks = int(float(toks[1]))
        except Exception:
            continue
        if n_blocks <= 0:
            continue
        payload = toks[2:]
        if not payload or len(payload) % n_blocks != 0:
            continue
        block_w = len(payload) // n_blocks
        if block_w < 4:
            continue
        entries: List[Tuple[int, np.ndarray]] = []
        for bi in range(n_blocks):
            base = bi * block_w
            block = payload[base : base + block_w]
            try:
                point_id = int(float(block[0]))
                direction = np.asarray(
                    [float(block[1]), float(block[2]), float(block[3])],
                    dtype=np.float64,
                )
            except Exception:
                continue
            if np.all(np.isfinite(direction)):
                entries.append((point_id, direction))
        out.append((row_id, entries))
    return out


def project_constraint_forces_to_world(
    *,
    lambdas: np.ndarray,
    rows: List[Tuple[int, List[Tuple[int, np.ndarray]]]],
    n_dofs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    per_dof = np.zeros((int(max(n_dofs, 0)), 3), dtype=np.float64)
    if per_dof.size == 0:
        return per_dof, np.zeros((3,), dtype=np.float64)

    for row_id, entries in rows:
        if row_id < 0 or row_id >= lambdas.shape[0]:
            continue
        lam = float(lambdas[row_id])
        for point_id, direction in entries:
            if 0 <= point_id < per_dof.shape[0]:
                per_dof[point_id] += direction * lam

    resultant_scene = per_dof.sum(axis=0)
    return per_dof, resultant_scene


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit H^T * lambda/dt projection against wall_total_force_vector_N.",
    )
    p.add_argument("--scene", choices=["plane_wall", "tube_wall"], default="plane_wall")
    p.add_argument("--probe", choices=["rigid_probe", "guidewire"], default="rigid_probe")
    p.add_argument("--mode", choices=["displacement", "open_loop_force"], default="displacement")
    p.add_argument("--tool-ref", default="ArchVarJShaped/JShaped_Default")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--friction", type=float, default=0.1)
    p.add_argument("--image-frequency-hz", type=float, default=7.5)
    p.add_argument("--alarm-distance", type=float, default=0.5)
    p.add_argument("--contact-distance", type=float, default=0.3)
    p.add_argument("--contact-epsilon", type=float, default=1e-7)
    p.add_argument("--lambda-active-eps", type=float, default=1e-6)
    p.add_argument("--min-consecutive-active-steps", type=int, default=10)
    p.add_argument("--insert-action", type=float, default=0.04)
    p.add_argument("--rotate-action", type=float, default=0.0)
    p.add_argument("--open-loop-force-n", type=float, default=1.0)
    p.add_argument("--open-loop-insert-action", type=float, default=1.0)
    p.add_argument("--open-loop-force-node-index", type=int, default=-1)
    p.add_argument("--phase-preload-steps", type=int, default=40)
    p.add_argument("--phase-hold-steps", type=int, default=20)
    p.add_argument("--phase-ramp-steps", type=int, default=40)
    p.add_argument("--phase-measure-steps", type=int, default=80)
    p.add_argument("--preload-insert-action", type=float, default=0.04)
    p.add_argument("--preload-force-n", type=float, default=0.0)
    p.add_argument("--hold-insert-action", type=float, default=0.0)
    p.add_argument("--hold-force-n", type=float, default=0.0)
    p.add_argument("--ramp-force-start-n", type=float, default=0.0)
    p.add_argument("--lcp-max-it", type=int, default=0, help="0 keeps current scene setting")
    p.add_argument("--lcp-tolerance", type=float, default=0.0, help="0 keeps current scene setting")
    p.add_argument(
        "--lcp-build-lcp",
        choices=["auto", "on", "off"],
        default="auto",
        help="Force LCP build_lcp toggle for debug runs.",
    )
    p.add_argument("--output-root", default="results/force_playground")
    p.add_argument("--run-name", default="constraint_audit")
    return p


def main() -> None:
    args = _make_parser().parse_args()
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{ts}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = ForcePlaygroundConfig(
        scene=args.scene,
        probe=args.probe,
        mode=args.mode,
        tool_ref=args.tool_ref,
        steps=1,
        seed=int(args.seed),
        friction=float(args.friction),
        image_frequency_hz=float(args.image_frequency_hz),
        alarm_distance=float(args.alarm_distance),
        contact_distance=float(args.contact_distance),
        contact_epsilon=float(args.contact_epsilon),
        show_sofa=False,
        plot=False,
        interactive=False,
        output_root=args.output_root,
        run_name=args.run_name,
        control=ControlConfig(
            insert_action=float(args.insert_action),
            rotate_action=float(args.rotate_action),
            open_loop_force_n=float(args.open_loop_force_n),
            open_loop_force_node_index=int(args.open_loop_force_node_index),
            open_loop_insert_action=float(args.open_loop_insert_action),
        ),
    )

    scene = build_scene(cfg, run_dir)
    sim = scene.simulation
    root = sim.root
    lcp = root.LCP
    collision_dofs = sim._instruments_combined.CollisionModel.CollisionDOFs
    dt_s = float(scene.dt_s)
    to_newton = float(unit_scale_to_si_newton(cfg.units))
    if int(args.lcp_max_it) > 0:
        try:
            lcp.maxIt.value = int(args.lcp_max_it)
        except Exception:
            pass
    if float(args.lcp_tolerance) > 0.0:
        try:
            lcp.tolerance.value = float(args.lcp_tolerance)
        except Exception:
            pass
    if str(args.lcp_build_lcp) != "auto":
        try:
            lcp.build_lcp.value = bool(str(args.lcp_build_lcp) == "on")
        except Exception:
            pass

    controller = build_controller(cfg, scene.wall_reference_normal)
    force_applicator = None
    if cfg.mode == "open_loop_force":
        force_applicator = ForceApplicator(
            sim,
            node_index=int(cfg.control.open_loop_force_node_index),
        )

    out_csv = run_dir / "constraint_audit.csv"
    print(f"[constraint-audit] run_dir={run_dir}")
    print(f"[constraint-audit] build_lcp={getattr(lcp.build_lcp, 'value', None)}")
    print(f"[constraint-audit] computeConstraintForces={getattr(lcp.computeConstraintForces, 'value', None)}")
    lmd = getattr(root, "localmindistance", None)
    print(
        "[constraint-audit] alarmDistance={a} contactDistance={c}".format(
            a=getattr(getattr(lmd, "alarmDistance", None), "value", None),
            c=getattr(getattr(lmd, "contactDistance", None), "value", None),
        )
    )
    print(f"[constraint-audit] lcp.maxIt={getattr(lcp.maxIt, 'value', None)}")
    print(f"[constraint-audit] lcp.tolerance={getattr(lcp.tolerance, 'value', None)}")
    print(f"[constraint-audit] dt_s={dt_s}")
    print(f"[constraint-audit] output={out_csv}")
    preload_steps = max(0, int(args.phase_preload_steps))
    hold_steps = max(0, int(args.phase_hold_steps))
    ramp_steps = max(0, int(args.phase_ramp_steps))
    measure_steps = max(0, int(args.phase_measure_steps))
    planned_steps = preload_steps + hold_steps + ramp_steps + measure_steps
    total_steps = int(args.steps) if int(args.steps) > 0 else planned_steps
    if total_steps <= 0:
        total_steps = planned_steps if planned_steps > 0 else 120
    print(
        "[constraint-audit] phases preload={a} hold={b} ramp={c} measure={d} total={t}".format(
            a=preload_steps,
            b=hold_steps,
            c=ramp_steps,
            d=measure_steps,
            t=total_steps,
        )
    )

    def phase_and_targets(step: int) -> tuple[str, float, float]:
        if step < preload_steps:
            return "preload", float(args.preload_insert_action), float(args.preload_force_n)
        if step < preload_steps + hold_steps:
            return "hold", float(args.hold_insert_action), float(args.hold_force_n)
        if step < preload_steps + hold_steps + ramp_steps:
            t = step - (preload_steps + hold_steps)
            if ramp_steps <= 1:
                alpha = 1.0
            else:
                alpha = float(t) / float(ramp_steps - 1)
            force_n = float(args.ramp_force_start_n) + alpha * (
                float(args.open_loop_force_n) - float(args.ramp_force_start_n)
            )
            return "ramp", float(args.hold_insert_action), float(force_n)
        return "measure", float(args.hold_insert_action), float(args.open_loop_force_n)

    gaps_active: List[float] = []
    active_step_count = 0
    current_consecutive = 0
    best_consecutive = 0
    best_consecutive_start = -1
    current_start = -1
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "step",
                "phase",
                "n_impulses",
                "n_rows",
                "lambda_active_rows",
                "lambda_abs_max",
                "wall_contact_detected",
                "wall_contact_count",
                "active_step",
                "consecutive_active_steps",
                "proj_fx_N",
                "proj_fy_N",
                "proj_fz_N",
                "proj_norm_N",
                "wall_fx_N",
                "wall_fy_N",
                "wall_fz_N",
                "wall_norm_N",
                "gap_N",
                "quality_tier",
                "association_method",
                "command_apply_status",
            ]
        )

        for step in range(total_steps):
            phase, insert_action, target_force_n = phase_and_targets(step)
            controller.insert_action = float(insert_action)
            if hasattr(controller, "target_force_n"):
                controller.target_force_n = float(target_force_n)

            command = controller.command(step)
            apply_status = "inactive"
            if force_applicator is not None:
                force_scene = np.asarray(command.commanded_force_vector_n, dtype=np.float64) / to_newton
                apply_status = force_applicator.apply_force_scene(force_scene.astype(np.float32))

            scene.intervention.step(np.asarray(command.action, dtype=np.float32))
            scene.force_info.step()

            info = scene.force_info.info
            impulses = np.asarray(lcp.constraintForces.value, dtype=np.float64).reshape((-1,))
            rows = parse_constraint_rows(collision_dofs.constraint.value)
            lambdas = impulses / dt_s if impulses.size else impulses

            pos = np.asarray(collision_dofs.position.value)
            n_dofs = int(pos.shape[0]) if pos.ndim >= 2 else 0
            _, projected_scene = project_constraint_forces_to_world(
                lambdas=lambdas,
                rows=rows,
                n_dofs=n_dofs,
            )

            # projected_scene is in scene units and opposite sign convention
            # to wall_total_force_vector_N.
            projected_n = -projected_scene * to_newton
            wall_n = np.asarray(
                info.get("wall_total_force_vector_N", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            ).reshape((3,))
            gap_n = float(np.linalg.norm(projected_n - wall_n))

            lambda_active_rows = int(np.count_nonzero(np.abs(impulses) > float(cfg.contact_epsilon)))
            lambda_abs_max = float(np.max(np.abs(impulses))) if impulses.size else 0.0
            wall_contact = bool(info.get("wall_contact_detected", False))
            active_step = bool(
                wall_contact
                and lambda_active_rows > 0
                and lambda_abs_max > float(args.lambda_active_eps)
            )
            if active_step:
                active_step_count += 1
                if current_consecutive == 0:
                    current_start = step
                current_consecutive += 1
                if current_consecutive > best_consecutive:
                    best_consecutive = current_consecutive
                    best_consecutive_start = current_start
            else:
                current_consecutive = 0
                current_start = -1

            if active_step:
                gaps_active.append(gap_n)

            writer.writerow(
                [
                    step,
                    phase,
                    int(impulses.shape[0]),
                    int(len(rows)),
                    lambda_active_rows,
                    lambda_abs_max,
                    int(wall_contact),
                    int(info.get("wall_contact_count", 0)),
                    int(active_step),
                    int(current_consecutive if active_step else 0),
                    float(projected_n[0]),
                    float(projected_n[1]),
                    float(projected_n[2]),
                    float(np.linalg.norm(projected_n)),
                    float(wall_n[0]),
                    float(wall_n[1]),
                    float(wall_n[2]),
                    float(np.linalg.norm(wall_n)),
                    gap_n,
                    str(info.get("wall_force_quality_tier", "")),
                    str(info.get("wall_force_association_method", "")),
                    str(apply_status),
                ]
            )

            print(
                f"[constraint-audit] step={step:03d} "
                f"phase={phase:7s} "
                f"imp={impulses.shape[0]:3d} rows={len(rows):3d} "
                f"lambda_rows={lambda_active_rows:3d} "
                f"lambda_max={lambda_abs_max:.6g} "
                f"active={int(active_step)} contact={int(wall_contact)} "
                f"gap_N={gap_n:.6g}"
            )

    if force_applicator is not None:
        force_applicator.clear()

    if gaps_active:
        arr = np.asarray(gaps_active, dtype=np.float64)
        print(
            "[constraint-audit] active_contact_gaps_N "
            f"count={arr.size} median={float(np.median(arr)):.6g} "
            f"p95={float(np.percentile(arr, 95)):.6g} max={float(np.max(arr)):.6g}"
        )
    else:
        print(
            "[constraint-audit] no active-contact steps "
            "(wall_contact && lambda_active_rows>0 && lambda_abs_max>lambda_active_eps)."
        )

    ratio = float(active_step_count) / float(total_steps) if total_steps > 0 else 0.0
    print(
        "[constraint-audit] active_summary "
        f"active_steps={active_step_count}/{total_steps} ({ratio:.2%}) "
        f"best_consecutive={best_consecutive} start={best_consecutive_start} "
        f"meets_min_consecutive={best_consecutive >= int(args.min_consecutive_active_steps)}"
    )


if __name__ == "__main__":
    main()
