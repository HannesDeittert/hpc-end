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
    """Parse SOFA MechanicalObject.constraint rows.

    Row format is expected as:
      row_id n_blocks [point_id dir_x dir_y dir_z (...optional extra...)] * n_blocks
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


def project_ht_lambda(
    *,
    lambda_force: np.ndarray,
    rows: List[Tuple[int, List[Tuple[int, np.ndarray]]]],
    n_dofs: int,
) -> np.ndarray:
    """Compute per-dof projected world force from H^T * lambda_force."""
    per_dof = np.zeros((int(max(n_dofs, 0)), 3), dtype=np.float64)
    if per_dof.size == 0:
        return per_dof

    for row_id, entries in rows:
        if row_id < 0 or row_id >= lambda_force.shape[0]:
            continue
        lam = float(lambda_force[row_id])
        for point_id, direction in entries:
            if 0 <= point_id < per_dof.shape[0]:
                per_dof[point_id] += direction * lam
    return per_dof


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Minimal maintainer-style check for wall+wire scene. "
            "Computes H^T*(constraintForces/dt) and compares with wall_total_force_vector_N."
        )
    )
    p.add_argument("--scene", choices=["plane_wall", "tube_wall"], default="plane_wall")
    p.add_argument("--probe", choices=["rigid_probe", "guidewire"], default="rigid_probe")
    p.add_argument("--mode", choices=["displacement", "open_loop_force"], default="open_loop_force")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--image-frequency-hz", type=float, default=7.5)
    p.add_argument("--friction", type=float, default=0.1)
    p.add_argument("--alarm-distance", type=float, default=0.5)
    p.add_argument("--contact-distance", type=float, default=0.3)
    p.add_argument("--contact-epsilon", type=float, default=1e-7)
    p.add_argument("--lambda-active-eps", type=float, default=1e-6)
    p.add_argument("--insert-action", type=float, default=1.0)
    p.add_argument("--open-loop-force-n", type=float, default=1.0)
    p.add_argument("--open-loop-force-node-index", type=int, default=-1)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--output-root", default="results/force_playground")
    p.add_argument("--run-name", default="simple_contact_forces")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{ts}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_csv = run_dir / "simple_contact_forces.csv"

    cfg = ForcePlaygroundConfig(
        scene=str(args.scene),
        probe=str(args.probe),
        mode=str(args.mode),
        steps=1,
        seed=int(args.seed),
        image_frequency_hz=float(args.image_frequency_hz),
        friction=float(args.friction),
        alarm_distance=float(args.alarm_distance),
        contact_distance=float(args.contact_distance),
        contact_epsilon=float(args.contact_epsilon),
        plot=False,
        interactive=False,
        show_sofa=False,
        output_root=str(args.output_root),
        run_name=str(args.run_name),
        control=ControlConfig(
            insert_action=float(args.insert_action),
            rotate_action=0.0,
            open_loop_force_n=float(args.open_loop_force_n),
            open_loop_force_node_index=int(args.open_loop_force_node_index),
            open_loop_insert_action=float(args.insert_action),
        ),
    )

    scene = build_scene(cfg, run_dir)
    sim = scene.simulation
    root = sim.root
    lcp = root.LCP
    collision_dofs = sim._instruments_combined.CollisionModel.CollisionDOFs
    dt_s = float(scene.dt_s)
    to_newton = float(unit_scale_to_si_newton(cfg.units))

    try:
        lcp.computeConstraintForces.value = True
    except Exception:
        pass

    controller = build_controller(cfg, scene.wall_reference_normal)
    controller.insert_action = float(args.insert_action)
    if hasattr(controller, "target_force_n"):
        controller.target_force_n = float(args.open_loop_force_n)

    force_applicator = None
    if cfg.mode == "open_loop_force":
        force_applicator = ForceApplicator(
            sim,
            node_index=int(cfg.control.open_loop_force_node_index),
        )

    print(f"[simple-contact] run_dir={run_dir}")
    print(f"[simple-contact] dt_s={dt_s}")
    print(
        "[simple-contact] solver class={cls} computeConstraintForces={ccf}".format(
            cls=lcp.getClassName(),
            ccf=getattr(getattr(lcp, "computeConstraintForces", None), "value", None),
        )
    )
    print(f"[simple-contact] output={out_csv}")

    active_steps = 0
    active_gaps: List[float] = []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "step",
                "n_impulses",
                "n_rows",
                "active_rows",
                "max_abs_lambda_over_dt",
                "proj_fx_N",
                "proj_fy_N",
                "proj_fz_N",
                "proj_norm_N",
                "wall_fx_N",
                "wall_fy_N",
                "wall_fz_N",
                "wall_norm_N",
                "gap_N",
                "contact_detected",
                "apply_status",
            ]
        )

        for step in range(int(args.steps)):
            cmd = controller.command(step + 1)
            apply_status = "inactive"
            if force_applicator is not None:
                force_scene = np.asarray(cmd.commanded_force_vector_n, dtype=np.float64) / to_newton
                apply_status = force_applicator.apply_force_scene(force_scene.astype(np.float32))

            scene.intervention.step(np.asarray(cmd.action, dtype=np.float32))
            scene.force_info.step()

            impulses = np.asarray(getattr(lcp.constraintForces, "value", []), dtype=np.float64).reshape((-1,))
            rows = parse_constraint_rows(getattr(collision_dofs.constraint, "value", ""))

            lambda_force = impulses / dt_s if (impulses.size and dt_s > 0.0) else impulses
            n_dofs = int(len(np.asarray(getattr(collision_dofs.position, "value", []))))
            per_dof = project_ht_lambda(lambda_force=lambda_force, rows=rows, n_dofs=n_dofs)

            # Reaction on wall is opposite sign of summed probe-side projection.
            proj_wall_N = -per_dof.sum(axis=0) * to_newton if per_dof.size else np.zeros((3,), dtype=np.float64)
            proj_norm_N = float(np.linalg.norm(proj_wall_N))

            info = scene.force_info.info
            wall_N = np.asarray(info.get("wall_total_force_vector_N", [0.0, 0.0, 0.0]), dtype=np.float64)
            wall_norm_N = float(np.linalg.norm(wall_N))

            if lambda_force.size:
                active_rows = int(np.count_nonzero(np.abs(lambda_force) > float(args.lambda_active_eps)))
                max_abs_lambda_over_dt = float(np.max(np.abs(lambda_force)))
            else:
                active_rows = 0
                max_abs_lambda_over_dt = 0.0

            gap_N = float(np.linalg.norm(proj_wall_N - wall_N))
            is_active = active_rows > 0 and max_abs_lambda_over_dt > float(args.lambda_active_eps)
            if is_active:
                active_steps += 1
                active_gaps.append(gap_N)

            if int(args.print_every) > 0 and (step % int(args.print_every) == 0):
                print(
                    "[simple-contact] step={s:03d} imp={i:3d} rows={r:3d} active={a:3d} "
                    "proj|F|={pn:.6g}N wall|F|={wn:.6g}N gap={g:.6g}N contact={c}".format(
                        s=step,
                        i=int(impulses.size),
                        r=int(len(rows)),
                        a=active_rows,
                        pn=proj_norm_N,
                        wn=wall_norm_N,
                        g=gap_N,
                        c=int(bool(info.get("wall_contact_detected", False))),
                    )
                )

            writer.writerow(
                [
                    int(step),
                    int(impulses.size),
                    int(len(rows)),
                    int(active_rows),
                    float(max_abs_lambda_over_dt),
                    float(proj_wall_N[0]),
                    float(proj_wall_N[1]),
                    float(proj_wall_N[2]),
                    float(proj_norm_N),
                    float(wall_N[0]),
                    float(wall_N[1]),
                    float(wall_N[2]),
                    float(wall_norm_N),
                    float(gap_N),
                    int(bool(info.get("wall_contact_detected", False))),
                    str(apply_status),
                ]
            )

    if force_applicator is not None:
        force_applicator.clear()

    print(
        "[simple-contact] summary active_steps={a}/{n}".format(
            a=int(active_steps),
            n=int(args.steps),
        )
    )
    if active_gaps:
        arr = np.asarray(active_gaps, dtype=np.float64)
        print(
            "[simple-contact] active_gap_N median={m:.6g} p95={p:.6g} max={x:.6g}".format(
                m=float(np.median(arr)),
                p=float(np.percentile(arr, 95.0)),
                x=float(np.max(arr)),
            )
        )
    else:
        print("[simple-contact] no active lambda steps.")
    print(f"[simple-contact] wrote {out_csv}")


if __name__ == "__main__":
    main()
