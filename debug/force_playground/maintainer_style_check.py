from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from steve_recommender.evaluation.force_si import unit_scale_to_si_newton

from .config import ControlConfig, ForcePlaygroundConfig
from .controllers import ForceApplicator, build_controller
from .scene_factory import build_scene


def _read_data(obj: object, name: str):
    try:
        data = getattr(obj, name)
    except Exception:
        return None
    if hasattr(data, "value"):
        try:
            return data.value
        except Exception:
            return None
    return data


def _parse_constraint_rows(raw: Any) -> List[Tuple[int, int, np.ndarray]]:
    """Parse MechanicalObject.constraint rows into (row_idx, dof_idx, coeff_xyz)."""
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


def _project_lambda_dt_to_world(
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Maintainer-style force check: print LCP constraintForces activity "
            "and constraint rows over time for the playground scene."
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

    p.add_argument("--insert-action", type=float, default=1.0)
    p.add_argument("--open-loop-force-n", type=float, default=1.0)
    p.add_argument("--open-loop-force-node-index", type=int, default=-1)

    p.add_argument("--output-root", default="results/force_playground")
    p.add_argument("--run-name", default="maintainer_style_check")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{ts}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

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
    controller = build_controller(cfg, scene.wall_reference_normal)
    controller.insert_action = float(args.insert_action)
    if hasattr(controller, "target_force_n"):
        controller.target_force_n = float(args.open_loop_force_n)

    lcp = sim.root.LCP
    collision_dofs = sim._instruments_combined.CollisionModel.CollisionDOFs
    wire_dofs = sim._instruments_combined.DOFs
    dt_s = float(scene.dt_s)
    force_scale = float(unit_scale_to_si_newton(cfg.units))

    force_applicator = None
    if cfg.mode == "open_loop_force":
        force_applicator = ForceApplicator(
            sim,
            node_index=int(cfg.control.open_loop_force_node_index),
        )

    out_csv = run_dir / "maintainer_style_steps.csv"
    print(f"[maintainer-check] run_dir={run_dir}")
    print(f"[maintainer-check] dt_s={dt_s}")
    print(f"[maintainer-check] computeConstraintForces={getattr(lcp.computeConstraintForces, 'value', None)}")
    print(
        "[maintainer-check] lambda_data_available "
        f"collision={_read_data(collision_dofs, 'lambda') is not None} "
        f"wire={_read_data(wire_dofs, 'lambda') is not None}"
    )
    print(f"[maintainer-check] output={out_csv}")

    active_steps = 0
    best_consecutive = 0
    current_consecutive = 0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "step",
                "n_constraint_forces",
                "n_constraint_rows",
                "active_rows",
                "max_abs_lambda",
                "sum_abs_lambda",
                "sum_abs_lambda_dt",
                "proj_fx_n",
                "proj_fy_n",
                "proj_fz_n",
                "proj_norm_n",
                "wall_fx_n",
                "wall_fy_n",
                "wall_fz_n",
                "wall_norm_n",
                "proj_wall_gap_n",
                "wall_contact_detected",
                "wall_contact_count",
                "quality_tier",
                "association_method",
                "association_coverage",
                "association_explicit_force_coverage",
                "force_channel",
                "active_constraint_step",
                "gap_dominant_class",
                "command_apply_status",
            ]
        )
        for step in range(int(args.steps)):
            cmd = controller.command(step + 1)
            apply_status = "inactive"
            if force_applicator is not None:
                force_scene = np.asarray(cmd.commanded_force_vector_n, dtype=np.float64) / force_scale
                apply_status = force_applicator.apply_force_scene(force_scene.astype(np.float32))
            scene.intervention.step(np.asarray(cmd.action, dtype=np.float32))
            scene.force_info.step()
            info = scene.force_info.info

            lcp_raw = _read_data(lcp, "constraintForces")
            impulses = np.asarray(lcp_raw, dtype=np.float64).reshape((-1,)) if lcp_raw is not None else np.zeros((0,))
            lambda_dt = impulses / dt_s if impulses.size and dt_s > 0.0 else np.zeros((0,), dtype=np.float64)
            constraint_raw = _read_data(collision_dofs, "constraint")
            row_entries = _parse_constraint_rows(constraint_raw)
            n_rows = len({int(row_idx) for row_idx, _dof_idx, _coeff in row_entries})

            active_rows = int(np.count_nonzero(np.abs(impulses) > float(cfg.contact_epsilon)))
            max_abs_lambda = float(np.max(np.abs(impulses))) if impulses.size else 0.0
            sum_abs_lambda = float(np.sum(np.abs(impulses))) if impulses.size else 0.0
            sum_abs_lambda_dt = float(sum_abs_lambda / dt_s) if dt_s > 0.0 else float("nan")

            pos = np.asarray(_read_data(collision_dofs, "position"))
            n_dofs = int(pos.shape[0]) if pos.ndim >= 2 else 0
            projected_scene = _project_lambda_dt_to_world(
                lambda_dt=lambda_dt,
                row_entries=row_entries,
                n_dofs=n_dofs,
            )
            # projection sign is opposite to wall reaction convention in wall_total_force_vector_N.
            projected_n = (-projected_scene * force_scale).reshape((3,))
            wall_n = np.asarray(
                info.get("wall_total_force_vector_N", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            ).reshape((3,))
            proj_wall_gap_n = float(np.linalg.norm(projected_n - wall_n))

            quality_tier = str(info.get("wall_force_quality_tier", "unavailable"))
            association_method = str(info.get("wall_force_association_method", "none"))
            association_coverage = float(info.get("wall_force_association_coverage", float("nan")))
            association_explicit_force_coverage = float(
                info.get("wall_force_association_explicit_force_coverage", float("nan"))
            )
            force_channel = str(info.get("wall_force_channel", "none"))
            wall_contact_detected = bool(info.get("wall_contact_detected", False))
            wall_contact_count = int(info.get("wall_contact_count", 0))
            active_constraint_step = bool(
                info.get(
                    "wall_force_active_constraint_step",
                    active_rows > 0 and max_abs_lambda > float(cfg.contact_epsilon),
                )
            )
            gap_dominant_class = str(info.get("wall_force_gap_dominant_class", "none"))

            is_active = active_constraint_step
            if is_active:
                active_steps += 1
                current_consecutive += 1
                if current_consecutive > best_consecutive:
                    best_consecutive = current_consecutive
            else:
                current_consecutive = 0

            print(
                f"[maintainer-check] step={step:03d} "
                f"imp={impulses.size:3d} rows={n_rows:3d} "
                f"active_rows={active_rows:3d} "
                f"max|lambda|={max_abs_lambda:.6g} "
                f"sum|lambda/dt|={sum_abs_lambda_dt:.6g} "
                f"gapN={proj_wall_gap_n:.6g} "
                f"quality={quality_tier} assoc={association_method}"
            )

            writer.writerow(
                [
                    step,
                    int(impulses.size),
                    int(n_rows),
                    active_rows,
                    max_abs_lambda,
                    sum_abs_lambda,
                    sum_abs_lambda_dt,
                    float(projected_n[0]),
                    float(projected_n[1]),
                    float(projected_n[2]),
                    float(np.linalg.norm(projected_n)),
                    float(wall_n[0]),
                    float(wall_n[1]),
                    float(wall_n[2]),
                    float(np.linalg.norm(wall_n)),
                    proj_wall_gap_n,
                    int(wall_contact_detected),
                    wall_contact_count,
                    quality_tier,
                    association_method,
                    association_coverage,
                    association_explicit_force_coverage,
                    force_channel,
                    int(active_constraint_step),
                    gap_dominant_class,
                    apply_status,
                ]
            )

    if force_applicator is not None:
        force_applicator.clear()

    print(
        "[maintainer-check] summary "
        f"active_steps={active_steps}/{int(args.steps)} "
        f"best_consecutive={best_consecutive}"
    )


if __name__ == "__main__":
    main()
