from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from steve_recommender.evaluation.force_si import unit_scale_to_si_newton

try:
    # Package/module execution: python -m debug.force_playground.maintainer_wire_simple
    from .config import ControlConfig, ForcePlaygroundConfig
    from .scene_factory import build_scene
except ImportError:
    # Direct script execution: python debug/force_playground/maintainer_wire_simple.py
    from debug.force_playground.config import ControlConfig, ForcePlaygroundConfig
    from debug.force_playground.scene_factory import build_scene


def _read_data(obj: Any, name: str) -> Any:
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


def _normalize_xyz_array(raw: Any) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.ndim == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.ndim == 1:
        if arr.size % 6 == 0:
            arr = arr.reshape((-1, 6))[:, :3]
        elif arr.size % 3 == 0:
            arr = arr.reshape((-1, 3))
        else:
            return np.zeros((0, 3), dtype=np.float64)
    elif arr.ndim >= 2:
        if arr.shape[1] < 3:
            return np.zeros((0, 3), dtype=np.float64)
        arr = arr[:, :3]
    return np.asarray(arr, dtype=np.float64)


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


def _project_lambda_dt_to_dofs(
    *,
    lambda_dt: np.ndarray,
    row_entries: List[Tuple[int, int, np.ndarray]],
    n_dofs: int,
) -> np.ndarray:
    if int(n_dofs) <= 0 or lambda_dt.size == 0 or not row_entries:
        return np.zeros((0, 3), dtype=np.float64)
    per_dof = np.zeros((int(n_dofs), 3), dtype=np.float64)
    for row_idx, dof_idx, coeff in row_entries:
        if row_idx < 0 or row_idx >= lambda_dt.shape[0]:
            continue
        if dof_idx < 0 or dof_idx >= per_dof.shape[0]:
            continue
        per_dof[dof_idx] += coeff * float(lambda_dt[row_idx])
    return per_dof


def _pick_lambda_xyz(sim: Any) -> Tuple[str, np.ndarray]:
    """Pick first available xyz lambda array from collision/wire DOFs."""
    candidates: List[Tuple[str, Any]] = []
    try:
        collision_dofs = sim._instruments_combined.CollisionModel.CollisionDOFs  # noqa: SLF001
        candidates.append(("collision.lambda", _read_data(collision_dofs, "lambda")))
    except Exception:
        pass
    try:
        wire_dofs = sim._instruments_combined.DOFs  # noqa: SLF001
        candidates.append(("wire.lambda", _read_data(wire_dofs, "lambda")))
    except Exception:
        pass

    for name, raw in candidates:
        arr = _normalize_xyz_array(raw)
        if arr.size:
            return name, arr
    return "none", np.zeros((0, 3), dtype=np.float64)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Minimal maintainer-style force debug for real wire + vascular scene. "
            "Prints LCP constraintForces and simple xyz force views."
        )
    )
    p.add_argument("--scene", choices=["plane_wall", "tube_wall"], default="plane_wall")
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--tool-ref", default="ArchVarJShaped/JShaped_Default")
    p.add_argument("--image-frequency-hz", type=float, default=7.5)
    p.add_argument("--friction", type=float, default=0.1)
    p.add_argument("--alarm-distance", type=float, default=0.5)
    p.add_argument("--contact-distance", type=float, default=0.3)
    p.add_argument("--contact-epsilon", type=float, default=1e-7)
    p.add_argument("--insert-action", type=float, default=0.2)
    p.add_argument("--rotate-action", type=float, default=0.0)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--output-root", default="results/force_playground")
    p.add_argument("--run-name", default="maintainer_wire_simple")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{ts}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_csv = run_dir / "maintainer_wire_simple.csv"

    cfg = ForcePlaygroundConfig(
        scene=str(args.scene),
        probe="guidewire",
        mode="displacement",
        tool_ref=str(args.tool_ref),
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
            rotate_action=float(args.rotate_action),
            open_loop_force_n=0.0,
            open_loop_force_node_index=-1,
            open_loop_insert_action=float(args.insert_action),
        ),
    )

    scene = build_scene(cfg, run_dir)
    sim = scene.simulation
    lcp = sim.root.LCP
    collision_dofs = sim._instruments_combined.CollisionModel.CollisionDOFs  # noqa: SLF001
    dt_s = float(scene.dt_s)
    force_scale_to_n = float(unit_scale_to_si_newton(cfg.units))
    action = np.asarray(
        [[float(args.insert_action), float(args.rotate_action)]], dtype=np.float32
    )

    try:
        lcp.computeConstraintForces.value = True
    except Exception:
        pass

    print(f"[wire-simple] run_dir={run_dir}")
    print(f"[wire-simple] dt_s={dt_s}")
    print(f"[wire-simple] tool_ref={cfg.tool_ref}")
    print(
        "[wire-simple] solver={cls} computeConstraintForces={ccf}".format(
            cls=lcp.getClassName(),
            ccf=getattr(getattr(lcp, "computeConstraintForces", None), "value", None),
        )
    )
    print(f"[wire-simple] output={out_csv}")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "step",
                "sim_time_s",
                "lambda_size",
                "active_rows",
                "lambda_abs_max",
                "proj_wall_fx_N",
                "proj_wall_fy_N",
                "proj_wall_fz_N",
                "proj_wall_norm_N",
                "lambda_xyz_source",
                "lambda_xyz_norm_sum",
                "wall_fx_N",
                "wall_fy_N",
                "wall_fz_N",
                "wall_norm_N",
                "contact_detected",
                "contact_count",
            ]
        )

        for step in range(int(args.steps)):
            scene.intervention.step(action)
            scene.force_info.step()
            info = scene.force_info.info

            lcp_raw = _read_data(lcp, "constraintForces")
            impulses = (
                np.asarray(lcp_raw, dtype=np.float64).reshape((-1,))
                if lcp_raw is not None
                else np.zeros((0,), dtype=np.float64)
            )
            lambda_dt = impulses / dt_s if impulses.size and dt_s > 0.0 else np.zeros((0,))
            active_rows = int(
                np.count_nonzero(np.abs(impulses) > float(cfg.contact_epsilon))
            )
            lambda_abs_max = float(np.max(np.abs(impulses))) if impulses.size else 0.0

            constraint_raw = _read_data(collision_dofs, "constraint")
            row_entries = _parse_constraint_rows(constraint_raw)
            positions = _normalize_xyz_array(_read_data(collision_dofs, "position"))
            proj_dofs_scene = _project_lambda_dt_to_dofs(
                lambda_dt=lambda_dt,
                row_entries=row_entries,
                n_dofs=int(positions.shape[0]),
            )
            proj_wall_N = (
                -np.sum(proj_dofs_scene, axis=0, dtype=np.float64) * force_scale_to_n
                if proj_dofs_scene.size
                else np.zeros((3,), dtype=np.float64)
            )
            proj_wall_norm_N = float(np.linalg.norm(proj_wall_N))

            lambda_xyz_source, lambda_xyz = _pick_lambda_xyz(sim)
            lambda_xyz_norm_sum = (
                float(np.linalg.norm(lambda_xyz, axis=1).sum()) if lambda_xyz.size else 0.0
            )

            wall_N = np.asarray(
                info.get("wall_total_force_vector_N", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            ).reshape((3,))
            wall_norm_N = float(np.linalg.norm(wall_N))
            sim_time = float(getattr(sim.root.time, "value", float("nan")))

            if int(args.print_every) > 0 and (step % int(args.print_every) == 0):
                if impulses.size > 0:
                    print(
                        f"[wire-simple] t={sim_time:.3f} lambda(n,t1,t2,...)={impulses}"
                    )
                else:
                    print(f"[wire-simple] t={sim_time:.3f} lambda=none")
                print(
                    "[wire-simple] xyz projected(H^T*lambda/dt) wall-reaction N="
                    f"{proj_wall_N} |norm|={proj_wall_norm_N:.6g}"
                )
                print(
                    "[wire-simple] xyz lambda source={src} sum_norm={sn:.6g}".format(
                        src=lambda_xyz_source,
                        sn=lambda_xyz_norm_sum,
                    )
                )
                print(
                    "[wire-simple] wall_total_force_vector_N={w} |norm|={n:.6g} "
                    "contact={c} count={k}".format(
                        w=wall_N,
                        n=wall_norm_N,
                        c=int(bool(info.get("wall_contact_detected", False))),
                        k=int(info.get("wall_contact_count", 0)),
                    )
                )

            writer.writerow(
                [
                    int(step),
                    sim_time,
                    int(impulses.size),
                    int(active_rows),
                    float(lambda_abs_max),
                    float(proj_wall_N[0]),
                    float(proj_wall_N[1]),
                    float(proj_wall_N[2]),
                    float(proj_wall_norm_N),
                    str(lambda_xyz_source),
                    float(lambda_xyz_norm_sum),
                    float(wall_N[0]),
                    float(wall_N[1]),
                    float(wall_N[2]),
                    float(wall_norm_N),
                    int(bool(info.get("wall_contact_detected", False))),
                    int(info.get("wall_contact_count", 0)),
                ]
            )

    print(f"[wire-simple] wrote {out_csv}")


if __name__ == "__main__":
    main()
