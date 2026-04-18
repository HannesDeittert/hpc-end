from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from steve_recommender.adapters import eve
from steve_recommender.devices import make_device
from steve_recommender.evaluation.config import ForceUnitsConfig
from steve_recommender.evaluation.force_si import unit_scale_to_si_newton
from steve_recommender.evaluation.info_collectors import SofaWallForceInfo


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


def _normalize_xyz_array(raw: Any, *, allow_6dof: bool = False) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.size == 0 or arr.ndim == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.ndim == 1:
        if allow_6dof and arr.size % 6 == 0:
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Keyboard-driven single-wire debug scene (single_jwire-like) with "
            "per-step tip force logging."
        )
    )
    p.add_argument("--tool-ref", default="ArchVarJShaped/JShaped_Default")
    p.add_argument("--arch-type", choices=["I", "II", "IV", "V", "VI", "VII"], default="I")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--friction", type=float, default=0.1)
    p.add_argument("--image-frequency-hz", type=float, default=7.5)
    p.add_argument("--image-rot-z-deg", type=float, default=20.0)
    p.add_argument("--image-rot-x-deg", type=float, default=5.0)
    p.add_argument("--target-threshold-mm", type=float, default=5.0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--insert-action", type=float, default=50.0)
    p.add_argument("--rotate-action", type=float, default=3.14)
    p.add_argument(
        "--force-mode",
        choices=["passive", "intrusive_lcp", "constraint_projected_si_validated"],
        default="passive",
    )
    p.add_argument("--plugin-path", default=None)
    p.add_argument("--contact-epsilon", type=float, default=1e-7)
    p.add_argument("--alarm-distance", type=float, default=0.5)
    p.add_argument("--contact-distance", type=float, default=0.3)
    p.add_argument(
        "--tip-index-mode",
        choices=["nearest", "first", "last"],
        default="nearest",
        help=(
            "How to map wire force rows to the tip. "
            "'nearest' maps by nearest wire node to fluoroscopy tip."
        ),
    )
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--output-root", default="results/force_playground")
    p.add_argument("--run-name", default="single_jwire_tip_force_debug")
    return p


def _build_intervention(args: argparse.Namespace) -> tuple[Any, Any]:
    arch_type = getattr(eve.intervention.vesseltree.ArchType, str(args.arch_type))
    vessel_tree = eve.intervention.vesseltree.AorticArch(
        arch_type=arch_type,
        seed=int(args.seed),
    )
    device = make_device(str(args.tool_ref))
    simulation = eve.intervention.simulation.SofaBeamAdapter(
        friction=float(args.friction),
    )
    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=float(args.image_frequency_hz),
        image_rot_zx=[float(args.image_rot_z_deg), float(args.image_rot_x_deg)],
    )
    target = eve.intervention.target.CenterlineRandom(
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
        threshold=float(args.target_threshold_mm),
        branches=["lcca", "rcca", "lsa", "rsa", "bct", "co"],
    )
    intervention = eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
    )
    intervention.make_non_mp()
    return intervention, simulation


def _configure_detection_distances(sim: Any, *, alarm_distance: float, contact_distance: float) -> None:
    lmd = getattr(getattr(sim, "root", None), "localmindistance", None)
    if lmd is None:
        return
    try:
        lmd.alarmDistance.value = float(alarm_distance)
    except Exception:
        pass
    try:
        lmd.contactDistance.value = float(contact_distance)
    except Exception:
        pass


def _extract_tip_force(
    *,
    intervention: Any,
    sim: Any,
    info: dict[str, Any],
    tip_index_mode: str,
    force_scale_to_n: float,
) -> tuple[np.ndarray, int, str]:
    wire_force_raw = info.get("wall_wire_force_vectors", np.zeros((0, 3), dtype=np.float32))
    wire_force = _normalize_xyz_array(wire_force_raw, allow_6dof=False)
    if wire_force.shape[0] <= 0:
        return np.zeros((3,), dtype=np.float64), -1, "missing_wire_force_vectors"

    # If already converted in collector, do not convert again.
    if bool(info.get("unit_converted_si", False)):
        wire_force_n = wire_force
    else:
        wire_force_n = np.asarray(wire_force, dtype=np.float64) * float(force_scale_to_n)

    if tip_index_mode == "first":
        idx = 0
        return wire_force_n[idx], idx, "first"
    if tip_index_mode == "last":
        idx = wire_force_n.shape[0] - 1
        return wire_force_n[idx], idx, "last"

    # nearest mode: use fluoroscopy tip and wire DOF positions when available.
    tip_pos = np.zeros((0,), dtype=np.float64)
    try:
        tip_pos = np.asarray(intervention.fluoroscopy.tracking3d[0], dtype=np.float64).reshape((3,))
    except Exception:
        pass

    wire_pos = np.zeros((0, 3), dtype=np.float64)
    try:
        wire_dofs = sim._instruments_combined.DOFs  # noqa: SLF001
        wire_pos = _normalize_xyz_array(_read_data(wire_dofs, "position"), allow_6dof=True)
    except Exception:
        wire_pos = np.zeros((0, 3), dtype=np.float64)

    n = int(min(wire_force_n.shape[0], wire_pos.shape[0]))
    if n > 0 and tip_pos.shape == (3,) and np.all(np.isfinite(tip_pos)):
        d = wire_pos[:n] - tip_pos.reshape((1, 3))
        idx = int(np.argmin(np.sum(d * d, axis=1)))
        return wire_force_n[idx], idx, "nearest_wire_dof"

    idx = wire_force_n.shape[0] - 1
    return wire_force_n[idx], idx, "last_fallback"


def main() -> None:
    args = _build_parser().parse_args()

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{ts}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_csv = run_dir / "single_jwire_tip_force_debug.csv"

    intervention, sim = _build_intervention(args)
    visualisation = eve.visualisation.SofaPygame(intervention=intervention)

    units = ForceUnitsConfig(length_unit="mm", mass_unit="kg", time_unit="s")
    force_scale_to_n = float(unit_scale_to_si_newton(units))
    force_info = SofaWallForceInfo(
        intervention=intervention,
        mode=str(args.force_mode),
        required=False,
        contact_epsilon=float(args.contact_epsilon),
        plugin_path=args.plugin_path,
        units=units,
        constraint_dt_s=1.0 / float(args.image_frequency_hz),
    )

    print(f"[tip-debug] run_dir={run_dir}")
    print(
        "[tip-debug] tool={tool} arch={arch} force_mode={mode} tip_index_mode={tim}".format(
            tool=args.tool_ref,
            arch=args.arch_type,
            mode=args.force_mode,
            tim=args.tip_index_mode,
        )
    )
    print(
        "[tip-debug] keys: arrows=wire | r+w/a/s/d=camera rotate | w/a/s/d=camera pan | e/q=zoom | Enter=reset | Esc=quit"
    )
    print(f"[tip-debug] output={out_csv}")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "step",
                "sim_time_s",
                "action_insert",
                "action_rotate",
                "tip_force_x_N",
                "tip_force_y_N",
                "tip_force_z_N",
                "tip_force_norm_N",
                "tip_force_wire_index",
                "tip_force_source",
                "wall_total_force_norm_N",
                "wall_contact_count",
                "wall_contact_detected",
                "wall_force_quality_tier",
                "wall_force_channel",
            ]
        )

        try:
            intervention.reset(int(args.seed))
            _configure_detection_distances(
                sim,
                alarm_distance=float(args.alarm_distance),
                contact_distance=float(args.contact_distance),
            )
            visualisation.reset(episode_nr=0)
            force_info.reset(episode_nr=0)

            step = 0
            while step < int(args.max_steps):
                trans = 0.0
                rot = 0.0
                camera_trans = np.array([0.0, 0.0, 0.0], dtype=np.float64)

                pygame.event.get()
                keys = pygame.key.get_pressed()

                if keys[pygame.K_ESCAPE]:
                    break
                if keys[pygame.K_UP]:
                    trans += float(args.insert_action)
                if keys[pygame.K_DOWN]:
                    trans -= float(args.insert_action)
                if keys[pygame.K_LEFT]:
                    rot += float(args.rotate_action)
                if keys[pygame.K_RIGHT]:
                    rot -= float(args.rotate_action)

                if keys[pygame.K_r]:
                    lao_rao = 0.0
                    cra_cau = 0.0
                    if keys[pygame.K_d]:
                        lao_rao += 10.0
                    if keys[pygame.K_a]:
                        lao_rao -= 10.0
                    if keys[pygame.K_w]:
                        cra_cau -= 10.0
                    if keys[pygame.K_s]:
                        cra_cau += 10.0
                    visualisation.rotate(lao_rao, cra_cau)
                else:
                    if keys[pygame.K_w]:
                        camera_trans += np.array([0.0, 0.0, 200.0], dtype=np.float64)
                    if keys[pygame.K_s]:
                        camera_trans -= np.array([0.0, 0.0, 200.0], dtype=np.float64)
                    if keys[pygame.K_a]:
                        camera_trans -= np.array([200.0, 0.0, 0.0], dtype=np.float64)
                    if keys[pygame.K_d]:
                        camera_trans += np.array([200.0, 0.0, 0.0], dtype=np.float64)
                    if np.any(camera_trans):
                        visualisation.translate(camera_trans)
                if keys[pygame.K_e]:
                    visualisation.zoom(1000.0)
                if keys[pygame.K_q]:
                    visualisation.zoom(-1000.0)

                action = (trans, rot)
                intervention.step(action)
                force_info.step()
                info = force_info.info

                tip_vec_n, tip_idx, tip_src = _extract_tip_force(
                    intervention=intervention,
                    sim=sim,
                    info=info,
                    tip_index_mode=str(args.tip_index_mode),
                    force_scale_to_n=force_scale_to_n,
                )
                tip_norm_n = float(np.linalg.norm(tip_vec_n))

                wall_total_norm_n = float(info.get("wall_total_force_norm_N", np.nan))
                contact_count = int(info.get("wall_contact_count", 0) or 0)
                contact_detected = bool(info.get("wall_contact_detected", False))
                quality_tier = str(info.get("wall_force_quality_tier", ""))
                force_channel = str(info.get("wall_force_channel", ""))

                sim_time = float(getattr(getattr(sim.root, "time", None), "value", float(step)))
                writer.writerow(
                    [
                        int(step),
                        sim_time,
                        float(trans),
                        float(rot),
                        float(tip_vec_n[0]),
                        float(tip_vec_n[1]),
                        float(tip_vec_n[2]),
                        float(tip_norm_n),
                        int(tip_idx),
                        tip_src,
                        wall_total_norm_n,
                        contact_count,
                        int(contact_detected),
                        quality_tier,
                        force_channel,
                    ]
                )

                if int(args.print_every) > 0 and (step % int(args.print_every) == 0):
                    print(
                        "[tip-debug] step={s} t={t:.3f} action=({a0:.3f},{a1:.3f}) "
                        "F_tip_N={f} |F_tip|={fn:.6g} idx={idx} src={src} "
                        "wall|F|={wn:.6g} contact={c} tier={tier} ch={ch}".format(
                            s=int(step),
                            t=sim_time,
                            a0=float(trans),
                            a1=float(rot),
                            f=np.asarray(tip_vec_n, dtype=np.float64).round(6).tolist(),
                            fn=tip_norm_n,
                            idx=int(tip_idx),
                            src=tip_src,
                            wn=wall_total_norm_n,
                            c=int(contact_detected),
                            tier=quality_tier,
                            ch=force_channel,
                        )
                    )

                visualisation.render()
                step += 1

                if keys[pygame.K_RETURN]:
                    intervention.reset(int(args.seed))
                    _configure_detection_distances(
                        sim,
                        alarm_distance=float(args.alarm_distance),
                        contact_distance=float(args.contact_distance),
                    )
                    visualisation.reset(episode_nr=0)
                    force_info.reset(episode_nr=0)
                    step = 0
                    print("[tip-debug] scene reset")
        except KeyboardInterrupt:
            print("[tip-debug] interrupted by user")
        finally:
            try:
                visualisation.close()
            except Exception:
                pass
            try:
                intervention.close()
            except Exception:
                pass

    print(f"[tip-debug] wrote {out_csv}")


if __name__ == "__main__":
    main()
