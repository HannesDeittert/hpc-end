from __future__ import annotations

import argparse

import numpy as np
import pygame

from steve_recommender.adapters import eve
from steve_recommender.devices import make_device
from steve_recommender.evaluation.config import ForceUnitsConfig
from steve_recommender.evaluation.force_si import unit_scale_to_si_newton
from steve_recommender.evaluation.info_collectors import SofaWallForceInfo


def _read_data_field(obj, name: str):
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
        return None


def _normalize_xyz_array(raw, *, allow_6dof: bool = False) -> np.ndarray:
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


def _collision_tip_force_estimate(
    *,
    coll_dofs,
    tip_position: np.ndarray,
    tip_node_count: int,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    coll_pos = _normalize_xyz_array(_read_data_field(coll_dofs, "position"), allow_6dof=False)
    coll_force = _normalize_xyz_array(_read_data_field(coll_dofs, "force"), allow_6dof=False)
    source = "collision.force"
    if coll_force.shape[0] <= 0:
        coll_force = _normalize_xyz_array(
            _read_data_field(coll_dofs, "externalForce"),
            allow_6dof=False,
        )
        source = "collision.externalForce"

    force_norm_sum = (
        float(np.linalg.norm(coll_force, axis=1).sum()) if coll_force.shape[0] > 0 else 0.0
    )

    n = int(min(coll_pos.shape[0], coll_force.shape[0]))
    if n <= 0:
        return np.zeros((3,), dtype=np.float64), np.zeros((0,), dtype=np.int64), "missing_collision_force", force_norm_sum

    k = int(max(1, min(int(tip_node_count), n)))
    d2 = np.sum((coll_pos[:n] - tip_position.reshape((1, 3))) ** 2, axis=1)
    nearest = np.argsort(d2)[:k].astype(np.int64)
    tip_force_on_wire = np.sum(coll_force[nearest], axis=0)
    tip_force_on_wall = -np.asarray(tip_force_on_wire, dtype=np.float64)
    return tip_force_on_wall, nearest, source, force_norm_sum


def _parse_constraint_rows(raw) -> list[tuple[int, int, np.ndarray]]:
    text = str(raw or "").strip()
    if not text:
        return []
    entries: list[tuple[int, int, np.ndarray]] = []
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
    row_entries: list[tuple[int, int, np.ndarray]],
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


def get_distal_collision_indices_by_length(
    coll_pos: np.ndarray,
    tip_length_mm: float,
) -> np.ndarray:
    arr = np.asarray(coll_pos, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 3:
        return np.zeros((0,), dtype=np.int64)
    arr = arr[:, :3]
    n = int(arr.shape[0])
    if not np.isfinite(float(tip_length_mm)):
        return np.zeros((0,), dtype=np.int64)
    if float(tip_length_mm) <= 0.0:
        return np.asarray([n - 1], dtype=np.int64)
    if n == 1:
        return np.asarray([0], dtype=np.int64)

    segment_lengths = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
    segment_lengths = np.where(np.isfinite(segment_lengths), segment_lengths, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    distal_s = float(cumulative[-1])
    dist_to_distal = distal_s - cumulative
    mask = dist_to_distal <= (float(tip_length_mm) + 1e-12)
    indices = np.nonzero(mask)[0].astype(np.int64)
    if indices.size <= 0:
        return np.asarray([n - 1], dtype=np.int64)
    return indices


def _constraint_tip_force_estimate(
    *,
    lcp,
    coll_dofs,
    tip_length_mm: float,
    tip_node_count: int,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float, np.ndarray]:
    try:
        impulses = np.asarray(
            lcp.getData("constraintForces").value,
            dtype=np.float64,
        ).reshape((-1,))
    except Exception:
        impulses = np.zeros((0,), dtype=np.float64)

    if impulses.size <= 0 or not np.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        return (
            np.zeros((3,), dtype=np.float64),
            np.zeros((3,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            "missing_constraint_forces",
            0.0,
            np.zeros((0, 3), dtype=np.float64),
        )

    lambda_dt = impulses / float(dt_s)

    constraint_raw = _read_data_field(coll_dofs, "constraint")
    row_entries = _parse_constraint_rows(constraint_raw)
    coll_pos = _normalize_xyz_array(_read_data_field(coll_dofs, "position"), allow_6dof=False)

    proj = _project_lambda_dt_to_dofs(
        lambda_dt=lambda_dt,
        row_entries=row_entries,
        n_dofs=int(coll_pos.shape[0]),
    )
    if proj.shape[0] <= 0:
        return (
            np.zeros((3,), dtype=np.float64),
            np.zeros((3,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            "missing_constraint_projection",
            0.0,
            np.zeros((0, 3), dtype=np.float64),
        )

    total_force_on_wall = -np.sum(proj, axis=0, dtype=np.float64)
    force_norm_sum = float(np.linalg.norm(proj, axis=1).sum())

    n = int(min(coll_pos.shape[0], proj.shape[0]))
    if n <= 0:
        return (
            np.zeros((3,), dtype=np.float64),
            total_force_on_wall,
            np.zeros((0,), dtype=np.int64),
            "missing_coll_pos_for_projection",
            force_norm_sum,
            np.zeros((0, 3), dtype=np.float64),
        )

    tip_indices = get_distal_collision_indices_by_length(
        coll_pos[:n],
        float(tip_length_mm),
    )
    source = "constraint_projection"
    if tip_indices.size <= 0:
        k = int(max(1, min(int(tip_node_count), n)))
        tip_indices = np.arange(n - k, n, dtype=np.int64)
        source = "constraint_projection_fallback_tip_node_count"

    tip_local_forces_on_wire = np.asarray(proj[tip_indices], dtype=np.float64)
    tip_local_forces_on_wall = -tip_local_forces_on_wire
    tip_force_on_wire = np.sum(tip_local_forces_on_wire, axis=0)
    tip_force_on_wall = -np.asarray(tip_force_on_wire, dtype=np.float64)
    return (
        tip_force_on_wall,
        total_force_on_wall,
        tip_indices,
        source,
        force_norm_sum,
        tip_local_forces_on_wall,
    )


def _tip_cancellation_metrics(local_vectors: np.ndarray) -> tuple[float, float, float, float]:
    vec = _normalize_xyz_array(local_vectors, allow_6dof=False)
    if vec.shape[0] <= 0:
        return 0.0, 0.0, 0.0, 0.0
    local_norms = np.linalg.norm(vec, axis=1)
    resultant_norm = float(np.linalg.norm(np.sum(vec, axis=0)))
    sum_local_norms = float(np.sum(local_norms))
    max_local_norm = float(np.max(local_norms))
    if sum_local_norms > 0.0 and np.isfinite(sum_local_norms):
        cancellation_ratio = 1.0 - (resultant_norm / sum_local_norms)
        cancellation_ratio = float(np.clip(cancellation_ratio, 0.0, 1.0))
    else:
        cancellation_ratio = 0.0
    return resultant_norm, sum_local_norms, max_local_norm, cancellation_ratio


class TipForceProbe:
    def __init__(self, wire_mo, *, tip_node_count: int = 3):
        self.wire_mo = wire_mo
        self.tip_node_count = int(max(1, tip_node_count))

    def read(self):
        pos = _normalize_xyz_array(_read_data_field(self.wire_mo, "position"), allow_6dof=True)
        if pos.shape[0] <= 0:
            return np.full((3,), np.nan), np.full((3,), np.nan), "missing_position", np.zeros((0,), dtype=np.int64)

        tip_position = np.asarray(pos[-1], dtype=np.float64)
        n_nodes = int(pos.shape[0])
        tip_start = max(0, n_nodes - int(self.tip_node_count))
        tip_indices = np.arange(tip_start, n_nodes, dtype=np.int64)

        lam_raw = _read_data_field(self.wire_mo, "lambda")
        lam = _normalize_xyz_array(lam_raw, allow_6dof=True)
        source = "lambda"
        if lam.shape[0] <= 0:
            # Fallback in scenes where lambda is not exposed on wire DOFs.
            lam = _normalize_xyz_array(_read_data_field(self.wire_mo, "force"), allow_6dof=True)
            source = "force_fallback"
        if lam.shape[0] <= 0:
            return tip_position, np.full((3,), np.nan), "missing_lambda_and_force", tip_indices

        tip_valid = tip_indices[tip_indices < lam.shape[0]]
        if tip_valid.size <= 0:
            return tip_position, np.full((3,), np.nan), f"{source}_index_mismatch", tip_indices

        tip_force_on_wire = np.sum(lam[tip_valid], axis=0)
        tip_force_on_wall = -np.asarray(tip_force_on_wire, dtype=np.float64)
        return tip_position, tip_force_on_wall, source, tip_valid


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Minimal single-wire SOFA scene with keyboard steering and simple tip-force readout."
    )
    p.add_argument("--tool-ref", default="ArchVarJShaped/JShaped_Default")
    p.add_argument("--arch-type", choices=["I", "II", "IV", "V", "VI", "VII"], default="I")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--friction", type=float, default=0.1)
    p.add_argument(
        "--stiffness-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply both young_modulus_straight and young_modulus_tip for debug runs. "
            "Use large values (e.g. 50, 100) for an intentionally very stiff wire."
        ),
    )
    p.add_argument(
        "--young-modulus-straight",
        type=float,
        default=None,
        help="Optional absolute override for straight-section Young's modulus.",
    )
    p.add_argument(
        "--young-modulus-tip",
        type=float,
        default=None,
        help="Optional absolute override for tip-section Young's modulus.",
    )
    p.add_argument("--image-frequency-hz", type=float, default=7.5)
    p.add_argument("--image-rot-z-deg", type=float, default=20.0)
    p.add_argument("--image-rot-x-deg", type=float, default=5.0)
    p.add_argument("--target-threshold-mm", type=float, default=5.0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--insert-action", type=float, default=50.0)
    p.add_argument("--rotate-action", type=float, default=3.14)
    p.add_argument(
        "--tip-length-mm",
        type=float,
        default=3.0,
        help="Distal collision-chain length used for tip-force aggregation.",
    )
    p.add_argument(
        "--tip-node-count",
        type=int,
        default=3,
        help="Fallback distal node count when tip-length selection is unavailable.",
    )
    p.add_argument(
        "--debug-every",
        type=int,
        default=10,
        help="Print constraint-side debug probe every N steps (0 disables).",
    )
    p.add_argument(
        "--contact-epsilon",
        type=float,
        default=1e-7,
        help="Contact epsilon used by SofaWallForceInfo validation probe.",
    )
    return p


def _build_device(args: argparse.Namespace):
    tool_ref = str(args.tool_ref)
    base_device = make_device(tool_ref)
    overrides: dict[str, float] = {}

    scale = float(args.stiffness_scale)
    if np.isfinite(scale) and scale > 0.0 and not np.isclose(scale, 1.0):
        if hasattr(base_device, "young_modulus_straight"):
            try:
                overrides["young_modulus_straight"] = float(
                    getattr(base_device, "young_modulus_straight")
                ) * scale
            except Exception:
                pass
        if hasattr(base_device, "young_modulus_tip"):
            try:
                overrides["young_modulus_tip"] = float(
                    getattr(base_device, "young_modulus_tip")
                ) * scale
            except Exception:
                pass

    if args.young_modulus_straight is not None:
        overrides["young_modulus_straight"] = float(args.young_modulus_straight)
    if args.young_modulus_tip is not None:
        overrides["young_modulus_tip"] = float(args.young_modulus_tip)

    if overrides:
        device = make_device(tool_ref, overrides=overrides)
    else:
        device = base_device
    return device, overrides


def _build_intervention(args: argparse.Namespace):
    arch_type = getattr(eve.intervention.vesseltree.ArchType, str(args.arch_type))
    vessel_tree = eve.intervention.vesseltree.AorticArch(
        arch_type=arch_type,
        seed=int(args.seed),
    )
    device, device_overrides = _build_device(args)
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
    return intervention, device, device_overrides


def main() -> None:
    args = _build_parser().parse_args()
    intervention, device, device_overrides = _build_intervention(args)
    visualisation = eve.visualisation.SofaPygame(intervention=intervention)

    print(
        "[basic-scene] keys: arrows=wire | r+w/a/s/d=camera rotate | "
        "w/a/s/d=camera pan | e/q=zoom | Enter=reset | Esc=quit"
    )
    if device_overrides:
        print(f"[basic-scene] [DEBUG] device overrides: {device_overrides}")
    try:
        ys = float(getattr(device, "young_modulus_straight"))
        yt = float(getattr(device, "young_modulus_tip"))
        print(f"[basic-scene] wire Young moduli: straight={ys:.6g} tip={yt:.6g}")
    except Exception:
        pass
    print(
        "[basic-scene] tip force: constraint-projected over last "
        f"{float(args.tip_length_mm):.3f} mm of collision chain (converted to Newton)"
    )
    print("[basic-scene] [DEBUG] runtime check: LCP.constraintForces + CollisionDOFs.constraint")
    print("[basic-scene] [DEBUG] validation: projected(raw, N) vs SofaWallForceInfo.total(N)")

    try:
        intervention.reset(int(args.seed))
        visualisation.reset(episode_nr=0)
        sim = intervention.simulation
        wire_mo = sim._instruments_combined.DOFs  # noqa: SLF001
        lcp = sim.root.LCP
        coll_dofs = sim._instruments_combined.CollisionModel.CollisionDOFs  # noqa: SLF001
        line_model = sim._instruments_combined.CollisionModel.LineCollisionModel  # noqa: SLF001
        point_model = sim._instruments_combined.CollisionModel.PointCollisionModel  # noqa: SLF001
        try:
            lcp.computeConstraintForces.value = True
            print("[basic-scene] [DEBUG] enabled LCP.computeConstraintForces=True")
        except Exception as exc:
            print(f"[basic-scene] [DEBUG] could not enable computeConstraintForces: {exc}")
        probe = TipForceProbe(wire_mo, tip_node_count=int(args.tip_node_count))
        dt_s = float(getattr(getattr(sim.root, "dt", None), "value", 0.0) or 0.0)
        # stEVE/BeamAdapter scene in this project uses mm,kg,s conventions.
        units_cfg = ForceUnitsConfig(length_unit="mm", mass_unit="kg", time_unit="s")
        force_scale_to_n = float(
            unit_scale_to_si_newton(
                units_cfg
            )
        )
        print(f"[basic-scene] force_scale_to_newton={force_scale_to_n:.6g}")
        force_info = None
        try:
            force_info = SofaWallForceInfo(
                intervention=intervention,
                mode="constraint_projected_si_validated",
                required=False,
                contact_epsilon=float(args.contact_epsilon),
                units=units_cfg,
                constraint_dt_s=float(dt_s) if np.isfinite(dt_s) and float(dt_s) > 0.0 else None,
            )
            force_info.reset(episode_nr=0)
            print(
                "[basic-scene] [DEBUG] enabled SofaWallForceInfo(mode=constraint_projected_si_validated)"
            )
        except Exception as exc:
            force_info = None
            print(f"[basic-scene] [DEBUG] SofaWallForceInfo init failed: {exc}")
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

            intervention.step((trans, rot))
            force_info_data = {}
            if force_info is not None:
                try:
                    force_info.step()
                    force_info_data = force_info.info
                except Exception:
                    force_info_data = {}
            tip_pos, tip_force_wall_simple, tip_src_simple, tip_ids_simple = probe.read()
            (
                proj_tip_force_wall,
                proj_total_force_wall,
                proj_tip_ids,
                proj_tip_src,
                proj_force_norm_sum,
                proj_tip_local_forces_on_wall,
            ) = _constraint_tip_force_estimate(
                lcp=lcp,
                coll_dofs=coll_dofs,
                tip_length_mm=float(args.tip_length_mm),
                tip_node_count=int(args.tip_node_count),
                dt_s=float(dt_s),
            )
            if proj_tip_src.startswith("constraint_projection"):
                tip_force_wall = np.asarray(proj_tip_force_wall, dtype=np.float64) * force_scale_to_n
                total_force_wall = np.asarray(proj_total_force_wall, dtype=np.float64) * force_scale_to_n
                tip_src = "constraint_projection"
                tip_ids = np.asarray(proj_tip_ids, dtype=np.int64)
            else:
                tip_force_wall = np.asarray(tip_force_wall_simple, dtype=np.float64) * force_scale_to_n
                total_force_wall = np.zeros((3,), dtype=np.float64)
                tip_src = f"{tip_src_simple}_fallback"
                tip_ids = np.asarray(tip_ids_simple, dtype=np.int64)

            # [DEBUG] Constraint-side runtime probe to verify live contact data availability.
            if int(args.debug_every) > 0 and (step % int(args.debug_every) == 0):
                try:
                    cf = np.asarray(
                        lcp.getData("constraintForces").value,
                        dtype=np.float64,
                    ).reshape((-1,))
                except Exception:
                    cf = np.zeros((0,), dtype=np.float64)
                try:
                    ccf = bool(lcp.getData("computeConstraintForces").value)
                except Exception:
                    ccf = False
                try:
                    constraint_raw = coll_dofs.getData("constraint").value
                    constraint_rows = [ln for ln in str(constraint_raw).splitlines() if ln.strip()]
                except Exception:
                    constraint_rows = []
                try:
                    line_contacts = int(line_model.getData("numberOfContacts").value)
                except Exception:
                    line_contacts = -1
                try:
                    point_contacts = int(point_model.getData("numberOfContacts").value)
                except Exception:
                    point_contacts = -1
                max_abs = float(np.max(np.abs(cf))) if cf.size else 0.0
                active = int(np.count_nonzero(np.abs(cf) > 1e-9)) if cf.size else 0
                info_total_n = np.asarray(
                    force_info_data.get("wall_total_force_vector_N", [np.nan, np.nan, np.nan]),
                    dtype=np.float64,
                ).reshape((3,))
                info_contact = int(bool(force_info_data.get("wall_contact_detected", False)))
                info_count = int(force_info_data.get("wall_contact_count", 0) or 0)
                info_channel = str(force_info_data.get("wall_force_channel", ""))
                info_tier = str(force_info_data.get("wall_force_quality_tier", ""))
                proj_total_n = np.asarray(proj_total_force_wall, dtype=np.float64) * float(
                    force_scale_to_n
                )
                tip_local_forces_n = (
                    np.asarray(proj_tip_local_forces_on_wall, dtype=np.float64) * float(force_scale_to_n)
                )
                (
                    tip_resultant_norm_n,
                    tip_sum_of_local_norms_n,
                    tip_max_local_norm_n,
                    tip_cancellation_ratio,
                ) = _tip_cancellation_metrics(tip_local_forces_n)
                if np.all(np.isfinite(info_total_n)) and np.all(np.isfinite(proj_total_n)):
                    delta_total_n = info_total_n - proj_total_n
                    delta_total_norm_n = float(np.linalg.norm(delta_total_n))
                else:
                    delta_total_n = np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
                    delta_total_norm_n = float("nan")
                coll_tip_force_wall, coll_tip_ids, coll_tip_src, coll_force_norm_sum = (
                    _collision_tip_force_estimate(
                        coll_dofs=coll_dofs,
                        tip_position=np.asarray(tip_pos, dtype=np.float64),
                        tip_node_count=int(args.tip_node_count),
                    )
                )
                print(
                    "[DEBUG] step={s:04d} constraintForces_size={n} active={a} "
                    "max_abs={m:.6g} collision_constraint_rows={r} "
                    "line_contacts={lc} point_contacts={pc} computeCF={ccf} "
                    "coll_force_norm_sum={cns:.6g} coll_tip_F=({cfx:.6g},{cfy:.6g},{cfz:.6g}) "
                    "coll_tip_src={cts} coll_tip_ids={cti} "
                    "proj_force_norm_sum={pns:.6g} "
                    "proj_tip_raw=({pfrx:.6g},{pfry:.6g},{pfrz:.6g}) "
                    "proj_tip_N=({pfnx:.6g},{pfny:.6g},{pfnz:.6g}) "
                    "proj_total_raw=({ptrx:.6g},{ptry:.6g},{ptrz:.6g}) "
                    "proj_total_N=({ptnx:.6g},{ptny:.6g},{ptnz:.6g}) "
                    "force_info_total_N=({itx:.6g},{ity:.6g},{itz:.6g}) "
                    "delta_total_N=({dtx:.6g},{dty:.6g},{dtz:.6g}) delta_norm_N={dn:.6g} "
                    "tip_resultant_norm_N={trn:.6g} tip_sum_of_local_norms_N={tsn:.6g} "
                    "tip_max_local_norm_N={tmn:.6g} tip_cancellation_ratio={tcr:.6g} "
                    "fi_contact={fic} fi_count={ficnt} fi_channel={fich} fi_tier={fitr} "
                    "proj_tip_src={pts} proj_tip_ids={pti}".format(
                        s=int(step),
                        n=int(cf.size),
                        a=int(active),
                        m=max_abs,
                        r=int(len(constraint_rows)),
                        lc=int(line_contacts),
                        pc=int(point_contacts),
                        ccf=int(ccf),
                        cns=float(coll_force_norm_sum),
                        cfx=float(coll_tip_force_wall[0]),
                        cfy=float(coll_tip_force_wall[1]),
                        cfz=float(coll_tip_force_wall[2]),
                        cts=coll_tip_src,
                        cti=coll_tip_ids.tolist(),
                        pns=float(proj_force_norm_sum),
                        pfrx=float(proj_tip_force_wall[0]),
                        pfry=float(proj_tip_force_wall[1]),
                        pfrz=float(proj_tip_force_wall[2]),
                        pfnx=float(proj_tip_force_wall[0] * force_scale_to_n),
                        pfny=float(proj_tip_force_wall[1] * force_scale_to_n),
                        pfnz=float(proj_tip_force_wall[2] * force_scale_to_n),
                        ptrx=float(proj_total_force_wall[0]),
                        ptry=float(proj_total_force_wall[1]),
                        ptrz=float(proj_total_force_wall[2]),
                        ptnx=float(proj_total_n[0]),
                        ptny=float(proj_total_n[1]),
                        ptnz=float(proj_total_n[2]),
                        itx=float(info_total_n[0]),
                        ity=float(info_total_n[1]),
                        itz=float(info_total_n[2]),
                        dtx=float(delta_total_n[0]),
                        dty=float(delta_total_n[1]),
                        dtz=float(delta_total_n[2]),
                        dn=float(delta_total_norm_n),
                        trn=float(tip_resultant_norm_n),
                        tsn=float(tip_sum_of_local_norms_n),
                        tmn=float(tip_max_local_norm_n),
                        tcr=float(tip_cancellation_ratio),
                        fic=int(info_contact),
                        ficnt=int(info_count),
                        fich=info_channel,
                        fitr=info_tier,
                        pts=proj_tip_src,
                        pti=proj_tip_ids.tolist(),
                    )
                )

            sim_time = float(getattr(getattr(sim.root, "time", None), "value", float(step)))
            print(
                "[basic-scene] step={s:04d} t={t:.3f} "
                "tip=({px:.4f},{py:.4f},{pz:.4f}) "
                "F_tip->wall_N=({fx:.6g},{fy:.6g},{fz:.6g}) "
                "F_total->wall_N=({tx:.6g},{ty:.6g},{tz:.6g}) "
                "src={src} nodes={n}".format(
                    s=int(step),
                    t=sim_time,
                    px=float(tip_pos[0]),
                    py=float(tip_pos[1]),
                    pz=float(tip_pos[2]),
                    fx=float(tip_force_wall[0]),
                    fy=float(tip_force_wall[1]),
                    fz=float(tip_force_wall[2]),
                    tx=float(total_force_wall[0]),
                    ty=float(total_force_wall[1]),
                    tz=float(total_force_wall[2]),
                    src=tip_src,
                    n=tip_ids.tolist(),
                )
            )
            visualisation.render()
            step += 1

            if keys[pygame.K_RETURN]:
                intervention.reset(int(args.seed))
                visualisation.reset(episode_nr=0)
                sim = intervention.simulation
                wire_mo = sim._instruments_combined.DOFs  # noqa: SLF001
                lcp = sim.root.LCP
                coll_dofs = sim._instruments_combined.CollisionModel.CollisionDOFs  # noqa: SLF001
                line_model = sim._instruments_combined.CollisionModel.LineCollisionModel  # noqa: SLF001
                point_model = sim._instruments_combined.CollisionModel.PointCollisionModel  # noqa: SLF001
                try:
                    lcp.computeConstraintForces.value = True
                    print("[basic-scene] [DEBUG] enabled LCP.computeConstraintForces=True")
                except Exception as exc:
                    print(f"[basic-scene] [DEBUG] could not enable computeConstraintForces: {exc}")
                probe = TipForceProbe(wire_mo, tip_node_count=int(args.tip_node_count))
                dt_s = float(getattr(getattr(sim.root, "dt", None), "value", 0.0) or 0.0)
                force_info = None
                try:
                    force_info = SofaWallForceInfo(
                        intervention=intervention,
                        mode="constraint_projected_si_validated",
                        required=False,
                        contact_epsilon=float(args.contact_epsilon),
                        units=units_cfg,
                        constraint_dt_s=float(dt_s)
                        if np.isfinite(dt_s) and float(dt_s) > 0.0
                        else None,
                    )
                    force_info.reset(episode_nr=0)
                    print(
                        "[basic-scene] [DEBUG] enabled SofaWallForceInfo(mode=constraint_projected_si_validated)"
                    )
                except Exception as exc:
                    force_info = None
                    print(f"[basic-scene] [DEBUG] SofaWallForceInfo init failed: {exc}")
                force_scale_to_n = float(
                    unit_scale_to_si_newton(
                        units_cfg
                    )
                )
                step = 0
                print("[basic-scene] reset")
    except KeyboardInterrupt:
        print("[basic-scene] interrupted by user")
    finally:
        try:
            visualisation.close()
        except Exception:
            pass
        try:
            intervention.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
