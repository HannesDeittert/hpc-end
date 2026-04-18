from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pygame

from steve_recommender.adapters import eve
from steve_recommender.devices import make_device
from steve_recommender.evaluation.config import ForceUnitsConfig
from steve_recommender.evaluation.force_si import unit_scale_to_si_newton


def _parse_rgba(arg: str) -> tuple[float, float, float, float]:
    text = str(arg).strip()
    if not text:
        return (0.0, 0.9, 1.0, 1.0)
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    vals = []
    for p in parts[:4]:
        try:
            vals.append(float(p))
        except Exception:
            pass
    if len(vals) == 3:
        vals.append(1.0)
    if len(vals) != 4:
        return (0.0, 0.9, 1.0, 1.0)
    return tuple(float(np.clip(v, 0.0, 1.0)) for v in vals)


def _read_data_field(obj: Any, name: str):
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


def _normalize_xyz_array(raw: Any, *, allow_6dof: bool = False) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.size == 0 or arr.ndim == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.ndim == 1:
        if allow_6dof and arr.size % 6 == 0:
            arr = arr.reshape((-1, 6))[:, :3]
        elif allow_6dof and arr.size % 7 == 0:
            arr = arr.reshape((-1, 7))[:, :3]
        elif arr.size % 3 == 0:
            arr = arr.reshape((-1, 3))
        else:
            return np.zeros((0, 3), dtype=np.float64)
    elif arr.ndim >= 2:
        if arr.shape[1] < 3:
            return np.zeros((0, 3), dtype=np.float64)
        arr = arr[:, :3]
    return np.asarray(arr, dtype=np.float64)


def _parse_constraint_rows(raw: Any) -> list[tuple[int, int, np.ndarray]]:
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

    seg = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
    seg = np.where(np.isfinite(seg), seg, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    distal_s = float(cumulative[-1])
    dist_to_distal = distal_s - cumulative
    mask = dist_to_distal <= (float(tip_length_mm) + 1e-12)
    indices = np.nonzero(mask)[0].astype(np.int64)
    if indices.size <= 0:
        return np.asarray([n - 1], dtype=np.int64)
    return indices


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


def _constraint_tip_force_estimate(
    *,
    lcp: Any,
    coll_dofs: Any,
    tip_length_mm: float,
    tip_node_count: int,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
    try:
        impulses = np.asarray(lcp.getData("constraintForces").value, dtype=np.float64).reshape((-1,))
    except Exception:
        impulses = np.zeros((0,), dtype=np.float64)

    if impulses.size <= 0 or not np.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        return (
            np.zeros((3,), dtype=np.float64),
            np.zeros((3,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            "missing_constraint_forces",
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
            np.zeros((0, 3), dtype=np.float64),
        )

    n = int(min(coll_pos.shape[0], proj.shape[0]))
    if n <= 0:
        return (
            np.zeros((3,), dtype=np.float64),
            np.zeros((3,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            "missing_coll_pos_for_projection",
            np.zeros((0, 3), dtype=np.float64),
        )

    tip_indices = get_distal_collision_indices_by_length(coll_pos[:n], float(tip_length_mm))
    source = "constraint_projection"
    if tip_indices.size <= 0:
        k = int(max(1, min(int(tip_node_count), n)))
        tip_indices = np.arange(n - k, n, dtype=np.int64)
        source = "constraint_projection_fallback_tip_node_count"

    tip_local_forces_on_wire = np.asarray(proj[tip_indices], dtype=np.float64)
    tip_local_forces_on_wall = -tip_local_forces_on_wire
    tip_force_on_wall = np.sum(tip_local_forces_on_wall, axis=0, dtype=np.float64)
    total_force_on_wall = -np.sum(proj, axis=0, dtype=np.float64)
    return tip_force_on_wall, total_force_on_wall, tip_indices, source, tip_local_forces_on_wall


def _write_plane_wall_mesh(
    mesh_path: Path,
    *,
    wall_x: float,
    wall_size_mm: float,
    wall_thickness_mm: float,
) -> None:
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    half = float(wall_size_mm) * 0.5
    x0 = float(wall_x)
    t = float(max(1e-6, wall_thickness_mm))
    x1 = x0 + t
    verts = [
        (x0, -half, -half),  # 1
        (x0, +half, -half),  # 2
        (x0, +half, +half),  # 3
        (x0, -half, +half),  # 4
        (x1, -half, -half),  # 5
        (x1, +half, -half),  # 6
        (x1, +half, +half),  # 7
        (x1, -half, +half),  # 8
    ]
    with mesh_path.open("w", encoding="utf-8") as f:
        f.write("# thin box wall mesh for single_wire_plane_wall\n")
        for vx, vy, vz in verts:
            f.write(f"v {vx:.9f} {vy:.9f} {vz:.9f}\n")
        # Front face (x = x0)
        f.write("f 1 2 3\n")
        f.write("f 1 3 4\n")
        # Back face (x = x1)
        f.write("f 8 7 6\n")
        f.write("f 8 6 5\n")
        # -Y face
        f.write("f 1 5 6\n")
        f.write("f 1 6 2\n")
        # +Y face
        f.write("f 4 3 7\n")
        f.write("f 4 7 8\n")
        # -Z face
        f.write("f 1 4 8\n")
        f.write("f 1 8 5\n")
        # +Z face
        f.write("f 2 6 7\n")
        f.write("f 2 7 3\n")


def _safe_float_attr(obj: Any, name: str) -> float:
    try:
        return float(getattr(obj, name))
    except Exception:
        return float("nan")


def _safe_int(value: Any, fallback: int = 1) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _override_device_wire_length(device: Any, wire_length_mm: float) -> tuple[float, float]:
    target = float(wire_length_mm)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError(f"wire_length_mm must be > 0, got {wire_length_mm!r}")

    sofa = getattr(device, "sofa_device", None)
    if sofa is None:
        raise ValueError("loaded device has no sofa_device")

    old_total = _safe_float_attr(sofa, "length")
    old_straight = _safe_float_attr(sofa, "straight_length")

    tip_radius = _safe_float_attr(device, "tip_radius")
    tip_angle = _safe_float_attr(device, "tip_angle")
    if np.isfinite(tip_radius) and np.isfinite(tip_angle) and tip_radius > 0.0 and tip_angle > 0.0:
        tip_length = float(tip_radius * tip_angle)
    elif np.isfinite(old_total) and np.isfinite(old_straight):
        tip_length = float(max(0.0, old_total - old_straight))
    else:
        tip_length = 0.0

    is_procedural = bool(getattr(sofa, "is_a_procedural_shape", False))
    if is_procedural:
        if target <= tip_length + 1e-9:
            raise ValueError(
                "wire_length_mm must be larger than tip arc length "
                f"({tip_length:.6g} mm), got {target:.6g} mm"
            )
        straight_new = target - tip_length
    else:
        straight_new = target

    updates = {}
    if hasattr(sofa, "length"):
        updates["length"] = float(target)
    if hasattr(sofa, "straight_length"):
        updates["straight_length"] = float(straight_new)

    # Visual edges.
    old_num_edges = _safe_int(getattr(sofa, "num_edges", 0), fallback=0)
    visu_rate = _safe_float_attr(device, "visu_edges_per_mm")
    if np.isfinite(visu_rate) and visu_rate > 0.0:
        num_edges_new = max(2, int(math.ceil(visu_rate * target)))
    elif np.isfinite(old_total) and old_total > 0.0 and old_num_edges > 0:
        num_edges_new = max(2, int(round(old_num_edges * target / old_total)))
    else:
        num_edges_new = max(2, old_num_edges if old_num_edges > 0 else 2)
    if hasattr(sofa, "num_edges"):
        updates["num_edges"] = int(num_edges_new)

    # Collision edges.
    old_collis = getattr(sofa, "num_edges_collis", None)
    collis_rate_tip = _safe_float_attr(device, "collis_edges_per_mm_tip")
    collis_rate_straight = _safe_float_attr(device, "collis_edges_per_mm_straight")
    if isinstance(old_collis, tuple):
        old_vals = [max(1, _safe_int(v, fallback=1)) for v in old_collis]
        if is_procedural and len(old_vals) >= 2:
            if (
                np.isfinite(collis_rate_straight)
                and collis_rate_straight > 0.0
                and np.isfinite(collis_rate_tip)
                and collis_rate_tip > 0.0
            ):
                straight_cnt = max(1, int(math.ceil(collis_rate_straight * straight_new)))
                tip_cnt = max(1, int(math.ceil(collis_rate_tip * tip_length)))
            else:
                if np.isfinite(old_straight) and old_straight > 0.0:
                    straight_cnt = max(
                        1,
                        int(round(old_vals[0] * (straight_new / old_straight))),
                    )
                else:
                    straight_cnt = old_vals[0]
                tip_cnt = old_vals[1]
            new_vals = [straight_cnt, tip_cnt, *old_vals[2:]]
        else:
            if np.isfinite(old_total) and old_total > 0.0:
                scale = target / old_total
                new_vals = [max(1, int(round(v * scale))) for v in old_vals]
            else:
                new_vals = old_vals
        updates["num_edges_collis"] = tuple(new_vals)
    elif isinstance(old_collis, int):
        if np.isfinite(old_total) and old_total > 0.0 and old_collis > 0:
            updates["num_edges_collis"] = max(1, int(round(old_collis * target / old_total)))
        elif old_collis > 0:
            updates["num_edges_collis"] = old_collis

    # Beam density.
    old_beams = getattr(sofa, "density_of_beams", None)
    beams_rate_tip = _safe_float_attr(device, "beams_per_mm_tip")
    beams_rate_straight = _safe_float_attr(device, "beams_per_mm_straight")
    if isinstance(old_beams, tuple):
        old_vals = [max(1, _safe_int(v, fallback=1)) for v in old_beams]
        if is_procedural and len(old_vals) >= 2:
            if (
                np.isfinite(beams_rate_straight)
                and beams_rate_straight > 0.0
                and np.isfinite(beams_rate_tip)
                and beams_rate_tip > 0.0
            ):
                straight_cnt = max(1, int(math.ceil(beams_rate_straight * straight_new)))
                tip_cnt = max(1, int(math.ceil(beams_rate_tip * tip_length)))
            else:
                if np.isfinite(old_straight) and old_straight > 0.0:
                    straight_cnt = max(
                        1,
                        int(round(old_vals[0] * (straight_new / old_straight))),
                    )
                else:
                    straight_cnt = old_vals[0]
                tip_cnt = old_vals[1]
            new_vals = [straight_cnt, tip_cnt, *old_vals[2:]]
        else:
            if np.isfinite(old_total) and old_total > 0.0:
                scale = target / old_total
                new_vals = [max(1, int(round(v * scale))) for v in old_vals]
            else:
                new_vals = old_vals
        updates["density_of_beams"] = tuple(new_vals)
    elif isinstance(old_beams, int):
        if np.isfinite(old_total) and old_total > 0.0 and old_beams > 0:
            updates["density_of_beams"] = max(1, int(round(old_beams * target / old_total)))
        elif old_beams > 0:
            updates["density_of_beams"] = old_beams

    # Key points.
    old_key_points = getattr(sofa, "key_points", None)
    if isinstance(old_key_points, tuple) and len(old_key_points) >= 2:
        if is_procedural and len(old_key_points) >= 3:
            key_points_new = (
                float(old_key_points[0]),
                float(straight_new),
                float(target),
                *[float(v) for v in old_key_points[3:]],
            )
        else:
            old_last = float(old_key_points[-1])
            if np.isfinite(old_last) and old_last > 0.0:
                key_points_scaled = [float(v) * (target / old_last) for v in old_key_points[:-1]]
                key_points_new = (*key_points_scaled, float(target))
            else:
                key_points_new = old_key_points
        updates["key_points"] = tuple(key_points_new)

    device.sofa_device = replace(sofa, **updates)
    if hasattr(device, "length"):
        device.length = float(target)
    return float(target), float(tip_length)


def _override_device_tip_shape(
    device: Any,
    *,
    tip_angle_deg: float | None = None,
    tip_angle_scale: float | None = None,
    tip_radius_mm: float | None = None,
) -> tuple[float, float, float]:
    sofa = getattr(device, "sofa_device", None)
    if sofa is None:
        raise ValueError("loaded device has no sofa_device")
    if not bool(getattr(sofa, "is_a_procedural_shape", False)):
        raise ValueError("tip-shape override requires procedural wire shape")

    total_length_mm = _safe_float_attr(sofa, "length")
    if not np.isfinite(total_length_mm) or total_length_mm <= 0.0:
        raise ValueError("device total length is invalid")

    cur_tip_angle = _safe_float_attr(device, "tip_angle")
    cur_tip_radius = _safe_float_attr(device, "tip_radius")
    if not np.isfinite(cur_tip_angle) or not np.isfinite(cur_tip_radius):
        raise ValueError("device does not expose finite tip_angle/tip_radius")

    new_tip_angle = float(cur_tip_angle)
    if tip_angle_scale is not None:
        scale = float(tip_angle_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"tip_angle_scale must be > 0, got {tip_angle_scale!r}")
        new_tip_angle *= scale
    if tip_angle_deg is not None:
        deg = float(tip_angle_deg)
        if not np.isfinite(deg) or deg <= 0.0:
            raise ValueError(f"tip_angle_deg must be > 0, got {tip_angle_deg!r}")
        new_tip_angle = float(np.deg2rad(deg))

    new_tip_radius = float(cur_tip_radius)
    if tip_radius_mm is not None:
        r = float(tip_radius_mm)
        if not np.isfinite(r) or r <= 0.0:
            raise ValueError(f"tip_radius_mm must be > 0, got {tip_radius_mm!r}")
        new_tip_radius = r

    if new_tip_angle <= 0.0:
        raise ValueError("effective tip angle must be > 0")

    # First: update explicit device parameters.
    if hasattr(device, "tip_angle"):
        device.tip_angle = float(new_tip_angle)
    if hasattr(device, "tip_radius"):
        device.tip_radius = float(new_tip_radius)

    # Update geometry parameter directly represented on sofa_device.
    try:
        device.sofa_device = replace(
            device.sofa_device,
            spire_diameter=float(2.0 * new_tip_radius),
        )
    except Exception:
        pass

    # Rebuild straight/tip split + discretization while preserving total length.
    _, tip_len_mm = _override_device_wire_length(device, float(total_length_mm))
    straight_len_mm = float(total_length_mm - tip_len_mm)
    return float(new_tip_angle), float(new_tip_radius), float(straight_len_mm)


def _print_buckling_estimate(device: Any) -> None:
    d_tip_mm = _safe_float_attr(device, "tip_outer_diameter")
    d_straight_mm = _safe_float_attr(device, "straight_outer_diameter")
    ym_tip_scene = _safe_float_attr(device, "young_modulus_tip")
    ym_straight_scene = _safe_float_attr(device, "young_modulus_straight")
    tip_radius_mm = _safe_float_attr(device, "tip_radius")
    tip_angle_rad = _safe_float_attr(device, "tip_angle")

    if not np.all(np.isfinite([d_tip_mm, d_straight_mm, ym_tip_scene, ym_straight_scene])):
        print("[plane-wall] Euler estimate unavailable (missing device fields)")
        return

    i_tip = math.pi * (d_tip_mm**4) / 64.0
    i_straight = math.pi * (d_straight_mm**4) / 64.0
    e_tip_phys = ym_tip_scene / 1000.0
    e_straight_phys = ym_straight_scene / 1000.0
    ei_tip = e_tip_phys * i_tip
    ei_straight = e_straight_phys * i_straight
    l_tip = tip_radius_mm * tip_angle_rad if np.isfinite(tip_radius_mm * tip_angle_rad) else float("nan")

    def _pcr(ei: float, k: float, l_mm: float) -> float:
        if not np.isfinite(ei) or not np.isfinite(k) or not np.isfinite(l_mm) or k <= 0.0 or l_mm <= 0.0:
            return float("nan")
        return float((math.pi**2) * ei / ((k * l_mm) ** 2))

    print("[plane-wall] Euler buckling estimate (N) using loaded wire parameters:")
    print(
        "[plane-wall] d_tip_mm={dt:.6g} d_straight_mm={ds:.6g} "
        "E_tip_phys_MPa={et:.6g} E_straight_phys_MPa={es:.6g}".format(
            dt=float(d_tip_mm),
            ds=float(d_straight_mm),
            et=float(e_tip_phys),
            es=float(e_straight_phys),
        )
    )
    print(
        "[plane-wall] I_tip_mm4={it:.6g} I_straight_mm4={is_:.6g} "
        "EI_tip_Nmm2={eit:.6g} EI_straight_Nmm2={eis:.6g}".format(
            it=float(i_tip),
            is_=float(i_straight),
            eit=float(ei_tip),
            eis=float(ei_straight),
        )
    )
    print("[plane-wall] straight-section Pcr table (N):")
    print("[plane-wall]   L_mm    K=2.0(fixed-free)    K=1.0(pinned-pinned)")
    for l_mm in [450.0, 300.0, 200.0, 150.0, 100.0, 75.0, 50.0]:
        p_k2 = _pcr(ei_straight, 2.0, l_mm)
        p_k1 = _pcr(ei_straight, 1.0, l_mm)
        print(f"[plane-wall]   {l_mm:6.1f} {p_k2:20.6g} {p_k1:24.6g}")
    if np.isfinite(l_tip) and l_tip > 0.0:
        print(
            "[plane-wall] tip-only local estimate (L_tip={lt:.6g} mm, EI_tip={ei:.6g} Nmm2): "
            "Pcr(K=2.0)={k2:.6g} N, Pcr(K=1.0)={k1:.6g} N".format(
                lt=float(l_tip),
                ei=float(ei_tip),
                k2=_pcr(ei_tip, 2.0, l_tip),
                k1=_pcr(ei_tip, 1.0, l_tip),
            )
        )


@dataclass
class _DummySpace:
    low: np.ndarray
    high: np.ndarray


@dataclass
class _DummyVesselTree:
    coordinate_space: _DummySpace


@dataclass
class _DummyFluoroscopy:
    image_frequency: float
    image_rot_zx: Tuple[float, float]
    image_center: list[float]


@dataclass
class _DummyTarget:
    threshold: float
    coordinates3d: np.ndarray


class _DummyIntervention:
    def __init__(
        self,
        *,
        simulation: Any,
        vessel_tree: _DummyVesselTree,
        fluoroscopy: _DummyFluoroscopy,
        target: _DummyTarget,
    ) -> None:
        self.simulation = simulation
        self.vessel_tree = vessel_tree
        self.fluoroscopy = fluoroscopy
        self.target = target


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone single-wire plane-wall debug scene for axial tip-force validation."
    )
    p.add_argument("--tool-ref", default="ArchVarJShaped/JShaped_Default")
    p.add_argument(
        "--wire-length-mm",
        type=float,
        default=None,
        help="Override total wire length in mm for this run.",
    )
    p.add_argument(
        "--tip-angle-deg",
        type=float,
        default=None,
        help="Override tip bend angle in degrees (smaller -> straighter tip).",
    )
    p.add_argument(
        "--tip-angle-scale",
        type=float,
        default=None,
        help="Scale current tip angle by this factor (e.g. 0.2 for near-straight tip).",
    )
    p.add_argument(
        "--tip-radius-mm",
        type=float,
        default=None,
        help="Override tip radius in mm (optional).",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--friction", type=float, default=0.1)
    p.add_argument("--dt-simulation", type=float, default=0.006)
    p.add_argument("--image-frequency-hz", type=float, default=7.5)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--insert-action", type=float, default=35.0)
    p.add_argument("--rotate-action", type=float, default=3.14)
    p.add_argument("--tip-length-mm", type=float, default=3.0)
    p.add_argument("--tip-node-count", type=int, default=3)
    p.add_argument(
        "--free-length-mm",
        type=float,
        default=None,
        help="Effective free length from insertion start to wall along +x (mm). Overrides --wall-x.",
    )
    p.add_argument(
        "--wall-x",
        type=float,
        default=120.0,
        help="Wall x-position in mm. Used when --free-length-mm is not set.",
    )
    p.add_argument("--wall-size-mm", type=float, default=260.0)
    p.add_argument("--wall-thickness-mm", type=float, default=2.0)
    p.add_argument("--contact-epsilon", type=float, default=1e-9)
    p.add_argument("--mesh-path", default="results/force_playground/plane_wall.obj")
    p.add_argument(
        "--wire-color",
        default="0.0,0.9,1.0,1.0",
        help="Debug visual color as r,g,b[,a] in [0,1]. Default is bright cyan.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    insertion_point = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    insertion_direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    if args.free_length_mm is not None:
        free_length_mm = float(args.free_length_mm)
        if not np.isfinite(free_length_mm) or free_length_mm <= 0.0:
            raise SystemExit(
                f"[plane-wall] invalid --free-length-mm: expected > 0, got {args.free_length_mm!r}"
            )
        wall_x = float(insertion_point[0] + free_length_mm)
        wall_x_source = "free_length_mm"
    else:
        wall_x = float(args.wall_x)
        wall_x_source = "wall_x"

    mesh_path = Path(args.mesh_path).expanduser().resolve()
    _write_plane_wall_mesh(
        mesh_path,
        wall_x=float(wall_x),
        wall_size_mm=float(args.wall_size_mm),
        wall_thickness_mm=float(args.wall_thickness_mm),
    )

    device = make_device(str(args.tool_ref))
    if (
        args.tip_angle_deg is not None
        or args.tip_angle_scale is not None
        or args.tip_radius_mm is not None
    ):
        try:
            tip_angle_rad, tip_radius_mm, straight_len_mm = _override_device_tip_shape(
                device,
                tip_angle_deg=args.tip_angle_deg,
                tip_angle_scale=args.tip_angle_scale,
                tip_radius_mm=args.tip_radius_mm,
            )
            print(
                "[plane-wall] tip-shape override: tip_angle_deg={a:.6g} tip_radius_mm={r:.6g} straight_len_mm={s:.6g}".format(
                    a=float(np.rad2deg(tip_angle_rad)),
                    r=float(tip_radius_mm),
                    s=float(straight_len_mm),
                )
            )
        except ValueError as exc:
            raise SystemExit(f"[plane-wall] invalid tip-shape override: {exc}") from exc
    if args.wire_length_mm is not None:
        try:
            length_used, tip_len_used = _override_device_wire_length(
                device,
                float(args.wire_length_mm),
            )
            print(
                "[plane-wall] wire length override: total={lt:.6g} mm tip_arc={lp:.6g} mm straight={ls:.6g} mm".format(
                    lt=float(length_used),
                    lp=float(tip_len_used),
                    ls=float(length_used - tip_len_used),
                )
            )
        except ValueError as exc:
            raise SystemExit(f"[plane-wall] invalid --wire-length-mm: {exc}") from exc
    device.color = _parse_rgba(str(args.wire_color))
    simulation = eve.intervention.simulation.SofaBeamAdapter(
        friction=float(args.friction),
        dt_simulation=float(args.dt_simulation),
    )
    coords_low = np.asarray(
        [-20.0, -0.5 * float(args.wall_size_mm), -0.5 * float(args.wall_size_mm)],
        dtype=np.float64,
    )
    coords_high = np.asarray(
        [float(wall_x) + 40.0, 0.5 * float(args.wall_size_mm), 0.5 * float(args.wall_size_mm)],
        dtype=np.float64,
    )

    dummy_intervention = _DummyIntervention(
        simulation=simulation,
        vessel_tree=_DummyVesselTree(coordinate_space=_DummySpace(low=coords_low, high=coords_high)),
        fluoroscopy=_DummyFluoroscopy(
            image_frequency=float(args.image_frequency_hz),
            image_rot_zx=(20.0, 5.0),
            image_center=[0.0, 0.0, 0.0],
        ),
        target=_DummyTarget(threshold=1.0, coordinates3d=np.asarray([float(wall_x), 0.0, 0.0])),
    )
    visualisation = eve.visualisation.SofaPygame(intervention=dummy_intervention)

    force_scale_to_n = float(
        unit_scale_to_si_newton(
            ForceUnitsConfig(length_unit="mm", mass_unit="kg", time_unit="s")
        )
    )

    print("[plane-wall] controls: Up=push | Down=retract | Left/Right=rotate | Enter=reset | Esc=quit")
    print("[plane-wall] camera: r+w/a/s/d=rotate | w/a/s/d=pan | e/q=zoom")
    print(f"[plane-wall] tool_ref={args.tool_ref}")
    print(f"[plane-wall] wire_color={tuple(device.color)}")
    print(
        "[plane-wall] wall_x={x:.6g} mm wall_size={s:.6g} mm wall_thickness={t:.6g} mm source={src}".format(
            x=float(wall_x),
            s=float(args.wall_size_mm),
            t=float(args.wall_thickness_mm),
            src=wall_x_source,
        )
    )
    print(
        "[plane-wall] effective_free_length_mm={f:.6g}".format(
            f=float(wall_x - float(insertion_point[0])),
        )
    )
    print(f"[plane-wall] friction(mu)={float(args.friction):.6g} dt_sim={float(args.dt_simulation):.6g} s")
    print(f"[plane-wall] force_scale_to_newton={force_scale_to_n:.6g}")
    _print_buckling_estimate(device)

    control_duration_s = 1.0 / float(args.image_frequency_hz)
    velocity_limits = np.asarray(getattr(device, "velocity_limit", (35.0, 3.14)), dtype=np.float64).reshape((2,))

    def _reset_scene(seed: int) -> tuple[Any, Any, Any, float]:
        simulation.reset(
            insertion_point=insertion_point,
            insertion_direction=insertion_direction,
            mesh_path=str(mesh_path),
            devices=[device],
            coords_high=coords_high,
            coords_low=coords_low,
            vessel_visual_path=None,
            seed=int(seed),
        )
        visualisation.reset(episode_nr=0)
        wire_mo = simulation._instruments_combined.DOFs  # noqa: SLF001
        lcp = simulation.root.LCP
        coll_dofs = simulation._instruments_combined.CollisionModel.CollisionDOFs  # noqa: SLF001
        try:
            lcp.computeConstraintForces.value = True
        except Exception:
            pass
        dt_s = float(getattr(getattr(simulation.root, "dt", None), "value", float(args.dt_simulation)))
        return wire_mo, lcp, coll_dofs, dt_s

    try:
        wire_mo, lcp, coll_dofs, dt_s = _reset_scene(int(args.seed))
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

            trans = float(np.clip(trans, -abs(velocity_limits[0]), abs(velocity_limits[0])))
            rot = float(np.clip(rot, -abs(velocity_limits[1]), abs(velocity_limits[1])))

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

            action = np.asarray([[trans, rot]], dtype=np.float64)
            simulation.step(action=action, duration=float(control_duration_s))

            wire_pos = _normalize_xyz_array(_read_data_field(wire_mo, "position"), allow_6dof=True)
            tip_pos = wire_pos[-1] if wire_pos.shape[0] else np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
            tip_to_wall_mm = float(wall_x) - float(tip_pos[0]) if np.all(np.isfinite(tip_pos)) else float("nan")

            (
                tip_force_wall_raw,
                total_force_wall_raw,
                tip_ids,
                force_src,
                tip_local_forces_wall_raw,
            ) = _constraint_tip_force_estimate(
                lcp=lcp,
                coll_dofs=coll_dofs,
                tip_length_mm=float(args.tip_length_mm),
                tip_node_count=int(args.tip_node_count),
                dt_s=float(dt_s),
            )
            tip_force_wall_n = np.asarray(tip_force_wall_raw, dtype=np.float64) * force_scale_to_n
            total_force_wall_n = np.asarray(total_force_wall_raw, dtype=np.float64) * force_scale_to_n
            tip_force_norm_n = float(np.linalg.norm(tip_force_wall_n))
            total_force_norm_n = float(np.linalg.norm(total_force_wall_n))

            tip_local_forces_wall_n = np.asarray(tip_local_forces_wall_raw, dtype=np.float64) * force_scale_to_n
            (
                tip_resultant_norm_n,
                tip_sum_of_local_norms_n,
                tip_max_local_norm_n,
                tip_cancellation_ratio,
            ) = _tip_cancellation_metrics(tip_local_forces_wall_n)

            try:
                cf = np.asarray(lcp.getData("constraintForces").value, dtype=np.float64).reshape((-1,))
            except Exception:
                cf = np.zeros((0,), dtype=np.float64)
            active_rows = int(np.count_nonzero(np.abs(cf) > float(args.contact_epsilon))) if cf.size else 0
            max_abs_cf = float(np.max(np.abs(cf))) if cf.size else 0.0

            sim_time = float(getattr(getattr(simulation.root, "time", None), "value", float(step)))
            print(
                "[plane-wall] step={s:04d} t={t:.3f} "
                "tip=({px:.4f},{py:.4f},{pz:.4f}) tip_to_wall_mm={dw:.4f} "
                "F_tip->wall_N=({fx:.6g},{fy:.6g},{fz:.6g}) |F_tip|_N={fn:.6g} "
                "F_total->wall_N=({tx:.6g},{ty:.6g},{tz:.6g}) |F_total|_N={tn:.6g} "
                "tip_sum_of_local_norms_N={tsn:.6g} tip_max_local_norm_N={tmn:.6g} "
                "tip_cancellation_ratio={tcr:.6g} src={src} nodes={nodes} "
                "active_contact_rows={ar} max_abs_constraintForces={mcf:.6g}".format(
                    s=int(step),
                    t=sim_time,
                    px=float(tip_pos[0]),
                    py=float(tip_pos[1]),
                    pz=float(tip_pos[2]),
                    dw=float(tip_to_wall_mm),
                    fx=float(tip_force_wall_n[0]),
                    fy=float(tip_force_wall_n[1]),
                    fz=float(tip_force_wall_n[2]),
                    fn=float(tip_force_norm_n),
                    tx=float(total_force_wall_n[0]),
                    ty=float(total_force_wall_n[1]),
                    tz=float(total_force_wall_n[2]),
                    tn=float(total_force_norm_n),
                    tsn=float(tip_sum_of_local_norms_n),
                    tmn=float(tip_max_local_norm_n),
                    tcr=float(tip_cancellation_ratio),
                    src=force_src,
                    nodes=np.asarray(tip_ids, dtype=np.int64).tolist(),
                    ar=int(active_rows),
                    mcf=float(max_abs_cf),
                )
            )

            visualisation.render()
            step += 1

            if keys[pygame.K_RETURN]:
                wire_mo, lcp, coll_dofs, dt_s = _reset_scene(int(args.seed))
                step = 0
                print("[plane-wall] reset")
    except KeyboardInterrupt:
        print("[plane-wall] interrupted by user")
    finally:
        try:
            visualisation.close()
        except Exception:
            pass
        try:
            simulation.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
