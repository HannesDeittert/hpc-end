from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from steve_recommender.evaluation.force_si import unit_scale_to_si_newton

from .config import ForcePlaygroundConfig
from .controllers import ForceApplicator, StepCommand
if TYPE_CHECKING:
    from .scene_factory import ForcePlaygroundScene

_VALIDATED_ASSOC = {
    "native_contact_export_triangle_id",
    "native_contact_export_triangle_id_global_row_bridge",
}


@dataclass
class StepTelemetry:
    step_record: Dict[str, Any]
    triangle_rows: List[Dict[str, Any]]


def _to_vec3(value: Any, default: Optional[np.ndarray] = None) -> np.ndarray:
    if default is None:
        default = np.zeros((3,), dtype=np.float32)
    try:
        arr = np.asarray(value, dtype=np.float32).reshape((3,))
        return arr
    except Exception:
        return np.asarray(default, dtype=np.float32).reshape((3,))


def _to_nx3(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            return np.zeros((0, 3), dtype=np.float32)
        arr = arr.reshape((-1, 3))
    if arr.ndim >= 2:
        arr = arr.reshape((arr.shape[0], -1))
        if arr.shape[1] < 3:
            return np.zeros((0, 3), dtype=np.float32)
        arr = arr[:, :3]
    return arr.astype(np.float32)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


class TelemetryCollector:
    def __init__(self, scene: "ForcePlaygroundScene", cfg: ForcePlaygroundConfig) -> None:
        self._scene = scene
        self._cfg = cfg
        self._force_scale_to_newton = float(unit_scale_to_si_newton(cfg.units))
        self._force_applicator: Optional[ForceApplicator] = None
        if cfg.mode == "open_loop_force":
            self._force_applicator = ForceApplicator(
                scene.simulation,
                node_index=int(cfg.control.open_loop_force_node_index),
            )

    def _read_lambda(self) -> Tuple[np.ndarray, np.ndarray, int]:
        sim = self._scene.simulation
        try:
            raw = sim.root.LCP.constraintForces.value
        except Exception:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0
        lam = np.asarray(raw, dtype=np.float32).reshape((-1,))
        dt_s = float(self._scene.dt_s) if float(self._scene.dt_s) > 0.0 else float("nan")
        if np.isfinite(dt_s) and dt_s > 0.0:
            lam_dt = lam / dt_s
        else:
            lam_dt = np.full_like(lam, np.nan)
        active = int(np.count_nonzero(np.abs(lam) > float(self._cfg.contact_epsilon)))
        return lam, lam_dt, active

    def _apply_commanded_force(self, command: StepCommand) -> Tuple[np.ndarray, np.ndarray, str]:
        cmd_n = _to_vec3(command.commanded_force_vector_n)
        if self._force_applicator is None:
            return cmd_n, np.zeros((3,), dtype=np.float32), "inactive"

        if not np.isfinite(self._force_scale_to_newton) or self._force_scale_to_newton <= 0.0:
            return cmd_n, np.zeros((3,), dtype=np.float32), "invalid_force_scale"

        cmd_scene = cmd_n / float(self._force_scale_to_newton)
        status = self._force_applicator.apply_force_scene(cmd_scene)
        return cmd_n, cmd_scene.astype(np.float32), status

    def _dense_triangle_forces(self, info: Dict[str, Any], n_triangles: int) -> np.ndarray:
        dense = _to_nx3(info.get("wall_segment_force_vectors_N", info.get("wall_segment_force_vectors")))
        if dense.shape[0] == n_triangles:
            return dense
        out = np.zeros((n_triangles, 3), dtype=np.float32)
        n_copy = int(min(n_triangles, dense.shape[0]))
        if n_copy > 0:
            out[:n_copy] = dense[:n_copy]
        return out

    @staticmethod
    def _normal_tangent(force_vec: np.ndarray, normal_vec: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
        n = _to_vec3(normal_vec)
        fn_scalar = float(np.dot(force_vec, n))
        fn_vec = n * fn_scalar
        ft_vec = force_vec - fn_vec
        fn_abs = float(abs(fn_scalar))
        ft_abs = float(np.linalg.norm(ft_vec))
        return fn_scalar, fn_vec.astype(np.float32), ft_vec.astype(np.float32), fn_abs, ft_abs

    def step(self, step_index: int, command: StepCommand) -> StepTelemetry:
        cmd_force_n, cmd_force_scene, applied_status = self._apply_commanded_force(command)

        self._scene.intervention.step(np.asarray(command.action, dtype=np.float32))
        self._scene.force_info.step()
        info = self._scene.force_info.info

        n_triangles = int(self._scene.wall_triangles.shape[0])
        tri_forces = self._dense_triangle_forces(info, n_triangles)
        tri_norms = np.linalg.norm(tri_forces, axis=1) if tri_forces.size else np.zeros((0,), dtype=np.float32)

        total_force_vec = _to_vec3(info.get("wall_total_force_vector_N", info.get("wall_total_force_vector")))
        sum_force_vec = np.sum(tri_forces, axis=0) if tri_forces.size else np.zeros((3,), dtype=np.float32)

        norm_sum_vector = float(np.linalg.norm(sum_force_vec))
        sum_norm = float(np.sum(tri_norms)) if tri_norms.size else 0.0
        peak_triangle_force = float(np.max(tri_norms)) if tri_norms.size else 0.0
        peak_triangle_id = int(np.argmax(tri_norms)) if tri_norms.size else -1

        normals = self._scene.wall_normals
        if normals.shape[0] != n_triangles:
            normals = np.zeros((n_triangles, 3), dtype=np.float32)

        triangle_rows: List[Dict[str, Any]] = []
        fn_vec_sum = np.zeros((3,), dtype=np.float32)
        ft_vec_sum = np.zeros((3,), dtype=np.float32)
        sum_abs_fn = 0.0
        sum_abs_ft = 0.0

        for tri_id in range(n_triangles):
            f_vec = tri_forces[tri_id]
            n_vec = normals[tri_id] if tri_id < normals.shape[0] else np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            fn_scalar, fn_vec, ft_vec, fn_abs, ft_abs = self._normal_tangent(f_vec, n_vec)
            fn_vec_sum += fn_vec
            ft_vec_sum += ft_vec
            sum_abs_fn += fn_abs
            sum_abs_ft += ft_abs
            ratio = float(ft_abs / fn_abs) if fn_abs > 1e-12 else float("nan")
            triangle_rows.append(
                {
                    "step": int(step_index),
                    "triangle_id": int(tri_id),
                    "active": int(float(np.linalg.norm(f_vec)) > float(self._cfg.contact_epsilon)),
                    "force_x": float(f_vec[0]),
                    "force_y": float(f_vec[1]),
                    "force_z": float(f_vec[2]),
                    "force_norm": float(np.linalg.norm(f_vec)),
                    "normal_x": float(n_vec[0]),
                    "normal_y": float(n_vec[1]),
                    "normal_z": float(n_vec[2]),
                    "fn_scalar": float(fn_scalar),
                    "fn_x": float(fn_vec[0]),
                    "fn_y": float(fn_vec[1]),
                    "fn_z": float(fn_vec[2]),
                    "fn_abs": float(fn_abs),
                    "ft_x": float(ft_vec[0]),
                    "ft_y": float(ft_vec[1]),
                    "ft_z": float(ft_vec[2]),
                    "ft_abs": float(ft_abs),
                    "ft_over_fn": float(ratio),
                }
            )

        lam, lam_dt, lambda_active_rows = self._read_lambda()
        lam_abs = np.abs(lam)
        lam_dt_abs = np.abs(lam_dt[np.isfinite(lam_dt)]) if lam_dt.size else np.zeros((0,), dtype=np.float32)

        association_method = str(info.get("wall_force_association_method", "none") or "none")
        association_coverage = _safe_float(info.get("wall_force_association_coverage", float("nan")))
        association_explicit_force_coverage = _safe_float(
            info.get("wall_force_association_explicit_force_coverage", float("nan"))
        )

        explicit_association = bool(
            association_method in _VALIDATED_ASSOC
            and np.isfinite(association_coverage)
            and association_coverage >= (1.0 - 1e-6)
            and np.isfinite(association_explicit_force_coverage)
            and association_explicit_force_coverage >= (1.0 - 1e-6)
        )
        internal_validated = bool(str(info.get("wall_force_quality_tier", "")) == "validated")
        si_converted = bool(info.get("unit_converted_si", False))

        step_record: Dict[str, Any] = {
            "step": int(step_index),
            "time_s": float(step_index) * float(self._scene.dt_s),
            "scene": self._cfg.scene,
            "probe": self._cfg.probe,
            "mode": self._cfg.mode,
            "insert_action": float(command.action.reshape((-1,))[0]),
            "rotate_action": float(command.action.reshape((-1,))[1]),
            "controller_status": str(command.controller_status),
            "commanded_force_vector_n": [float(cmd_force_n[0]), float(cmd_force_n[1]), float(cmd_force_n[2])],
            "commanded_force_scalar_n": float(command.commanded_force_scalar_n),
            "commanded_force_vector_scene": [
                float(cmd_force_scene[0]),
                float(cmd_force_scene[1]),
                float(cmd_force_scene[2]),
            ],
            "command_apply_status": str(applied_status),
            # Required step metrics.
            "norm_sum_vector": float(norm_sum_vector),
            "sum_norm": float(sum_norm),
            "peak_triangle_force": float(peak_triangle_force),
            "peak_triangle_id": int(peak_triangle_id),
            "total_force_vector": [
                float(total_force_vec[0]),
                float(total_force_vec[1]),
                float(total_force_vec[2]),
            ],
            "total_force_norm": float(np.linalg.norm(total_force_vec)),
            "sum_force_vector": [
                float(sum_force_vec[0]),
                float(sum_force_vec[1]),
                float(sum_force_vec[2]),
            ],
            "sum_force_gap_norm": float(np.linalg.norm(sum_force_vec - total_force_vec)),
            "sum_abs_fn": float(sum_abs_fn),
            "sum_abs_ft": float(sum_abs_ft),
            "fn_vector_sum": [float(fn_vec_sum[0]), float(fn_vec_sum[1]), float(fn_vec_sum[2])],
            "ft_vector_sum": [float(ft_vec_sum[0]), float(ft_vec_sum[1]), float(ft_vec_sum[2])],
            "decomposition_gap_norm": float(np.linalg.norm((fn_vec_sum + ft_vec_sum) - sum_force_vec)),
            # lambda and lambda/dt in parallel.
            "lambda_abs_sum": float(np.sum(lam_abs)) if lam_abs.size else 0.0,
            "lambda_abs_max": float(np.max(lam_abs)) if lam_abs.size else 0.0,
            "lambda_dt_abs_sum": float(np.sum(lam_dt_abs)) if lam_dt_abs.size else 0.0,
            "lambda_dt_abs_max": float(np.max(lam_dt_abs)) if lam_dt_abs.size else 0.0,
            "lambda_active_rows_count": int(lambda_active_rows),
            "lambda_values": [float(x) for x in lam.tolist()],
            "lambda_dt_values": [float(x) for x in lam_dt.tolist()],
            # Explicit validation stages.
            "si_converted": bool(si_converted),
            "explicit_association": bool(explicit_association),
            "internal_validated": bool(internal_validated),
            "oracle_physical_pass": None,
            "oracle_reason": "not_evaluated",
            # Keep stable quality diagnostics from production collector.
            "quality_tier": str(info.get("wall_force_quality_tier", "")),
            "association_method": association_method,
            "association_coverage": float(association_coverage),
            "association_explicit_force_coverage": float(association_explicit_force_coverage),
            "association_ordering_stable": bool(info.get("wall_force_association_ordering_stable", False)),
            "active_constraint_step": bool(info.get("wall_force_active_constraint_step", False)),
            "gap_active_projected_count": int(info.get("wall_force_gap_active_projected_count", 0)),
            "gap_explicit_mapped_count": int(info.get("wall_force_gap_explicit_mapped_count", 0)),
            "gap_unmapped_count": int(info.get("wall_force_gap_unmapped_count", 0)),
            "gap_contact_mode": str(info.get("wall_force_gap_contact_mode", "none")),
            "native_contact_export_status": str(info.get("wall_native_contact_export_status", "")),
            "wall_contact_count": int(info.get("wall_contact_count", 0)),
            "wall_contact_detected": bool(info.get("wall_contact_detected", False)),
            "force_units": {
                "length_unit": self._cfg.units.length_unit,
                "mass_unit": self._cfg.units.mass_unit,
                "time_unit": self._cfg.units.time_unit,
            },
            "dt_s": float(self._scene.dt_s),
            "wall_reference_normal": [
                float(self._scene.wall_reference_normal[0]),
                float(self._scene.wall_reference_normal[1]),
                float(self._scene.wall_reference_normal[2]),
            ],
        }

        return StepTelemetry(step_record=step_record, triangle_rows=triangle_rows)

    def close(self) -> None:
        if self._force_applicator is not None:
            self._force_applicator.clear()
