from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from third_party.stEVE.eve.intervention.vesseltree.util.meshing import load_mesh
from .models import ForceTelemetrySpec, ForceTelemetrySummary

DEFAULT_TIP_THRESHOLD_MM = 3.0
"""Default distal-tip arc-length threshold in millimeters.

Tip semantics: absolute arc length from the distal end of the wire, in mm. A
collision DOF is classified as tip when its arc-length distance from the distal
end is less than or equal to this threshold.
"""

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_monitor_plugin_path(plugin_override: Optional[Path]) -> Optional[Path]:
    candidates: list[Path] = []
    if plugin_override is not None:
        candidates.append(Path(plugin_override).expanduser())

    env_path = os.environ.get("STEVE_WALL_FORCE_MONITOR_PLUGIN")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.append(
        _repo_root()
        / "native"
        / "sofa_wire_force_monitor"
        / "build"
        / "libSofaWireForceMonitor.so"
    )

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except Exception:
            continue
    return None


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


def _set_data_field(obj: Any, name: str, value: Any) -> bool:
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


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _unit_scale_to_newton(units: Any) -> float:
    length_scale = {"m": 1.0, "mm": 1e-3}[str(units.length_unit)]
    mass_scale = {"kg": 1.0, "g": 1e-3}[str(units.mass_unit)]
    time_scale = {"s": 1.0, "ms": 1e-3}[str(units.time_unit)]
    return float(mass_scale * length_scale / (time_scale * time_scale))


def collision_dof_arc_lengths_from_distal_mm(
    wire_positions_mm: Any,
    *,
    collision_dof_wire_indices: Optional[Sequence[int]] = None,
) -> dict[int, float]:
    """Map collision DOF index to arc length from the distal wire end.

    Parameters:
        wire_positions_mm: Wire or collision DOF positions in millimeters,
            shaped as `(n, 3)`. When `collision_dof_wire_indices` is omitted,
            these positions are interpreted as the collision DOF positions.
        collision_dof_wire_indices: Optional wire-DOF indices for each collision
            DOF, in collision-DOF order. When provided, arc length is computed
            along `wire_positions_mm` and sampled at these wire indices.

    Returns:
        A mapping from collision DOF index to arc length from the distal end in
        millimeters. The verified SOFA BeamAdapter convention is that the last
        collision DOF is distal, so the last collision DOF maps to `0.0` mm.

    Unit convention:
        Arc lengths are geometric lengths in millimeters and are independent of
        the scene-force-to-SI-force conversion.
    """

    positions = np.asarray(wire_positions_mm, dtype=np.float64).reshape((-1, 3))
    if positions.shape[0] == 0:
        return {}
    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative_from_start = np.concatenate(([0.0], np.cumsum(deltas)))

    if collision_dof_wire_indices is None:
        selected_indices = np.arange(positions.shape[0], dtype=np.int64)
    else:
        selected_indices = np.asarray(collision_dof_wire_indices, dtype=np.int64).reshape((-1,))
        if selected_indices.size == 0:
            return {}
        if np.any(selected_indices < 0) or np.any(selected_indices >= positions.shape[0]):
            raise ValueError(
                "collision_dof_wire_indices must be valid indices into wire_positions_mm"
            )

    distal_wire_index = int(selected_indices[-1])
    distal_arc_mm = float(cumulative_from_start[distal_wire_index])
    return {
        int(collision_dof_index): float(max(distal_arc_mm - cumulative_from_start[int(wire_index)], 0.0))
        for collision_dof_index, wire_index in enumerate(selected_indices)
    }


def _parse_constraint_rows(raw: Any) -> list[tuple[int, int, np.ndarray]]:
    """Parse SOFA constraint rows into (row_idx, dof_idx, coeff_xyz).

    Accepts the textual serialization stored in MechanicalObject.constraint.
    """
    text = str(raw or "").strip()
    if not text:
        return []
    entries: list[tuple[int, int, np.ndarray]] = []
    for line in text.splitlines():
        if "Constraint ID" in line and "dof ID" in line and "value" in line:
            pattern = re.compile(
                r"Constraint ID\s*:\s*(?P<row>[-+0-9.eE]+)\s*/\s*dof ID\s*:\s*(?P<dof>[-+0-9.eE]+)\s*/\s*value\s*:\s*(?P<vals>.*?)(?=(?:Constraint ID\s*:)|$)",
                re.IGNORECASE,
            )
            for match in pattern.finditer(line):
                try:
                    row_idx = int(float(match.group("row")))
                    dof_idx = int(float(match.group("dof")))
                except Exception:
                    continue
                float_tokens = re.findall(r"[-+0-9.eE]+", match.group("vals") or "")
                if len(float_tokens) < 3:
                    continue
                try:
                    coeff = np.asarray(
                        [float(float_tokens[0]), float(float_tokens[1]), float(float_tokens[2])],
                        dtype=np.float32,
                    )
                except Exception:
                    continue
                if not np.all(np.isfinite(coeff)):
                    continue
                entries.append((row_idx, dof_idx, coeff))
            continue

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
                coeff = np.asarray([float(block[1]), float(block[2]), float(block[3])], dtype=np.float32)
            except Exception:
                continue
            if not np.all(np.isfinite(coeff)):
                continue
            entries.append((row_idx, dof_idx, coeff))
    return entries


def _project_constraint_forces(
    lcp_forces: np.ndarray, constraint_raw: Any, n_points: int, dt_s: Optional[float] = None
) -> tuple[np.ndarray, list[dict]]:
    """Project LCP scalar forces onto per-point xyz vectors using constraint rows.

    Returns (per_point_forces [n_points,3], row_contribs list).
    """
    n = int(max(n_points, 0))
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32), []
    out = np.zeros((n, 3), dtype=np.float32)
    if lcp_forces is None:
        return out, []
    lcp = np.asarray(lcp_forces, dtype=np.float32).reshape((-1,))
    rows = _parse_constraint_rows(constraint_raw)
    if not rows:
        return out, []
    scale = 1.0
    if dt_s is not None and np.isfinite(dt_s) and float(dt_s) > 0.0:
        scale = 1.0 / float(dt_s)

    row_dof_force: dict[tuple[int, int], np.ndarray] = {}
    for row_idx, dof_idx, coeff in rows:
        if row_idx < 0 or row_idx >= lcp.shape[0]:
            continue
        if dof_idx < 0 or dof_idx >= n:
            continue
        contrib = coeff * float(lcp[row_idx]) * float(scale)
        out[dof_idx] += contrib
        key = (int(row_idx), int(dof_idx))
        if key not in row_dof_force:
            row_dof_force[key] = np.zeros((3,), dtype=np.float32)
        row_dof_force[key] += contrib.astype(np.float32)

    row_contribs: list[dict] = []
    for row_idx, dof_idx in sorted(row_dof_force.keys()):
        force_vec = row_dof_force[(row_idx, dof_idx)].astype(np.float32)
        row_contribs.append({
            "row_idx": int(row_idx),
            "dof_idx": int(dof_idx),
            "force_vec": force_vec,
            "force_norm": float(np.linalg.norm(force_vec)),
        })
    return out, row_contribs


@dataclass(frozen=True)
class ForceRuntimeStatus:
    configured: bool
    source: str
    error: str = ""


class EvalV2ForceTelemetryCollector:
    """Collect force telemetry directly from SOFA runtime objects for eval_v2."""

    def __init__(
        self,
        *,
        spec: ForceTelemetrySpec,
        action_dt_s: float,
        anatomy_mesh_path: Optional[Path] = None,
    ) -> None:
        self._spec = spec
        self._action_dt_s = float(max(action_dt_s, 1e-9))
        self._tip_threshold_mm = float(spec.tip_threshold_mm)
        self._anatomy_mesh_path = (
            None if anatomy_mesh_path is None else Path(anatomy_mesh_path)
        )
        self._tip_arc_length_fallback_logged = False
        self._status = ForceRuntimeStatus(False, "uninitialized", "")
        self._last_root_id: Optional[int] = None

        self._available_any = False
        self._contact_detected_any = False
        self._active_constraint_any = False
        self._ordering_stable = True
        self._source = ""
        self._channel = "none"
        self._quality_tier = "unavailable"
        self._association_method = "none"
        self._association_coverage: Optional[float] = None
        self._monitor_nonzero_detected = False

        self._contact_count_max = 0
        self._segment_count_max = 0

        self._total_force_norm_samples: list[float] = []
        self._lcp_max_abs_samples: list[float] = []
        self._lcp_sum_abs_samples: list[float] = []

        self._peak_segment_force_norm: Optional[float] = None
        self._peak_segment_force_step: Optional[int] = None
        self._peak_segment_force_segment_id: Optional[int] = None

        self._mapped_contact_rows_any = False
        self._lcp_contact_export_partial_any = False
        self._lcp_mapped_force_like_max_samples: list[float] = []
        self._lcp_mapped_wall_row_count_max = 0
        self._lcp_contact_export_contact_count_max = 0
        self._lcp_contact_export_coverage_samples: list[float] = []

        self._last_status_text: str = ""
        self._si_conversion_enabled = bool(
            self._spec.mode == "constraint_projected_si_validated"
            and self._spec.units is not None
        )
        self._force_scale_to_newton = (
            _unit_scale_to_newton(self._spec.units)
            if self._si_conversion_enabled and self._spec.units is not None
            else 1.0
        )
        self._lcp_force_like_max_samples: list[float] = []
        # Per-timestep sparse records for later tracing/export
        self._triangle_force_records: list[dict] = []
        self._wire_force_records: list[dict] = []
        self._tip_force_proxy_per_step_N: list[float] = []
        self._triangle_normals_by_id: dict[int, np.ndarray] = {}

    @property
    def status(self) -> ForceRuntimeStatus:
        return self._status

    @staticmethod
    def _import_plugin(sim: Any, plugin_path: Path) -> tuple[bool, str]:
        errors: list[str] = []
        try:
            import SofaRuntime  # type: ignore

            if hasattr(SofaRuntime, "importPlugin"):
                SofaRuntime.importPlugin(str(plugin_path))
                return True, ""
        except Exception as exc:
            errors.append(f"SofaRuntime.importPlugin: {exc}")

        try:
            sim.root.addObject("RequiredPlugin", pluginName=str(plugin_path))
            return True, ""
        except Exception as exc:
            errors.append(f"RequiredPlugin addObject: {exc}")
        return False, "; ".join(errors)

    def _ensure_intrusive_lcp(self, sim: Any) -> ForceRuntimeStatus:
        lcp = getattr(getattr(sim, "root", None), "LCP", None)
        if lcp is None:
            return ForceRuntimeStatus(False, "intrusive_lcp_missing_lcp", "")
        configured = _set_data_field(lcp, "build_lcp", True)
        configured = _set_data_field(lcp, "computeConstraintForces", True) or configured
        if not configured:
            return ForceRuntimeStatus(False, "intrusive_lcp_unconfigured", "")
        return ForceRuntimeStatus(True, "intrusive_lcp", "")

    def _ensure_contact_export(self, sim: Any) -> None:
        """Attach WireWallContactExport to root, best-effort. Failure does not block monitor path.

        vesselNode="@." scans from root so DefaultContactManager contact nodes
        (children of root = LCA of vesselTree and InstrumentCombined) are found.
        """
        export = getattr(sim.root, "wire_wall_contact_export", None)
        if export is not None:
            return
        try:
            sim.root.addObject(
                "WireWallContactExport",
                name="wire_wall_contact_export",
                vesselNode="@.",
                vesselTopology="@vesselTree/MeshTopology",
                collisionMechanicalObject="@InstrumentCombined/CollisionModel/CollisionDOFs",
                contactEpsilon=float(self._spec.contact_epsilon),
            )
            logger.info("[force_telemetry] WireWallContactExport attached to root")
        except Exception as exc:
            logger.warning(f"[force_telemetry] WireWallContactExport attach failed: {exc}")

    def _ensure_triangle_normals_loaded(self) -> None:
        if self._triangle_normals_by_id or self._anatomy_mesh_path is None:
            return
        mesh_path = Path(self._anatomy_mesh_path)
        if not mesh_path.exists():
            logger.warning(
                "[force_telemetry] anatomy mesh path missing for triangle normals: %s",
                mesh_path,
            )
            return
        try:
            mesh = load_mesh(str(mesh_path)).extract_surface().triangulate()
            faces = np.asarray(mesh.faces, dtype=np.int64).reshape((-1, 4))
            if faces.shape[1] != 4:
                return
            triangle_indices = np.asarray(faces[:, 1:4], dtype=np.int64)
            vertices = np.asarray(mesh.points, dtype=np.float64).reshape((-1, 3))
            normals: dict[int, np.ndarray] = {}
            for triangle_id, triangle in enumerate(triangle_indices):
                p0 = vertices[int(triangle[0])]
                p1 = vertices[int(triangle[1])]
                p2 = vertices[int(triangle[2])]
                normal = np.cross(p1 - p0, p2 - p0)
                norm = float(np.linalg.norm(normal))
                if norm <= 0.0:
                    continue
                normals[int(triangle_id)] = (normal / norm).astype(np.float64)
            self._triangle_normals_by_id = normals
        except Exception:
            logger.exception("[force_telemetry] failed to load anatomy mesh normals")

    def _ensure_passive_monitor(self, sim: Any) -> ForceRuntimeStatus:
        lcp = getattr(getattr(sim, "root", None), "LCP", None)
        if lcp is not None:
            _set_data_field(lcp, "computeConstraintForces", True)

        monitor = getattr(sim.root, "wire_wall_force_monitor", None)
        if monitor is not None:
            logger.info("[force_telemetry] Passive monitor already exists on root")
            self._ensure_contact_export(sim)
            return ForceRuntimeStatus(True, "passive_monitor", "")

        plugin_path = resolve_monitor_plugin_path(self._spec.plugin_path)
        if plugin_path is None:
            logger.warning("[force_telemetry] Passive monitor plugin not found")
            return ForceRuntimeStatus(False, "passive_plugin_missing", "plugin not found")

        logger.info(f"[force_telemetry] Resolved plugin path: {plugin_path}")
        ok, error = self._import_plugin(sim, plugin_path)
        if not ok:
            logger.warning(f"[force_telemetry] Plugin import failed: {error}")
            return ForceRuntimeStatus(False, "passive_plugin_load_failed", error)

        logger.info("[force_telemetry] Plugin imported successfully")

        try:
            logger.info("[force_telemetry] Attaching WireWallForceMonitor to scene root")
            sim.root.addObject(
                "WireWallForceMonitor",
                name="wire_wall_force_monitor",
                collisionMechanicalObject="@InstrumentCombined/CollisionModel/CollisionDOFs",
                wireMechanicalObject="@InstrumentCombined/DOFs",
                vesselMechanicalObject="@vesselTree/dofs",
                vesselTopology="@vesselTree/MeshTopology",
                contactEpsilon=float(self._spec.contact_epsilon),
            )
        except Exception as exc:
            logger.error(f"[force_telemetry] Failed to attach monitor: {exc}")
            return ForceRuntimeStatus(False, "passive_monitor_attach_failed", str(exc))

        monitor = getattr(sim.root, "wire_wall_force_monitor", None)
        if monitor is None:
            logger.error("[force_telemetry] Monitor object not found after addObject")
            return ForceRuntimeStatus(False, "passive_monitor_missing_after_attach", "")

        logger.info("[force_telemetry] Passive monitor attached successfully")
        self._ensure_contact_export(sim)
        return ForceRuntimeStatus(True, "passive_monitor", "")

    def ensure_runtime(self, *, intervention: Any) -> ForceRuntimeStatus:
        simulation = getattr(intervention, "simulation", None)
        root = getattr(simulation, "root", None)
        if root is None:
            logger.error("[force_telemetry] Simulation root is None")
            self._status = ForceRuntimeStatus(False, "simulation_root_missing", "")
            return self._status

        current_root_id = id(root)
        if self._last_root_id == current_root_id and self._status.source != "uninitialized":
            logger.info(f"[force_telemetry] Reusing cached status: {self._status.source}")
            return self._status

        logger.info(f"[force_telemetry] ensure_runtime called with mode={self._spec.mode}")
        # Preserve previous ensure_runtime semantics but ensure contact export is
        # attached for mapping when possible. Do not change passive behavior.
        if self._spec.mode == "intrusive_lcp":
            self._status = self._ensure_intrusive_lcp(simulation)
        elif self._spec.mode in {"passive", "constraint_projected_si_validated"}:
            self._status = self._ensure_passive_monitor(simulation)
        else:
            self._status = ForceRuntimeStatus(False, "unsupported_mode", self._spec.mode)
            logger.error(f"[force_telemetry] Unsupported mode: {self._spec.mode}")

        # Best-effort: attach contact export to enable mapping of LCP rows to wall
        # triangles even when running in passive mode. Failure is non-fatal.
        try:
            self._ensure_contact_export(simulation)
        except Exception:
            pass

        logger.info(f"[force_telemetry] ensure_runtime result: {self._status.source} (configured={self._status.configured})")
        self._last_root_id = current_root_id
        return self._status

    def _sample_lcp(self, root: Any) -> None:
        lcp = getattr(root, "LCP", None)
        if lcp is None:
            return
        raw = _read_data_field(lcp, "constraintForces", None)
        if raw is None:
            return
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return
        abs_arr = np.abs(arr[np.isfinite(arr)])
        if abs_arr.size == 0:
            return
        self._active_constraint_any = True
        self._lcp_max_abs_samples.append(float(np.max(abs_arr)))
        self._lcp_sum_abs_samples.append(float(np.sum(abs_arr)))

        dt_s = _to_float(_read_data_field(root, "dt", None))
        if dt_s is None or dt_s <= 0.0:
            dt_s = self._action_dt_s
        if dt_s > 0.0:
            # SOFA LCP constraintForces are impulse-like quantities; convert to
            # force-like values using lambda / dt (see SOFA discussion #3812).
            force_like = abs_arr / float(dt_s)
            force_like *= float(self._force_scale_to_newton)
            if force_like.size > 0:
                self._lcp_force_like_max_samples.append(float(np.max(force_like)))

    def _sample_constraint_projection(self, root: Any, step_index: int) -> None:
        """Project constraint rows to world-space per-dof forces and assemble
        per-triangle and per-wire sparse records when mapping is available.
        """
        self._ensure_triangle_normals_loaded()
        lcp = getattr(root, "LCP", None)
        if lcp is None:
            return
        raw = _read_data_field(lcp, "constraintForces", None)
        if raw is None:
            return
        lcp_arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if lcp_arr.size == 0:
            return

        # Resolve dt
        dt_s = _to_float(_read_data_field(root, "dt", None))
        if dt_s is None or dt_s <= 0.0:
            dt_s = self._action_dt_s

        # Try to find collision DOFs object for parsing constraint rows
        collision_obj = None
        try:
            collision_obj = getattr(root, "InstrumentCombined").CollisionModel.CollisionDOFs
        except Exception:
            try:
                collision_obj = getattr(root, "CollisionDOFs")
            except Exception:
                collision_obj = None

        constraint_raw = None
        n_points = 0
        positions = None
        if collision_obj is not None:
            constraint_raw = _read_data_field(collision_obj, "constraint", None)
            pos_raw = _read_data_field(collision_obj, "position", None)
            positions = np.asarray(pos_raw, dtype=np.float64).reshape(-1, 3) if pos_raw is not None else None
            n_points = int(positions.shape[0]) if positions is not None and positions.ndim == 2 else 0

        # Project LCP scalars to per-dof force vectors using constraint rows, if possible
        row_contribs: list[dict] = []
        per_dof_forces = np.zeros((0, 3), dtype=np.float32)
        if constraint_raw is not None and n_points > 0:
            proj, row_contribs = _project_constraint_forces(
                lcp_forces=lcp_arr, constraint_raw=constraint_raw, n_points=n_points, dt_s=dt_s
            )
            per_dof_forces = proj.astype(np.float32)

        # Now assemble mapping using contact export if present
        export = getattr(root, "wire_wall_contact_export", None)
        mapped_rows = set()
        unmapped_rows = set()
        triangle_force_acc: dict[int, np.ndarray] = {}
        triangle_row_count: dict[int, int] = {}
        wire_force_list: list[dict] = []
        arc_lengths_from_distal_mm: dict[int, float] = {}
        if positions is not None and positions.size > 0:
            if not self._tip_arc_length_fallback_logged:
                logger.info(
                    "[force_telemetry] WireBeamInterpolation arc-length query unavailable; "
                    "falling back to cumulative Euclidean distance over collision DOFs"
                )
                self._tip_arc_length_fallback_logged = True
            arc_lengths_from_distal_mm = collision_dof_arc_lengths_from_distal_mm(positions)

        if export is not None and bool(_read_data_field(export, "available", False)):
            # Read export arrays
            row_indices_raw = _read_data_field(export, "constraintRowIndices", None)
            row_valid_raw = _read_data_field(export, "constraintRowValidFlags", None)
            wall_ids_raw = _read_data_field(export, "wallTriangleIds", None)
            triangle_valid_raw = _read_data_field(export, "triangleIdValidFlags", None)
            collision_dofs_raw = _read_data_field(export, "collisionDofIndices", None)
            collision_valid_raw = _read_data_field(export, "collisionDofValidFlags", None)

            if row_indices_raw is None or row_valid_raw is None:
                row_indices_arr = None
                row_valid_arr = None
            else:
                row_indices_arr = np.asarray(row_indices_raw, dtype=np.int32).reshape(-1)
                row_valid_arr = np.asarray(row_valid_raw, dtype=np.uint32).reshape(-1)

            wall_ids_arr = np.asarray(wall_ids_raw, dtype=np.int32).reshape(-1) if wall_ids_raw is not None else None
            triangle_valid_arr = np.asarray(triangle_valid_raw, dtype=np.uint32).reshape(-1) if triangle_valid_raw is not None else None
            collision_dofs_arr = np.asarray(collision_dofs_raw, dtype=np.int32).reshape(-1) if collision_dofs_raw is not None else None
            collision_valid_arr = np.asarray(collision_valid_raw, dtype=np.uint32).reshape(-1) if collision_valid_raw is not None else None

            # Build mapping from row -> set(triangle_ids), and row -> set(collision_dofs)
            row_to_tri: dict[int, set] = {}
            row_to_collisiondofs: dict[int, set] = {}
            if row_indices_arr is not None and row_valid_arr is not None:
                nrec = min(row_indices_arr.size, row_valid_arr.size)
                for i in range(nrec):
                    ridx = int(row_indices_arr[i])
                    valid = bool(int(row_valid_arr[i]))
                    if not valid or ridx < 0:
                        self._lcp_contact_export_partial_any = True
                        continue
                    if wall_ids_arr is not None and i < wall_ids_arr.size:
                        tri_valid = True
                        if triangle_valid_arr is not None and i < triangle_valid_arr.size:
                            tri_valid = bool(int(triangle_valid_arr[i]))
                        tri = int(wall_ids_arr[i])
                        if tri_valid:
                            row_to_tri.setdefault(ridx, set()).add(tri)
                        else:
                            self._lcp_contact_export_partial_any = True
                    if collision_dofs_arr is not None and i < collision_dofs_arr.size:
                        cd_valid = True
                        if collision_valid_arr is not None and i < collision_valid_arr.size:
                            cd_valid = bool(int(collision_valid_arr[i]))
                        cd = int(collision_dofs_arr[i])
                        if cd_valid:
                            row_to_collisiondofs.setdefault(ridx, set()).add(cd)
                        else:
                            self._lcp_contact_export_partial_any = True

            coverage = _to_float(_read_data_field(export, "explicitCoverage", None))
            if coverage is not None:
                self._lcp_contact_export_coverage_samples.append(coverage)
                if coverage < 1.0:
                    self._lcp_contact_export_partial_any = True

            # Iterate row contributions and distribute forces
            for rc in row_contribs:
                r = int(rc.get("row_idx", -1))
                fv = np.asarray(rc.get("force_vec", np.zeros((3,))), dtype=np.float32).reshape((3,))
                if r in row_to_tri and row_to_tri[r]:
                    mapped_rows.add(r)
                    for tri in row_to_tri[r]:
                        triangle_force_acc.setdefault(tri, np.zeros((3,), dtype=np.float32))
                        triangle_force_acc[tri] += fv
                        triangle_row_count[tri] = triangle_row_count.get(tri, 0) + 1
                else:
                    unmapped_rows.add(r)

                # wire/collision dof mapping: create per-wire record for each associated collision dof
                if r in row_to_collisiondofs and row_to_collisiondofs[r]:
                    for cd in row_to_collisiondofs[r]:
                        world_pos = None
                        if positions is not None and 0 <= int(cd) < positions.shape[0]:
                            p = positions[int(cd)].astype(np.float32)
                            world_pos = p.tolist()
                        arc_length_mm = arc_lengths_from_distal_mm.get(int(cd))
                        is_tip = (
                            arc_length_mm is not None
                            and float(arc_length_mm) <= float(self._tip_threshold_mm)
                        )
                        wire_force_list.append(
                            {
                                "timestep": int(step_index),
                                "wire_collision_dof": int(cd),
                                "world_pos": world_pos,
                                "arc_length_from_distal_mm": None if arc_length_mm is None else float(arc_length_mm),
                                "is_tip": bool(is_tip),
                                "row_idx": int(r),
                                "fx_scene": float(fv[0]),
                                "fy_scene": float(fv[1]),
                                "fz_scene": float(fv[2]),
                                "norm_scene": float(np.linalg.norm(fv)),
                                "fx_N": float(fv[0] * float(self._force_scale_to_newton)),
                                "fy_N": float(fv[1] * float(self._force_scale_to_newton)),
                                "fz_N": float(fv[2] * float(self._force_scale_to_newton)),
                                "norm_N": float(np.linalg.norm(fv) * float(self._force_scale_to_newton)),
                                "mapped": True,
                            }
                        )

        else:
            # No export: treat all nonzero row_contribs as unmapped diagnostic
            for rc in row_contribs:
                r = int(rc.get("row_idx", -1))
                fv = np.asarray(rc.get("force_vec", np.zeros((3,))), dtype=np.float32).reshape((3,))
                if float(np.linalg.norm(fv)) > 0.0:
                    unmapped_rows.add(r)

        # Assemble triangle records
        for tri, vec in triangle_force_acc.items():
            norm = float(np.linalg.norm(vec))
            self._triangle_force_records.append(
                {
                    "timestep": int(step_index),
                    "triangle_id": int(tri),
                    "fx_N": float(vec[0] * float(self._force_scale_to_newton)),
                    "fy_N": float(vec[1] * float(self._force_scale_to_newton)),
                    "fz_N": float(vec[2] * float(self._force_scale_to_newton)),
                    "norm_N": float(norm * float(self._force_scale_to_newton)),
                    "contributing_rows": int(triangle_row_count.get(tri, 0)),
                    "mapped": True,
                }
            )

        # Append wire records
        for rec in wire_force_list:
            self._wire_force_records.append(rec)

        tip_force_step_max_N = 0.0
        if wire_force_list and self._triangle_normals_by_id:
            tip_row_ids = {
                int(record["row_idx"])
                for record in wire_force_list
                if bool(record.get("is_tip", False))
            }
            row_to_force_N: dict[int, np.ndarray] = {}
            for rc in row_contribs:
                row_idx = int(rc.get("row_idx", -1))
                row_to_force_N[row_idx] = np.asarray(
                    rc.get("force_vec", np.zeros((3,))), dtype=np.float64
                ).reshape((3,)) * float(self._force_scale_to_newton)
            for row_idx in tip_row_ids:
                for triangle_id in row_to_tri.get(row_idx, set()) if 'row_to_tri' in locals() else ():
                    normal = self._triangle_normals_by_id.get(int(triangle_id))
                    force_vec_N = row_to_force_N.get(int(row_idx))
                    if normal is None or force_vec_N is None:
                        continue
                    wire_to_wall_force_N = -force_vec_N
                    compressive_normal_N = max(
                        0.0, float(np.dot(wire_to_wall_force_N, normal))
                    )
                    if compressive_normal_N > tip_force_step_max_N:
                        tip_force_step_max_N = compressive_normal_N
        self._tip_force_proxy_per_step_N.append(float(tip_force_step_max_N))

        # Update mapping counters
        self._lcp_mapped_wall_row_count_max = max(self._lcp_mapped_wall_row_count_max, int(len(mapped_rows)))
        self._last_constraint_projection = {
            "total_rows": int(lcp_arr.size),
            "nonzero_rows": int(np.count_nonzero(np.abs(lcp_arr) > 0.0)),
            "mapped_rows": int(len(mapped_rows)),
            "unmapped_rows": int(len(unmapped_rows)),
            "mapping_coverage": None if (len(mapped_rows) + len(unmapped_rows)) == 0 else float(len(mapped_rows) / (len(mapped_rows) + len(unmapped_rows))),
        }

    def _sample_contact_export(self, root: Any) -> None:
        """Read WireWallContactExport and accumulate mapped wall-contact LCP force samples."""
        export = getattr(root, "wire_wall_contact_export", None)
        if export is None:
            return
        if not bool(_read_data_field(export, "available", False)):
            return

        row_indices_raw = _read_data_field(export, "constraintRowIndices", None)
        row_valid_raw = _read_data_field(export, "constraintRowValidFlags", None)
        if row_indices_raw is None or row_valid_raw is None:
            return

        row_indices = np.asarray(row_indices_raw, dtype=np.int32).reshape(-1)
        row_valid = np.asarray(row_valid_raw, dtype=np.uint32).reshape(-1)
        n = min(row_indices.size, row_valid.size)
        if n == 0:
            return

        contact_count = int(_read_data_field(export, "contactCount", 0) or 0)
        self._lcp_contact_export_contact_count_max = max(
            self._lcp_contact_export_contact_count_max,
            contact_count,
        )

        valid_mask = row_valid[:n].astype(bool)
        # Deduplicate: C++ emits one record per (contact_point, row) pair; rows can repeat.
        valid_rows = np.unique(row_indices[:n][valid_mask])
        valid_rows = valid_rows[valid_rows >= 0]
        if valid_rows.size == 0:
            return

        lcp = getattr(root, "LCP", None)
        if lcp is None:
            return
        raw_lcp = _read_data_field(lcp, "constraintForces", None)
        if raw_lcp is None:
            return
        lcp_arr = np.asarray(raw_lcp, dtype=np.float64).reshape(-1)
        if lcp_arr.size == 0:
            return

        in_range = valid_rows[valid_rows < lcp_arr.size]
        if in_range.size == 0:
            return

        mapped_forces = lcp_arr[in_range]
        finite_mask = np.isfinite(mapped_forces)
        if not np.any(finite_mask):
            return

        mapped_abs = np.abs(mapped_forces[finite_mask])
        if mapped_abs.size == 0:
            return

        self._mapped_contact_rows_any = True
        self._lcp_mapped_wall_row_count_max = max(
            self._lcp_mapped_wall_row_count_max, int(in_range.size)
        )

        coverage = _to_float(_read_data_field(export, "explicitCoverage", None))
        if coverage is not None:
            self._lcp_contact_export_coverage_samples.append(coverage)

        dt_s = _to_float(_read_data_field(root, "dt", None))
        if dt_s is None or dt_s <= 0.0:
            dt_s = self._action_dt_s
        if dt_s > 0.0:
            force_like = mapped_abs / float(dt_s) * float(self._force_scale_to_newton)
            if force_like.size > 0:
                self._lcp_mapped_force_like_max_samples.append(float(np.max(force_like)))

        # populate basic mapping stats for downstream per-timestep assembly
        try:
            wall_ids_raw = _read_data_field(export, "wallTriangleIds", None)
            collision_dofs_raw = _read_data_field(export, "collisionDofIndices", None)
        except Exception:
            wall_ids_raw = None
            collision_dofs_raw = None
        self._last_export_rows = dict(
            row_indices=row_indices if 'row_indices' in locals() else None,
            row_valid=row_valid if 'row_valid' in locals() else None,
            wall_triangle_ids=np.asarray(wall_ids_raw, dtype=np.int32).reshape(-1) if wall_ids_raw is not None else None,
            collision_dof_indices=np.asarray(collision_dofs_raw, dtype=np.int32).reshape(-1) if collision_dofs_raw is not None else None,
        )

    def capture_step(self, *, intervention: Any, step_index: int) -> None:
        simulation = getattr(intervention, "simulation", None)
        root = getattr(simulation, "root", None)
        if root is None:
            return

        self._sample_lcp(root)
        self._sample_contact_export(root)
        # Project constraint rows to per-dof forces and assemble per-timestep records
        try:
            self._sample_constraint_projection(root, step_index)
        except Exception:
            logger.exception("[force_telemetry] constraint projection sampling failed")
        monitor = getattr(root, "wire_wall_force_monitor", None)
        # Monitor is treated as a legacy/diagnostic data source only. We still
        # read its fields for diagnostics, but we do not rely on it for
        # validated wall-force scoring. Primary source is LCP + contact export.
        if monitor is not None:
            monitor_available = bool(_read_data_field(monitor, "available", False))
            self._available_any = self._available_any or monitor_available
            self._source = str(_read_data_field(monitor, "source", self._source or "passive_monitor"))
            self._last_status_text = str(_read_data_field(monitor, "status", ""))

            total_norm = _to_float(_read_data_field(monitor, "totalForceNorm", None))
            if total_norm is not None:
                if self._si_conversion_enabled:
                    total_norm *= float(self._force_scale_to_newton)
                self._total_force_norm_samples.append(total_norm)
                if total_norm > float(self._spec.contact_epsilon):
                    self._monitor_nonzero_detected = True

            contact_count = int(_read_data_field(monitor, "contactCount", 0) or 0)
            wall_segment_count = int(_read_data_field(monitor, "wallSegmentCount", 0) or 0)
            self._contact_count_max = max(self._contact_count_max, contact_count)
            self._segment_count_max = max(self._segment_count_max, wall_segment_count)
            if contact_count > 0 or (total_norm is not None and total_norm > float(self._spec.contact_epsilon)):
                self._contact_detected_any = True

            segment_forces = np.asarray(_read_data_field(monitor, "segmentForceVectors", []), dtype=np.float64)
            if segment_forces.ndim == 2 and segment_forces.shape[1] >= 3 and segment_forces.shape[0] > 0:
                if self._si_conversion_enabled:
                    segment_forces = segment_forces * float(self._force_scale_to_newton)
                norms = np.linalg.norm(segment_forces[:, :3], axis=1)
                if norms.size > 0 and np.any(np.isfinite(norms)):
                    peak_idx = int(np.nanargmax(norms))
                    peak_norm = float(norms[peak_idx])
                    if self._peak_segment_force_norm is None or peak_norm > self._peak_segment_force_norm:
                        self._peak_segment_force_norm = peak_norm
                        self._peak_segment_force_step = int(step_index)
                        self._peak_segment_force_segment_id = peak_idx
                    if peak_norm > float(self._spec.contact_epsilon):
                        self._monitor_nonzero_detected = True

    def build_summary(self) -> ForceTelemetrySummary:
        monitor_has_signal = bool(self._total_force_norm_samples and self._monitor_nonzero_detected)
        mapped_lcp_has_signal = bool(self._mapped_contact_rows_any and self._lcp_mapped_force_like_max_samples)
        raw_lcp_has_signal = bool(self._lcp_force_like_max_samples)

        # New priority: mapped LCP (scoreable) > unmapped LCP (diagnostic) > legacy monitor (diagnostic only)
        use_mapped_lcp = mapped_lcp_has_signal
        use_unmapped_lcp = (not use_mapped_lcp) and raw_lcp_has_signal
        mapped_lcp_partial = bool(use_mapped_lcp and self._lcp_contact_export_partial_any)

        if use_mapped_lcp and not mapped_lcp_partial:
            effective_force_samples = self._lcp_mapped_force_like_max_samples
            available_for_score = bool(effective_force_samples)
            # Keep legacy validation_status 'ok' when mapped rows exist and produce samples
            validation_status = "ok" if available_for_score else "configured_no_samples"
        elif use_mapped_lcp and mapped_lcp_partial:
            effective_force_samples = self._lcp_mapped_force_like_max_samples
            available_for_score = False
            validation_status = "partial"
        elif use_unmapped_lcp:
            effective_force_samples = self._lcp_force_like_max_samples
            available_for_score = False
            validation_status = "lcp_only_unmapped"
        else:
            # No LCP signal; monitor may have diagnostic data but cannot be used for scoring
            effective_force_samples = self._total_force_norm_samples
            available_for_score = False
            if self._spec.required:
                validation_status = "required_missing"
            elif self._status.configured:
                validation_status = "configured_no_samples"
            else:
                validation_status = "not_collected"

        lcp_max_abs_max = (
            float(np.max(self._lcp_max_abs_samples)) if self._lcp_max_abs_samples else None
        )
        lcp_sum_abs_mean = (
            float(np.mean(self._lcp_sum_abs_samples)) if self._lcp_sum_abs_samples else None
        )

        # Diagnostic totals: prefer monitor samples for human-readable diagnostics
        # even when scoring uses mapped LCP. This preserves historical report fields.
        if monitor_has_signal:
            total_force_norm_max = (
                float(np.max(self._total_force_norm_samples)) if self._total_force_norm_samples else None
            )
            total_force_norm_mean = (
                float(np.mean(self._total_force_norm_samples)) if self._total_force_norm_samples else None
            )
        else:
            total_force_norm_max = float(np.max(effective_force_samples)) if effective_force_samples else None
            total_force_norm_mean = float(np.mean(effective_force_samples)) if effective_force_samples else None

        # "degraded" means data is present but not at highest confidence.
        # LCP fallback gets "degraded" because constraint rows are collected but may be
        # unmapped or only partially mapped.
        # "unavailable" means no data at all.
        if use_mapped_lcp and not mapped_lcp_partial:
            quality_tier = "validated"
        elif use_mapped_lcp or use_unmapped_lcp:
            quality_tier = "degraded"
        else:
            quality_tier = "unavailable"

        if "nearest_triangle_centroid" in self._last_status_text:
            self._association_method = "force_points_nearest_triangle"
            self._association_coverage = 0.0

        tip_force_records = tuple(
            dict(record) for record in self._wire_force_records if bool(record.get("is_tip", False))
        )
        tip_force_total_vector = (
            np.asarray(
                [
                    [
                        float(record.get("fx_N", 0.0)),
                        float(record.get("fy_N", 0.0)),
                        float(record.get("fz_N", 0.0)),
                    ]
                    for record in tip_force_records
                ],
                dtype=np.float64,
            ).sum(axis=0)
            if tip_force_records
            else np.zeros((3,), dtype=np.float64)
        )
        tip_force_available = bool(tip_force_records)

        source = self._source or self._status.source
        if use_mapped_lcp:
            source = f"{source}:fallback_lcp_dt_mapped"
            channel = "lcp.constraintForces/dt"
        elif use_unmapped_lcp:
            source = f"{source}:fallback_lcp_dt_unmapped"
            channel = "lcp.constraintForces/dt"
        else:
            channel = "none"

        return ForceTelemetrySummary(
            available_for_score=available_for_score,
            validation_status=validation_status,
            validation_error=(self._status.error or None),
            source=source,
            channel=channel,
            quality_tier=quality_tier,
            association_method=self._association_method,
            association_coverage=self._association_coverage,
            ordering_stable=self._ordering_stable,
            active_constraint_any=self._active_constraint_any,
            contact_detected_any=self._contact_detected_any,
            contact_count_max=int(self._contact_count_max),
            segment_count_max=int(self._segment_count_max),
            lcp_max_abs_max=lcp_max_abs_max,
            lcp_sum_abs_mean=lcp_sum_abs_mean,
            total_force_norm_max=total_force_norm_max,
            total_force_norm_mean=total_force_norm_mean,
            total_force_norm_max_newton=total_force_norm_max,
            total_force_norm_mean_newton=total_force_norm_mean,
            peak_segment_force_norm=self._peak_segment_force_norm,
            peak_segment_force_norm_newton=self._peak_segment_force_norm,
            peak_segment_force_step=self._peak_segment_force_step,
            peak_segment_force_segment_id=self._peak_segment_force_segment_id,
            peak_segment_force_time_s=(
                None
                if self._peak_segment_force_step is None
                else float(self._peak_segment_force_step) * self._action_dt_s
            ),
            tip_force_available=tip_force_available,
            tip_force_validation_status="ok" if tip_force_available else "unmapped",
            tip_force_records=tip_force_records,
            tip_force_total_vector_N=(
                float(tip_force_total_vector[0]),
                float(tip_force_total_vector[1]),
                float(tip_force_total_vector[2]),
            ),
            tip_force_total_norm_N=float(np.linalg.norm(tip_force_total_vector)),
            tip_force_peak_normal_N=(
                float(np.max(self._tip_force_proxy_per_step_N))
                if self._tip_force_proxy_per_step_N
                else None
            ),
            tip_force_total_mean_N=(
                float(np.mean(self._tip_force_proxy_per_step_N))
                if self._tip_force_proxy_per_step_N
                else None
            ),
            lcp_mapped_wall_row_count_max=int(self._lcp_mapped_wall_row_count_max),
            lcp_contact_export_coverage=(
                float(np.max(self._lcp_contact_export_coverage_samples))
                if self._lcp_contact_export_coverage_samples
                else None
            ),
        )


__all__ = [
    "DEFAULT_TIP_THRESHOLD_MM",
    "EvalV2ForceTelemetryCollector",
    "ForceRuntimeStatus",
    "collision_dof_arc_lengths_from_distal_mm",
    "resolve_monitor_plugin_path",
]
