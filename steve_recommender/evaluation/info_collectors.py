from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import ForceUnitsConfig
from .force_si import unit_scale_to_si_newton, units_to_dict
from .sofa_force_monitor import SofaForceMonitorRuntime


class TipStateInfo:
    """Collect basic per-step tip state from the intervention.

    This is implemented as an *eve Info* compatible object, but we keep it local
    to avoid editing upstream stEVE packages.

    Important:
    - stEVE's `eve.Env` only requires an "Info-like" object with:
        * `info` property (dict)
        * `step()` called after every `env.step()`
        * `reset(episode_nr)` called at the start of every episode
        * `__call__()` returning the dict

    We intentionally do *not* subclass `eve.info.Info` here so importing this
    module never relies on custom sys.path hacks.
    """

    def __init__(self, intervention, name_prefix: str = "tip") -> None:
        if not isinstance(name_prefix, str) or not name_prefix:
            raise ValueError("name_prefix must be a non-empty string")

        self.name = name_prefix
        self._intervention = intervention
        self._name_prefix = name_prefix

        self.reset()

    def reset(self, episode_nr: int = 0) -> None:
        _ = episode_nr
        # Cached values updated on every step().
        self._pos3d = np.full((3,), np.nan, dtype=np.float32)
        self._inserted_length = float("nan")
        self._rotation = float("nan")

        # Best-effort: try to capture an initial state after the intervention reset.
        try:
            self.step()
        except Exception:
            pass

    def step(self) -> None:
        # tracking3d[0] is the tip (distal-most point) in tracking coordinates.
        try:
            self._pos3d = np.asarray(
                self._intervention.fluoroscopy.tracking3d[0],
                dtype=np.float32,
            )
        except Exception:
            self._pos3d = np.full((3,), np.nan, dtype=np.float32)

        # Device state is accessible via intervention properties.
        try:
            self._inserted_length = float(self._intervention.device_lengths_inserted[0])
        except Exception:
            self._inserted_length = float("nan")

        try:
            self._rotation = float(self._intervention.device_rotations[0])
        except Exception:
            self._rotation = float("nan")

    @property
    def info(self) -> Dict[str, Any]:
        return {
            f"{self._name_prefix}_pos3d": self._pos3d,
            f"{self._name_prefix}_inserted_length": self._inserted_length,
            f"{self._name_prefix}_rotation": self._rotation,
        }

    def __call__(self) -> Dict[str, Any]:
        return self.info


class SofaWallForceInfo:
    """Collect wall-contact telemetry from SOFA.

    Modes:
    - ``passive``: read vectors from WireWallForceMonitor C++ plugin.
    - ``intrusive_lcp``: derive vectors from collision DOF forces + LCP telemetry.
    - ``constraint_projected_si_validated``: passive runtime + SI conversion + stricter checks.
    """

    def __init__(
        self,
        intervention,
        *,
        mode: str = "passive",
        required: bool = False,
        contact_epsilon: float = 1e-7,
        plugin_path: Optional[str] = None,
        units: Optional[ForceUnitsConfig] = None,
        constraint_dt_s: Optional[float] = None,
        name: str = "wall_forces",
    ) -> None:
        self.name = name
        self._intervention = intervention
        self._mode = str(mode)
        self._required = bool(required)
        self._contact_epsilon = float(max(contact_epsilon, 0.0))
        self._units = units
        self._constraint_dt_s = (
            float(constraint_dt_s)
            if constraint_dt_s is not None and np.isfinite(constraint_dt_s) and float(constraint_dt_s) > 0.0
            else None
        )
        self._unit_converted_si = bool(
            self._mode == "constraint_projected_si_validated" and self._units is not None
        )
        self._si_force_scale = (
            unit_scale_to_si_newton(self._units)
            if self._unit_converted_si and self._units is not None
            else 1.0
        )
        self._runtime = SofaForceMonitorRuntime(
            mode=self._mode,
            contact_epsilon=self._contact_epsilon,
            plugin_path=plugin_path,
        )
        # Deterministic explicit association cache:
        # collision-force sample index -> wall triangle id
        self._force_idx_to_wall_triangle: Dict[int, int] = {}
        self._cache_max_surface_distance_mm = 2.5
        self._cache_neighbor_window = 2

        self.reset()

    def reset(self, episode_nr: int = 0) -> None:
        _ = episode_nr
        self._available = False
        self._source = "unavailable"
        self._error = ""
        self._contact_count = 0
        self._wall_segment_count = 0
        self._wall_active_segment_ids = np.zeros((0,), dtype=np.int32)
        self._wall_active_segment_force_vectors = np.zeros((0, 3), dtype=np.float32)
        self._contact_force_vectors = np.zeros((0, 3), dtype=np.float32)
        self._contact_segment_indices = np.zeros((0,), dtype=np.int32)
        self._segment_force_vectors = np.zeros((0, 3), dtype=np.float32)
        self._total_force_vector = np.zeros((3,), dtype=np.float32)
        self._total_force_norm = float("nan")
        self._lcp_sum_abs = float("nan")
        self._lcp_max_abs = float("nan")
        self._lcp_active_count = 0
        self._wire_force_norm = float("nan")
        self._collis_force_norm = float("nan")
        self._wire_force_vectors = np.zeros((0, 3), dtype=np.float32)
        self._collision_force_vectors = np.zeros((0, 3), dtype=np.float32)
        self._wire_force_vectors_source = ""
        self._collision_force_vectors_source = ""
        self._contact_detected = False
        self._force_source = "none"
        self._force_norm_sum = float("nan")
        self._wall_field_force_vector = np.zeros((3,), dtype=np.float32)
        self._wall_field_force_norm = float("nan")
        self._tip_force_vector = np.zeros((3,), dtype=np.float32)
        self._tip_force_norm = float("nan")
        self._tip_force_sample_index = -1
        self._tip_force_source = "none"
        self._status = ""
        self._gap_active_projected_count = 0
        self._gap_explicit_mapped_count = 0
        self._gap_unmapped_count = 0
        self._gap_class_counts = {}
        self._gap_dominant_class = "none"
        self._gap_contact_mode = "none"
        self._gap_row_bridge_mode = "none"
        self._gap_row_bridge_offset = 0
        self._gap_row_bridge_applied = False
        self._gap_row_bridge_deterministic = True
        self._gap_row_bridge_pair_count = 0
        self._gap_row_bridge_dof_hits = 0
        self._association_method = "none"
        self._association_explicit_ratio = float("nan")
        self._association_coverage = float("nan")
        self._association_explicit_force_coverage = float("nan")
        self._association_ordering_stable = False
        self._active_constraint_step = False
        self._native_contact_export_available = False
        self._native_contact_export_source = ""
        self._native_contact_export_status = ""
        self._native_contact_export_explicit_coverage = float("nan")
        self._gap_active_projected_count = 0
        self._gap_explicit_mapped_count = 0
        self._gap_unmapped_count = 0
        self._gap_class_counts: Dict[str, int] = {}
        self._gap_dominant_class = "none"
        self._gap_contact_mode = "none"
        self._quality_tier = "unavailable"
        self._association_method = "none"
        self._association_explicit_ratio = float("nan")
        self._association_coverage = float("nan")
        self._association_explicit_force_coverage = float("nan")
        self._association_ordering_stable = False
        self._active_constraint_step = False
        self._native_contact_export_available = False
        self._native_contact_export_source = ""
        self._native_contact_export_status = ""
        self._native_contact_export_explicit_coverage = float("nan")
        self._gap_active_projected_count = 0
        self._gap_explicit_mapped_count = 0
        self._gap_unmapped_count = 0
        self._gap_class_counts = {}
        self._gap_dominant_class = "none"
        self._gap_contact_mode = "none"
        self._gap_row_bridge_mode = "none"
        self._gap_row_bridge_offset = 0
        self._gap_row_bridge_applied = False
        self._gap_row_bridge_deterministic = True
        self._gap_row_bridge_pair_count = 0
        self._gap_row_bridge_dof_hits = 0
        self._unit_scale_to_newton = float(self._si_force_scale) if self._unit_converted_si else float("nan")
        self._force_idx_to_wall_triangle.clear()

        # Best-effort: capture forces at the initial pose.
        try:
            self.step()
        except Exception:
            pass

    @staticmethod
    def _safe_norm(arr: Any) -> float:
        try:
            return float(np.linalg.norm(np.asarray(arr)))
        except Exception:
            return float("nan")

    @staticmethod
    def _read_data(obj: Any, attr: str) -> Any:
        try:
            value = getattr(obj, attr)
        except Exception:
            return None
        if hasattr(value, "value"):
            try:
                return value.value
            except Exception:
                return None
        return value

    def _extract_lcp(self, sim: Any) -> tuple[float, float, int]:
        try:
            lcp = sim.root.LCP
            forces = self._read_data(lcp, "constraintForces")
            forces_arr = np.asarray(forces, dtype=np.float32)
            if forces_arr.size == 0:
                return 0.0, 0.0, 0
            max_abs = float(np.max(np.abs(forces_arr)))
            active = int(np.count_nonzero(np.abs(forces_arr) > self._contact_epsilon))
            return float(np.sum(np.abs(forces_arr))), max_abs, active
        except Exception:
            return float("nan"), float("nan"), 0

    def _resolve_constraint_dt_s(self, sim: Any) -> Optional[float]:
        """Resolve dt for lambda->force conversion.

        Priority:
        1) live SOFA root dt (simulation substep)
        2) configured fallback dt from constructor
        """

        try:
            root = getattr(sim, "root", None)
            raw_dt = self._read_data(root, "dt") if root is not None else None
            dt = float(raw_dt)
            if np.isfinite(dt) and dt > 0.0:
                return float(dt)
        except Exception:
            pass
        return self._constraint_dt_s

    @staticmethod
    def _extract_tip_force_from_samples(
        *,
        tip_pos: Any,
        candidate_forces: np.ndarray,
        candidate_positions: np.ndarray,
        contact_epsilon: float,
    ) -> Tuple[np.ndarray, float, int, str]:
        """Pick the force sample nearest to the current tip position."""

        out_zero = np.zeros((3,), dtype=np.float32)
        try:
            tip = np.asarray(tip_pos, dtype=np.float32).reshape((3,))
        except Exception:
            return out_zero, float("nan"), -1, "none"
        if not np.all(np.isfinite(tip)):
            return out_zero, float("nan"), -1, "none"

        forces = np.asarray(candidate_forces, dtype=np.float32)
        positions = np.asarray(candidate_positions, dtype=np.float32)
        if forces.ndim != 2 or positions.ndim != 2 or forces.shape[1] < 3 or positions.shape[1] < 3:
            return out_zero, float("nan"), -1, "none"
        n = int(min(forces.shape[0], positions.shape[0]))
        if n <= 0:
            return out_zero, float("nan"), -1, "none"

        forces = forces[:n, :3]
        positions = positions[:n, :3]
        d2 = np.sum((positions - tip.reshape((1, 3))) ** 2, axis=1)
        idx = int(np.argmin(d2))
        vec = forces[idx].astype(np.float32)
        norm = float(np.linalg.norm(vec))
        eps = float(max(contact_epsilon, 0.0))
        if not np.isfinite(norm) or norm <= eps:
            return out_zero, 0.0, idx, "nearest_tip_sample_below_epsilon"
        return vec, norm, idx, "nearest_tip_sample"

    def _apply_si_conversion_if_needed(self) -> None:
        if not self._unit_converted_si:
            return
        s = float(self._si_force_scale)
        if not np.isfinite(s) or s <= 0.0:
            return

        self._contact_force_vectors = np.asarray(self._contact_force_vectors, dtype=np.float32) * s
        self._wall_active_segment_force_vectors = (
            np.asarray(self._wall_active_segment_force_vectors, dtype=np.float32) * s
        )
        self._segment_force_vectors = np.asarray(self._segment_force_vectors, dtype=np.float32) * s
        self._total_force_vector = np.asarray(self._total_force_vector, dtype=np.float32) * s
        self._wire_force_vectors = np.asarray(self._wire_force_vectors, dtype=np.float32) * s
        self._collision_force_vectors = np.asarray(self._collision_force_vectors, dtype=np.float32) * s
        self._tip_force_vector = np.asarray(self._tip_force_vector, dtype=np.float32) * s

        if np.isfinite(self._total_force_norm):
            self._total_force_norm = float(self._total_force_norm * s)
        if np.isfinite(self._wire_force_norm):
            self._wire_force_norm = float(self._wire_force_norm * s)
        if np.isfinite(self._collis_force_norm):
            self._collis_force_norm = float(self._collis_force_norm * s)
        if np.isfinite(self._force_norm_sum):
            self._force_norm_sum = float(self._force_norm_sum * s)
        if np.isfinite(self._wall_field_force_norm):
            self._wall_field_force_norm = float(self._wall_field_force_norm * s)
        if np.isfinite(self._tip_force_norm):
            self._tip_force_norm = float(self._tip_force_norm * s)
        self._wall_field_force_vector = np.asarray(self._wall_field_force_vector, dtype=np.float32) * s

    @staticmethod
    def _safe_int(x: Any, default: int = 0) -> int:
        try:
            return int(x)
        except Exception:
            return int(default)

    def _extract_contact_count(self, sim: Any) -> int:
        counts = []
        try:
            point_model = sim._instruments_combined.CollisionModel.PointCollisionModel  # noqa: SLF001
            counts.append(self._safe_int(self._read_data(point_model, "numberOfContacts"), 0))
        except Exception:
            pass
        try:
            line_model = sim._instruments_combined.CollisionModel.LineCollisionModel  # noqa: SLF001
            counts.append(self._safe_int(self._read_data(line_model, "numberOfContacts"), 0))
        except Exception:
            pass
        return int(sum(c for c in counts if c > 0))

    def _listener_wall_model_on_second(self, listener: Any) -> bool:
        model1 = str(self._read_data(listener, "collisionModel1") or "")
        model2 = str(self._read_data(listener, "collisionModel2") or "")

        def _is_wall_model(s: str) -> bool:
            return ("vesselTree" in s) or ("TriangleCollisionModel" in s)

        is_wall_1 = _is_wall_model(model1)
        is_wall_2 = _is_wall_model(model2)
        if is_wall_2 and not is_wall_1:
            return True
        if is_wall_1 and not is_wall_2:
            return False
        # Runtime wiring uses (wire, wall) as the preferred order.
        return True

    @classmethod
    def _extract_ints_flat(cls, value: Any, out: Optional[List[int]] = None) -> List[int]:
        if out is None:
            out = []
        if isinstance(value, (int, np.integer)):
            out.append(int(value))
            return out
        if isinstance(value, (tuple, list)):
            for item in value:
                cls._extract_ints_flat(item, out)
            return out
        for attr in ("id", "id1", "id2", "index", "first", "second"):
            if hasattr(value, attr):
                try:
                    out.append(int(getattr(value, attr)))
                except Exception:
                    pass
        return out

    @staticmethod
    def _entry_get(entry: Any, key: str) -> Any:
        if isinstance(entry, dict):
            return entry.get(key, None)
        if hasattr(entry, key):
            try:
                return getattr(entry, key)
            except Exception:
                return None
        return None

    @classmethod
    def _extract_listener_element_pair(cls, entry: Any) -> Tuple[Any, Any]:
        # Most common representations first.
        if isinstance(entry, (tuple, list)):
            if len(entry) >= 2:
                return entry[0], entry[1]
            if len(entry) == 1:
                return entry[0], None

        for k1, k2 in (
            ("element1", "element2"),
            ("elem1", "elem2"),
            ("first", "second"),
            ("model1", "model2"),
        ):
            e1 = cls._entry_get(entry, k1)
            e2 = cls._entry_get(entry, k2)
            if e1 is not None or e2 is not None:
                return e1, e2

        elem = cls._entry_get(entry, "elem")
        if isinstance(elem, (tuple, list)) and len(elem) >= 2:
            return elem[0], elem[1]

        # Unknown shape: keep as first element; second unknown.
        return entry, None

    @classmethod
    def _extract_wall_triangle_id_from_element(
        cls,
        element: Any,
        *,
        wall_triangle_count: int,
    ) -> Optional[int]:
        n_wall = int(max(wall_triangle_count, 0))
        if n_wall <= 0 or element is None:
            return None

        if isinstance(element, (int, np.integer)):
            v = int(element)
            return v if 0 <= v < n_wall else None

        for attr in (
            "triangleIndex",
            "triangle_id",
            "triangleId",
            "elementIndex",
            "index",
            "idx",
            "id",
        ):
            v = cls._entry_get(element, attr)
            if isinstance(v, (int, np.integer)):
                tri = int(v)
                if 0 <= tri < n_wall:
                    return tri

        flat_ids = cls._extract_ints_flat(element)
        for tri in flat_ids:
            if 0 <= int(tri) < n_wall:
                return int(tri)
        return None

    def _read_listener_wall_triangle_ids(
        self, listener: Any, wall_triangle_count: int
    ) -> List[Tuple[Optional[int], str]]:
        try:
            entries = listener.getContactElements()
        except Exception:
            entries = self._read_data(listener, "contactElements") or []

        wall_on_second = self._listener_wall_model_on_second(listener)
        ids_out: List[Tuple[Optional[int], str]] = []
        n_wall = int(max(wall_triangle_count, 0))
        for entry in list(entries) if entries is not None else []:
            tri_id: Optional[int] = None
            source = "none"

            e1, e2 = self._extract_listener_element_pair(entry)
            preferred = e2 if wall_on_second else e1
            tri_id = self._extract_wall_triangle_id_from_element(
                preferred, wall_triangle_count=n_wall
            )
            if tri_id is not None:
                source = "contact_element_preferred_side"
            else:
                alternate = e1 if wall_on_second else e2
                tri_id = self._extract_wall_triangle_id_from_element(
                    alternate, wall_triangle_count=n_wall
                )
                if tri_id is not None:
                    source = "contact_element_alternate_side"

            if tri_id is None:
                # Last fallback for exotic listener payload layouts.
                flat_ids = self._extract_ints_flat(entry)
                for candidate in flat_ids:
                    if 0 <= int(candidate) < n_wall:
                        tri_id = int(candidate)
                        source = "contact_element_flat_fallback"
                        break

            ids_out.append((tri_id, source))
        return ids_out

    @staticmethod
    def _normalize_point_coord_array(arr: Any, *, allow_6dof: bool) -> np.ndarray:
        try:
            arr = np.asarray(arr, dtype=np.float32)
        except Exception:
            return np.zeros((0, 3), dtype=np.float32)
        if arr.ndim == 1:
            if allow_6dof and arr.size % 7 == 0:
                arr = arr.reshape((-1, 7))
            elif allow_6dof and arr.size % 6 == 0:
                arr = arr.reshape((-1, 6))
            elif arr.size % 3 == 0:
                arr = arr.reshape((-1, 3))
            else:
                return np.zeros((0, 3), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return np.zeros((0, 3), dtype=np.float32)
        return arr[:, :3].astype(np.float32)

    @staticmethod
    def _vec3_or_none(value: Any) -> Optional[np.ndarray]:
        try:
            arr = np.asarray(value, dtype=np.float32).reshape((-1,))
        except Exception:
            return None
        if arr.size < 3:
            return None
        out = arr[:3].astype(np.float32)
        if not np.all(np.isfinite(out)):
            return None
        return out

    def _read_listener_contact_points(
        self, listener: Any
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int]:
        points: List[Tuple[np.ndarray, np.ndarray]] = []
        count = 0
        try:
            count = self._safe_int(listener.getNumberOfContacts(), 0)
        except Exception:
            pass

        entries: Any = []
        try:
            entries = listener.getContactPoints()
        except Exception:
            entries = self._read_data(listener, "contactPoints") or []

        for entry in list(entries) if entries is not None else []:
            p_wire: Optional[np.ndarray] = None
            p_wall: Optional[np.ndarray] = None
            if isinstance(entry, (tuple, list)):
                if len(entry) >= 4:
                    p_wire = self._vec3_or_none(entry[1])
                    p_wall = self._vec3_or_none(entry[3])
                elif len(entry) >= 2:
                    p_wire = self._vec3_or_none(entry[0])
                    p_wall = self._vec3_or_none(entry[1])
            if p_wire is None or p_wall is None:
                for a_wire, a_wall in (
                    ("pointA", "pointB"),
                    ("point1", "point2"),
                    ("p1", "p2"),
                ):
                    if hasattr(entry, a_wire) and hasattr(entry, a_wall):
                        p_wire = self._vec3_or_none(getattr(entry, a_wire))
                        p_wall = self._vec3_or_none(getattr(entry, a_wall))
                        if p_wire is not None and p_wall is not None:
                            break
            if p_wire is not None and p_wall is not None:
                points.append((p_wire, p_wall))

        count = max(count, len(points))
        return points, count

    def _read_listener_contact_records(
        self, listener: Any, wall_triangle_count: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        pairs, count = self._read_listener_contact_points(listener)
        tri_ids = self._read_listener_wall_triangle_ids(listener, wall_triangle_count)
        records: List[Dict[str, Any]] = []
        for i, (p_wire, p_wall) in enumerate(pairs):
            tri_id: Optional[int] = None
            tri_source = "none"
            if i < len(tri_ids):
                tri_id, tri_source = tri_ids[i]
            records.append(
                {
                    "wire_point": p_wire,
                    "wall_point": p_wall,
                    "wall_triangle_id": tri_id,
                    "triangle_source": tri_source,
                }
            )
        count = max(count, len(records))
        return records, count

    def _extract_contact_records_from_listeners(
        self, sim: Any, wall_triangle_count: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        listeners = [
            getattr(sim.root, "wire_wall_contact_line", None),
            getattr(sim.root, "wire_wall_contact_point", None),
        ]
        records: List[Dict[str, Any]] = []
        total_count = 0
        for listener in listeners:
            if listener is None:
                continue
            listener_records, listener_count = self._read_listener_contact_records(
                listener, wall_triangle_count=wall_triangle_count
            )
            records.extend(listener_records)
            total_count += int(listener_count)
        if total_count <= 0 and records:
            total_count = len(records)
        return records, int(total_count)

    def _read_contact_points_from_contact_node(self, node: Any) -> List[np.ndarray]:
        points: List[np.ndarray] = []
        if node is None:
            return points
        try:
            objects = list(getattr(node, "objects", []))
        except Exception:
            objects = []
        for obj in objects:
            pos_raw = self._read_data(obj, "position")
            # Depending on the SOFA contact response object type, positions can be
            # exposed as 3D or rigid-like (6/7 dof) rows. We always keep xyz only.
            pos = self._normalize_point_coord_array(pos_raw, allow_6dof=True)
            if pos.size == 0:
                continue
            # Contact response nodes also contain helper objects without meaningful positions.
            # Keep only points that are finite and look like actual contact samples.
            for p in pos:
                if np.all(np.isfinite(p)):
                    points.append(p.astype(np.float32))
        return points

    def _extract_contact_records_from_contact_nodes(
        self, sim: Any
    ) -> Tuple[List[Dict[str, Any]], int]:
        records: List[Dict[str, Any]] = []
        try:
            vessel_tree = sim.root.vesselTree
        except Exception:
            return records, 0

        node_names = (
            "TriangleCollisionModel-PointCollisionModel",
            "TriangleCollisionModel-LineCollisionModel",
            "PointCollisionModel-TriangleCollisionModel",
            "LineCollisionModel-TriangleCollisionModel",
        )
        seen = set()
        for node_name in node_names:
            node = getattr(vessel_tree, node_name, None)
            if node is None:
                continue
            contact_points = self._read_contact_points_from_contact_node(node)
            for p in contact_points:
                key = tuple(np.round(np.asarray(p, dtype=np.float64), 8).tolist())
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    {
                        "wire_point": None,
                        "wall_point": p.astype(np.float32),
                        "wall_triangle_id": None,
                        "triangle_source": "contact_node_wall_point",
                    }
                )
        return records, len(records)

    def _extract_contact_records_from_native_export(
        self,
        sim: Any,
        *,
        wall_triangle_count: int,
    ) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
        exporter = getattr(sim.root, "wire_wall_contact_export", None)
        if exporter is None:
            return [], 0, {
                "available": False,
                "source": "native_contact_export_missing",
                "status": "missing",
                "explicit_coverage": float("nan"),
                "ordering_stable": False,
            }

        available = bool(self._read_data(exporter, "available"))
        source = str(self._read_data(exporter, "source") or "native_contact_export")
        status = str(self._read_data(exporter, "status") or "")
        ordering_stable = bool(self._read_data(exporter, "orderingStable"))
        try:
            explicit_coverage = float(self._read_data(exporter, "explicitCoverage"))
        except Exception:
            explicit_coverage = float("nan")

        try:
            wall_points = self._normalize_point_coord_array(
                self._read_data(exporter, "wallPoints"),
                allow_6dof=False,
            )
        except Exception:
            wall_points = np.zeros((0, 3), dtype=np.float32)
        if wall_points.size == 0:
            return [], 0, {
                "available": available,
                "source": source,
                "status": status or "ok:no_contacts",
                "explicit_coverage": explicit_coverage,
                "ordering_stable": ordering_stable,
            }

        n = int(wall_points.shape[0])

        def _int_array(name: str, default: int = -1) -> np.ndarray:
            raw = self._read_data(exporter, name)
            if raw is None:
                return np.full((n,), int(default), dtype=np.int64)
            arr = np.asarray(raw, dtype=np.int64).reshape((-1,))
            if arr.size < n:
                arr = np.pad(arr, (0, n - arr.size), mode="constant", constant_values=int(default))
            return arr[:n]

        tri_ids = _int_array("wallTriangleIds", default=-1)
        local_indices = _int_array("contactLocalIndices", default=-1)
        model_sides = _int_array("modelSides", default=-1)
        constraint_rows = _int_array("constraintRowIndices", default=-1)
        collision_dofs = _int_array("collisionDofIndices", default=-1)
        tri_valid = _int_array("triangleIdValidFlags", default=0)
        row_valid = _int_array("constraintRowValidFlags", default=0)
        collision_dof_valid = _int_array("collisionDofValidFlags", default=0)
        in_range = _int_array("inRangeFlags", default=0)
        mapping_complete = _int_array("mappingCompleteFlags", default=0)
        ordering_flags = _int_array("orderingStableFlags", default=0)

        raw_tags = self._read_data(exporter, "sourceNodeTags")
        if raw_tags is None:
            source_tags = [""] * n
        else:
            source_tags = [str(x) for x in list(raw_tags)]
            if len(source_tags) < n:
                source_tags.extend([""] * (n - len(source_tags)))
            source_tags = source_tags[:n]
        raw_kinds = self._read_data(exporter, "contactKinds")
        if raw_kinds is None:
            contact_kinds = ["unknown"] * n
        else:
            contact_kinds = [str(x).strip().lower() or "unknown" for x in list(raw_kinds)]
            if len(contact_kinds) < n:
                contact_kinds.extend(["unknown"] * (n - len(contact_kinds)))
            contact_kinds = contact_kinds[:n]

        records: List[Dict[str, Any]] = []
        for i in range(n):
            tri = int(tri_ids[i])
            tri_valid_i = bool(int(tri_valid[i])) and bool(int(in_range[i]))
            tri_in_range = 0 <= tri < int(max(wall_triangle_count, 0))
            if tri_valid_i and not tri_in_range:
                tri_valid_i = False
            row_idx = int(constraint_rows[i])
            row_valid_i = bool(int(row_valid[i])) and row_idx >= 0
            collision_dof_idx = int(collision_dofs[i])
            collision_dof_valid_i = (
                bool(int(collision_dof_valid[i])) and collision_dof_idx >= 0
            )
            records.append(
                {
                    "wire_point": None,
                    "wall_point": wall_points[i].astype(np.float32),
                    "wall_triangle_id": tri if tri_valid_i else None,
                    "triangle_source": (
                        "native_contact_export_triangle_id"
                        if tri_valid_i
                        else "native_contact_export_missing"
                    ),
                    "source_node_tag": source_tags[i],
                    "model_side": int(model_sides[i]),
                    "contact_local_index": int(local_indices[i]),
                    "contact_kind": contact_kinds[i],
                    "constraint_row_index": row_idx if row_valid_i else None,
                    "collision_dof_index": (
                        collision_dof_idx if collision_dof_valid_i else None
                    ),
                    "integrity_triangle_id_valid": tri_valid_i,
                    "integrity_constraint_row_valid": row_valid_i,
                    "integrity_collision_dof_valid": collision_dof_valid_i,
                    "integrity_in_range": bool(int(in_range[i])),
                    "integrity_mapping_complete": bool(int(mapping_complete[i])),
                    "integrity_ordering_stable": bool(int(ordering_flags[i])),
                }
            )

        records = self._deduplicate_contact_records(records)
        return records, len(records), {
            "available": available,
            "source": source,
            "status": status,
            "explicit_coverage": explicit_coverage,
            "ordering_stable": ordering_stable,
        }

    @staticmethod
    def _deduplicate_contact_records(
        records: List[Dict[str, Any]],
        *,
        rounding_digits: int = 6,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for rec in records:
            wall_tri = rec.get("wall_triangle_id", None)
            tri_key = int(wall_tri) if isinstance(wall_tri, (int, np.integer)) else -1
            row_idx = rec.get("constraint_row_index", None)
            row_key = int(row_idx) if isinstance(row_idx, (int, np.integer)) else -1
            dof_idx = rec.get("collision_dof_index", None)
            dof_key = int(dof_idx) if isinstance(dof_idx, (int, np.integer)) else -1
            wire_point = rec.get("wire_point", None)
            wall_point = rec.get("wall_point", None)
            if wire_point is None:
                wire_key = None
            else:
                try:
                    wire_key = tuple(
                        np.round(np.asarray(wire_point, dtype=np.float64).reshape((3,)), rounding_digits).tolist()
                    )
                except Exception:
                    wire_key = None
            if wall_point is None:
                wall_key = None
            else:
                try:
                    wall_key = tuple(
                        np.round(np.asarray(wall_point, dtype=np.float64).reshape((3,)), rounding_digits).tolist()
                    )
                except Exception:
                    wall_key = None
            key = (tri_key, row_key, dof_key, wire_key, wall_key)
            if key in seen:
                continue
            seen.add(key)
            out.append(rec)
        return out

    def _read_vessel_wall_geometry(
        self, sim: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            vessel_dofs = sim.root.vesselTree.dofs
            vessel_topo = sim.root.vesselTree.MeshTopology
        except Exception:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
            )
        try:
            pos = self._normalize_point_coord_array(
                self._read_data(vessel_dofs, "position"),
                allow_6dof=False,
            )
            tri = np.asarray(self._read_data(vessel_topo, "triangles"), dtype=np.int32)
            tri = tri.reshape((-1, 3))
        except Exception:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
            )
        if pos.size == 0 or tri.size == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
            )
        if np.max(tri) >= pos.shape[0] or np.min(tri) < 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
            )
        centroids = pos[tri].mean(axis=1).astype(np.float32)
        return pos.astype(np.float32), tri.astype(np.int32), centroids

    @staticmethod
    def _closest_point_on_triangle(
        p: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        """Return closest point on triangle ABC to point P (all in R^3).

        Based on the standard region tests from Real-Time Collision Detection.
        """

        ab = b - a
        ac = c - a
        ap = p - a
        d1 = float(np.dot(ab, ap))
        d2 = float(np.dot(ac, ap))
        if d1 <= 0.0 and d2 <= 0.0:
            return a

        bp = p - b
        d3 = float(np.dot(ab, bp))
        d4 = float(np.dot(ac, bp))
        if d3 >= 0.0 and d4 <= d3:
            return b

        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            return a + v * ab

        cp = p - c
        d5 = float(np.dot(ab, cp))
        d6 = float(np.dot(ac, cp))
        if d6 >= 0.0 and d5 <= d6:
            return c

        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            return a + w * ac

        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            denom = (d4 - d3) + (d5 - d6)
            if denom == 0.0:
                return b
            w = (d4 - d3) / denom
            return b + w * (c - b)

        denom = va + vb + vc
        if denom == 0.0:
            return a
        inv = 1.0 / denom
        v = vb * inv
        w = vc * inv
        return a + ab * v + ac * w

    @classmethod
    def _find_nearest_triangle_surface(
        cls,
        point: np.ndarray,
        wall_vertices: np.ndarray,
        wall_triangles: np.ndarray,
    ) -> Optional[int]:
        if wall_vertices.size == 0 or wall_triangles.size == 0:
            return None
        p = np.asarray(point, dtype=np.float32).reshape((3,))
        best_idx: Optional[int] = None
        best_d2 = float("inf")
        for tri_idx in range(int(wall_triangles.shape[0])):
            tri = wall_triangles[tri_idx]
            try:
                a = wall_vertices[int(tri[0])]
                b = wall_vertices[int(tri[1])]
                c = wall_vertices[int(tri[2])]
            except Exception:
                continue
            cp = cls._closest_point_on_triangle(p, a, b, c)
            d = p - cp
            d2 = float(np.dot(d, d))
            if d2 < best_d2:
                best_d2 = d2
                best_idx = tri_idx
        return int(best_idx) if best_idx is not None else None

    @staticmethod
    def _parse_constraint_rows(raw: Any) -> List[Tuple[int, int, np.ndarray]]:
        """Parse SOFA constraint rows into (row_idx, dof_idx, coeff_xyz).

        Expected row text shape (examples):
        - ``row n dof cx cy cz``
        - ``row n dof cx cy cz rx ry rz dof ...``
        """

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
                        dtype=np.float32,
                    )
                except Exception:
                    continue
                if not np.all(np.isfinite(coeff)):
                    continue
                entries.append((row_idx, dof_idx, coeff))
        return entries

    def _project_constraint_forces(
        self,
        *,
        lcp_forces: np.ndarray,
        constraint_raw: Any,
        n_points: int,
        dt_s: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Project LCP constraint scalars onto per-point xyz vectors.

        Returns:
        - aggregated per-point force vectors [N,3]
        - deterministic per-(row,dof) contributions with fields:
          ``row_idx``, ``dof_idx``, ``force_vec``, ``force_norm``.
        """

        n = int(max(n_points, 0))
        if n <= 0:
            return np.zeros((0, 3), dtype=np.float32), []
        out = np.zeros((n, 3), dtype=np.float32)
        if lcp_forces.size == 0:
            return out, []
        rows = self._parse_constraint_rows(constraint_raw)
        if not rows:
            return out, []

        lcp = np.asarray(lcp_forces, dtype=np.float32).reshape((-1,))
        scale = 1.0
        if dt_s is not None and np.isfinite(dt_s) and float(dt_s) > 0.0:
            # constraintForces represent impulse-like multipliers in many LCP scenes.
            # Convert to force-like quantity via lambda / dt.
            scale = 1.0 / float(dt_s)

        row_dof_force: Dict[Tuple[int, int], np.ndarray] = {}
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

        row_contribs: List[Dict[str, Any]] = []
        for row_idx, dof_idx in sorted(row_dof_force.keys()):
            force_vec = row_dof_force[(row_idx, dof_idx)].astype(np.float32)
            row_contribs.append(
                {
                    "row_idx": int(row_idx),
                    "dof_idx": int(dof_idx),
                    "force_vec": force_vec,
                    "force_norm": float(np.linalg.norm(force_vec)),
                }
            )
        return out, row_contribs

    def _collect_constraint_projection_candidates(
        self, sim: Any
    ) -> List[Tuple[str, np.ndarray, np.ndarray, float, List[Dict[str, Any]]]]:
        candidates: List[
            Tuple[str, np.ndarray, np.ndarray, float, List[Dict[str, Any]]]
        ] = []
        lcp_raw = self._read_data(sim.root.LCP, "constraintForces")
        if lcp_raw is None:
            lcp_forces = np.zeros((0,), dtype=np.float32)
        else:
            lcp_forces = np.asarray(lcp_raw, dtype=np.float32)

        resolved_dt_s = (
            self._resolve_constraint_dt_s(sim)
            if self._mode == "constraint_projected_si_validated"
            else None
        )

        def add_state_projection(prefix: str, state_obj: Any, *, allow_6dof_pos: bool) -> None:
            if state_obj is None:
                return
            positions = self._normalize_point_coord_array(
                self._read_data(state_obj, "position"),
                allow_6dof=allow_6dof_pos,
            )
            if positions.size == 0:
                return
            constraint_raw = self._read_data(state_obj, "constraint")
            proj, row_contribs = self._project_constraint_forces(
                lcp_forces=lcp_forces,
                constraint_raw=constraint_raw,
                n_points=int(positions.shape[0]),
                dt_s=resolved_dt_s,
            )
            if proj.size == 0:
                return
            proj = proj[: positions.shape[0]].astype(np.float32)
            norm_sum = float(np.linalg.norm(proj, axis=1).sum()) if proj.size else 0.0
            candidates.append(
                (
                    f"{prefix}.constraintProjection",
                    proj,
                    positions.astype(np.float32),
                    norm_sum,
                    row_contribs,
                )
            )

        try:
            add_state_projection(
                "collision",
                sim._instruments_combined.CollisionModel.CollisionDOFs,  # noqa: SLF001
                allow_6dof_pos=False,
            )
        except Exception:
            pass
        try:
            add_state_projection(
                "wire",
                sim._instruments_combined.DOFs,  # noqa: SLF001
                allow_6dof_pos=True,
            )
        except Exception:
            pass
        return candidates

    def _collect_force_candidates(
        self, sim: Any
    ) -> List[Tuple[str, np.ndarray, np.ndarray, float, List[Dict[str, Any]]]]:
        """Return force candidates as (source, forces[N,3], positions[N,3], norm_sum, row_contribs)."""

        candidates: List[
            Tuple[str, np.ndarray, np.ndarray, float, List[Dict[str, Any]]]
        ] = []
        lcp_raw = self._read_data(sim.root.LCP, "constraintForces")
        if lcp_raw is None:
            lcp_forces = np.zeros((0,), dtype=np.float32)
        else:
            lcp_forces = np.asarray(lcp_raw, dtype=np.float32)

        def add_state_candidates(prefix: str, state_obj: Any, *, allow_6dof_pos: bool) -> None:
            if state_obj is None:
                return
            positions = self._normalize_point_coord_array(
                self._read_data(state_obj, "position"),
                allow_6dof=allow_6dof_pos,
            )
            if positions.size == 0:
                return
            for attr in ("force", "externalForce"):
                forces = self._normalize_point_force_array(
                    self._read_data(state_obj, attr),
                    allow_6dof=allow_6dof_pos,
                )
                if forces.size == 0:
                    continue
                n = min(forces.shape[0], positions.shape[0])
                if n <= 0:
                    continue
                f = forces[:n].astype(np.float32)
                p = positions[:n].astype(np.float32)
                norm_sum = float(np.linalg.norm(f, axis=1).sum()) if f.size else 0.0
                candidates.append((f"{prefix}.{attr}", f, p, norm_sum, []))

            # Constraint projection fallback:
            # Some BeamAdapter scenes keep MechanicalObject.force at zero while LCP
            # constraintForces carry the active contact reaction. We project these
            # scalars onto xyz vectors using the per-state constraint rows.
            try:
                constraint_raw = self._read_data(state_obj, "constraint")
                proj, row_contribs = self._project_constraint_forces(
                    lcp_forces=lcp_forces,
                    constraint_raw=constraint_raw,
                    n_points=int(positions.shape[0]),
                )
                if proj.size:
                    proj = proj[: positions.shape[0]].astype(np.float32)
                    proj_norm_sum = (
                        float(np.linalg.norm(proj, axis=1).sum()) if proj.size else 0.0
                    )
                    candidates.append(
                        (
                            f"{prefix}.constraintProjection",
                            proj,
                            positions.astype(np.float32),
                            proj_norm_sum,
                            row_contribs,
                        )
                    )
            except Exception:
                pass

        try:
            add_state_candidates(
                "collision",
                sim._instruments_combined.CollisionModel.CollisionDOFs,  # noqa: SLF001
                allow_6dof_pos=False,
            )
        except Exception:
            pass
        try:
            add_state_candidates(
                "wire",
                sim._instruments_combined.DOFs,  # noqa: SLF001
                allow_6dof_pos=True,
            )
        except Exception:
            pass
        return candidates

    @staticmethod
    def _map_contact_forces_to_wall_segments(
        *,
        contact_pairs: List[Tuple[np.ndarray, np.ndarray]],
        wall_centroids: np.ndarray,
        candidate_forces: np.ndarray,
        candidate_positions: np.ndarray,
        contact_epsilon: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map contact points to nearest wall triangle and nearest force sample."""

        n_wall = int(wall_centroids.shape[0])
        wall_segment_forces = np.zeros((n_wall, 3), dtype=np.float32)
        if (
            n_wall <= 0
            or not contact_pairs
            or candidate_forces.size == 0
            or candidate_positions.size == 0
        ):
            return (
                wall_segment_forces,
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((3,), dtype=np.float32),
            )

        eps = float(max(contact_epsilon, 0.0))
        for wire_p, wall_p in contact_pairs:
            if wire_p is None or wall_p is None:
                continue
            d_force = candidate_positions - wire_p.reshape((1, 3))
            idx_force = int(np.argmin(np.sum(d_force * d_force, axis=1)))
            force_vec = candidate_forces[idx_force].astype(np.float32)
            if float(np.linalg.norm(force_vec)) <= eps:
                continue

            d_wall = wall_centroids - wall_p.reshape((1, 3))
            idx_tri = int(np.argmin(np.sum(d_wall * d_wall, axis=1)))
            wall_segment_forces[idx_tri] -= force_vec

        norms = (
            np.linalg.norm(wall_segment_forces, axis=1)
            if wall_segment_forces.size
            else np.zeros((0,), dtype=np.float32)
        )
        active_mask = norms > eps
        active_ids = np.nonzero(active_mask)[0].astype(np.int32)
        active_forces = wall_segment_forces[active_ids].astype(np.float32)
        total = wall_segment_forces.sum(axis=0).astype(np.float32)
        return wall_segment_forces, active_ids, active_forces, total

    @staticmethod
    def _contact_mode_from_records(contact_records: List[Dict[str, Any]]) -> str:
        kinds = {
            str(rec.get("contact_kind", "")).strip().lower()
            for rec in contact_records
            if str(rec.get("contact_kind", "")).strip().lower() in {"line", "point"}
        }
        if not kinds:
            return "none"
        if len(kinds) == 1:
            return next(iter(kinds))
        return "mixed"

    @classmethod
    def _map_row_projected_forces_to_wall_segments(
        cls,
        *,
        row_contribs: List[Dict[str, Any]],
        contact_records: List[Dict[str, Any]],
        wall_vertices: np.ndarray,
        wall_triangles: np.ndarray,
        wall_centroids: np.ndarray,
        contact_epsilon: float,
        allow_surface_fallback: bool = True,
        native_no_contacts_step: bool = False,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        str,
        int,
        float,
        int,
        int,
        Dict[str, Any],
    ]:
        n_wall = int(wall_centroids.shape[0])
        wall_segment_forces = np.zeros((n_wall, 3), dtype=np.float32)
        eps = float(max(contact_epsilon, 0.0))
        if n_wall <= 0:
            return (
                wall_segment_forces,
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((3,), dtype=np.float32),
                "none",
                0,
                0.0,
                0,
                0,
                {
                    "active_projected_count": 0,
                    "explicit_mapped_count": 0,
                    "unmapped_count": 0,
                    "class_counts": {},
                    "dominant_class": "none",
                    "contact_mode": "none",
                },
            )

        relevant_rows: List[Dict[str, Any]] = []
        for rc in row_contribs:
            row_idx = rc.get("row_idx", None)
            dof_idx = rc.get("dof_idx", None)
            force_vec = np.asarray(
                rc.get("force_vec", np.zeros((3,), dtype=np.float32)), dtype=np.float32
            ).reshape((3,))
            if not isinstance(row_idx, (int, np.integer)) or int(row_idx) < 0:
                continue
            if not isinstance(dof_idx, (int, np.integer)) or int(dof_idx) < 0:
                continue
            if float(np.linalg.norm(force_vec)) <= eps:
                continue
            relevant_rows.append(
                {
                    "row_idx": int(row_idx),
                    "dof_idx": int(dof_idx),
                    "force_vec": force_vec,
                }
            )

        explicit_contact_dofs = {
            int(rec.get("collision_dof_index"))
            for rec in contact_records
            if isinstance(rec.get("collision_dof_index", None), (int, np.integer))
            and int(rec.get("collision_dof_index")) >= 0
        }
        if native_no_contacts_step and not contact_records:
            relevant_rows = []
        if explicit_contact_dofs:
            filtered_rows = [
                rc
                for rc in relevant_rows
                if int(rc.get("dof_idx", -1)) in explicit_contact_dofs
            ]
            if filtered_rows:
                relevant_rows = filtered_rows

        active_projected_count = int(len(relevant_rows))
        if active_projected_count <= 0:
            return (
                wall_segment_forces,
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((3,), dtype=np.float32),
                "none",
                0,
                0.0,
                0,
                0,
                {
                    "active_projected_count": 0,
                    "explicit_mapped_count": 0,
                    "unmapped_count": 0,
                    "class_counts": {},
                    "dominant_class": "none",
                    "contact_mode": cls._contact_mode_from_records(contact_records),
                },
            )

        row_to_records_local: Dict[int, List[Dict[str, Any]]] = {}
        for rec in contact_records:
            row_idx = rec.get("constraint_row_index", None)
            if isinstance(row_idx, (int, np.integer)) and int(row_idx) >= 0:
                row_to_records_local.setdefault(int(row_idx), []).append(rec)

        class_counts: Dict[str, int] = {}

        def _bump(reason: str) -> None:
            class_counts[reason] = int(class_counts.get(reason, 0)) + 1

        projected_row_to_dofs: Dict[int, set] = {}
        for rc in relevant_rows:
            projected_row_to_dofs.setdefault(int(rc["row_idx"]), set()).add(int(rc["dof_idx"]))
        local_row_to_dofs: Dict[int, set] = {}
        for local_row, recs in row_to_records_local.items():
            dof_set = {
                int(rec.get("collision_dof_index"))
                for rec in recs
                if isinstance(rec.get("collision_dof_index", None), (int, np.integer))
                and int(rec.get("collision_dof_index")) >= 0
            }
            if dof_set:
                local_row_to_dofs[int(local_row)] = dof_set

        # Build deterministic local-row -> global-row mapping.
        # Priority:
        # 1) Identity mapping for rows already in global projection domain.
        # 2) DOF-overlap mapping (most specific bridge).
        # 3) Dominant constant offset inferred from resolved pairs.
        # 4) Row-overlap-only offset fallback (still explicit, but lower confidence).
        row_bridge_offset = 0
        row_bridge_applied = False
        row_bridge_deterministic = True
        row_bridge_mode = "identity"
        local_to_global: Dict[int, int] = {}
        assigned_global_rows: set = set()

        for local_row in sorted(row_to_records_local.keys()):
            if int(local_row) in projected_row_to_dofs:
                local_to_global[int(local_row)] = int(local_row)
                assigned_global_rows.add(int(local_row))

        if local_row_to_dofs and projected_row_to_dofs:
            best_per_local: Dict[int, Tuple[int, int, int]] = {}
            for local_row, dof_set in sorted(local_row_to_dofs.items()):
                if int(local_row) in local_to_global:
                    continue
                scored: List[Tuple[int, int, int]] = []
                for global_row, global_dofs in sorted(projected_row_to_dofs.items()):
                    if int(global_row) in assigned_global_rows:
                        continue
                    overlap = int(len(set(dof_set) & set(global_dofs)))
                    if overlap <= 0:
                        continue
                    # score tuple: (overlap, negative distance preference via second key)
                    scored.append((overlap, abs(int(global_row) - int(local_row)), int(global_row)))
                if not scored:
                    continue
                scored.sort(key=lambda x: (-int(x[0]), int(x[1]), int(x[2])))
                best = scored[0]
                same_score = [x for x in scored if int(x[0]) == int(best[0]) and int(x[1]) == int(best[1])]
                if len(same_score) > 1:
                    row_bridge_deterministic = False
                    continue
                best_per_local[int(local_row)] = best

            for local_row, best in sorted(
                best_per_local.items(),
                key=lambda kv: (-int(kv[1][0]), int(kv[1][1]), int(kv[0]), int(kv[1][2])),
            ):
                global_row = int(best[2])
                if global_row in assigned_global_rows:
                    continue
                local_to_global[int(local_row)] = global_row
                assigned_global_rows.add(global_row)

        offset_votes: Dict[int, int] = {}
        for local_row, global_row in local_to_global.items():
            offset = int(global_row) - int(local_row)
            offset_votes[offset] = int(offset_votes.get(offset, 0)) + 1
        if offset_votes:
            best_count = max(offset_votes.values())
            best_offsets = sorted(
                int(offset) for offset, count in offset_votes.items() if int(count) == int(best_count)
            )
            if len(best_offsets) == 1:
                row_bridge_offset = int(best_offsets[0])
                for local_row in sorted(row_to_records_local.keys()):
                    if int(local_row) in local_to_global:
                        continue
                    global_row = int(local_row) + int(row_bridge_offset)
                    if (
                        global_row in projected_row_to_dofs
                        and global_row not in assigned_global_rows
                    ):
                        local_to_global[int(local_row)] = int(global_row)
                        assigned_global_rows.add(int(global_row))
            else:
                row_bridge_deterministic = False

        if not local_to_global and row_to_records_local and projected_row_to_dofs:
            local_rows = sorted(int(x) for x in row_to_records_local.keys())
            projected_rows = sorted(int(x) for x in projected_row_to_dofs.keys())
            candidate_counts: Dict[int, int] = {}
            for projected_row in projected_rows:
                for local_row in local_rows:
                    offset = int(projected_row) - int(local_row)
                    if offset in candidate_counts:
                        continue
                    shifted = {int(row) + int(offset) for row in local_rows}
                    candidate_counts[int(offset)] = int(
                        len(shifted & set(projected_rows))
                    )
            if candidate_counts:
                best_count = max(candidate_counts.values())
                best_offsets = sorted(
                    int(offset)
                    for offset, count in candidate_counts.items()
                    if int(count) == int(best_count)
                )
                if len(best_offsets) == 1 and int(best_count) > 0:
                    row_bridge_offset = int(best_offsets[0])
                    for local_row in local_rows:
                        global_row = int(local_row) + int(row_bridge_offset)
                        if global_row in projected_row_to_dofs:
                            local_to_global[int(local_row)] = int(global_row)
                    if local_to_global:
                        row_bridge_mode = "row_overlap_offset"
                else:
                    row_bridge_deterministic = False

        row_to_records: Dict[int, List[Dict[str, Any]]] = {}
        for local_row, recs in row_to_records_local.items():
            global_row = local_to_global.get(int(local_row), None)
            if global_row is None:
                continue
            row_to_records.setdefault(int(global_row), []).extend(recs)
        dof_to_records: Dict[int, List[Dict[str, Any]]] = {}
        for rec in contact_records:
            dof_idx = rec.get("collision_dof_index", None)
            if isinstance(dof_idx, (int, np.integer)) and int(dof_idx) >= 0:
                dof_to_records.setdefault(int(dof_idx), []).append(rec)

        if local_to_global:
            row_bridge_applied = any(
                int(global_row) != int(local_row)
                for local_row, global_row in local_to_global.items()
            )
            if row_bridge_mode == "identity" and row_bridge_applied:
                row_bridge_mode = "dof_overlap"
        if not row_bridge_deterministic:
            _bump("projection_row_not_resolvable")

        mapped_explicit = 0
        mapped_total = 0
        mapped_force_explicit_count = 0
        mapped_force_count = 0
        mapped_surface = 0
        mapped_centroid = 0
        dof_bridge_hits = 0

        for rc in relevant_rows:
            row_idx = int(rc["row_idx"])
            dof_idx = int(rc["dof_idx"])
            force_vec = rc["force_vec"].astype(np.float32)
            records = row_to_records.get(row_idx, [])
            if not records:
                records = dof_to_records.get(dof_idx, [])
                if records:
                    dof_bridge_hits += 1
            if not records:
                _bump("no_native_record_for_active_row")
                continue

            if not all(bool(r.get("integrity_ordering_stable", False)) for r in records):
                _bump("ordering_or_integrity_violation")
                continue
            if not all(bool(r.get("integrity_mapping_complete", False)) for r in records):
                _bump("ordering_or_integrity_violation")
                continue

            dof_candidates = {
                int(r.get("collision_dof_index"))
                for r in records
                if isinstance(r.get("collision_dof_index", None), (int, np.integer))
                and int(r.get("collision_dof_index")) >= 0
            }
            if dof_candidates and int(dof_idx) not in dof_candidates:
                alt_records = dof_to_records.get(dof_idx, [])
                if alt_records and all(
                    bool(r.get("integrity_ordering_stable", False))
                    and bool(r.get("integrity_mapping_complete", False))
                    for r in alt_records
                ):
                    alt_dofs = {
                        int(r.get("collision_dof_index"))
                        for r in alt_records
                        if isinstance(r.get("collision_dof_index", None), (int, np.integer))
                        and int(r.get("collision_dof_index")) >= 0
                    }
                    if not alt_dofs or int(dof_idx) in alt_dofs:
                        records = alt_records
                        dof_bridge_hits += 1
                        dof_candidates = alt_dofs
                if dof_candidates and int(dof_idx) not in dof_candidates:
                    _bump("domain_mismatch_row_vs_force_index")
                    continue

            tri_candidates = {
                int(r.get("wall_triangle_id"))
                for r in records
                if isinstance(r.get("wall_triangle_id", None), (int, np.integer))
                and 0 <= int(r.get("wall_triangle_id")) < n_wall
            }
            tri_out_of_range_present = any(
                isinstance(r.get("wall_triangle_id", None), (int, np.integer))
                and int(r.get("wall_triangle_id")) >= n_wall
                for r in records
            )
            if tri_out_of_range_present:
                _bump("triangle_id_out_of_range")
                continue
            if len(tri_candidates) > 1:
                _bump("row_to_triangle_ambiguous")
                continue
            if len(tri_candidates) == 0:
                _bump("native_record_missing_triangle_id")
                continue

            tri = int(next(iter(tri_candidates)))
            wall_segment_forces[tri] -= force_vec
            mapped_explicit += 1
            mapped_total += 1
            mapped_force_explicit_count += 1
            mapped_force_count += 1

        if dof_bridge_hits > 0:
            row_bridge_applied = True
            row_bridge_mode = "dof_key"

        if allow_surface_fallback and mapped_explicit < active_projected_count:
            # Map remaining rows geometrically for degraded continuity only.
            mapped_rows = {
                int(rc["row_idx"])
                for rc in relevant_rows
                if int(rc["row_idx"]) in row_to_records
                and any(
                    isinstance(r.get("wall_triangle_id", None), (int, np.integer))
                    and 0 <= int(r.get("wall_triangle_id")) < n_wall
                    for r in row_to_records.get(int(rc["row_idx"]), [])
                )
            }
            for rc in relevant_rows:
                row_idx = int(rc["row_idx"])
                if row_idx in mapped_rows:
                    continue
                records = row_to_records.get(row_idx, [])
                if not records:
                    continue
                wall_p = next(
                    (
                        np.asarray(r.get("wall_point"), dtype=np.float32)
                        for r in records
                        if r.get("wall_point", None) is not None
                    ),
                    None,
                )
                if wall_p is None:
                    continue
                tri_idx = cls._find_nearest_triangle_surface(
                    wall_p,
                    wall_vertices,
                    wall_triangles,
                )
                if tri_idx is not None:
                    wall_segment_forces[int(tri_idx)] -= rc["force_vec"].astype(np.float32)
                    mapped_total += 1
                    mapped_force_count += 1
                    mapped_surface += 1
                else:
                    d_wall = wall_centroids - wall_p.reshape((1, 3))
                    tri_idx = int(np.argmin(np.sum(d_wall * d_wall, axis=1)))
                    wall_segment_forces[tri_idx] -= rc["force_vec"].astype(np.float32)
                    mapped_total += 1
                    mapped_force_count += 1
                    mapped_centroid += 1

        norms = (
            np.linalg.norm(wall_segment_forces, axis=1)
            if wall_segment_forces.size
            else np.zeros((0,), dtype=np.float32)
        )
        active_mask = norms > eps
        active_ids = np.nonzero(active_mask)[0].astype(np.int32)
        active_forces = wall_segment_forces[active_ids].astype(np.float32)
        total = wall_segment_forces.sum(axis=0).astype(np.float32)

        explicit_ratio = (
            float(mapped_explicit) / float(mapped_total)
            if mapped_total > 0
            else 0.0
        )
        if mapped_explicit == active_projected_count and mapped_total == active_projected_count:
            method = (
                "native_contact_export_triangle_id_global_row_bridge"
                if row_bridge_applied
                else "native_contact_export_triangle_id"
            )
        elif mapped_explicit > 0 and mapped_surface == 0 and mapped_centroid == 0:
            method = (
                "partial_native_contact_export_triangle_id_global_row_bridge"
                if row_bridge_applied
                else "partial_native_contact_export_triangle_id"
            )
        elif mapped_explicit > 0 and (mapped_surface > 0 or mapped_centroid > 0):
            method = "mixed_native_and_surface"
        elif mapped_explicit == 0 and mapped_total > 0:
            method = "degraded_row_nearest_triangle_surface"
        else:
            method = "none"

        dominant_class = "none"
        if class_counts:
            dominant_class = max(
                class_counts.keys(),
                key=lambda k: int(class_counts.get(k, 0)),
            )
        gap = {
            "active_projected_count": int(active_projected_count),
            "explicit_mapped_count": int(mapped_explicit),
            "unmapped_count": int(max(0, active_projected_count - mapped_explicit)),
            "class_counts": class_counts,
            "dominant_class": dominant_class,
            "contact_mode": cls._contact_mode_from_records(contact_records),
            "row_bridge_offset": int(row_bridge_offset),
            "row_bridge_applied": bool(row_bridge_applied),
            "row_bridge_deterministic": bool(row_bridge_deterministic),
            "row_bridge_mode": str(row_bridge_mode),
            "row_bridge_pair_count": int(len(local_to_global)),
            "row_bridge_dof_hits": int(dof_bridge_hits),
        }
        return (
            wall_segment_forces,
            active_ids,
            active_forces,
            total,
            method,
            int(mapped_total),
            float(explicit_ratio),
            int(mapped_force_count),
            int(mapped_force_explicit_count),
            gap,
        )

    @classmethod
    def _map_contact_records_to_wall_segments(
        cls,
        *,
        contact_records: List[Dict[str, Any]],
        wall_vertices: np.ndarray,
        wall_triangles: np.ndarray,
        wall_centroids: np.ndarray,
        candidate_forces: np.ndarray,
        candidate_positions: np.ndarray,
        contact_epsilon: float,
        dof_triangle_cache: Optional[Dict[int, int]] = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        str,
        int,
        float,
        int,
        int,
    ]:
        """Map explicitly detected contacts to wall triangles.

        Returns:
        - dense wall segment vectors
        - active segment ids
        - active segment vectors
        - total wall vector
        - association method
        - mapped contact count
        - explicit-id ratio in [0,1]
        - mapped unique non-zero force sample count
        - explicit-mapped unique non-zero force sample count
        """

        n_wall = int(wall_centroids.shape[0])
        wall_segment_forces = np.zeros((n_wall, 3), dtype=np.float32)
        if (
            n_wall <= 0
            or not contact_records
            or candidate_forces.size == 0
            or candidate_positions.size == 0
        ):
            return (
                wall_segment_forces,
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((3,), dtype=np.float32),
                "none",
                0,
                0.0,
                0,
                0,
            )

        eps = float(max(contact_epsilon, 0.0))
        force_norms = np.linalg.norm(candidate_forces, axis=1)
        active_force_indices = np.nonzero(force_norms > eps)[0].astype(np.int32)
        if candidate_positions.shape[0] < candidate_forces.shape[0]:
            active_force_indices = active_force_indices[
                active_force_indices < int(candidate_positions.shape[0])
            ]
        if active_force_indices.size <= 0:
            return (
                wall_segment_forces,
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((3,), dtype=np.float32),
                "none",
                0,
                0.0,
                0,
                0,
            )
        # Deterministic record order for reproducible force-index association.
        sorted_records = sorted(
            contact_records,
            key=lambda r: (
                str(r.get("source_node_tag", "")),
                int(r.get("model_side", -1))
                if isinstance(r.get("model_side", None), (int, np.integer))
                else -1,
                int(r.get("contact_local_index", -1))
                if isinstance(r.get("contact_local_index", None), (int, np.integer))
                else -1,
                int(r.get("wall_triangle_id", -1))
                if isinstance(r.get("wall_triangle_id", None), (int, np.integer))
                else -1,
                tuple(
                    np.round(
                        np.asarray(
                            r.get("wall_point", np.zeros((3,), dtype=np.float32)),
                            dtype=np.float64,
                        ).reshape((3,)),
                        6,
                    ).tolist()
                ),
            ),
        )
        anchor_records: List[Tuple[np.ndarray, Dict[str, Any]]] = []
        for rec in sorted_records:
            wire_p = rec.get("wire_point", None)
            wall_p = rec.get("wall_point", None)
            force_anchor = wire_p if wire_p is not None else wall_p
            if force_anchor is None:
                continue
            try:
                force_anchor_arr = np.asarray(force_anchor, dtype=np.float32).reshape((3,))
            except Exception:
                continue
            if not np.all(np.isfinite(force_anchor_arr)):
                continue
            anchor_records.append((force_anchor_arr, rec))
        if not anchor_records:
            return (
                wall_segment_forces,
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((3,), dtype=np.float32),
                "none",
                0,
                0.0,
                0,
                0,
            )
        mapped_explicit = 0
        mapped_explicit_native = 0
        mapped_explicit_listener = 0
        mapped_surface = 0
        mapped_centroid = 0
        mapped_total = 0
        mapped_force_indices: set[int] = set()
        mapped_force_indices_explicit: set[int] = set()
        for idx_force in active_force_indices.tolist():
            idx_force = int(idx_force)
            force_vec = candidate_forces[idx_force].astype(np.float32)
            p_force = candidate_positions[idx_force].astype(np.float32).reshape((1, 3))
            d_anchor = np.asarray(
                [np.sum((p_force - anchor.reshape((1, 3))) ** 2) for anchor, _ in anchor_records],
                dtype=np.float64,
            )
            idx_anchor = int(np.argmin(d_anchor))
            _force_anchor, rec = anchor_records[idx_anchor]
            wall_p = rec.get("wall_point", None)
            wall_tri = rec.get("wall_triangle_id", None)
            tri_source = str(rec.get("triangle_source", "") or "")

            idx_tri: Optional[int] = None
            try:
                tri_int = int(wall_tri)
                if 0 <= tri_int < n_wall:
                    idx_tri = tri_int
                    mapped_explicit += 1
                    if tri_source.startswith("native_contact_export_triangle_id"):
                        mapped_explicit_native += 1
                    elif tri_source.startswith("contact_element"):
                        mapped_explicit_listener += 1
            except Exception:
                idx_tri = None

            if idx_tri is None and wall_p is not None:
                idx_tri = cls._find_nearest_triangle_surface(
                    np.asarray(wall_p, dtype=np.float32),
                    wall_vertices,
                    wall_triangles,
                )
                if idx_tri is not None:
                    mapped_surface += 1
            if idx_tri is None and wall_p is not None:
                d_wall = wall_centroids - np.asarray(wall_p, dtype=np.float32).reshape((1, 3))
                idx_tri = int(np.argmin(np.sum(d_wall * d_wall, axis=1)))
                mapped_centroid += 1
            if idx_tri is None:
                continue

            # Contact force on wall is opposite to force applied to wire/collision point.
            wall_segment_forces[idx_tri] -= force_vec
            mapped_force_indices.add(int(idx_force))
            if tri_source.startswith("native_contact_export") or tri_source.startswith(
                "contact_element"
            ):
                mapped_force_indices_explicit.add(int(idx_force))
            if dof_triangle_cache is not None:
                dof_triangle_cache[int(idx_force)] = int(idx_tri)
            mapped_total += 1

        norms = (
            np.linalg.norm(wall_segment_forces, axis=1)
            if wall_segment_forces.size
            else np.zeros((0,), dtype=np.float32)
        )
        active_mask = norms > eps
        active_ids = np.nonzero(active_mask)[0].astype(np.int32)
        active_forces = wall_segment_forces[active_ids].astype(np.float32)
        total = wall_segment_forces.sum(axis=0).astype(np.float32)
        mapped_force_count = int(len(mapped_force_indices))
        mapped_force_explicit_count = int(len(mapped_force_indices_explicit))

        explicit_ratio = (
            float(mapped_explicit) / float(mapped_total)
            if mapped_total > 0
            else 0.0
        )
        if mapped_explicit > 0 and mapped_surface == 0 and mapped_centroid == 0:
            if mapped_explicit_native == mapped_explicit and mapped_explicit_listener == 0:
                method = "native_contact_export_triangle_id"
            elif mapped_explicit_listener == mapped_explicit and mapped_explicit_native == 0:
                method = "contact_element_triangle_id"
            elif mapped_explicit_native > 0 and mapped_explicit_listener > 0:
                method = "mixed_native_and_listener_triangle_id"
            else:
                method = "explicit_triangle_id"
        elif mapped_explicit > 0 and mapped_surface > 0 and mapped_centroid == 0:
            if mapped_explicit_native > 0 and mapped_explicit_listener == 0:
                method = "mixed_native_and_surface"
            elif mapped_explicit_listener > 0 and mapped_explicit_native == 0:
                method = "mixed_contact_element_and_surface"
            else:
                method = "mixed_explicit_and_surface"
        elif mapped_explicit == 0 and mapped_surface > 0 and mapped_centroid == 0:
            method = "contact_point_nearest_triangle_surface"
        elif mapped_explicit > 0 and mapped_centroid > 0:
            if mapped_explicit_native > 0 and mapped_explicit_listener == 0:
                method = "mixed_native_surface_and_centroid"
            elif mapped_explicit_listener > 0 and mapped_explicit_native == 0:
                method = "mixed_contact_element_surface_and_centroid"
            else:
                method = "mixed_explicit_surface_and_centroid"
        elif mapped_surface > 0 and mapped_centroid > 0:
            method = "mixed_surface_and_centroid"
        elif mapped_centroid > 0:
            method = "contact_point_nearest_centroid"
        else:
            method = "none"
        return (
            wall_segment_forces,
            active_ids,
            active_forces,
            total,
            method,
            int(mapped_total),
            float(explicit_ratio),
            mapped_force_count,
            mapped_force_explicit_count,
        )

    @classmethod
    def _map_force_points_with_cached_triangles(
        cls,
        *,
        wall_vertices: np.ndarray,
        wall_triangles: np.ndarray,
        wall_centroids: np.ndarray,
        candidate_forces: np.ndarray,
        candidate_positions: np.ndarray,
        contact_epsilon: float,
        force_idx_to_wall_triangle: Dict[int, int],
        max_surface_distance: float,
        neighbor_window: int = 0,
        allow_surface_fallback: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int, float]:
        """Map non-zero force samples using cached explicit triangle ids.

        This is used when active constraints exist but no fresh contact records are
        exposed in the current step. The cache is only populated by explicit
        contact-association steps.
        """

        n_wall = int(wall_centroids.shape[0])
        wall_segment_forces = np.zeros((n_wall, 3), dtype=np.float32)
        if (
            n_wall <= 0
            or candidate_forces.size == 0
            or candidate_positions.size == 0
            or not force_idx_to_wall_triangle
        ):
            return (
                wall_segment_forces,
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((3,), dtype=np.float32),
                "none",
                0,
                0.0,
            )

        eps = float(max(contact_epsilon, 0.0))
        max_d = float(max(max_surface_distance, 0.0))
        nbh = int(max(neighbor_window, 0))
        mapped_cached = 0
        mapped_surface = 0
        mapped_total = 0

        def _distance_to_cached_triangle(point: np.ndarray, tri_int: int) -> Optional[float]:
            if tri_int < 0 or tri_int >= n_wall:
                return None
            tri = wall_triangles[tri_int]
            a = wall_vertices[int(tri[0])]
            b = wall_vertices[int(tri[1])]
            c = wall_vertices[int(tri[2])]
            cp = cls._closest_point_on_triangle(point, a, b, c)
            d = point - cp
            return float(np.linalg.norm(d))

        n = min(int(candidate_forces.shape[0]), int(candidate_positions.shape[0]))
        for idx in range(n):
            force_vec = candidate_forces[idx].astype(np.float32)
            if float(np.linalg.norm(force_vec)) <= eps:
                continue

            tri_idx: Optional[int] = None
            tri_cached = force_idx_to_wall_triangle.get(int(idx), None)
            p = candidate_positions[idx].astype(np.float32)
            if tri_cached is not None:
                tri_int = int(tri_cached)
                d_norm = _distance_to_cached_triangle(p, tri_int)
                if d_norm is not None and d_norm <= max_d:
                    tri_idx = tri_int
                    mapped_cached += 1
            if tri_idx is None and nbh > 0:
                best_nbh_tri: Optional[int] = None
                best_nbh_dist = float("inf")
                for j in range(int(idx) - nbh, int(idx) + nbh + 1):
                    tri_nbh = force_idx_to_wall_triangle.get(int(j), None)
                    if tri_nbh is None:
                        continue
                    tri_int = int(tri_nbh)
                    d_norm = _distance_to_cached_triangle(p, tri_int)
                    if d_norm is None:
                        continue
                    if d_norm < best_nbh_dist:
                        best_nbh_dist = d_norm
                        best_nbh_tri = tri_int
                if best_nbh_tri is not None and best_nbh_dist <= max_d:
                    tri_idx = int(best_nbh_tri)
                    mapped_cached += 1
                    force_idx_to_wall_triangle[int(idx)] = int(best_nbh_tri)

            if tri_idx is None and allow_surface_fallback:
                tri_idx = cls._find_nearest_triangle_surface(
                    p,
                    wall_vertices,
                    wall_triangles,
                )
                if tri_idx is not None:
                    mapped_surface += 1
                    force_idx_to_wall_triangle[int(idx)] = int(tri_idx)

            if tri_idx is None:
                continue

            wall_segment_forces[int(tri_idx)] -= force_vec
            mapped_total += 1

        norms = (
            np.linalg.norm(wall_segment_forces, axis=1)
            if wall_segment_forces.size
            else np.zeros((0,), dtype=np.float32)
        )
        active_mask = norms > eps
        active_ids = np.nonzero(active_mask)[0].astype(np.int32)
        active_forces = wall_segment_forces[active_ids].astype(np.float32)
        total = wall_segment_forces.sum(axis=0).astype(np.float32)

        explicit_ratio = (
            float(mapped_cached) / float(mapped_total)
            if mapped_total > 0
            else 0.0
        )
        if mapped_cached > 0 and mapped_surface == 0:
            method = "cached_contact_triangle_id"
        elif mapped_cached > 0 and mapped_surface > 0:
            method = "mixed_cached_and_surface"
        elif mapped_cached == 0 and mapped_surface > 0:
            method = "force_point_nearest_triangle_surface"
        else:
            method = "none"
        return (
            wall_segment_forces,
            active_ids,
            active_forces,
            total,
            method,
            int(mapped_total),
            float(explicit_ratio),
        )

    @staticmethod
    def _map_force_points_to_wall_segments(
        *,
        wall_centroids: np.ndarray,
        candidate_forces: np.ndarray,
        candidate_positions: np.ndarray,
        contact_epsilon: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fallback mapping without ContactListener pairs.

        Maps each non-zero force sample to the nearest wall triangle centroid.
        """

        n_wall = int(wall_centroids.shape[0])
        wall_segment_forces = np.zeros((n_wall, 3), dtype=np.float32)
        if (
            n_wall <= 0
            or candidate_forces.size == 0
            or candidate_positions.size == 0
        ):
            return (
                wall_segment_forces,
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((3,), dtype=np.float32),
            )

        eps = float(max(contact_epsilon, 0.0))
        for idx in range(min(candidate_forces.shape[0], candidate_positions.shape[0])):
            force_vec = candidate_forces[idx].astype(np.float32)
            if float(np.linalg.norm(force_vec)) <= eps:
                continue
            p = candidate_positions[idx].astype(np.float32).reshape((1, 3))
            d_wall = wall_centroids - p
            idx_tri = int(np.argmin(np.sum(d_wall * d_wall, axis=1)))
            wall_segment_forces[idx_tri] -= force_vec

        norms = (
            np.linalg.norm(wall_segment_forces, axis=1)
            if wall_segment_forces.size
            else np.zeros((0,), dtype=np.float32)
        )
        active_mask = norms > eps
        active_ids = np.nonzero(active_mask)[0].astype(np.int32)
        active_forces = wall_segment_forces[active_ids].astype(np.float32)
        total = wall_segment_forces.sum(axis=0).astype(np.float32)
        return wall_segment_forces, active_ids, active_forces, total

    def _build_from_collision_point_forces(self, point_forces: np.ndarray) -> None:
        point_forces = np.asarray(point_forces, dtype=np.float32)
        if point_forces.ndim != 2 or point_forces.shape[1] != 3:
            self._contact_force_vectors = np.zeros((0, 3), dtype=np.float32)
            self._contact_segment_indices = np.zeros((0,), dtype=np.int32)
            self._segment_force_vectors = np.zeros((0, 3), dtype=np.float32)
            self._total_force_vector = np.zeros((3,), dtype=np.float32)
            self._total_force_norm = float("nan")
            return

        if point_forces.shape[0] >= 2:
            self._segment_force_vectors = (
                0.5 * (point_forces[:-1] + point_forces[1:])
            ).astype(np.float32)
        else:
            self._segment_force_vectors = np.zeros((0, 3), dtype=np.float32)

        norms = np.linalg.norm(point_forces, axis=1) if point_forces.size else np.zeros((0,), dtype=np.float32)
        mask = norms > self._contact_epsilon
        self._contact_force_vectors = point_forces[mask].astype(np.float32)
        if self._segment_force_vectors.shape[0] > 0:
            max_seg = self._segment_force_vectors.shape[0] - 1
            indices = np.clip(np.nonzero(mask)[0], 0, max_seg)
            self._contact_segment_indices = indices.astype(np.int32)
        else:
            self._contact_segment_indices = np.zeros((0,), dtype=np.int32)

        self._total_force_vector = np.sum(point_forces, axis=0).astype(np.float32)
        self._total_force_norm = float(np.linalg.norm(self._total_force_vector))

    @staticmethod
    def _normalize_point_force_array(arr: Any, *, allow_6dof: bool) -> np.ndarray:
        try:
            arr = np.asarray(arr, dtype=np.float32)
        except Exception:
            return np.zeros((0, 3), dtype=np.float32)
        if arr.ndim == 1:
            if allow_6dof and arr.size % 6 == 0:
                arr = arr.reshape((-1, 6))
            elif arr.size % 3 == 0:
                arr = arr.reshape((-1, 3))
            else:
                return np.zeros((0, 3), dtype=np.float32)
        if arr.ndim != 2:
            return np.zeros((0, 3), dtype=np.float32)
        if arr.shape[1] >= 3:
            return arr[:, :3].astype(np.float32)
        return np.zeros((0, 3), dtype=np.float32)

    @classmethod
    def _pick_best_force_array(
        cls,
        arrays: Dict[str, np.ndarray],
    ) -> tuple[np.ndarray, str]:
        best_name = ""
        best_arr = np.zeros((0, 3), dtype=np.float32)
        best_norm = -1.0
        for name, arr in arrays.items():
            arr = cls._normalize_point_force_array(arr, allow_6dof=False)
            if arr.size == 0:
                continue
            norm = float(np.linalg.norm(np.sum(arr, axis=0)))
            if norm > best_norm:
                best_norm = norm
                best_name = name
                best_arr = arr
        return best_arr, best_name

    @classmethod
    def _read_collision_point_forces(cls, sim: Any) -> tuple[np.ndarray, str]:
        arrays: Dict[str, np.ndarray] = {}
        try:
            coll_dofs = sim._instruments_combined.CollisionModel.CollisionDOFs  # noqa: SLF001
        except Exception:
            return np.zeros((0, 3), dtype=np.float32), ""
        for attr in ("force", "externalForce"):
            try:
                arrays[attr] = np.asarray(getattr(coll_dofs, attr).value, dtype=np.float32)
            except Exception:
                continue
        if not arrays:
            return np.zeros((0, 3), dtype=np.float32), ""
        return cls._pick_best_force_array(arrays)

    @classmethod
    def _read_wire_point_forces(cls, sim: Any) -> tuple[np.ndarray, str]:
        arrays: Dict[str, np.ndarray] = {}
        try:
            wire_dofs = sim._instruments_combined.DOFs  # noqa: SLF001
        except Exception:
            return np.zeros((0, 3), dtype=np.float32), ""
        for attr in ("force", "externalForce"):
            try:
                raw = np.asarray(getattr(wire_dofs, attr).value, dtype=np.float32)
            except Exception:
                continue
            arr = cls._normalize_point_force_array(raw, allow_6dof=True)
            if arr.size:
                arrays[attr] = arr
        if not arrays:
            return np.zeros((0, 3), dtype=np.float32), ""
        # Wire DOFs may include 6DoF rows; arrays are already normalized to (N,3).
        best_name = ""
        best_arr = np.zeros((0, 3), dtype=np.float32)
        best_norm = -1.0
        for name, arr in arrays.items():
            norm = float(np.linalg.norm(np.sum(arr, axis=0)))
            if norm > best_norm:
                best_norm = norm
                best_name = name
                best_arr = arr
        return best_arr, best_name

    def step(self) -> None:
        sim = getattr(self._intervention, "simulation", None)

        # If the intervention is still in mp mode, we cannot access SOFA objects.
        if sim is None or not hasattr(sim, "root"):
            self._available = False
            self._source = "unavailable"
            self._error = "simulation-root-unavailable"
            self._contact_count = 0
            self._contact_detected = False
            self._lcp_sum_abs = float("nan")
            self._lcp_max_abs = float("nan")
            self._lcp_active_count = 0
            self._wire_force_norm = float("nan")
            self._collis_force_norm = float("nan")
            self._wire_force_vectors = np.zeros((0, 3), dtype=np.float32)
            self._collision_force_vectors = np.zeros((0, 3), dtype=np.float32)
            self._wire_force_vectors_source = ""
            self._collision_force_vectors_source = ""
            self._force_source = "none"
            self._force_norm_sum = float("nan")
            self._status = "err:simulation-root-unavailable"
            self._quality_tier = "unavailable"
            self._association_method = "none"
            self._association_explicit_ratio = float("nan")
            return

        runtime = self._runtime.ensure(sim)
        if runtime.error and not self._error:
            self._error = runtime.error
        if runtime.source and runtime.source != "uninitialized":
            self._source = runtime.source
        if self._required and not runtime.configured:
            raise RuntimeError(
                "force extraction is required but runtime setup failed "
                f"(mode={self._mode}, source={runtime.source}, error={runtime.error})"
            )

        geometry_contact_count = self._extract_contact_count(sim)
        self._contact_count = geometry_contact_count
        self._lcp_sum_abs, self._lcp_max_abs, self._lcp_active_count = self._extract_lcp(sim)
        self._wall_segment_count = 0
        self._wall_active_segment_ids = np.zeros((0,), dtype=np.int32)
        self._wall_active_segment_force_vectors = np.zeros((0, 3), dtype=np.float32)
        self._contact_force_vectors = np.zeros((0, 3), dtype=np.float32)
        self._contact_segment_indices = np.zeros((0,), dtype=np.int32)
        self._segment_force_vectors = np.zeros((0, 3), dtype=np.float32)
        self._contact_detected = False
        self._force_source = "none"
        self._force_norm_sum = float("nan")
        self._wire_force_vectors = np.zeros((0, 3), dtype=np.float32)
        self._collision_force_vectors = np.zeros((0, 3), dtype=np.float32)
        self._wire_force_vectors_source = ""
        self._collision_force_vectors_source = ""
        self._wall_field_force_vector = np.zeros((3,), dtype=np.float32)
        self._wall_field_force_norm = float("nan")
        self._status = ""

        if self._mode in {"passive", "constraint_projected_si_validated"}:
            monitor = getattr(sim.root, "wire_wall_force_monitor", None)
            if monitor is None:
                self._available = False
                self._source = runtime.source or "passive_monitor_missing"
                self._error = runtime.error or "wire_wall_force_monitor object missing"
                self._status = f"err:{self._source}"
                self._quality_tier = "unavailable"
                self._association_method = "none"
                self._association_explicit_ratio = float("nan")
                self._wire_force_norm = float("nan")
                self._collis_force_norm = float("nan")
                self._wire_force_vectors = np.zeros((0, 3), dtype=np.float32)
                self._collision_force_vectors = np.zeros((0, 3), dtype=np.float32)
                self._wire_force_vectors_source = ""
                self._collision_force_vectors_source = ""
                self._total_force_norm = float("nan")
                self._total_force_vector = np.zeros((3,), dtype=np.float32)
                return

            monitor_available = bool(self._read_data(monitor, "available"))
            monitor_source = str(
                self._read_data(monitor, "source") or runtime.source or "passive_monitor"
            )
            monitor_status = str(self._read_data(monitor, "status") or runtime.error or "")

            wall_vertices, wall_triangles, wall_centroids = self._read_vessel_wall_geometry(sim)
            self._wall_segment_count = int(wall_centroids.shape[0])
            (
                native_records,
                native_count,
                native_meta,
            ) = self._extract_contact_records_from_native_export(
                sim,
                wall_triangle_count=self._wall_segment_count,
            )
            self._native_contact_export_available = bool(
                native_meta.get("available", False)
            )
            self._native_contact_export_source = str(
                native_meta.get("source", "")
            )
            self._native_contact_export_status = str(
                native_meta.get("status", "")
            )
            self._native_contact_export_explicit_coverage = float(
                native_meta.get("explicit_coverage", float("nan"))
            )
            native_ordering_stable = bool(native_meta.get("ordering_stable", False))
            native_no_contacts_step = str(native_meta.get("status", "")).startswith(
                "ok:no_contacts"
            )
            listener_records, listener_count = self._extract_contact_records_from_listeners(
                sim, wall_triangle_count=self._wall_segment_count
            )
            contact_node_records, contact_node_count = self._extract_contact_records_from_contact_nodes(
                sim
            )
            fallback_records = self._deduplicate_contact_records(
                list(listener_records) + list(contact_node_records)
            )
            using_native_records = False
            contact_records_source = "none"
            if self._mode == "constraint_projected_si_validated":
                # Native exporter is the only path that can lead to validated quality.
                # Fallback records are kept for degraded operation only.
                if native_records:
                    contact_records = list(native_records)
                    using_native_records = True
                    contact_records_source = "native_contact_export"
                else:
                    contact_records = list(fallback_records)
                    contact_records_source = "fallback_contact_records"
            else:
                contact_records = self._deduplicate_contact_records(
                    list(native_records) + list(fallback_records)
                )
                if native_records:
                    contact_records_source = "mixed_native_and_fallback"
                elif fallback_records:
                    contact_records_source = "fallback_contact_records"
                else:
                    contact_records_source = "none"
            # numberOfContacts on collision models is noisy in this scene (can include
            # non wall-wire pairs / stale broad-phase counts). For force validation we
            # trust ContactListener + monitor-side counters + active LCP rows.
            monitor_contact_count = self._safe_int(self._read_data(monitor, "contactCount"), 0)
            self._contact_count = max(
                int(listener_count),
                int(contact_node_count),
                int(native_count),
                int(monitor_contact_count),
                int(len(contact_records)),
            )
            self._contact_detected = bool(
                int(self._contact_count) > 0
                or (
                    np.isfinite(self._lcp_max_abs)
                    and float(self._lcp_max_abs) > float(self._contact_epsilon)
                )
            )

            # Optional auxiliary metric: wall field force (can be zero for static walls).
            try:
                vessel_dofs = sim.root.vesselTree.dofs
                wall_field = self._normalize_point_force_array(
                    self._read_data(vessel_dofs, "force"), allow_6dof=False
                )
                if wall_field.size:
                    self._wall_field_force_vector = np.sum(wall_field, axis=0).astype(
                        np.float32
                    )
                    self._wall_field_force_norm = float(
                        np.linalg.norm(self._wall_field_force_vector)
                    )
            except Exception:
                pass

            # Canonical source in validated mode: constraint projection only.
            # Legacy passive mode keeps broader candidates for compatibility.
            if self._mode == "constraint_projected_si_validated":
                candidates = self._collect_constraint_projection_candidates(sim)
            else:
                candidates = self._collect_force_candidates(sim)
            best_source = "none"
            best_force_norm_sum = 0.0
            best_forces = np.zeros((0, 3), dtype=np.float32)
            best_positions = np.zeros((0, 3), dtype=np.float32)
            best_row_contribs: List[Dict[str, Any]] = []
            if candidates:
                if self._mode == "constraint_projected_si_validated":
                    # No source mixing in validated mode: prefer collision projection as
                    # source-of-truth; only use wire projection as explicitly degraded fallback.
                    preferred = [c for c in candidates if c[0].startswith("collision.constraintProjection")]
                    degraded = [c for c in candidates if c[0].startswith("wire.constraintProjection")]
                    if preferred:
                        (
                            best_source,
                            best_forces,
                            best_positions,
                            best_force_norm_sum,
                            best_row_contribs,
                        ) = max(
                            preferred, key=lambda c: float(c[3])
                        )
                    elif degraded:
                        (
                            best_source,
                            best_forces,
                            best_positions,
                            best_force_norm_sum,
                            best_row_contribs,
                        ) = max(
                            degraded, key=lambda c: float(c[3])
                        )
                else:
                    (
                        best_source,
                        best_forces,
                        best_positions,
                        best_force_norm_sum,
                        best_row_contribs,
                    ) = max(
                        candidates, key=lambda c: float(c[3])
                    )

            tip_pos = None
            try:
                tip_pos = np.asarray(
                    self._intervention.fluoroscopy.tracking3d[0],
                    dtype=np.float32,
                )
            except Exception:
                tip_pos = None
            (
                self._tip_force_vector,
                self._tip_force_norm,
                self._tip_force_sample_index,
                self._tip_force_source,
            ) = self._extract_tip_force_from_samples(
                tip_pos=tip_pos,
                candidate_forces=best_forces,
                candidate_positions=best_positions,
                contact_epsilon=self._contact_epsilon,
            )
            if best_source and self._tip_force_source != "none":
                self._tip_force_source = f"{self._tip_force_source}[{best_source}]"

            collision_candidates = [
                c for c in candidates if c[0].startswith("collision.constraintProjection")
            ]
            if collision_candidates:
                (
                    _collision_source,
                    collision_forces,
                    _collision_positions,
                    _collision_norm,
                    collision_row_contribs,
                ) = max(collision_candidates, key=lambda c: float(c[3]))
            else:
                collision_forces = np.zeros((0, 3), dtype=np.float32)
                collision_row_contribs = []
            collision_nonzero_force_point_count = (
                int(
                    np.count_nonzero(
                        np.linalg.norm(collision_forces, axis=1) > float(self._contact_epsilon)
                    )
                )
                if collision_forces.size
                else 0
            )
            collision_active_rows = [
                rc
                for rc in collision_row_contribs
                if float(rc.get("force_norm", 0.0)) > float(self._contact_epsilon)
                and isinstance(rc.get("row_idx", None), (int, np.integer))
                and int(rc.get("row_idx")) >= 0
            ]
            collision_active_projected_count = int(len(collision_active_rows))
            lambda_active = bool(
                np.isfinite(self._lcp_max_abs)
                and float(self._lcp_max_abs) > float(self._contact_epsilon)
            )
            self._active_constraint_step = bool(
                lambda_active and collision_active_projected_count > 0
            )

            used_fallback_mapping = False
            association_method = "none"
            mapped_contact_count = 0
            mapped_force_count = 0
            mapped_force_explicit_count = 0
            association_explicit_ratio = 0.0
            association_coverage = float("nan")
            association_explicit_force_coverage = float("nan")
            association_ordering_stable = False
            self._gap_active_projected_count = int(collision_active_projected_count)
            self._gap_explicit_mapped_count = 0
            self._gap_unmapped_count = 0
            self._gap_class_counts = {}
            self._gap_dominant_class = "none"
            self._gap_contact_mode = self._contact_mode_from_records(contact_records)
            nonzero_force_point_count = int(
                np.count_nonzero(
                    np.linalg.norm(best_forces, axis=1) > float(self._contact_epsilon)
                )
            ) if best_forces.size else 0
            if self._mode == "constraint_projected_si_validated":
                association_ordering_stable = bool(
                    using_native_records
                    and native_ordering_stable
                    and all(
                        bool(rec.get("integrity_ordering_stable", False))
                        for rec in contact_records
                    )
                )
                (
                    self._segment_force_vectors,
                    self._wall_active_segment_ids,
                    self._wall_active_segment_force_vectors,
                    self._total_force_vector,
                    association_method,
                    mapped_contact_count,
                    association_explicit_ratio,
                    mapped_force_count,
                    mapped_force_explicit_count,
                    gap_info,
                ) = self._map_row_projected_forces_to_wall_segments(
                    row_contribs=collision_active_rows,
                    contact_records=contact_records,
                    wall_vertices=wall_vertices,
                    wall_triangles=wall_triangles,
                    wall_centroids=wall_centroids,
                    contact_epsilon=self._contact_epsilon,
                    allow_surface_fallback=True,
                    native_no_contacts_step=bool(native_no_contacts_step),
                )
                self._gap_active_projected_count = int(
                    gap_info.get("active_projected_count", 0)
                )
                self._gap_explicit_mapped_count = int(
                    gap_info.get("explicit_mapped_count", 0)
                )
                self._gap_unmapped_count = int(gap_info.get("unmapped_count", 0))
                self._gap_class_counts = dict(gap_info.get("class_counts", {}) or {})
                self._gap_dominant_class = str(
                    gap_info.get("dominant_class", "none") or "none"
                )
                self._gap_contact_mode = str(
                    gap_info.get("contact_mode", "none") or "none"
                )
                self._gap_row_bridge_mode = str(
                    gap_info.get("row_bridge_mode", "none") or "none"
                )
                self._gap_row_bridge_offset = int(gap_info.get("row_bridge_offset", 0))
                self._gap_row_bridge_applied = bool(
                    gap_info.get("row_bridge_applied", False)
                )
                self._gap_row_bridge_deterministic = bool(
                    gap_info.get("row_bridge_deterministic", True)
                )
                self._gap_row_bridge_pair_count = int(
                    gap_info.get("row_bridge_pair_count", 0)
                )
                self._gap_row_bridge_dof_hits = int(
                    gap_info.get("row_bridge_dof_hits", 0)
                )
                if mapped_force_count > 0 and collision_active_projected_count > 0:
                    association_coverage = float(mapped_force_count) / float(
                        collision_active_projected_count
                    )
                if (
                    mapped_force_explicit_count > 0
                    and collision_active_projected_count > 0
                ):
                    association_explicit_force_coverage = float(
                        mapped_force_explicit_count
                    ) / float(collision_active_projected_count)
                if (
                    mapped_contact_count <= 0
                    and best_forces.size > 0
                    and best_positions.size > 0
                    and np.isfinite(self._lcp_max_abs)
                    and float(self._lcp_max_abs) > float(self._contact_epsilon)
                ):
                    (
                        self._segment_force_vectors,
                        self._wall_active_segment_ids,
                        self._wall_active_segment_force_vectors,
                        self._total_force_vector,
                        association_method,
                        mapped_contact_count,
                        association_explicit_ratio,
                    ) = self._map_force_points_with_cached_triangles(
                        wall_vertices=wall_vertices,
                        wall_triangles=wall_triangles,
                        wall_centroids=wall_centroids,
                        candidate_forces=best_forces,
                        candidate_positions=best_positions,
                        contact_epsilon=self._contact_epsilon,
                        force_idx_to_wall_triangle=self._force_idx_to_wall_triangle,
                        max_surface_distance=self._cache_max_surface_distance_mm,
                        neighbor_window=self._cache_neighbor_window,
                        allow_surface_fallback=False,
                    )
                    mapped_force_count = int(mapped_contact_count)
                    mapped_force_explicit_count = int(
                        round(
                            float(max(0.0, min(1.0, association_explicit_ratio)))
                            * float(mapped_force_count)
                        )
                    )
                    if mapped_force_count > 0 and collision_active_projected_count > 0:
                        association_coverage = float(mapped_force_count) / float(
                            collision_active_projected_count
                        )
                    if (
                        mapped_force_explicit_count > 0
                        and collision_active_projected_count > 0
                    ):
                        association_explicit_force_coverage = float(
                            mapped_force_explicit_count
                        ) / float(collision_active_projected_count)
                if (
                    mapped_contact_count <= 0
                    and best_forces.size > 0
                    and best_positions.size > 0
                    and np.isfinite(self._lcp_max_abs)
                    and float(self._lcp_max_abs) > float(self._contact_epsilon)
                ):
                    used_fallback_mapping = True
                    (
                        self._segment_force_vectors,
                        self._wall_active_segment_ids,
                        self._wall_active_segment_force_vectors,
                        self._total_force_vector,
                    ) = self._map_force_points_to_wall_segments(
                        wall_centroids=wall_centroids,
                        candidate_forces=best_forces,
                        candidate_positions=best_positions,
                        contact_epsilon=self._contact_epsilon,
                    )
                    mapped_contact_count = int(self._wall_active_segment_ids.size)
                    mapped_force_count = int(nonzero_force_point_count)
                    mapped_force_explicit_count = 0
                    association_method = "degraded_force_point_nearest_centroid"
                    association_explicit_ratio = 0.0
                    if collision_active_projected_count > 0:
                        association_coverage = float(mapped_force_count) / float(
                            collision_active_projected_count
                        )
                    association_explicit_force_coverage = 0.0
            else:
                contact_pairs = [
                    (rec["wire_point"], rec["wall_point"])
                    for rec in contact_records
                    if rec.get("wire_point", None) is not None and rec.get("wall_point", None) is not None
                ]
                (
                    self._segment_force_vectors,
                    self._wall_active_segment_ids,
                    self._wall_active_segment_force_vectors,
                    self._total_force_vector,
                ) = self._map_contact_forces_to_wall_segments(
                    contact_pairs=contact_pairs,
                    wall_centroids=wall_centroids,
                    candidate_forces=best_forces,
                    candidate_positions=best_positions,
                    contact_epsilon=self._contact_epsilon,
                )
                if (
                    not contact_pairs
                    and best_forces.size > 0
                    and best_positions.size > 0
                    and np.isfinite(self._lcp_max_abs)
                    and float(self._lcp_max_abs) > float(self._contact_epsilon)
                ):
                    used_fallback_mapping = True
                    (
                        self._segment_force_vectors,
                        self._wall_active_segment_ids,
                        self._wall_active_segment_force_vectors,
                        self._total_force_vector,
                    ) = self._map_force_points_to_wall_segments(
                        wall_centroids=wall_centroids,
                        candidate_forces=best_forces,
                        candidate_positions=best_positions,
                        contact_epsilon=self._contact_epsilon,
                    )
                    nnz_force_points = int(
                        np.count_nonzero(
                            np.linalg.norm(best_forces, axis=1) > float(self._contact_epsilon)
                        )
                    )
                    self._contact_count = max(int(self._contact_count), nnz_force_points)
                association_method = (
                    "force_points_nearest_triangle"
                    if used_fallback_mapping
                    else "contact_point_nearest_centroid"
                )
                mapped_contact_count = int(self._wall_active_segment_ids.size)
                mapped_force_count = int(nonzero_force_point_count)
                mapped_force_explicit_count = 0
            self._total_force_norm = self._safe_norm(self._total_force_vector)
            self._contact_force_vectors = self._wall_active_segment_force_vectors
            self._contact_segment_indices = self._wall_active_segment_ids

            # Channel stats (used for debugging/validation).
            coll_forces_arr, coll_source = self._read_collision_point_forces(sim)
            wire_forces_arr, wire_source = self._read_wire_point_forces(sim)
            self._wire_force_vectors = np.asarray(wire_forces_arr, dtype=np.float32).reshape(
                (-1, 3)
            )
            self._collision_force_vectors = np.asarray(
                coll_forces_arr, dtype=np.float32
            ).reshape((-1, 3))
            self._wire_force_vectors_source = str(wire_source or "")
            self._collision_force_vectors_source = str(coll_source or "")
            self._collis_force_norm = self._safe_norm(coll_forces_arr)
            self._wire_force_norm = self._safe_norm(wire_forces_arr)
            self._force_source = best_source
            self._force_norm_sum = float(best_force_norm_sum)

            self._available = bool(runtime.configured and monitor_available)
            self._source = f"passive_monitor_wall_triangles[{best_source}]"
            self._association_method = association_method
            self._association_explicit_ratio = (
                float(association_explicit_ratio)
                if np.isfinite(association_explicit_ratio)
                else float("nan")
            )
            if (
                self._mode == "constraint_projected_si_validated"
                and self._active_constraint_step
                and collision_active_projected_count > 0
            ):
                # Coverage gating denominator for validated mode:
                # active constraint step => collision.constraintProjection non-zero contributions.
                self._association_coverage = float(mapped_force_count) / float(
                    collision_active_projected_count
                )
                self._association_explicit_force_coverage = float(
                    mapped_force_explicit_count
                ) / float(collision_active_projected_count)
            else:
                self._association_coverage = (
                    float(association_coverage)
                    if np.isfinite(association_coverage)
                    else float("nan")
                )
                self._association_explicit_force_coverage = (
                    float(association_explicit_force_coverage)
                    if np.isfinite(association_explicit_force_coverage)
                    else float("nan")
                )
            self._association_ordering_stable = bool(association_ordering_stable)

            sum_tolerance = max(1e-6, 10.0 * float(self._contact_epsilon))
            if self._wall_active_segment_force_vectors.size > 0:
                seg_sum = np.sum(self._wall_active_segment_force_vectors, axis=0).astype(np.float32)
            else:
                seg_sum = np.zeros((3,), dtype=np.float32)
            sum_consistent = bool(
                np.allclose(
                    seg_sum,
                    np.asarray(self._total_force_vector, dtype=np.float32).reshape((3,)),
                    atol=float(sum_tolerance),
                    rtol=1e-3,
                )
            )

            if best_source == "none":
                if self._contact_detected:
                    self._quality_tier = "unavailable"
                    self._status = "err:no_force_candidate"
                    self._error = "contact_detected_without_constraint_projection"
                else:
                    self._quality_tier = "validated"
                    self._status = "ok:no_force_candidate:no_contact"
                    self._error = ""
            else:
                if self._mode == "constraint_projected_si_validated":
                    is_collision_projection = best_source.startswith("collision.constraintProjection")
                    validated_association = association_method in {
                        "native_contact_export_triangle_id",
                        "native_contact_export_triangle_id_global_row_bridge",
                    }
                    full_coverage = bool(
                        np.isfinite(self._association_coverage)
                        and float(self._association_coverage) >= (1.0 - 1e-6)
                    )
                    full_explicit_force_coverage = bool(
                        np.isfinite(self._association_explicit_force_coverage)
                        and float(self._association_explicit_force_coverage) >= (1.0 - 1e-6)
                    )
                    if self._active_constraint_step and not is_collision_projection:
                        self._quality_tier = "unavailable"
                        self._status = "err:active_constraint_without_collision_projection"
                        self._error = "active_constraint_without_collision_projection"
                    elif self._active_constraint_step and mapped_force_count <= 0:
                        self._quality_tier = "unavailable"
                        self._status = "err:no_contact_association"
                        self._error = "active_constraint_without_contact_association"
                    elif (
                        self._active_constraint_step
                        and validated_association
                        and bool(self._native_contact_export_available)
                        and bool(using_native_records)
                        and bool(self._association_ordering_stable)
                        and full_coverage
                        and full_explicit_force_coverage
                        and sum_consistent
                    ):
                        self._quality_tier = "validated"
                        self._status = (
                            f"ok:{best_source}:{association_method}:coverage=1"
                        )
                    elif not self._active_constraint_step and not self._contact_detected:
                        self._quality_tier = "validated"
                        self._status = f"ok:{best_source}:no_contact"
                    elif not self._active_constraint_step:
                        # Geometry-only contacts are non-binding for coverage gating.
                        self._quality_tier = "validated"
                        self._status = f"ok:{best_source}:no_active_constraint_step"
                    else:
                        self._quality_tier = "degraded"
                        self._status = f"warn:{best_source}:{association_method}"
                        if not self._error:
                            self._error = (
                                "degraded_contact_association"
                                if mapped_force_count > 0
                                else "degraded_missing_explicit_association"
                            )
                else:
                    self._quality_tier = "degraded"
                    self._status = (
                        f"ok:{best_source}:{association_method}"
                        if best_source != "none"
                        else "err:no_force_candidate"
                    )

            if monitor_status:
                self._status += f";monitor={monitor_status}"
            self._status += f";contacts_src={contact_records_source}"
            self._status += (
                f";native_export={int(self._native_contact_export_available)}"
                f":{self._native_contact_export_status or 'n/a'}"
            )
            if self._mode == "constraint_projected_si_validated":
                self._status += (
                    f";gap=act{int(self._gap_active_projected_count)}"
                    f"/exp{int(self._gap_explicit_mapped_count)}"
                    f"/unm{int(self._gap_unmapped_count)}"
                    f":{self._gap_dominant_class}"
                )
            if not contact_records and not used_fallback_mapping:
                self._status += ";contacts=none"
            if used_fallback_mapping:
                self._status += ";contacts=fallback_force_points"
            if geometry_contact_count > self._contact_count:
                self._status += (
                    f";geom_contacts_ignored={int(geometry_contact_count)}"
                )
            if runtime.error:
                self._error = str(runtime.error)
            elif monitor_status.startswith("err:"):
                self._error = monitor_status
            elif best_source != "none" and self._quality_tier != "degraded":
                self._error = ""
            self._apply_si_conversion_if_needed()
            return

        # intrusive_lcp mode (explicit opt-in fallback)
        if not runtime.configured and self._required:
            self._available = False
            self._source = runtime.source
            self._error = runtime.error
            self._status = f"err:{runtime.source}"
            self._quality_tier = "unavailable"
            self._association_method = "none"
            return
        try:
            coll_forces_arr, coll_source = self._read_collision_point_forces(sim)
            wire_forces_arr, wire_source = self._read_wire_point_forces(sim)
            self._wire_force_vectors = np.asarray(wire_forces_arr, dtype=np.float32).reshape(
                (-1, 3)
            )
            self._collision_force_vectors = np.asarray(
                coll_forces_arr, dtype=np.float32
            ).reshape((-1, 3))
            self._wire_force_vectors_source = str(wire_source or "")
            self._collision_force_vectors_source = str(coll_source or "")
            self._build_from_collision_point_forces(coll_forces_arr)
            self._collis_force_norm = self._safe_norm(coll_forces_arr)
            self._wire_force_norm = self._safe_norm(wire_forces_arr)
            self._force_source = (
                f"intrusive_lcp[{coll_source}]" if coll_source else "intrusive_lcp"
            )
            self._force_norm_sum = (
                float(np.linalg.norm(coll_forces_arr, axis=1).sum())
                if coll_forces_arr.size
                else 0.0
            )
            self._available = True
            self._source = runtime.source or "intrusive_lcp"
            self._status = f"ok:{self._force_source}:direct_point_forces"
            self._quality_tier = "degraded"
            self._association_method = "direct_point_forces"
            self._association_explicit_ratio = float("nan")
            if coll_source:
                self._source = f"{self._source}[{coll_source}]"
            self._error = runtime.error
            if (not np.isfinite(self._total_force_norm)) or (
                abs(float(self._total_force_norm)) <= self._contact_epsilon
            ):
                wire_point_forces, wire_source = self._read_wire_point_forces(sim)
                if wire_point_forces.size:
                    wire_norm = float(np.linalg.norm(np.sum(wire_point_forces, axis=0)))
                    if wire_norm > self._contact_epsilon:
                        self._build_from_collision_point_forces(wire_point_forces)
                        self._collis_force_norm = self._safe_norm(self._contact_force_vectors)
                        self._wire_force_norm = self._safe_norm(self._segment_force_vectors)
                        self._contact_count = max(
                            int(self._contact_count),
                            int(self._contact_force_vectors.shape[0]),
                        )
                        if wire_source:
                            self._source = "intrusive_lcp_wiredofs_fallback[" + wire_source + "]"
                        else:
                            self._source = "intrusive_lcp_wiredofs_fallback"
                        self._force_source = self._source
                        self._status = f"ok:{self._force_source}:wiredofs_fallback"
                        self._association_method = "wiredofs_fallback"
                        self._association_explicit_ratio = float("nan")
                        self._force_norm_sum = (
                            float(np.linalg.norm(wire_point_forces, axis=1).sum())
                            if wire_point_forces.size
                            else 0.0
                        )
            self._wall_segment_count = int(self._segment_force_vectors.shape[0])
            seg_norms = (
                np.linalg.norm(self._segment_force_vectors, axis=1)
                if self._segment_force_vectors.size
                else np.zeros((0,), dtype=np.float32)
            )
            active = seg_norms > self._contact_epsilon
            if np.any(active):
                self._wall_active_segment_ids = np.nonzero(active)[0].astype(np.int32)
                self._wall_active_segment_force_vectors = self._segment_force_vectors[
                    self._wall_active_segment_ids
                ]
            else:
                self._wall_active_segment_ids = np.zeros((0,), dtype=np.int32)
                self._wall_active_segment_force_vectors = np.zeros(
                    (0, 3), dtype=np.float32
                )
            self._contact_detected = bool(
                int(self._contact_count) > 0
                or (
                    np.isfinite(self._lcp_max_abs)
                    and float(self._lcp_max_abs) > float(self._contact_epsilon)
                )
            )
            try:
                tip_pos = np.asarray(
                    self._intervention.fluoroscopy.tracking3d[0],
                    dtype=np.float32,
                )
            except Exception:
                tip_pos = None
            coll_positions = np.zeros((0, 3), dtype=np.float32)
            try:
                coll_dofs = sim._instruments_combined.CollisionModel.CollisionDOFs  # noqa: SLF001
                coll_positions = self._normalize_point_coord_array(
                    self._read_data(coll_dofs, "position"),
                    allow_6dof=False,
                )
            except Exception:
                pass
            (
                self._tip_force_vector,
                self._tip_force_norm,
                self._tip_force_sample_index,
                self._tip_force_source,
            ) = self._extract_tip_force_from_samples(
                tip_pos=tip_pos,
                candidate_forces=self._collision_force_vectors,
                candidate_positions=coll_positions,
                contact_epsilon=self._contact_epsilon,
            )
            if self._tip_force_source != "none":
                self._tip_force_source = "nearest_tip_sample[intrusive_collision]"
            self._apply_si_conversion_if_needed()
        except Exception as exc:
            self._available = False
            self._source = "intrusive_lcp_error"
            self._error = str(exc)
            self._wire_force_norm = float("nan")
            self._collis_force_norm = float("nan")
            self._wire_force_vectors = np.zeros((0, 3), dtype=np.float32)
            self._collision_force_vectors = np.zeros((0, 3), dtype=np.float32)
            self._wire_force_vectors_source = ""
            self._collision_force_vectors_source = ""
            self._total_force_norm = float("nan")
            self._total_force_vector = np.zeros((3,), dtype=np.float32)
            self._contact_force_vectors = np.zeros((0, 3), dtype=np.float32)
            self._contact_segment_indices = np.zeros((0,), dtype=np.int32)
            self._segment_force_vectors = np.zeros((0, 3), dtype=np.float32)
            self._contact_detected = False
            self._force_source = "intrusive_lcp_error"
            self._force_norm_sum = float("nan")
            self._status = f"err:{self._source}"
            self._quality_tier = "unavailable"
            self._association_method = "none"
            self._association_explicit_ratio = float("nan")
            self._apply_si_conversion_if_needed()

    @property
    def info(self) -> Dict[str, Any]:
        units = units_to_dict(self._units) if self._units is not None else {}
        return {
            "wall_force_mode": self._mode,
            "wall_force_available": bool(self._available),
            "wall_force_source": self._source,
            "wall_force_status": self._status,
            "wall_force_error": self._error,
            "wall_force_quality_tier": self._quality_tier,
            "wall_force_association_method": self._association_method,
            "wall_force_association_explicit_ratio": self._association_explicit_ratio,
            "wall_force_association_coverage": self._association_coverage,
            "wall_force_association_explicit_force_coverage": self._association_explicit_force_coverage,
            "wall_force_association_ordering_stable": bool(self._association_ordering_stable),
            "wall_force_active_constraint_step": bool(self._active_constraint_step),
            "wall_force_gap_active_projected_count": int(self._gap_active_projected_count),
            "wall_force_gap_explicit_mapped_count": int(self._gap_explicit_mapped_count),
            "wall_force_gap_unmapped_count": int(self._gap_unmapped_count),
            "wall_force_gap_class_counts": dict(self._gap_class_counts),
            "wall_force_gap_dominant_class": self._gap_dominant_class,
            "wall_force_gap_contact_mode": self._gap_contact_mode,
            "wall_force_gap_row_bridge_mode": self._gap_row_bridge_mode,
            "wall_force_gap_row_bridge_offset": int(self._gap_row_bridge_offset),
            "wall_force_gap_row_bridge_applied": bool(self._gap_row_bridge_applied),
            "wall_force_gap_row_bridge_deterministic": bool(
                self._gap_row_bridge_deterministic
            ),
            "wall_force_gap_row_bridge_pair_count": int(
                self._gap_row_bridge_pair_count
            ),
            "wall_force_gap_row_bridge_dof_hits": int(self._gap_row_bridge_dof_hits),
            "wall_contact_count": int(self._contact_count),
            "wall_contact_detected": bool(self._contact_detected),
            "wall_force_channel": self._force_source,
            "wall_force_norm_sum": self._force_norm_sum,
            "wall_native_contact_export_available": bool(self._native_contact_export_available),
            "wall_native_contact_export_source": self._native_contact_export_source,
            "wall_native_contact_export_status": self._native_contact_export_status,
            "wall_native_contact_export_explicit_coverage": self._native_contact_export_explicit_coverage,
            "wall_segment_count": int(self._wall_segment_count),
            "wall_active_segment_ids": self._wall_active_segment_ids,
            "wall_active_segment_force_vectors": self._wall_active_segment_force_vectors,
            "wall_contact_force_vectors": self._contact_force_vectors,
            "wall_contact_segment_indices": self._contact_segment_indices,
            "wall_segment_force_vectors": self._segment_force_vectors,
            "wall_total_force_vector": self._total_force_vector,
            "wall_total_force_norm": self._total_force_norm,
            "wall_lcp_sum_abs": self._lcp_sum_abs,
            "wall_lcp_max_abs": self._lcp_max_abs,
            "wall_lcp_active_count": int(self._lcp_active_count),
            "wall_wire_force_norm": self._wire_force_norm,
            "wall_collision_force_norm": self._collis_force_norm,
            "wall_wire_force_vectors": self._wire_force_vectors,
            "wall_collision_force_vectors": self._collision_force_vectors,
            "wall_wire_force_vectors_source": self._wire_force_vectors_source,
            "wall_collision_force_vectors_source": self._collision_force_vectors_source,
            "wall_field_force_vector": self._wall_field_force_vector,
            "wall_field_force_norm": self._wall_field_force_norm,
            "wall_tip_force_vector": self._tip_force_vector,
            "wall_tip_force_norm": self._tip_force_norm,
            "wall_tip_force_sample_index": int(self._tip_force_sample_index),
            "wall_tip_force_source": self._tip_force_source,
            "wall_total_force_vector_N": self._total_force_vector,
            "wall_total_force_norm_N": self._total_force_norm,
            "wall_active_segment_force_vectors_N": self._wall_active_segment_force_vectors,
            "wall_segment_force_vectors_N": self._segment_force_vectors,
            "wall_tip_force_vector_N": self._tip_force_vector,
            "wall_tip_force_norm_N": self._tip_force_norm,
            "force_units": units,
            "unit_converted_si": bool(self._unit_converted_si),
            "force_unit_scale_to_newton": self._unit_scale_to_newton,
        }

    def __call__(self) -> Dict[str, Any]:
        return self.info
