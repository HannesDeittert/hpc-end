from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .models import ForceTelemetrySpec, ForceTelemetrySummary


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


@dataclass(frozen=True)
class ForceRuntimeStatus:
    configured: bool
    source: str
    error: str = ""


class EvalV2ForceTelemetryCollector:
    """Collect force telemetry directly from SOFA runtime objects for eval_v2."""

    def __init__(self, *, spec: ForceTelemetrySpec, action_dt_s: float) -> None:
        self._spec = spec
        self._action_dt_s = float(max(action_dt_s, 1e-9))
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

    def _ensure_passive_monitor(self, sim: Any) -> ForceRuntimeStatus:
        lcp = getattr(getattr(sim, "root", None), "LCP", None)
        if lcp is not None:
            _set_data_field(lcp, "computeConstraintForces", True)

        monitor = getattr(sim.root, "wire_wall_force_monitor", None)
        if monitor is not None:
            return ForceRuntimeStatus(True, "passive_monitor", "")

        plugin_path = resolve_monitor_plugin_path(self._spec.plugin_path)
        if plugin_path is None:
            return ForceRuntimeStatus(False, "passive_plugin_missing", "plugin not found")
        ok, error = self._import_plugin(sim, plugin_path)
        if not ok:
            return ForceRuntimeStatus(False, "passive_plugin_load_failed", error)

        try:
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
            return ForceRuntimeStatus(False, "passive_monitor_attach_failed", str(exc))

        monitor = getattr(sim.root, "wire_wall_force_monitor", None)
        if monitor is None:
            return ForceRuntimeStatus(False, "passive_monitor_missing_after_attach", "")
        return ForceRuntimeStatus(True, "passive_monitor", "")

    def ensure_runtime(self, *, intervention: Any) -> ForceRuntimeStatus:
        simulation = getattr(intervention, "simulation", None)
        root = getattr(simulation, "root", None)
        if root is None:
            self._status = ForceRuntimeStatus(False, "simulation_root_missing", "")
            return self._status

        current_root_id = id(root)
        if self._last_root_id == current_root_id and self._status.source != "uninitialized":
            return self._status

        if self._spec.mode == "intrusive_lcp":
            self._status = self._ensure_intrusive_lcp(simulation)
        elif self._spec.mode in {"passive", "constraint_projected_si_validated"}:
            self._status = self._ensure_passive_monitor(simulation)
        else:
            self._status = ForceRuntimeStatus(False, "unsupported_mode", self._spec.mode)

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

    def capture_step(self, *, intervention: Any, step_index: int) -> None:
        simulation = getattr(intervention, "simulation", None)
        root = getattr(simulation, "root", None)
        if root is None:
            return

        self._sample_lcp(root)
        monitor = getattr(root, "wire_wall_force_monitor", None)
        if monitor is None:
            return

        monitor_available = bool(_read_data_field(monitor, "available", False))
        self._available_any = self._available_any or monitor_available
        if monitor_available:
            self._channel = "wire_wall_force_monitor"
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
        use_lcp_fallback = (
            bool(self._lcp_max_abs_samples)
            and (not self._total_force_norm_samples or not self._monitor_nonzero_detected)
        )
        effective_force_samples = (
            self._lcp_force_like_max_samples if use_lcp_fallback else self._total_force_norm_samples
        )
        available_for_score = bool(self._available_any and effective_force_samples)
        if self._status.configured and available_for_score:
            validation_status = "ok"
        elif self._spec.required and not available_for_score:
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
        total_force_norm_max = (
            float(np.max(effective_force_samples)) if effective_force_samples else None
        )
        total_force_norm_mean = (
            float(np.mean(effective_force_samples)) if effective_force_samples else None
        )

        quality_tier = "unavailable"
        if available_for_score:
            quality_tier = "degraded"
        if "nearest_triangle_centroid" in self._last_status_text:
            self._association_method = "force_points_nearest_triangle"
            self._association_coverage = 0.0

        source = self._source or self._status.source
        channel = self._channel if available_for_score else "none"
        if use_lcp_fallback:
            source = f"{source}:fallback_lcp_dt"
            channel = "lcp.constraintForces/dt"

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
        )


__all__ = [
    "EvalV2ForceTelemetryCollector",
    "ForceRuntimeStatus",
    "resolve_monitor_plugin_path",
]
