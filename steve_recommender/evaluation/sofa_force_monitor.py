from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_monitor_plugin_path(plugin_override: Optional[str]) -> Optional[Path]:
    """Resolve monitor plugin location with deterministic precedence."""

    candidates = []
    if plugin_override:
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


@dataclass(frozen=True)
class ForceRuntimeStatus:
    configured: bool
    source: str
    error: str


class SofaForceMonitorRuntime:
    """Runtime adapter to enable wall-force extraction without patching stEVE."""

    def __init__(
        self,
        *,
        mode: str,
        contact_epsilon: float,
        plugin_path: Optional[str],
    ) -> None:
        self._mode = str(mode)
        self._contact_epsilon = float(max(contact_epsilon, 0.0))
        self._plugin_override = plugin_path
        self._enable_contact_listeners = (
            str(os.environ.get("STEVE_FORCE_ENABLE_CONTACT_LISTENER", "0")).strip().lower()
            in {"1", "true", "yes", "on"}
        )

        self._last_root_id: Optional[int] = None
        self._status = ForceRuntimeStatus(False, "uninitialized", "")

    @staticmethod
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

    @staticmethod
    def _import_plugin(sim: Any, plugin_path: Path) -> tuple[bool, str]:
        errors = []

        try:
            import SofaRuntime  # type: ignore

            if hasattr(SofaRuntime, "importPlugin"):
                SofaRuntime.importPlugin(str(plugin_path))
                return True, ""
        except Exception as exc:  # pragma: no cover - depends on runtime env
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
            return ForceRuntimeStatus(
                False,
                "intrusive_lcp_missing_lcp",
                "LCP solver object not found on simulation root",
            )

        ok_build = self._set_data_field(lcp, "build_lcp", True)
        ok_forces = self._set_data_field(lcp, "computeConstraintForces", True)
        if not (ok_build or ok_forces):
            return ForceRuntimeStatus(
                False,
                "intrusive_lcp_unconfigured",
                "could not set LCP build_lcp/computeConstraintForces",
            )

        return ForceRuntimeStatus(True, "intrusive_lcp", "")

    def _ensure_passive_monitor(self, sim: Any) -> ForceRuntimeStatus:
        # Request constraint-force export from the existing solver while keeping
        # build_lcp untouched (passive mode). This improves availability of
        # collision force vectors without switching to intrusive_lcp mode.
        lcp = getattr(getattr(sim, "root", None), "LCP", None)
        if lcp is not None:
            self._set_data_field(lcp, "computeConstraintForces", True)

        monitor = getattr(sim.root, "wire_wall_force_monitor", None)
        if monitor is not None:
            self._ensure_native_contact_export(sim)
            if not self._enable_contact_listeners:
                return ForceRuntimeStatus(
                    True,
                    "passive_monitor",
                    "",
                )
            listener_ok, listener_info = self._ensure_contact_listeners(sim)
            if not listener_ok:
                return ForceRuntimeStatus(
                    True,
                    "passive_monitor",
                    "",
                )
            return ForceRuntimeStatus(True, "passive_monitor", "")

        plugin_path = resolve_monitor_plugin_path(self._plugin_override)
        if plugin_path is None:
            return ForceRuntimeStatus(
                False,
                "passive_plugin_missing",
                (
                    "WireWallForceMonitor plugin not found "
                    "(set force_extraction.plugin_path or STEVE_WALL_FORCE_MONITOR_PLUGIN)"
                ),
            )

        ok_plugin, plugin_error = self._import_plugin(sim, plugin_path)
        if not ok_plugin:
            return ForceRuntimeStatus(
                False,
                "passive_plugin_load_failed",
                plugin_error,
            )

        try:
            sim.root.addObject(
                "WireWallForceMonitor",
                name="wire_wall_force_monitor",
                collisionMechanicalObject="@InstrumentCombined/CollisionModel/CollisionDOFs",
                wireMechanicalObject="@InstrumentCombined/DOFs",
                vesselMechanicalObject="@vesselTree/dofs",
                vesselTopology="@vesselTree/MeshTopology",
                contactEpsilon=self._contact_epsilon,
            )
        except Exception as exc:
            return ForceRuntimeStatus(
                False,
                "passive_monitor_attach_failed",
                str(exc),
            )

        monitor = getattr(sim.root, "wire_wall_force_monitor", None)
        if monitor is None:
            return ForceRuntimeStatus(
                False,
                "passive_monitor_missing_after_attach",
                "monitor object did not appear on root after addObject",
            )
        self._ensure_native_contact_export(sim)
        if not self._enable_contact_listeners:
            return ForceRuntimeStatus(
                True,
                "passive_monitor",
                "",
            )
        listener_ok, listener_info = self._ensure_contact_listeners(sim)
        if not listener_ok:
            return ForceRuntimeStatus(True, "passive_monitor", "")
        return ForceRuntimeStatus(True, "passive_monitor", "")

    def _ensure_native_contact_export(self, sim: Any) -> None:
        export = getattr(sim.root, "wire_wall_contact_export", None)
        if export is not None:
            return
        try:
            sim.root.addObject(
                "WireWallContactExport",
                name="wire_wall_contact_export",
                vesselNode="@vesselTree",
                vesselTopology="@vesselTree/MeshTopology",
                collisionMechanicalObject="@InstrumentCombined/CollisionModel/CollisionDOFs",
                contactEpsilon=self._contact_epsilon,
            )
        except Exception:
            # Keep force monitor operational even if explicit export is unavailable.
            return

    @staticmethod
    def _ensure_required_plugin(sim: Any, plugin_name: str) -> None:
        try:
            sim.root.addObject("RequiredPlugin", pluginName=plugin_name)
        except Exception:
            # Plugin may already be loaded or unavailable as component package name.
            pass

    def _ensure_contact_listeners(self, sim: Any) -> tuple[bool, str]:
        self._ensure_required_plugin(sim, "Sofa.Component.Collision.Response.Contact")

        # Point-vs-triangle listener is the most stable option in this scene.
        # A line-vs-triangle ContactListener can segfault on some setups when the
        # collision pipeline mutates contact outputs across frames.
        listeners = [
            (
                "wire_wall_contact_point",
                "@InstrumentCombined/CollisionModel/PointCollisionModel",
                "@vesselTree/TriangleCollisionModel",
            ),
        ]

        created = []
        errors = []
        existing = 0
        for name, model_a, model_b in listeners:
            listener_obj = getattr(sim.root, name, None)
            if listener_obj is not None:
                # ContactListener only receives collision events when "listening" is enabled.
                # In some SOFA builds this defaults to false for scripted addObject usage.
                self._set_data_field(listener_obj, "listening", True)
                existing += 1
                continue
            try:
                sim.root.addObject(
                    "ContactListener",
                    name=name,
                    collisionModel1=model_a,
                    collisionModel2=model_b,
                )
                listener_obj = getattr(sim.root, name, None)
                if listener_obj is not None:
                    self._set_data_field(listener_obj, "listening", True)
                created.append(name)
                continue
            except Exception as exc_ab:
                try:
                    sim.root.addObject(
                        "ContactListener",
                        name=name,
                        collisionModel1=model_b,
                        collisionModel2=model_a,
                    )
                    listener_obj = getattr(sim.root, name, None)
                    if listener_obj is not None:
                        self._set_data_field(listener_obj, "listening", True)
                    created.append(name)
                    continue
                except Exception as exc_ba:
                    errors.append(f"{name}:{exc_ab} | swapped:{exc_ba}")

        ready = sum(
            1 for name, _, _ in listeners if getattr(sim.root, name, None) is not None
        )
        if ready <= 0:
            return False, "; ".join(errors) if errors else "no contact listener attached"
        info = f"listeners_ready={ready} existing={existing} created={','.join(created)}"
        if errors:
            info += "; errors=" + "; ".join(errors)
        return True, info

    def ensure(self, sim: Any) -> ForceRuntimeStatus:
        """Ensure runtime force-extraction wiring on the current SOFA root."""

        root = getattr(sim, "root", None)
        if root is None:
            self._status = ForceRuntimeStatus(
                False,
                "simulation_root_missing",
                "simulation root is not available yet",
            )
            return self._status

        current_root_id = id(root)
        if self._last_root_id == current_root_id and self._status.source != "uninitialized":
            return self._status

        if self._mode == "intrusive_lcp":
            status = self._ensure_intrusive_lcp(sim)
        elif self._mode in {"passive", "constraint_projected_si_validated"}:
            status = self._ensure_passive_monitor(sim)
        else:
            status = ForceRuntimeStatus(
                False,
                "unsupported_mode",
                f"unsupported force mode: {self._mode}",
            )

        self._last_root_id = current_root_id
        self._status = status
        return status
