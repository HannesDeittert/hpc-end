"""Force telemetry adapter for train_v2 rewards."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from steve_recommender.eval_v2.force_telemetry import (
    EvalV2ForceTelemetryCollector,
    ForceRuntimeStatus,
)
from steve_recommender.eval_v2.models import ForceTelemetrySpec, ForceUnits
from third_party.stEVE.eve.intervention.simulation.simulationmp import SimulationMP

from ..config import RewardSpec

_DEFAULT_UNITS = ForceUnits(length_unit="mm", mass_unit="kg", time_unit="s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForceRewardSample:
    """One force sample exposed to reward components."""

    wire_force_normal_instant_N: float
    wire_force_normal_trial_max_N: float
    tip_force_normal_instant_N: float
    tip_force_normal_trial_max_N: float


class ForceRuntime:
    """Thin adapter around the eval_v2 force collector for training rewards."""

    def __init__(self, *, reward_spec: RewardSpec, action_dt_s: float) -> None:
        needs_units = (
            reward_spec.force_telemetry_mode == "constraint_projected_si_validated"
        )
        spec = ForceTelemetrySpec(
            mode=reward_spec.force_telemetry_mode,
            units=_DEFAULT_UNITS if needs_units else None,
            write_full_trace=False,
            write_diagnostics=False,
        )
        self._collector = EvalV2ForceTelemetryCollector(
            spec=spec,
            action_dt_s=action_dt_s,
        )
        self._last_status = ForceRuntimeStatus(False, "uninitialized", "")
        self._logged_status_signature: tuple[str, str, bool] | None = None

    def ensure_runtime(self, *, intervention: Any) -> ForceRuntimeStatus:
        self._maybe_bind_anatomy_mesh_path(intervention=intervention)
        simulation = getattr(intervention, "simulation", None)
        if isinstance(simulation, SimulationMP):
            status = ForceRuntimeStatus(
                False,
                "simulation_mp_wrapper",
                "force telemetry requires a direct Sofa simulation, not SimulationMP",
            )
            self._log_status_once(status, level="warning")
            self._last_status = status
            return status

        root = getattr(simulation, "root", None)
        if root is None:
            status = ForceRuntimeStatus(
                False,
                "simulation_root_not_ready",
                "simulation root not created yet; retry after env.reset",
            )
            self._log_status_once(status, level="info")
            self._last_status = status
            return status

        status = self._collector.ensure_runtime(intervention=intervention)
        if status.configured:
            self._log_status_once(status, level="info")
        else:
            self._log_status_once(status, level="warning")
        self._last_status = status
        return status

    def _maybe_bind_anatomy_mesh_path(self, *, intervention: Any) -> None:
        current_mesh_path = getattr(self._collector, "_anatomy_mesh_path", None)
        if current_mesh_path is not None:
            return
        vessel_tree = getattr(intervention, "vessel_tree", None)
        candidate_paths = (
            getattr(vessel_tree, "mesh_path", None),
            getattr(vessel_tree, "visu_mesh_path", None),
        )
        for raw_path in candidate_paths:
            if raw_path is None:
                continue
            path = Path(str(raw_path))
            setattr(self._collector, "_anatomy_mesh_path", path)
            logger.info(
                "[train_v2 force telemetry] using anatomy mesh path for normals: %s",
                path,
            )
            return

    def sample_step(self, *, intervention: Any, step_index: int) -> ForceRewardSample:
        status = self.ensure_runtime(intervention=intervention)
        if not status.configured:
            raise RuntimeError(
                "train_v2 force telemetry is unavailable during a live step: "
                f"{status.source} ({status.error})"
            )
        self._collector.capture_step(intervention=intervention, step_index=step_index)
        summary = self._collector.build_summary()
        return ForceRewardSample(
            wire_force_normal_instant_N=float(
                summary.wire_force_normal_instant_N or 0.0
            ),
            wire_force_normal_trial_max_N=float(
                summary.wire_force_normal_trial_max_N or 0.0
            ),
            tip_force_normal_instant_N=float(
                summary.tip_force_normal_instant_N or 0.0
            ),
            tip_force_normal_trial_max_N=float(
                summary.tip_force_normal_trial_max_N or 0.0
            ),
        )

    def _log_status_once(self, status: ForceRuntimeStatus, *, level: str) -> None:
        signature = (status.source, status.error, bool(status.configured))
        if signature == self._logged_status_signature:
            return
        self._logged_status_signature = signature
        message = (
            "[train_v2 force telemetry] %s: %s"
            % (status.source, status.error or "configured")
        )
        if level == "warning":
            logger.warning(message)
        else:
            logger.info(message)
