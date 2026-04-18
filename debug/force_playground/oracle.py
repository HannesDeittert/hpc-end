from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .config import ForcePlaygroundConfig


@dataclass
class OracleStepResult:
    step: int
    applicable: bool
    used_in_window: bool
    f_ref_n: float
    f_meas_n: float
    abs_error: float
    rel_error: float
    abs_tol: float
    rel_tol: float
    passed: Optional[bool]
    reason: str


class NormalForceBalanceOracle:
    """Physical plausibility oracle for normal-force balance on plane wall."""

    def __init__(self, cfg: ForcePlaygroundConfig, wall_reference_normal: np.ndarray) -> None:
        self._cfg = cfg
        self._oracle_cfg = cfg.oracle
        self._wall_n = np.asarray(wall_reference_normal, dtype=np.float32).reshape((3,))
        self._results: List[OracleStepResult] = []
        self._applicable = bool(
            cfg.oracle.enabled
            and cfg.oracle.oracle_type == "normal_force_balance"
            and cfg.scene == "plane_wall"
            and cfg.probe == "rigid_probe"
            and cfg.mode == "open_loop_force"
        )
        self._applicability_reason = ""
        if not self._applicable:
            self._applicability_reason = (
                "normal_force_balance_oracle_v1_requires "
                "scene=plane_wall, probe=rigid_probe, mode=open_loop_force"
            )

    @property
    def applicable(self) -> bool:
        return bool(self._applicable)

    @property
    def applicability_reason(self) -> str:
        return str(self._applicability_reason)

    def _expected_wall_reaction_n(self, commanded_force_vector_n: np.ndarray) -> float:
        # Commanded force acts on probe; wall reaction should oppose probe load.
        return float(-np.dot(commanded_force_vector_n, self._wall_n))

    def _step_result(self, step_record: Dict[str, Any]) -> OracleStepResult:
        step = int(step_record.get("step", 0))
        if not self._applicable:
            return OracleStepResult(
                step=step,
                applicable=False,
                used_in_window=False,
                f_ref_n=float("nan"),
                f_meas_n=float("nan"),
                abs_error=float("nan"),
                rel_error=float("nan"),
                abs_tol=float(self._oracle_cfg.abs_tol_n),
                rel_tol=float(self._oracle_cfg.rel_tol),
                passed=None,
                reason=self._applicability_reason or "oracle_disabled",
            )

        cmd_vec = np.asarray(step_record.get("commanded_force_vector_n", [0.0, 0.0, 0.0]), dtype=np.float32)
        total_force = np.asarray(step_record.get("total_force_vector", [0.0, 0.0, 0.0]), dtype=np.float32)

        f_ref_n = self._expected_wall_reaction_n(cmd_vec)
        # Collected total wall force vector is probe-on-wall.
        # Convert to wall-reaction-on-probe for direct comparison.
        f_meas_n = float(-np.dot(total_force, self._wall_n))
        abs_error = float(abs(f_meas_n - f_ref_n))

        near_zero = abs(f_ref_n) < float(self._oracle_cfg.near_zero_ref_n)
        if near_zero:
            rel_error = float("nan")
            limit = float(self._oracle_cfg.abs_tol_n)
        else:
            rel_error = float(abs_error / abs(f_ref_n))
            limit = max(float(self._oracle_cfg.abs_tol_n), float(self._oracle_cfg.rel_tol) * abs(f_ref_n))

        if step <= int(self._oracle_cfg.warmup_steps):
            return OracleStepResult(
                step=step,
                applicable=True,
                used_in_window=False,
                f_ref_n=float(f_ref_n),
                f_meas_n=float(f_meas_n),
                abs_error=float(abs_error),
                rel_error=float(rel_error),
                abs_tol=float(self._oracle_cfg.abs_tol_n),
                rel_tol=float(self._oracle_cfg.rel_tol),
                passed=None,
                reason="warmup",
            )

        wall_contact = bool(step_record.get("wall_contact_detected", False))
        active_rows = int(step_record.get("lambda_active_rows_count", 0))
        total_force_norm = float(abs(step_record.get("total_force_norm", 0.0)))
        if (not wall_contact) or active_rows <= 0 or total_force_norm <= float(self._cfg.contact_epsilon):
            return OracleStepResult(
                step=step,
                applicable=True,
                used_in_window=False,
                f_ref_n=float(f_ref_n),
                f_meas_n=float(f_meas_n),
                abs_error=float(abs_error),
                rel_error=float(rel_error),
                abs_tol=float(self._oracle_cfg.abs_tol_n),
                rel_tol=float(self._oracle_cfg.rel_tol),
                passed=None,
                reason="no_contact",
            )

        return OracleStepResult(
            step=step,
            applicable=True,
            used_in_window=False,
            f_ref_n=float(f_ref_n),
            f_meas_n=float(f_meas_n),
            abs_error=float(abs_error),
            rel_error=float(rel_error),
            abs_tol=float(self._oracle_cfg.abs_tol_n),
            rel_tol=float(self._oracle_cfg.rel_tol),
            passed=bool(abs_error <= limit),
            reason="ok" if abs_error <= limit else "outside_tolerance",
        )

    def evaluate_step(self, step_record: Dict[str, Any]) -> Dict[str, Any]:
        result = self._step_result(step_record)
        self._results.append(result)
        step_record["oracle_f_ref_n"] = float(result.f_ref_n)
        step_record["oracle_f_meas_n"] = float(result.f_meas_n)
        step_record["oracle_abs_error"] = float(result.abs_error)
        step_record["oracle_rel_error"] = float(result.rel_error)
        step_record["oracle_abs_tol"] = float(result.abs_tol)
        step_record["oracle_rel_tol"] = float(result.rel_tol)
        step_record["oracle_physical_pass"] = result.passed
        step_record["oracle_reason"] = result.reason
        return step_record

    def finalize(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "oracle_type": self._oracle_cfg.oracle_type,
            "applicable": bool(self._applicable),
            "applicability_reason": self._applicability_reason,
            "rel_tol": float(self._oracle_cfg.rel_tol),
            "abs_tol_n": float(self._oracle_cfg.abs_tol_n),
            "near_zero_ref_n": float(self._oracle_cfg.near_zero_ref_n),
            "warmup_steps": int(self._oracle_cfg.warmup_steps),
            "window_steps": int(self._oracle_cfg.window_steps),
            "window": {},
            "passed": None,
            "results": [asdict(r) for r in self._results],
        }

        if not self._applicable:
            return out

        candidates = [
            r
            for r in self._results
            if r.applicable and r.passed is not None and str(r.reason) in {"ok", "outside_tolerance"}
        ]
        if not candidates:
            out["passed"] = False
            out["window"] = {
                "count": 0,
                "passes": 0,
                "fails": 0,
                "reason": "no_oracle_candidate_steps",
            }
            return out

        w = int(max(1, self._oracle_cfg.window_steps))
        window = candidates[-w:]
        for r in window:
            r.used_in_window = True

        pass_count = int(sum(1 for r in window if bool(r.passed)))
        fail_count = int(len(window) - pass_count)
        passed = bool(fail_count == 0)

        out["passed"] = passed
        out["window"] = {
            "count": int(len(window)),
            "passes": int(pass_count),
            "fails": int(fail_count),
            "first_step": int(window[0].step),
            "last_step": int(window[-1].step),
            "max_abs_error": float(max(r.abs_error for r in window)),
            "max_rel_error": float(
                np.nanmax(np.asarray([r.rel_error for r in window], dtype=np.float64))
            ),
            "mean_abs_error": float(np.mean(np.asarray([r.abs_error for r in window], dtype=np.float64))),
        }
        out["results"] = [asdict(r) for r in self._results]
        return out
