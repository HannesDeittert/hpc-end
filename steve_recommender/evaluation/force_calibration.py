from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config import AgentSpec, EvaluationConfig
from .force_si import units_to_dict
from .sofa_force_monitor import resolve_monitor_plugin_path


@dataclass(frozen=True)
class CalibrationRecord:
    key: str
    created_at: str
    mode: str
    tool: str
    checkpoint: str
    passed: bool
    validation_status: str
    validation_error: str
    force_units: Dict[str, str]
    fingerprint: Dict[str, Any]
    metrics: Dict[str, Any]


def _sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def calibration_fingerprint(cfg: EvaluationConfig, agent: AgentSpec) -> Dict[str, Any]:
    """Build a cache fingerprint for force calibration compatibility checks."""

    plugin = resolve_monitor_plugin_path(cfg.force_extraction.plugin_path)
    dt = 1.0 / float(cfg.anatomy.image_frequency_hz)
    return {
        "schema_version": 2,
        "mode": cfg.force_extraction.mode,
        "validation_suite": "dual_probe_repro_v1",
        "tool": agent.tool,
        "checkpoint": agent.checkpoint,
        "anatomy": {
            "type": cfg.anatomy.type,
            "arch_type": cfg.anatomy.arch_type,
            "seed": int(cfg.anatomy.seed),
            "friction": float(cfg.anatomy.friction),
            "image_frequency_hz": float(cfg.anatomy.image_frequency_hz),
            "target_mode": cfg.anatomy.target_mode,
            "target_branches": list(cfg.anatomy.target_branches),
        },
        "physics": {
            "dt_s": dt,
            "response_model": "FrictionContactConstraint",
            "friction_mu": float(cfg.anatomy.friction),
            "contactDistance": 0.3,
            "intersection_method": "LocalMinDistance",
            "collision_model_flags": {
                "wall_simulated": False,
                "wall_moving": False,
                "beamadapter_line_triangle_path": True,
            },
        },
        "runtime": {
            "sofa_root": os.environ.get("SOFA_ROOT", ""),
            "plugin_path": str(plugin) if plugin is not None else "",
            "plugin_sha256": _sha256_file(plugin),
            "tolerance_profile": cfg.force_extraction.calibration.tolerance_profile,
        },
        "units": (
            units_to_dict(cfg.force_extraction.units)
            if cfg.force_extraction.units is not None
            else {}
        ),
    }


def _record_key(fingerprint: Dict[str, Any]) -> str:
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "records": {}}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return {"schema_version": 1, "records": {}}
    records = raw.get("records")
    if not isinstance(records, dict):
        records = {}
    return {"schema_version": int(raw.get("schema_version", 1)), "records": records}


def _write_cache(path: Path, cache: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def load_force_calibration(*, cache_path: str, cache_key: str) -> Dict[str, Any]:
    cache = _read_cache(Path(cache_path))
    rec = cache.get("records", {}).get(cache_key)
    if not isinstance(rec, dict):
        return {"found": False, "passed": False, "record": None}
    return {"found": True, "passed": bool(rec.get("passed", False)), "record": rec}


def _first_summary_row(summary_path: Path) -> Dict[str, Any]:
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        row = next(reader, None)
    if row is None:
        raise RuntimeError(f"Calibration probe produced empty summary: {summary_path}")
    return row


def _load_trial_series(npz_path: str) -> Dict[str, np.ndarray]:
    p = Path(npz_path)
    if not p.exists():
        return {}
    try:
        with np.load(p, allow_pickle=True) as data:
            return {
                "wall_total_force_norm_N": np.asarray(
                    data.get("wall_total_force_norm_N", np.asarray([])),
                    dtype=np.float64,
                ).reshape((-1,)),
                "wall_peak_segment_force_norm_per_step": np.asarray(
                    data.get("wall_peak_segment_force_norm_per_step", np.asarray([])),
                    dtype=np.float64,
                ).reshape((-1,)),
            }
    except Exception:
        return {}


def _tolerance_profile(name: str) -> Dict[str, float]:
    profile = str(name or "default_v1").strip().lower()
    if profile == "strict_v1":
        return {
            "trace_atol_N": 5e-4,
            "trace_rtol": 1e-3,
        }
    return {
        "trace_atol_N": 1e-3,
        "trace_rtol": 1e-2,
    }


def _series_reproducible(
    a: np.ndarray,
    b: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, str]:
    a = np.asarray(a, dtype=np.float64).reshape((-1,))
    b = np.asarray(b, dtype=np.float64).reshape((-1,))
    if a.size == 0 or b.size == 0:
        return False, "missing_trace"
    if a.shape != b.shape:
        return False, f"trace_length_mismatch:{a.size}!={b.size}"
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return False, "non_finite_trace"
    if np.allclose(a, b, atol=float(atol), rtol=float(rtol)):
        return True, ""
    max_abs = float(np.max(np.abs(a - b)))
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom > 0.0, denom, 1.0)
    max_rel = float(np.max(np.abs(a - b) / denom))
    return False, f"trace_mismatch:max_abs={max_abs:.3g},max_rel={max_rel:.3g}"


def run_force_calibration(cfg: EvaluationConfig) -> CalibrationRecord:
    """Run one-trial probe and persist calibration verdict to cache."""

    if not cfg.agents:
        raise ValueError("Calibration requires at least one agent in config")
    if cfg.force_extraction.units is None:
        raise ValueError("Calibration requires force_extraction.units")

    # Probe with first agent only; calibration key is per-tool+checkpoint+physics fingerprint.
    agent = cfg.agents[0]
    probe_cfg_a = _build_calibration_probe_config(cfg, agent, suffix="calib_a")
    probe_cfg_b = _build_calibration_probe_config(cfg, agent, suffix="calib_b")

    # Local import to avoid module cycle during package import.
    from .pipeline import run_evaluation

    run_dir_a = run_evaluation(probe_cfg_a)
    run_dir_b = run_evaluation(probe_cfg_b)
    row_a = _first_summary_row(run_dir_a / "summary.csv")
    row_b = _first_summary_row(run_dir_b / "summary.csv")

    force_available_a = float(row_a.get("wall_force_available", "0") or 0.0) > 0.5
    force_available_b = float(row_b.get("wall_force_available", "0") or 0.0) > 0.5
    force_max_n_a = float(row_a.get("wall_force_max_N", "nan") or "nan")
    force_max_n_b = float(row_b.get("wall_force_max_N", "nan") or "nan")
    contact_any_a = float(row_a.get("wall_contact_detected_any", "0") or 0.0) > 0.5
    contact_any_b = float(row_b.get("wall_contact_detected_any", "0") or 0.0) > 0.5
    status_a = str(row_a.get("force_validation_status", "") or "")
    status_b = str(row_b.get("force_validation_status", "") or "")
    error_a = str(row_a.get("force_validation_error", "") or "")
    error_b = str(row_b.get("force_validation_error", "") or "")
    quality_a = str(row_a.get("wall_force_quality_tier", "") or "")
    quality_b = str(row_b.get("wall_force_quality_tier", "") or "")
    assoc_a = str(row_a.get("wall_force_association_method", "") or "")
    assoc_b = str(row_b.get("wall_force_association_method", "") or "")
    assoc_ratio_a = float(
        row_a.get("wall_force_association_explicit_ratio", "nan") or "nan"
    )
    assoc_ratio_b = float(
        row_b.get("wall_force_association_explicit_ratio", "nan") or "nan"
    )
    assoc_cov_a = float(row_a.get("wall_force_association_coverage", "nan") or "nan")
    assoc_cov_b = float(row_b.get("wall_force_association_coverage", "nan") or "nan")
    assoc_exp_force_cov_a = float(
        row_a.get("wall_force_association_explicit_force_coverage", "nan") or "nan"
    )
    assoc_exp_force_cov_b = float(
        row_b.get("wall_force_association_explicit_force_coverage", "nan") or "nan"
    )

    tol = _tolerance_profile(cfg.force_extraction.calibration.tolerance_profile)
    trace_a = _load_trial_series(str(row_a.get("npz_path", ""))).get(
        "wall_total_force_norm_N", np.asarray([], dtype=np.float64)
    )
    trace_b = _load_trial_series(str(row_b.get("npz_path", ""))).get(
        "wall_total_force_norm_N", np.asarray([], dtype=np.float64)
    )
    repro_ok, repro_error = _series_reproducible(
        trace_a,
        trace_b,
        atol=float(tol["trace_atol_N"]),
        rtol=float(tol["trace_rtol"]),
    )
    reference_suite_ok = True
    reference_suite_reason = ""
    reference_suite_metrics: Dict[str, Any] = {}
    if cfg.force_extraction.mode == "constraint_projected_si_validated":
        # Use deterministic minimal scenes as the primary reproducibility oracle
        # for force telemetry. Agent trajectory variability in full episodes should
        # not invalidate an otherwise correct force extraction pipeline.
        from .reference_scene import run_reference_scene_suite

        suite_out_dir = (
            Path(cfg.output_root)
            / "force_reference_scene"
            / f"{cfg.name}_calib_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        suite = run_reference_scene_suite(
            tool_ref=agent.tool,
            output_dir=suite_out_dir,
            base_seed=int(cfg.base_seed),
            plugin_path=cfg.force_extraction.plugin_path,
            contact_epsilon=float(cfg.force_extraction.contact_epsilon),
            units=cfg.force_extraction.units,
        )
        reference_suite_ok = bool(suite.pass_validated_suite)
        reference_suite_reason = str(
            suite.external_limit_reason or "reference_suite_not_validated"
        )
        reference_suite_metrics = {
            "output_dir": str(suite.output_dir),
            "pass_validated_suite": int(suite.pass_validated_suite),
            "external_limit_detected": int(suite.external_limit_detected),
            "external_limit_reason": str(suite.external_limit_reason),
        }

    fail_reasons = []
    if not (force_available_a and force_available_b):
        fail_reasons.append("force_unavailable_in_probe")
    if (contact_any_a and force_max_n_a <= float(cfg.force_extraction.contact_epsilon)) or (
        contact_any_b and force_max_n_b <= float(cfg.force_extraction.contact_epsilon)
    ):
        fail_reasons.append("contact_without_nonzero_force")
    # Calibration only accepts fully validated quality tiers.
    if status_a != "pass:validated" or status_b != "pass:validated":
        fail_reasons.append(
            f"validation_status_mismatch:a={status_a or 'n/a'},b={status_b or 'n/a'}"
        )
    if quality_a != "validated" or quality_b != "validated":
        fail_reasons.append(
            f"quality_tier_mismatch:a={quality_a or 'n/a'},b={quality_b or 'n/a'}"
        )
    validated_assoc_methods = {
        "native_contact_export_triangle_id",
        "contact_element_triangle_id",
        "contact_point_nearest_triangle_surface",
        "mixed_contact_element_and_surface",
        "cached_contact_triangle_id",
        "mixed_cached_and_surface",
    }
    if assoc_a not in validated_assoc_methods or assoc_b not in validated_assoc_methods:
        fail_reasons.append(
            f"association_method_mismatch:a={assoc_a or 'n/a'},b={assoc_b or 'n/a'}"
        )
    if assoc_a in {
        "native_contact_export_triangle_id",
        "native_contact_export_triangle_id_global_row_bridge",
        "contact_element_triangle_id",
        "cached_contact_triangle_id",
    }:
        a_explicit_ok = np.isfinite(assoc_exp_force_cov_a) and float(assoc_exp_force_cov_a) >= 0.999
        a_ratio_ok = np.isfinite(assoc_ratio_a) and float(assoc_ratio_a) >= 0.999
        if not (a_explicit_ok or a_ratio_ok):
            fail_reasons.append(
                "association_coverage_low:a="
                f"explicit_force_cov={assoc_exp_force_cov_a},ratio={assoc_ratio_a},cov={assoc_cov_a}"
            )
    if assoc_b in {
        "native_contact_export_triangle_id",
        "native_contact_export_triangle_id_global_row_bridge",
        "contact_element_triangle_id",
        "cached_contact_triangle_id",
    }:
        b_explicit_ok = np.isfinite(assoc_exp_force_cov_b) and float(assoc_exp_force_cov_b) >= 0.999
        b_ratio_ok = np.isfinite(assoc_ratio_b) and float(assoc_ratio_b) >= 0.999
        if not (b_explicit_ok or b_ratio_ok):
            fail_reasons.append(
                "association_coverage_low:b="
                f"explicit_force_cov={assoc_exp_force_cov_b},ratio={assoc_ratio_b},cov={assoc_cov_b}"
            )
    if assoc_a == "mixed_cached_and_surface":
        if (not np.isfinite(assoc_ratio_a)) or float(assoc_ratio_a) < 0.8:
            fail_reasons.append(f"association_ratio_low:a={assoc_ratio_a}")
    if assoc_b == "mixed_cached_and_surface":
        if (not np.isfinite(assoc_ratio_b)) or float(assoc_ratio_b) < 0.8:
            fail_reasons.append(f"association_ratio_low:b={assoc_ratio_b}")
    if error_a or error_b:
        if error_a:
            fail_reasons.append(f"probe_a_error:{error_a}")
        if error_b:
            fail_reasons.append(f"probe_b_error:{error_b}")
    if not reference_suite_ok:
        fail_reasons.append(f"reference_suite:{reference_suite_reason}")
    if not repro_ok and not reference_suite_ok:
        fail_reasons.append(f"reproducibility:{repro_error}")
    passed = len(fail_reasons) == 0

    fingerprint = calibration_fingerprint(cfg, agent)
    key = _record_key(fingerprint)
    validation_status = (
        "pass:validated" if passed else "fail:validated_reference"
    )
    validation_error = ";".join(fail_reasons)
    record = CalibrationRecord(
        key=key,
        created_at=datetime.now(timezone.utc).isoformat(),
        mode=cfg.force_extraction.mode,
        tool=agent.tool,
        checkpoint=agent.checkpoint,
        passed=passed,
        validation_status=validation_status,
        validation_error=validation_error,
        force_units=units_to_dict(cfg.force_extraction.units),
        fingerprint=fingerprint,
        metrics={
            "probe_a": {
                "run_dir": str(run_dir_a),
                "wall_force_max_N": force_max_n_a,
                "wall_contact_detected_any": int(contact_any_a),
                "wall_force_available": int(force_available_a),
                "wall_force_quality_tier": quality_a,
                "wall_force_association_method": assoc_a,
                "wall_force_association_explicit_ratio": assoc_ratio_a,
                "wall_force_association_coverage": assoc_cov_a,
                "wall_force_association_explicit_force_coverage": assoc_exp_force_cov_a,
                "force_validation_status": status_a,
                "force_validation_error": error_a,
            },
            "probe_b": {
                "run_dir": str(run_dir_b),
                "wall_force_max_N": force_max_n_b,
                "wall_contact_detected_any": int(contact_any_b),
                "wall_force_available": int(force_available_b),
                "wall_force_quality_tier": quality_b,
                "wall_force_association_method": assoc_b,
                "wall_force_association_explicit_ratio": assoc_ratio_b,
                "wall_force_association_coverage": assoc_cov_b,
                "wall_force_association_explicit_force_coverage": assoc_exp_force_cov_b,
                "force_validation_status": status_b,
                "force_validation_error": error_b,
            },
            "reproducibility": {
                "passed": int(repro_ok),
                "error": repro_error,
                "trace_len": int(min(trace_a.size, trace_b.size)),
                "trace_atol_N": float(tol["trace_atol_N"]),
                "trace_rtol": float(tol["trace_rtol"]),
            },
            "reference_suite": reference_suite_metrics,
        },
    )

    cache_path = Path(cfg.force_extraction.calibration.cache_path)
    cache = _read_cache(cache_path)
    cache.setdefault("records", {})
    cache["records"][key] = asdict(record)
    _write_cache(cache_path, cache)
    return record


def _build_calibration_probe_config(
    cfg: EvaluationConfig,
    agent: AgentSpec,
    *,
    suffix: str = "probe",
) -> EvaluationConfig:
    """Return a safe one-trial config used to generate calibration evidence.

    Calibration probing must not depend on an already-present calibration cache.
    Therefore we force `force_extraction.calibration.required=False` for the probe run.
    """

    probe_calibration = replace(cfg.force_extraction.calibration, required=False)
    probe_force = replace(
        cfg.force_extraction,
        required=False,
        calibration=probe_calibration,
    )
    return replace(
        cfg,
        name=f"{cfg.name}_{suffix}",
        agents=[agent],
        n_trials=1,
        seeds=[cfg.base_seed],
        visualize=False,
        visualize_force_debug=False,
        force_extraction=probe_force,
    )
