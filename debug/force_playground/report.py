from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .config import ForcePlaygroundConfig


def _max_of(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, float("nan"))) for r in rows]
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmax(arr))


def _mean_of(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, float("nan"))) for r in rows]
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def build_summary_markdown(
    cfg: ForcePlaygroundConfig,
    step_records: List[Dict[str, Any]],
    oracle_report: Dict[str, Any],
) -> str:
    n_steps = int(len(step_records))
    q_tiers = sorted({str(r.get("quality_tier", "")) for r in step_records})

    max_norm_sum_vector = _max_of(step_records, "norm_sum_vector")
    max_sum_norm = _max_of(step_records, "sum_norm")
    max_peak_triangle = _max_of(step_records, "peak_triangle_force")
    max_fn = _max_of(step_records, "sum_abs_fn")
    max_ft = _max_of(step_records, "sum_abs_ft")
    max_lambda = _max_of(step_records, "lambda_abs_sum")
    max_lambda_dt = _max_of(step_records, "lambda_dt_abs_sum")

    mean_sum_gap = _mean_of(step_records, "sum_force_gap_norm")
    mean_decomp_gap = _mean_of(step_records, "decomposition_gap_norm")

    n_si = int(sum(1 for r in step_records if bool(r.get("si_converted", False))))
    n_explicit = int(sum(1 for r in step_records if bool(r.get("explicit_association", False))))
    n_validated = int(sum(1 for r in step_records if bool(r.get("internal_validated", False))))
    n_oracle_true = int(sum(1 for r in step_records if r.get("oracle_physical_pass", None) is True))
    n_oracle_false = int(sum(1 for r in step_records if r.get("oracle_physical_pass", None) is False))

    lines = [
        "# Force Playground Summary",
        "",
        "## Run",
        f"- scene: `{cfg.scene}`",
        f"- probe: `{cfg.probe}`",
        f"- mode: `{cfg.mode}`",
        f"- tool_ref: `{cfg.tool_ref}`",
        f"- steps_recorded: `{n_steps}`",
        "",
        "## Required Metrics",
        f"- max(norm(sum(v_i))): `{max_norm_sum_vector:.8g}`",
        f"- max(sum(norm(v_i))): `{max_sum_norm:.8g}`",
        f"- max(peak_triangle_force): `{max_peak_triangle:.8g}`",
        f"- max(sum|F_n|): `{max_fn:.8g}`",
        f"- max(sum|F_t|): `{max_ft:.8g}`",
        f"- max(sum|lambda|): `{max_lambda:.8g}`",
        f"- max(sum|lambda/dt|): `{max_lambda_dt:.8g}`",
        "",
        "## Consistency",
        f"- mean(||sum(triangle_forces)-total_force_vector||): `{mean_sum_gap:.8g}`",
        f"- mean(||(F_n+F_t)-F_total||): `{mean_decomp_gap:.8g}`",
        f"- quality_tiers_seen: `{', '.join(q_tiers)}`",
        "",
        "## Validation Stages",
        "- `si_converted`: telemetry converted using explicit unit metadata",
        "- `explicit_association`: explicit native wall-triangle mapping used with full coverage",
        "- `internal_validated`: collector quality gate (`quality_tier=validated`) passed",
        "- `oracle_physical_pass`: physical plausibility check (`normal_force_balance`)",
        f"- counts: si_converted={n_si}, explicit_association={n_explicit}, internal_validated={n_validated}, oracle_pass={n_oracle_true}, oracle_fail={n_oracle_false}",
        "",
        "## Oracle",
        f"- applicable: `{oracle_report.get('applicable', False)}`",
        f"- passed: `{oracle_report.get('passed', None)}`",
        f"- window: `{oracle_report.get('window', {})}`",
    ]

    return "\n".join(lines) + "\n"
