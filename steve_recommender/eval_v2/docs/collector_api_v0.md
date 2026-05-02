# Collector API Snapshot v0

This document freezes the current public surface of `EvalV2ForceTelemetryCollector`
before the force-telemetry TDD work adds more behavior.

It is intentionally narrow: the goal is to record what the collector exposes today,
not to define the future design.

## Class

`EvalV2ForceTelemetryCollector`

Location:

- [steve_recommender/eval_v2/force_telemetry.py](../force_telemetry.py)

### Constructor

```python
EvalV2ForceTelemetryCollector(*, spec: ForceTelemetrySpec, action_dt_s: float) -> None
```

Arguments:

- `spec`: the force telemetry specification from `models.py`
- `action_dt_s`: action-step duration used as fallback when `root.dt` is missing or invalid

Effects:

- stores the spec and step duration
- initializes runtime status and all telemetry accumulators
- derives SI scaling from `spec.units` when the validated SI mode is active

### Public property

```python
status -> ForceRuntimeStatus
```

Returns:

- the cached runtime status for the most recent `ensure_runtime(...)` call

### Public methods

```python
ensure_runtime(*, intervention: Any) -> ForceRuntimeStatus
```

Arguments:

- `intervention`: live intervention object with a `simulation.root`

Returns:

- `ForceRuntimeStatus(configured: bool, source: str, error: str)`

Behavior:

- resolves the active SOFA root
- configures the intrusive LCP path when `mode == "intrusive_lcp"`
- configures the passive monitor path when `mode in {"passive", "constraint_projected_si_validated"}`
- attaches `WireWallContactExport` best-effort for mapping support
- caches the status for the current root object

```python
capture_step(*, intervention: Any, step_index: int) -> None
```

Arguments:

- `intervention`: live intervention object with a `simulation.root`
- `step_index`: 1-based step counter used for trace records

Behavior:

- samples `root.LCP.constraintForces`
- samples the passive monitor when present
- samples `WireWallContactExport` when present
- projects constraint rows into per-DOF and per-triangle sparse records
- appends in-memory telemetry samples used by `build_summary()`

Returns:

- `None`

```python
build_summary() -> ForceTelemetrySummary
```

Returns:

- the per-trial summary dataclass used by the eval_v2 service layer

Behavior:

- combines monitor samples, raw LCP samples, and mapped-LCP samples
- chooses the current scoring channel and validation status
- exposes diagnostic fields for report generation and downstream tests

## In-memory state fields

The collector currently keeps the following private state.

### Runtime and status

- `_spec`
- `_action_dt_s`
- `_status`
- `_last_root_id`

### Monitor / LCP activity flags

- `_available_any`
- `_contact_detected_any`
- `_active_constraint_any`
- `_ordering_stable`
- `_source`
- `_channel`
- `_quality_tier`
- `_association_method`
- `_association_coverage`
- `_monitor_nonzero_detected`

### Contact counters and sampled maxima

- `_contact_count_max`
- `_segment_count_max`
- `_total_force_norm_samples`
- `_lcp_max_abs_samples`
- `_lcp_sum_abs_samples`
- `_peak_segment_force_norm`
- `_peak_segment_force_step`
- `_peak_segment_force_segment_id`

### Mapping-related accumulators

- `_mapped_contact_rows_any`
- `_lcp_mapped_force_like_max_samples`
- `_lcp_mapped_wall_row_count_max`
- `_lcp_contact_export_contact_count_max`
- `_lcp_contact_export_coverage_samples`
- `_last_export_rows`
- `_last_constraint_projection`
- `_triangle_force_records`
- `_wire_force_records`

### Status text and unit scaling

- `_last_status_text`
- `_si_conversion_enabled`
- `_force_scale_to_newton`
- `_lcp_force_like_max_samples`

## Helper functions used by the collector

These are private module helpers, but they are part of the current API surface
relevant to tests and callers inside `eval_v2`.

- `resolve_monitor_plugin_path(plugin_override: Optional[Path]) -> Optional[Path]`
- `_unit_scale_to_newton(units: Any) -> float`
- `_parse_constraint_rows(raw: Any) -> list[tuple[int, int, np.ndarray]]`
- `_project_constraint_forces(lcp_forces: np.ndarray, constraint_raw: Any, n_points: int, dt_s: Optional[float] = None) -> tuple[np.ndarray, list[dict]]`

## Return dataclasses

The collector returns `ForceRuntimeStatus` from `ensure_runtime(...)` and
`ForceTelemetrySummary` from `build_summary()`.

`ForceRuntimeStatus` fields:

- `configured: bool`
- `source: str`
- `error: str`

The current summary dataclass is defined in `models.py` and includes the
collector-facing fields for scoring, validation, and diagnostics.
