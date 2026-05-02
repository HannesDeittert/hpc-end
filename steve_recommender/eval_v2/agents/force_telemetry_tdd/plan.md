# eval_v2 Force Telemetry TDD Plan

This folder is the handoff location for the new force-telemetry validation work.
It exists so a later agent can open one file, see the plan, and continue from the
latest recorded state without having to reconstruct the intent from chat history.

## Goal

Stabilize `eval_v2` force telemetry with a TDD split across three independent axes:

- Physics: the projection from `MechanicalObject.constraint` plus `LCP.constraintForces` to per-DOF Newton forces.
- Mapping: row-to-triangle and row-to-wire-DOF attribution through a synthetic exporter first, then the real SOFA export path.
- Artefacts: deterministic per-step traces, persistence schema, and regression fixtures.

The key convention is:

- `scene_force [kg·mm/s²] = m [kg] * g [mm/s²]`
- `SI_force [N] = scene_force * 1e-3`
- `mN` is only a readability alias for SI values, not a scene-unit label.

## Working principle

The order matters:

1. Freeze the public collector API and capture real runtime fixtures.
2. Lock the physics math with pure unit tests that do not boot SOFA.
3. Lock mapping with a fake exporter before touching the live scene.
4. Validate end-to-end physics in a minimal sphere-on-plane SOFA scene.
5. Regress the real intervention traces and their scoreable coverage.
6. Add persistence last.

This avoids mixing three different failure modes in the same test layer.

## Phase 0 - Preparation

### Task 0.1 - API snapshot of the collector

Deliverable:

- `steve_recommender/eval_v2/docs/collector_api_v0.md`

Record:

- class signature
- public methods
- arguments and return types
- in-memory state fields

Status:

- Completed: wrote `steve_recommender/eval_v2/docs/collector_api_v0.md` with the current public collector surface, constructor, property, methods, return dataclasses, and in-memory state fields.

### Task 0.2 - Real constraint snapshot

Deliverable:

- `steve_recommender/eval_v2/tests/fixtures/constraint_string_v23_06_real.txt`
- companion README with date, scene, `dt`, and row count
- `lcp_constraintForces_real.npy` from the same step

Status:

- Completed: captured a real contact step from `debug.force_playground.maintainer_example_exact`, wrote `steve_recommender/eval_v2/tests/fixtures/constraint_string_v23_06_real.txt`, `steve_recommender/eval_v2/tests/fixtures/lcp_constraintForces_real.npy`, and the companion README with step, `dt`, and row count.

## Phase A - Pure unit tests

This phase is SOFA-free and should run in milliseconds in CI.
It validates the physics axis only.

### Task A.1 - Parser robustness

Goal:

- `_parse_constraint_rows` accepts both legacy whitespace numeric format and the newer `Constraint ID : / dof ID : / value :` format.
- Both formats produce the same sparse representation.

Planned tests:

- `test_parse_old_format_single_dof_per_row`
- `test_parse_new_format_single_dof_per_row`
- `test_parse_old_and_new_yield_identical_rows`
- `test_parse_multidof_row_keeps_all_dofs`
- `test_parse_empty_string_returns_empty_rows`
- `test_parse_real_v23_06_fixture`

Status:

- Completed: added parser coverage for legacy whitespace rows, label-based rows, multidof rows, empty input, and the real v23.06 fixture snapshot.

### Task A.2 - Projection mathematics

Goal:

- `_project_constraint_forces` computes `H^T * lambda / dt` deterministically and with the correct sign.

Planned tests:

- `test_project_unit_lambda_recovers_h_row`
- `test_project_zero_lambda_yields_zero_forces`
- `test_project_three_friction_rows_recompose_3d_vector`
- `test_project_multidof_row_distributes_with_barycentric_weights`
- `test_project_dt_scales_inversely`

Status:

- Completed: added deterministic projection coverage for unit lambda, zero lambda, friction-row composition, barycentric splitting, and inverse dt scaling.

### Task A.3 - Unit invariant

Goal:

- eliminate 1000x unit mistakes permanently.

Planned tests:

- `test_unit_invariant_scene_to_si_factor_is_exactly_1e_minus_3`
- `test_unit_scaling_applied_exactly_once`
- `test_unit_invariant_holds_for_triangle_records`
- `test_unit_invariant_holds_for_wire_records`

Status:

- Completed: locked the scene-to-SI factor and verified that triangle and wire records are converted exactly once.

### Task A.4 - Availability semantics

Goal:

- make `available_for_score` and `validation_status` deterministic and documented.

Planned tests:

- `test_status_ok_when_mapping_complete`
- `test_status_lcp_only_unmapped_when_export_missing`
- `test_status_partial_when_some_rows_invalid`
- `test_native_monitor_always_diagnostic_only`

Status:

- Completed: complete, partial, unmapped, and diagnostic-only monitor cases are now deterministic in the collector and covered by tests.

## Phase A.5 - Mapping with a synthetic exporter

This phase is still unit-level, but it isolates the mapping axis from SOFA.

### Task A.5.1 - Fake exporter harness

Deliverable:

- `steve_recommender/eval_v2/tests/helpers/fake_contact_export.py`

Stub fields:

- `constraintRowIndices`
- `constraintRowValidFlags`
- `wallTriangleIds`
- `triangleIdValidFlags`
- `collisionDofIndices`
- `collisionDofValidFlags`
- `contactCount`
- `explicitCoverage`

Status:

- Completed: added a reusable `FakeContactExport` helper with normalized row, triangle, and collision-DOF validity flags.

### Task A.5.2 - Row to triangle aggregation

Planned tests:

- `test_single_row_maps_to_single_triangle`
- `test_multiple_rows_same_triangle_accumulate`
- `test_rows_with_invalid_flag_excluded_from_aggregation`
- `test_explicit_coverage_false_marks_summary_unscoreable`

Status:

- Completed: row-to-triangle aggregation, invalid-row exclusion, and explicit-coverage failure behavior are covered by the synthetic mapping tests.

### Task A.5.3 - Row to wire DOF aggregation

Planned tests:

- `test_row_maps_to_correct_wire_collision_dof`
- `test_wire_dof_record_force_matches_projection`
- `test_distal_dof_filter_isolates_tip_records`

#### Plumbing inspection

- Collector constructed at: `steve_recommender/eval_v2/runner.py:463`
- Config dataclass passed in: `ForceTelemetrySpec` in `steve_recommender/eval_v2/models.py`
- CLI builds it via: `_handle_run` in `steve_recommender/eval_v2/cli.py`
- GUI builds it via: `_build_job_from_state` in `steve_recommender/eval_v2/ui_wizard.py`

Finding:

- CLI and GUI both build an `EvaluationJob` containing an `EvaluationScenario`.
- `EvaluationScenario.force_telemetry` is the shared `ForceTelemetrySpec` dataclass passed to the collector by `runner.py`.
- Outcome: same dataclass, one config chain. Add `tip_threshold_mm` to `ForceTelemetrySpec`; no unification refactor is needed.

Definitions to lock:

- Tip semantics: absolute arc length from the distal end of the wire in mm. A collision DOF is "tip" when its arc-length distance from the distal end is less than or equal to `tip_threshold_mm`.
- Distal direction: verified in the real intervention scene on 2026-04-28. At contact step 12, `CollisionDOFs[0]` was near insertion at `[0.0639, 0.0046, 1.9989]` and `CollisionDOFs[-1]` was the distal tip/contact end at `[4.5029, 5.8389, 43.1529]`; therefore the last collision DOF is distal.
- Default constant: `DEFAULT_TIP_THRESHOLD_MM = 3.0`, imported by all consumers.

Status:

- Completed: row-to-wire DOF mapping, projected-force checks, distal-tip arc-length filtering, tip record flags, summary aggregation, and real-fixture smoke coverage are complete.

### Task A.5.4 - Many-to-many robustness

Planned tests:

- `test_one_row_to_many_dofs_distributes_correctly`
- `test_many_rows_to_one_dof_accumulate`
- `test_record_count_matches_expected_sparse_size`

Status:

- Completed: fan-out and sparse record-count behavior are covered by the many-to-many mapping tests.
## Phase B - Sphere-on-plane validation scene

This is the first booted SOFA validation scene.
It validates the physics axis end-to-end, not the mapping axis.

### Task B.1 - Scene setup

Deliverable:

- `steve_recommender/eval_v2/tests/scenes/validation_sphere_on_plane.py`

Status:

- Completed: added a minimal sphere-on-plane validation scene scaffold and a structure test that imports without SOFA.
### Task B.2 - Analytical ground truth test

Planned test:

- `test_validation_scene_recovers_mg_within_1pct`

Status:

- Completed: B.2.a completed the SOFA boot/contact slice, and B.2.b now validates that projecting `LCP.constraintForces / dt` through the scene's real constraint rows recovers `m*g` within 1%.

### Task B.3 - Friction variant

Planned test:

- `test_validation_scene_with_friction_normal_force_unchanged`

Status:

- Completed: added a friction variant test that verifies the normal force remains equal to `m*g` with nonzero friction.

### Task B.4 - dt sweep

Planned test:

- `test_validation_scene_force_invariant_across_dt`

Status:

- Completed: added a dt sweep test that verifies projected force remains invariant across `dt = 0.005, 0.01, 0.02`.

## Phase C - Real intervention smoke and regression

This phase checks artefacts and reproducibility on the real intervention pipeline.

### Task C.1 - Trace regression

Deliverable:

- `steve_recommender/eval_v2/tests/fixtures/trace_smoke_<seed>.json`

Planned test:

- `test_force_trace_smoke_matches_golden`

Status:

- Completed: replaced the synthetic trace shape fixture with a live 20-step eval_v2 runner force trace for seed 123 and added `test_force_trace_smoke_matches_golden`.

### Task C.2 - Coverage statistics

Planned test:

- `test_real_scene_has_nonzero_scoreable_coverage`

Status:

- Completed: added `test_real_scene_has_nonzero_scoreable_coverage` against the live seed-123 trace fixture.

## Phase D - Persistence

This is intentionally last.
Only once the earlier layers are stable should the output format be frozen.

### Task D.1 - Writer

Deliverable:

- `steve_recommender/eval_v2/docs/persistence_schema.md`
- NPZ writer for arrays
- JSONL writer for sparse per-step records

Status:

- Completed: added the persistence schema doc and the NPZ/JSONL writer module.

### Task D.2 - Round-trip tests

Planned tests:

- `test_persistence_roundtrip_preserves_records`
- `test_persistence_schema_version_in_header`

Status:

- Completed: added round-trip and schema-version tests for the force-trace persistence layer.

## Current progress log

This section is the handoff note for future agents.
After each completed step, append one short entry here that records what was done and what remains.

### Completed so far

- Completed Task A.1 by implementing parser support for both legacy and label-based constraint formats and validating it with six parser-focused tests.
- Completed Task A.2 by validating the constraint-force projection math with five focused projection tests.
- Completed Task A.3 by locking the unit conversion factor and record-level SI scaling with new mapping tests.
- Completed Task A.4 by making availability semantics deterministic for complete, partial, and unmapped exports.
- Completed Task A.5 by adding the synthetic contact-export harness and the row-to-triangle / row-to-wire mapping tests.
- Completed Task B.1 by adding a minimal sphere-on-plane validation scene scaffold.
- Completed Task B.2.a by porting the SOFA-supported mesh collision wiring into the validation scene, enabling `computeConstraintForces` on the LCP solver, and adding boot-test assertions for populated geometry, emitted contact rows, and non-empty solver impulses.
- Completed Task B.2.b by projecting real SOFA constraint rows and LCP impulses back to scene forces and asserting the result equals `m*g` within 1%.
- Completed Task B.3 by adding the friction variant that keeps the normal force invariant.
- Completed Task B.4 by adding the dt sweep that keeps projected force invariant across tested timesteps.
- Completed Task C.1 by capturing a live seed-123 trace-smoke fixture and locking its deterministic summary/representative records with a golden test.
- Completed Task C.2 by requiring the live trace fixture to have scoreable validated coverage with mapped rows and nonzero force.
- Completed Task D.1 and Task D.2 by adding the persistence writer, schema doc, and round-trip tests.
- Completed Task A.5.3.0 by adding the shared `DEFAULT_TIP_THRESHOLD_MM`, validating positive `tip_threshold_mm` in `ForceTelemetrySpec`, storing it in the collector, and plumbing the CLI `--tip-threshold-mm` flag through the shared scenario config.
- Completed Task A.5.3.a by adding the collision-DOF arc-length helper with the verified convention that the last collision DOF is distal.
- Completed Task A.5.3.b by adding `arc_length_from_distal_mm` and inclusive `is_tip` classification to wire force records.
- Completed Task A.5.3.c by adding summary-level tip records, total tip force vector aggregation, and unit-invariant tip record fields.
- Completed Task A.5.3.d by running the distal-tip filter on the real v23.06 constraint/LCP fixtures with a synthetic arc-length map.
- Completed the A.5.3 report round-trip edge by preserving the new tip summary fields when persisted reports are loaded from disk.

### What remains

- Nothing in the planned force-telemetry TDD workstream.

### Deferred

- GUI tip-threshold knob: GUI does not currently expose `tip_threshold_mm`; it uses `DEFAULT_TIP_THRESHOLD_MM = 3.0`. To expose later: add a numeric field to the execution config page in `ui_wizard_pages.py` and pass through the existing config object; no collector changes needed.
- Known unrelated EGL visual test failure: `tests/eval_v2/test_runner.py::SingleTrialRunnerIntegrationTests::test_run_single_trial_emits_real_sofa_rgb_frames_when_visualized` can fail in environments without a valid EGL device.

Closing note: All planned phases complete: Phase 0, A, A.5, B, C, D.

## Notes for later agents

- Keep physics, mapping, and artefacts separated in the test hierarchy.
- Do not mix the sphere-on-plane scene with mapping assertions.
- Do not rename the unit convention without updating this plan and the tests together.
- When a step is completed, record it in the progress log before moving on.
