# eval_v2 Trial Trace Persistence v2 Plan

This folder is the handoff location for the Phase E trial-trace persistence
workstream. It exists so a later agent can open one file, see the plan, and
continue from the latest recorded state without reconstructing intent from chat
history.

## Goal

Add always-on, per-trial HDF5 trace persistence to `eval_v2` so every trial
emits one self-contained, machine-readable trace file that supports both full
load and random step access without changing the writer.

The trace must capture:

- scene starting state
- every wire DOF position per step
- every action
- every contact
- every per-step force telemetry record
- enough metadata to reconstruct the trial offline

The locked format decisions are:

- HDF5 via `h5py`, one file per trial
- default behavior is always-on, with `--no-write-trace` as opt-out
- schema version `2` for new traces
- dense per-step fields stored as chunked 2D+ datasets
- one step per chunk for dense per-step arrays
- gzip level `4` on bulk position/contact arrays only
- `float32` for positions and forces, `int32` for indices, `bool` for flags
- each worker writes only its own trace files
- service pre-writes shared anatomy mesh files before worker spawn

## Working principle

The order matters:

1. Freeze the schema version and documentation first.
2. Lock CSR ragged-contact encoding with pure unit tests before touching HDF5.
3. Implement the writer with strong boundary validation and partial-file
   semantics.
4. Add the reader with both full-load and random-access APIs against the same
   file format.
5. Prove the format survives the multiprocessing worker model.
6. Wire the recorder into production only after the storage layer is locked.

This keeps schema, pure data transforms, I/O, parallelism, and production
integration separated.

## Deliverable layout

Per-job output layout:

```text
output_root/
└── <job_name>_<timestamp>/
    ├── report.json
    ├── summary.csv
    ├── report.md
    ├── meshes/
    │     anatomy_<id>.h5
    └── traces/
          trial_<candidate>_<env_seed>_<policy_seed>.h5
```

Per-trial HDF5 layout:

```text
/meta
/scenario
/scene_static
/steps
/contacts
/diagnostics
```

`/diagnostics` stays opt-in behind `write_diagnostics=False` by default.

## Phase E - Always-on trial trace persistence

### Task E.1 - Schema doc, version constant, dependency

Deliverables:

- `steve_recommender/eval_v2/docs/persistence_schema_v2.md`
- rename `persistence_schema.md` to `persistence_schema_v1.md`
- new `TRACE_SCHEMA_VERSION = 2`
- retained `LEGACY_TRACE_SCHEMA_VERSION = 1`
- `pyproject.toml` dependency on `h5py>=3.0`
- `ruff` and `black` installed in the `master-project` conda env

Planned tests:

- `test_trace_schema_version_constant_pinned_to_2`
- `test_legacy_schema_version_constant_pinned_to_1`
- `test_v1_reader_still_loads_existing_fixtures`
- `test_h5py_dependency_importable`

Status:

- Completed: v2/v1 schema docs are split, schema version constants are pinned, `h5py` is added to project dependencies, and the legacy v1 readers remain intact.

### Task E.2 - Sparse CSR encoder / decoder

Goal:

- convert ragged per-step contact lists to and from CSR-style `(step_offsets,
  flat_arrays)` without any HDF5 I/O

Planned tests:

- `test_csr_round_trip_preserves_records`
- `test_csr_zero_contact_steps_have_zero_length_slices`
- `test_csr_single_contact_step`
- `test_csr_offsets_array_length_is_n_plus_1`
- `test_csr_decode_step_k_returns_only_step_k_records`
- `test_csr_handles_empty_trial_zero_steps`

Status:

- Completed: added canonical persistence-layer contact dataclasses and a pure CSR encoder/decoder with locked dtypes and per-step random-access decoding.

### Task E.3 - TrialTraceRecorder writer

Goal:

- add `TrialTraceRecorder` as a context-managed HDF5 writer with buffered step
  flushes, partial/complete status tracking, and one-writer-per-file semantics

Planned tests:

- `test_recorder_writes_all_required_datasets`
- `test_recorder_attributes_match_scenario_config`
- `test_recorder_chunking_one_step_per_chunk_for_dense_fields`
- `test_recorder_compression_applied_to_bulk_arrays_only`
- `test_recorder_flush_boundary_does_not_drop_steps`
- `test_recorder_close_finalizes_status_to_complete`
- `test_recorder_unclosed_file_has_partial_status`
- `test_recorder_rejects_invalid_scenario_config`

Status:

- Completed: added a context-managed HDF5 trial writer with boundary validation, partial/complete status handling, one-step chunking for dense arrays, and gzip compression on the bulk arrays required by the schema.

### Task E.4 - TraceReader full-load and step access

Goal:

- add `TraceReader` with `load_all()` and `step(k)` over the same HDF5 layout

Planned tests:

- `test_reader_round_trip_full_load_preserves_records`
- `test_reader_step_access_returns_correct_data`
- `test_random_step_access_does_not_load_full_dataset`
- `test_sequential_full_scan_is_chunk_efficient`
- `test_reader_handles_partial_trace_gracefully`
- `test_reader_handles_corrupt_file_with_clear_error`
- `test_reader_contacts_for_step_uses_csr_offsets`

Status:

- Completed: added `TraceReader` with full-load, per-step chunked access, CSR-backed contact slicing, partial-trace handling, and a dedicated corrupt-file exception type.

### Task E.5 - Parallel-writer correctness

Goal:

- prove the per-trial HDF5 choice survives the multiprocessing worker model and
  shared mesh pre-write scheme

Planned tests:

- `test_four_workers_write_distinct_trial_files_concurrently`
- `test_concurrent_writes_do_not_corrupt_files`
- `test_shared_mesh_file_written_by_service_before_workers`
- `test_worker_crash_leaves_identifiable_partial_file`
- `test_parallel_writes_scale_roughly_linearly`

Status:

- Completed: added spawn-based concurrency tests for per-worker trace files, locked partial-file crash behavior, and added `write_anatomy_mesh(...)` with atomic publish and overwrite rejection semantics for shared mesh files.

### Task E.6 - Production wire-in

Goal:

- wire trace writing into CLI, GUI defaults, runner, and service without
  changing the closed force-telemetry collector behavior

Planned changes:

- `ForceTelemetrySpec.write_full_trace: bool = True`
- `ForceTelemetrySpec.write_diagnostics: bool = False`
- CLI flags `--no-write-trace` and `--write-diagnostics`
- GUI keeps defaults and does not expose either flag
- `runner.py` creates one `TrialTraceRecorder` per trial when enabled
- `service.py` pre-writes meshes before worker spawn

Planned tests:

- `test_runner_writes_trace_file_per_trial`
- `test_no_write_trace_flag_suppresses_file_creation`
- `test_full_job_produces_n_trace_files_for_n_trials`
- `test_cli_passes_no_write_trace_flag_correctly`
- `test_cli_passes_write_diagnostics_flag_correctly`
- `test_gui_default_path_writes_trace`
- `test_service_pre_writes_meshes_before_workers`
- `test_meshes_written_once_per_anatomy_not_per_trial`

Status:

- Completed: added `write_full_trace`/`write_diagnostics` plumbing to the
  config and CLI, wired `TrialTraceRecorder` into `run_single_trial(...)`,
  pre-wrote anatomy meshes in the service layer, and made trace-write failures
  non-fatal to trials with warning capture on the trial result.

### Task E.5.5 - Schema split to match producer aggregates

Goal:

- split v2 contact persistence into independent wire-contact and triangle-contact
  ragged tables so the schema matches the closed collector's two aggregate views

Status:

- Completed: rewrote the v2 schema to use `/contacts/wire/*` and
  `/contacts/triangle/*`, refactored the CSR/writer/reader stack to match, and
  removed the fabricated joined contact row from persistence.

## Current progress log

This section is the handoff note for later agents. After each completed step,
append one short entry here that records what was done and what remains.

### Completed so far

- Plan created from the locked Phase E decisions and task order; implementation has not started yet.
- Completed Task E.1 by splitting the persistence docs into v1/v2, pinning `TRACE_SCHEMA_VERSION = 2` and `LEGACY_TRACE_SCHEMA_VERSION = 1`, adding `h5py>=3.0`, and proving the legacy v1 readers still load existing v1 artifacts.
- Completed Task E.2 by formalizing `TriangleContactRecord`, `WireContactRecord`, `ContactRecord`, and `CSREncoded` in `force_trace_persistence.py` and locking the CSR ragged-contact encoding with pure unit tests.
- Completed Task E.3 by adding `ScenarioConfig`, `SceneStaticState`, `StepData`, and `TrialTraceRecorder`, then verifying the produced `.h5` file structure, attrs, chunking, compression, flush behavior, and partial-file semantics.
- Completed Task E.4 by adding `TraceReader` and `TraceFileCorruptError`, then locking full-load, chunked step access, CSR contact slicing, partial-trace reads, and corrupt-file handling with reader tests.
- Completed Task E.5 by proving distinct worker processes can write trace files concurrently without corruption and by adding `write_anatomy_mesh(...)` as the service-layer mesh pre-write primitive.
- Completed Task E.5.5 by splitting contact persistence into independent wire and triangle CSR tables, updating the HDF5 layout and reader/writer APIs to match the collector's real aggregate outputs.
- Completed Task E.6 by wiring always-on HDF5 traces into the CLI, runner, and service layers, pre-writing shared anatomy meshes once per job, and treating trace-write failures as warnings instead of failed trials.
- Final verification: `tests/eval_v2` passed with the known EGL visual test deselected, touched-file `ruff`/`black` checks were clean, and a real two-trial CLI smoke wrote `meshes/`, `traces/`, and report artifacts under `/tmp/eval_v2_trace_smoke/eval_v2_job`.
- Post-close validation: a real 50-trial, 4-worker, 100-step benchmark on `Tree_00` with `archvar_original_best` produced 50 complete traces totaling 26.64 MiB, while traces-on (`157.50 s`) and traces-off (`158.85 s`) wall times were effectively identical on the same seed schedule.

### What remains

- Nothing planned remains in Phase E.

### Deferred

- GUI viewer with scrubbable timeline. Storage layer only in this workstream.
- Trace retention policy or automatic deletion.
- Cross-trial aggregate analysis tooling.
- Cloud or remote storage integration.
- Migration of existing v1 fixtures to v2.
- GUI trace-writing toggles. GUI uses `ForceTelemetrySpec` defaults only.
- Repo-wide lint and format drift outside the touched Phase E files. Per-file
  `ruff`/`black` checks are clean for this workstream; unrelated backlog remains
  elsewhere in `eval_v2`.

## Closing retrospective

- The storage layer now matches the closed collector exactly: wire and triangle
  contacts are persisted as separate CSR-indexed views rather than a fabricated
  joined row type.
- The production runner writes one HDF5 trace per trial by default and keeps
  trial scoring resilient when trace persistence fails.
- The service layer pre-writes shared anatomy meshes once per job so parallel
  workers never contend on mesh outputs.
- All planned phases complete: E.1 -> E.6.

## Notes for later agents

- Do not modify the closed force-telemetry producer API unless a Phase E task
  explicitly requires a runner-side hook.
- Keep unit conventions aligned with the closed workstream:
  `scene_force [kg·mm/s²] * 1e-3 = SI_force [N]`.
- Dense per-step arrays stay step-major and chunked one step at a time.
- Each worker must own its trace file exclusively from start to finish.
- Canonical persistence-layer record and config dataclasses live in `force_trace_persistence.py` so the writer and reader share one typed schema boundary.
- Update this plan after every completed sub-step before moving on.
