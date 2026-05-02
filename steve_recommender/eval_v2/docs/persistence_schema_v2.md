# Trial Trace Persistence Schema v2

This document defines the Phase E trial-trace persistence schema for
`eval_v2`. New traces are written as one HDF5 file per trial.

## Version

- `schema_version = 2`
- container format: HDF5 via `h5py`

## Output layout

Per-job layout:

```text
output_root/
└── <job_name>_<timestamp>/
    ├── manifest.json
    ├── candidate_summaries.csv
    ├── candidate_summaries.json
    ├── trials.h5
    ├── report.md
    ├── meshes/
    │     anatomy_<id>.h5
    └── traces/
          trial_<candidate>_<env_seed>_<policy_seed>.h5
```

`manifest.json` is the archive entrypoint. `trials.h5` stores one row per
executed trial in column-oriented form. This document still describes the
per-trial HDF5 trace files under `traces/`.

Per-trial layout:

```text
/meta
/scenario
/scene_static
/steps
/contacts
/diagnostics
```

The writer is always the same. Readers may either load the full trial or seek
to one step at a time.

## Locked storage decisions

- one HDF5 file per trial
- dense per-step arrays are stored as chunked 2D+ datasets, indexed by step
- dense per-step datasets use one step per chunk
- gzip level `4` is used on:
  - `/steps/wire_positions`
  - `/steps/wire_collision_positions`
  - `/contacts/*` bulk arrays
- positions and forces use `float32`
- indices use `int32`
- flags use `bool`
- `/diagnostics` is optional and is only written when `write_diagnostics=True`

## `/meta`

Attributes only:

- `schema_version: int = 2`
- `eval_v2_sha: str`
- `sofa_version: str`
- `created_at: str` as ISO 8601 timestamp
- `trial_status: str` in `{"complete", "partial"}`

`trial_status` starts as `"partial"` when the file is created and flips to
`"complete"` only after successful close.

## `/scenario`

Attributes only:

- `anatomy_id: str`
- `wire_id: str`
- `target_spec_json: str`
- `env_seed: int`
- `policy_seed: int | null`
- `dt_s: float`
- `friction_mu: float`
- `tip_threshold_mm: float`
- `max_episode_steps: int`
- `mesh_ref: str`

`mesh_ref` is a relative path to `../meshes/anatomy_<id>.h5`.

## `/scene_static`

Datasets:

- `wire_initial_position`: shape `(n_dofs, 3)`, dtype `float32`
- `wire_initial_rotation`: shape `(4,)`, dtype `float32`

`wire_initial_rotation` stores a quaternion.

## `/steps`

All datasets have length `N`, one entry per simulation step.

Datasets:

- `step_index`: shape `(N,)`, dtype `int32`
- `sim_time_s`: shape `(N,)`, dtype `float32`
- `wire_positions`: shape `(N, n_dofs, 3)`, dtype `float32`,
  compression `gzip=4`, chunk `(1, n_dofs, 3)`
- `wire_collision_positions`: shape `(N, n_coll, 3)`, dtype `float32`,
  compression `gzip=4`, chunk `(1, n_coll, 3)`
- `action`: shape `(N, action_dim)`, dtype `float32`
- `total_wall_force_N`: shape `(N,)`, dtype `float32`
- `tip_force_norm_N`: shape `(N,)`, dtype `float32`
- `contact_count`: shape `(N,)`, dtype `int16`
- `scoreable`: shape `(N,)`, dtype `bool`

The format does not use per-step groups such as `/steps/step_0/...`.

## `/contacts`

The collector measures contacts in two physically meaningful aggregations: per
wire collision DOF and per wall triangle. A single LCP constraint row
contributes to both views, but the producer does not retain the per-row
association, only the aggregations. Persisting both views faithfully is
preferable to fabricating a joined per-row record at write time. Consumers that
need both wire and wall views can read both tables; consumers that need one or
the other read only what they need.

Both contact views are stored as CSR-style ragged arrays so step `k` can be
recovered with `step_offsets[k]:step_offsets[k + 1]`.

### `/contacts/wire`

Per-DOF wire-contact aggregates. `M_wire` is the total number of flattened
wire-contact records across the trial.

Datasets:

- `step_offsets`: shape `(N + 1,)`, dtype `int32`
- `wire_dof_index`: shape `(M_wire,)`, dtype `int32`, compression `gzip=4`
- `wire_dof_force_xyz_N`: shape `(M_wire, 3)`, dtype `float32`,
  compression `gzip=4`
- `arc_length_from_distal_mm`: shape `(M_wire,)`, dtype `float32`,
  compression `gzip=4`
- `is_tip`: shape `(M_wire,)`, dtype `bool`

### `/contacts/triangle`

Per-triangle wall-contact aggregates. `M_tri` is the total number of flattened
triangle-contact records across the trial.

Datasets:

- `step_offsets`: shape `(N + 1,)`, dtype `int32`
- `triangle_id`: shape `(M_tri,)`, dtype `int32`, compression `gzip=4`
- `triangle_force_xyz_N`: shape `(M_tri, 3)`, dtype `float32`,
  compression `gzip=4`

The two tables share the same step index axis but are otherwise independent:
`M_wire` and `M_tri` generally differ, and there is no row-by-row
correspondence between `/contacts/wire/*` and `/contacts/triangle/*`.

## `/diagnostics`

Optional group, only written when `write_diagnostics=True`.

Planned datasets:

- `lcp_constraint_forces`
- `constraint_rows`

These are debug-heavy fields and should remain absent from normal traces.

## Units and invariants

- force fields with `_N` are stored in SI Newtons
- lengths with `_mm` are stored in millimeters
- `tip_threshold_mm` and `arc_length_from_distal_mm` use the same distal-tip
  semantics as the force-telemetry workstream
- dense per-step arrays are step-major to support both full-scan and random
  step access
