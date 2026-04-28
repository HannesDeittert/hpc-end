# Force Trace Persistence Schema

This document freezes the lightweight trace format used by the eval_v2 force-telemetry TDD work.

## Version

- `schema_version = 1`

## NPZ bundle

The NPZ writer stores dense arrays for the primary sparse record fields and keeps a JSON copy of the original records for exact round-tripping.

### Required arrays

- `schema_version`
- `triangle_timestep`
- `triangle_id`
- `triangle_fx_N`
- `triangle_fy_N`
- `triangle_fz_N`
- `triangle_norm_N`
- `triangle_contributing_rows`
- `wire_timestep`
- `wire_collision_dof`
- `wire_row_idx`
- `wire_fx_N`
- `wire_fy_N`
- `wire_fz_N`
- `wire_norm_N`

### JSON payloads in the bundle

- `triangle_records_json`
- `wire_records_json`
- `metadata_json`

## JSONL stream

The JSONL writer emits one header row followed by one record per line.

### Header row

```json
{"record_type": "header", "schema_version": 1, "metadata": {}}
```

### Record rows

Each subsequent line is a plain JSON object representing a sparse trace record.

## Conventions

- Force values are stored in Newtons for persisted trace artefacts.
- `triangle_*` records describe wall-contact aggregation.
- `wire_*` records describe collision-DOF aggregation.
- Sparse records are kept as dictionaries to preserve the collector output without lossy reshaping.
