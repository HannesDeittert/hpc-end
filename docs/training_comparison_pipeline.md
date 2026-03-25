# Training Comparison Pipeline (ArchVar)

This document describes a practical Python-first pipeline for comparing long-running/resumed training jobs.

## Goals
- Preserve complete checkpoint history across resumed runs.
- Preserve progress/quality/restart timelines from `main.log`.
- Build a lineage-aware index (resume chains, not isolated runs).
- Keep storage format simple and scriptable (JSONL + CSV).

## Implemented Components

### 1) History extraction
Script: `scripts/export_archvar_history.py`

- Scans run folders under a configurable root.
- Parses each `main.log` for:
  - tool selection
  - resume checkpoint path
  - loaded counters
  - update events
  - evaluation events
  - quality events
  - restart events
  - optional heartbeat events
- Parses checkpoint directory for all `.everl` files and checkpoint step numbers.
- Optionally scans worker logs for timeout-kill / traceback / SIGSEGV counters.
- Optionally parses run CSV (`<run_name>.csv`) for quality/config cross-checks.
- Resolves parent-child lineage via `Loading checkpoint: ...` path matching.

### 2) Normalized index
Written to an output folder as both JSONL and CSV:

- `runs.jsonl` / `runs.csv`: one run summary per row
- `checkpoints.jsonl` / `checkpoints.csv`: one checkpoint file per row
- `events.jsonl` / `events.csv`: parsed log events per row
- `chains.jsonl` / `chains.csv`: lineage-level aggregate rows
- `manifest.json`: generation metadata + file list

### 3) Chain comparison
Script: `scripts/compare_archvar_history.py`

- Reads `chains.jsonl` and prints a sortable comparison table.
- Supports regex filtering by chain id or tool.
- Supports CSV export of filtered chain rows.

## Why JSONL + CSV
- JSONL: robust typed records, easy to stream, easy in Python.
- CSV: quick ad-hoc use in shell/Excel.
- No heavy dependency requirements (pandas optional but not required).

## Recommended Workflow

1. Extract index:

```bash
python3 scripts/export_archvar_history.py \
  --runs-root /home/woody/.../results/eve_paper/neurovascular/aorta/gw_only/arch_vmr_94 \
  --out-dir logs/hpc/history_index
```

2. Compare chains:

```bash
python3 scripts/compare_archvar_history.py \
  --index-dir logs/hpc/history_index \
  --top 20
```

3. Focus one experiment family (example):

```bash
python3 scripts/compare_archvar_history.py \
  --index-dir logs/hpc/history_index \
  --tool-regex "jlongtip45|jshaped_default"
```

## Physical Comparison Pipeline (Checkpoint-Linked)

Use this when you want a clean physical comparison where each checkpoint is
explicitly tied to the tool used during training.

`scripts/eval_checkpoint_manifest.py` is a compatibility wrapper and now routes
through the shared comparison/evaluation core.

1. Build an evaluation manifest from the index:

```bash
python3 scripts/build_checkpoint_eval_manifest.py \
  --index-dir logs/hpc/history_index \
  --selection latest_per_chain \
  --out logs/hpc/history_index/checkpoint_eval_manifest.json
```

2. Inspect (no simulation yet):

```bash
python3 scripts/eval_checkpoint_manifest.py \
  --manifest logs/hpc/history_index/checkpoint_eval_manifest.json \
  --dry-run
```

3. Run physical evaluation:

```bash
python3 scripts/eval_checkpoint_manifest.py \
  --manifest logs/hpc/history_index/checkpoint_eval_manifest.json \
  --n-trials 20 \
  --base-seed 123 \
  --max-episode-steps 1000 \
  --policy-device cuda \
  --name archvar_physical_compare
```

Outputs:
- `results/eval_runs/<timestamp>_archvar_physical_compare/summary.csv`
- `results/eval_runs/<timestamp>_archvar_physical_compare/ranking.csv`

## Next Investigation/Extension Plan

1. Throughput decomposition
- Add per-stage throughput summaries from update/eval/heartbeat events.
- Report explore-steps/hour and update-steps/hour per run and chain.

2. Failure diagnostics table
- Join chain rows with Slurm accounting (`sacct`) and `*.out` parsing.
- Add normalized failure cause labels (`TIMEOUT`, `OOM`, `IMPORT_ERROR`, `FD_LIMIT`).

3. Quality-at-step interpolation
- Build comparison at matched exploration-step anchors (fairer than "latest vs latest").
- Add trend metrics (rolling slope, recovery after resume).

4. Incremental indexing mode
- Store stable run hash + mtime and re-parse only changed runs.
- Useful for continuous daily updates on HPC data.

5. Optional parquet backend
- Keep JSONL/CSV as baseline.
- Optionally write parquet when pandas/pyarrow are available.

## Notes on current U45/Baseline behavior
- U45 is slower mostly due to lower worker count and restart overhead.
- Baseline chain shows higher daily step gain.
- OOM and file descriptor pressure should be tracked as first-class chain diagnostics.
