# Evaluation (Agent Benchmark)

Repo-local evaluation pipeline to benchmark **trained RL agents** (checkpoints) on a **fixed anatomy + start/target** setup.

This module intentionally lives **outside** upstream stEVE repos so it can be:
- executed from the **terminal** (CLI),
- imported from the **UI** code,
- and extended without touching upstream stEVE code.

## Quick start (CLI)

Prerequisites are the same as training:

```bash
conda activate master-project
source scripts/sofa_env.sh
```

Edit the example config and set your checkpoint paths:
- `docs/eval_example.yml`

Run:

```bash
steve-eval --config docs/eval_example.yml
```

Or use the comparison entrypoint (registry-aware candidate resolution):

```bash
steve-compare --config docs/compare_example.yml
```

Enable Sofa window rendering from CLI (without editing YAML):

```bash
steve-compare --config docs/compare_example.yml --visualize --visualize-trials-per-agent 1
steve-eval --config docs/eval_example.yml --visualize --visualize-trials-per-agent 1
```

Enable live contact/force debug while rendering:

```bash
steve-compare --config docs/compare_example.yml --visualize \
  --visualize-force-debug --visualize-force-top-k 8
```

List available `agent_ref` values:

```bash
steve-compare --list-agent-refs
```

Outputs:
- `results/eval_runs/<timestamp>_<name>/summary.csv` (per trial)
- `results/eval_runs/<timestamp>_<name>/trials/*.npz` (time series)
- `results/eval_runs/<timestamp>_<name>/report.{md,csv,json}` (per-agent aggregation + ranking)
- `results/force_calibration/cache.json` (validation cache for SI-validated mode)

## Configuration (YAML)

Top-level keys (see `docs/eval_example.yml`):
- `name`: run name (used for output folder)
- `agents`: list of `{name, tool, checkpoint}`
- `n_trials`, `base_seed`
- `seeds` (optional explicit seed list, overrides `n_trials/base_seed`)
- `max_episode_steps`
- `policy_device`: `"cuda"` or `"cpu"`
- `use_non_mp_sim`: set `true` if you want SOFA force scalars
- `stochastic_eval`: `false` by default (deterministic policy eval)
- `visualize`: open Sofa window while evaluating (optional)
- `visualize_trials_per_agent`: how many early trials per agent are rendered
- `visualize_force_debug`: if `true`, show live contact/force debug in window title
- `visualize_force_debug_top_k`: how many top-force segments to include in debug output
- `force_extraction`: passive/intrusive/validated mode + required/optional policy
- `output_root`
- `anatomy`: currently only `type: aortic_arch`
- `scoring`: optional scoring configuration (default is `mode: default_v1`)

## Anatomy / target

Currently supported:
- `aortic_arch` (stEVE built-in generator)
  - start: default `InsertionPoint`
  - target: `BranchEnd` on configured branch end(s), e.g. `["lcca"]`

## Force extraction (important note)

- `mode: passive` (preferred): dual-channel telemetry.
  - Contact detection via `ContactListener` + `LCP constraintForces` (detection/intensity only).
  - Force vectors from available mechanical states (`collision`/`wire`, `force`/`externalForce`).
  - Contact points are mapped to nearest wall triangles (v1: nearest centroid).
- `mode: intrusive_lcp` (explicit fallback): enables LCP force telemetry and may change trajectories.
- `mode: constraint_projected_si_validated`: uses constraint projection (`J^T * lambda / dt`) and SI conversion.
  - **Source of truth**: constraint projection only (`collision.constraintProjection` preferred), no mixing with `force/externalForce`.
  - Wall-triangle association in validated mode:
    - preferred: native C++ explicit contact export (`WireWallContactExport`) with explicit `wallTriangleId` provenance.
    - fallback-only (`degraded`): listener/contact-node/surface/centroid approximations.
    - deterministic continuity path: when a step has active constraints but no fresh contact records, previously explicit triangle ids can be reused per force-sample index (`cached_contact_triangle_id`) with geometric sanity checks.
  - ContactListener remains optional (`STEVE_FORCE_ENABLE_CONTACT_LISTENER=1`) and is never required for the validated path.
  - Requires explicit `force_extraction.units` (`length_unit`, `mass_unit`, `time_unit`).
  - Uses calibration cache (`force_extraction.calibration.cache_path`).
  - Enforces quality tiers:
    - `validated`: active-constraint step with:
      - `collision.constraintProjection`
      - explicit association method `native_contact_export_triangle_id`
      - 100% force-contribution coverage
      - stable ordering + segment-sum consistency
    - `degraded`: association/source fallback (stored, but not used for safety scoring)
    - `unavailable`: missing or inconsistent telemetry
  - If `required: true`, only `validated` quality is accepted.
- `required: true`: fail fast if force telemetry is missing/inconsistent.
- `required: false`: continue run and exclude safety term from score (weights renormalized).

Wall-force telemetry is written sparsely per step and is fully reconstructable:
- `wall_segment_count`
- `wall_active_segment_ids`
- `wall_active_segment_force_vectors`
- `wall_total_force_vector` / `wall_total_force_norm`
- `wall_contact_detected`
- `wall_force_channel` / `wall_force_norm_sum`
- `wall_force_quality_tier`
- `wall_force_association_method`
- `wall_force_association_explicit_ratio`
- `wall_force_association_coverage`
- `wall_force_association_explicit_force_coverage`
- `wall_force_association_ordering_stable`
- `wall_force_active_constraint_step`
- `wall_force_gap_active_projected_count`
- `wall_force_gap_explicit_mapped_count`
- `wall_force_gap_unmapped_count`
- `wall_force_gap_class_counts`
- `wall_force_gap_dominant_class`
- `wall_force_gap_contact_mode`
- `wall_native_contact_export_available`
- `wall_native_contact_export_source`
- `wall_native_contact_export_status`
- `wall_native_contact_export_explicit_coverage`
- `wall_field_force_norm` (auxiliary only; can stay zero for static/collision-only walls)
- `wall_active_segment_force_vectors_N`
- `wall_total_force_vector_N` / `wall_total_force_norm_N`
- `wall_peak_segment_force_norm` (max local segment force over all steps)
- `wall_peak_segment_force_norm_N` (same in validated SI mode)
- `wall_peak_segment_force_step`, `wall_peak_segment_force_segment_id`, `wall_peak_segment_force_time_s`
- `force_units`
- `force_validation_status` / `force_validation_error`
  - `pass:validated`: fully validated quality
  - `pass:degraded`: force available but degraded association/source quality
  - `fail:unavailable`: force telemetry unusable for validated safety metrics

Passive mode requires the native plugin:

```bash
# Ubuntu: sudo apt install libboost-all-dev cmake build-essential
scripts/build_wall_force_monitor.sh
export STEVE_WALL_FORCE_MONITOR_PLUGIN=/absolute/path/to/libSofaWireForceMonitor.so
```

For validated SI mode, create/update calibration cache:

```bash
steve-force-calibrate --config docs/eval_example.yml
```

Calibration runs a deterministic **two-probe reference suite** (same seed/config twice) and only passes when:
- both probes reach `force_validation_status=pass:validated`
- association method is explicit native export (`native_contact_export_triangle_id`)
- explicit coverage is ~1.0 in active-constraint steps
- SI force traces are reproducible within tolerance (`tolerance_profile`)

Calibration is fingerprinted per physics/tool/checkpoint combination.
For comparison runs with multiple candidates, ensure each candidate has a
passing cache entry (run calibration once per candidate/tool variant).

Deterministic reference scenes (minimal correctness probe):

```bash
steve-force-reference --tool TestModel_StandardJ035/StandardJ035_PTFE
```

This runs two fixed SOFA scenes with the strict validated mode unchanged:
- `point_vs_triangle` (planar wall)
- `line_vs_triangle` (tube wall with offset shaft contact)

Artifacts are written to `results/force_reference_scene/<timestamp>/`:
- `reference_scene_report.json` (suite verdict + per-case failure reasons)
- `<case>_run_a.csv`, `<case>_run_b.csv` (step-wise telemetry traces)

Interpretation:
- `pass_validated_suite=1`: strict `validated` is reproducibly reachable in the minimal scenes.
- `pass_validated_suite=0` + `external_limit=1`: reproducible external/runtime limit detected
  (e.g. no explicit native contact records despite active constraints).

Gap diagnostics are also exported per trial:
- `force_gap_reports/<agent>_trialXXXX_seedY.json`
- `force_gap_reports/<agent>_trialXXXX_seedY.csv`

## Scoring + reports

Each trial gets a scalar `score` plus components:
- `score_success`, `score_efficiency`, `score_safety`, `score_smoothness`

These are aggregated into per-agent ranking reports:
- `report.md` (human-readable)
- `report.csv` (spreadsheet)
- `report.json` (UI / programmatic consumption)

The scoring is intentionally **relative**; force scales are configurable because their absolute units can depend on scene/unit conventions.
Safety scoring uses `wall_total_force_norm` (wall-segment resultant), while legacy wire-force signals remain in outputs for diagnostics.

## UI integration

You can call the pipeline directly:

```python
from steve_recommender.evaluation import load_config, run_evaluation

cfg = load_config("docs/eval_example.yml")
run_dir = run_evaluation(cfg)
```

Or build a config dict in the UI and convert it:

```python
from steve_recommender.evaluation.config import config_from_dict
cfg = config_from_dict(raw_dict)
```

Comparison core (registry-aware):

```python
from steve_recommender.services import comparison_from_dict, run_comparison

cfg = comparison_from_dict(raw_dict)
run_dir = run_comparison(cfg)
```

### UI checkpoint discovery

The UI "Evaluate" tab can auto-discover trained checkpoints if you start training via
`scripts/train_paper.sh`. That wrapper prints a small structured header into
`results/paper_runs/nohup_*.log` which includes `tool=...` and the checkpoint folder path.

The Evaluate tab uses this to:
- filter the tool list to "tools with checkpoints"
- auto-fill a recommended checkpoint (prefers `best_checkpoint.everl`)

## Visual playback (Sofa/Pygame viewer)

You can **watch a trained agent act in the simulator** on a chosen aortic arch via
the small `play_agent.py` helper and an interactive Sofa/Pygame window.

### 1) CLI usage

Example with your Standard‑J wire and the latest checkpoint:

```bash
conda activate master-project
source scripts/sofa_env.sh

python -m steve_recommender.evaluation.play_agent \
  --tool TestModel_StandardJ035/StandardJ035_PTFE \
  --checkpoint results/paper_runs/2025-12-17_111848_paper_standardj/checkpoints/checkpoint8525996.everl \
  --arch-record arch_009999 \
  --device cuda \
  --episodes 1 \
  --max-episode-steps 500
```

Notes:
- `--tool` must match the tool reference used during training (`model/wire`).
- `--checkpoint` is any EveRL checkpoint (`.everl`) from `results/paper_runs/...`.
- `--arch-record` selects an existing aortic arch from the dataset generated by
  `generate_aortic_arch_dataset.py` (e.g. `arch_000123`, `arch_009999`, …).
- `--device` controls the policy device (`cuda` or `cpu`); SOFA itself remains CPU‑based.

The script builds an intervention for the chosen anatomy and wire, wraps it in
`BenchEnv(visualisation=True)` and opens a Sofa/Pygame window.

### 2) UI usage

In the **Evaluate** tab:
- Select an anatomy from the aortic arch dataset.
- Add at least one agent row (tool + checkpoint).
- Click **“Play first agent (Sofa window)”**.

The UI spawns `play_agent.py` in a separate process with the selected anatomy and the
first configured agent; logs appear in the Evaluate tab while the Sofa window is open.

### 3) Camera controls in the Sofa window

The viewer uses an interactive wrapper around `eve.visualisation.SofaPygame`
(`InteractiveSofaPygame`) with simple keyboard controls:

- Arrow left / right: rotate LAO / RAO (around z‑axis)
- Arrow up / down: rotate CRA / CAU (around x‑axis)
- `W`: zoom in
- `S`: zoom out

Tips:
- Hold keys to change the view continuously while the agent runs.
- Close the window or press `Ctrl+C` in the terminal to stop playback early.

## More docs

See `docs/evaluation_pipeline.md` for full details on outputs and file formats.
