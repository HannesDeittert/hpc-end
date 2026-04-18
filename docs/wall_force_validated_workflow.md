# Wall-Force Workflow (Strict Validated, Final)

This is the recommended end-to-end workflow for force telemetry in stEVE/SOFA with BeamAdapter.

## 1) Operational Workflow (Copy-Paste)

### 1.1 Environment setup (required)
```bash
cd ~/dev/Uni/master-project
source ~/miniconda3/etc/profile.d/conda.sh
conda activate master-project

export SOFA_ROOT=/home/hannes-deittert/opt/Sofa_v23.06.00_Linux/SOFA_v23.06.00_Linux
source scripts/sofa_env.sh
export MPLCONFIGDIR=/tmp/mplconfig && mkdir -p "$MPLCONFIGDIR"
```

### 1.2 Build and export native monitor plugin (required)
```bash
bash scripts/build_wall_force_monitor.sh
export STEVE_WALL_FORCE_MONITOR_PLUGIN="$PWD/native/sofa_wire_force_monitor/build/libSofaWireForceMonitor.so"
```

### 1.3 Run deterministic reference suite (required validation oracle)
```bash
python -m steve_recommender.evaluation.reference_scene_cli \
  --plugin-path "$STEVE_WALL_FORCE_MONITOR_PLUGIN" \
  --output-dir "results/force_reference_scene/manual_$(date +%Y%m%d_%H%M%S)"
```

Expected key line:
- `pass_validated_suite=1`

### 1.4 Run calibration for both compare fingerprints (required before strict compare)
```bash
python -m steve_recommender.evaluation.calibrate_cli --config docs/eval_hq_validated_smoke.yml
python -m steve_recommender.evaluation.calibrate_cli --config docs/eval_hq_validated_smoke_thirdparty.yml
```

Expected key lines per config:
- `passed=1 status=pass:validated`
- `reference_suite passed=1 external_limit=0`

### 1.5 Run strict compare (required execution path)
```bash
python -m steve_recommender.comparison.run_cli \
  --config docs/compare_hq_validated_smoke.yml \
  --force-mode constraint_projected_si_validated \
  --force-required \
  --force-plugin-path "$STEVE_WALL_FORCE_MONITOR_PLUGIN"
```

Expected:
- no `RuntimeError`
- final line with `done: results/eval_runs/<timestamp>_compare_hq_validated_smoke`

---

## 2) Result Interpretation

## 2.1 What to check first
- Calibration cache: `results/force_calibration/cache_hq_validated_smoke.json`
- Compare summary: `results/eval_runs/<run>/summary.csv`
- Compare trial arrays: `results/eval_runs/<run>/trials/*.npz`
- Compare report: `results/eval_runs/<run>/report.md`
- Reference suite report: `results/eval_runs/force_reference_scene/<suite_run>/reference_scene_report.json`

## 2.2 Fields proving strict validated path
In `summary.csv`, check:
- `wall_force_channel` → must be `collision.constraintProjection`
- `wall_force_association_method` → native explicit method (e.g. `native_contact_export_triangle_id`)
- `wall_force_quality_tier` → `validated`
- `force_validation_status` → `pass:validated`
- `wall_force_association_coverage` → should be `1` in strict validated cases
- `wall_force_association_explicit_force_coverage` → should be `1`
- `wall_native_contact_export_available` → `1`

## 2.3 Core force metrics
- `wall_force_max_N` (trial peak)
- `wall_total_force_norm_max_N`, `wall_total_force_norm_mean_N`
- `wall_peak_segment_force_norm_N`
- `wall_peak_segment_force_step`, `wall_peak_segment_force_segment_id`, `wall_peak_segment_force_time_s`
- Gap diagnostics:
  - `wall_force_gap_active_projected_count_sum`
  - `wall_force_gap_explicit_mapped_count_sum`
  - `wall_force_gap_unmapped_count_sum`
  - `wall_force_gap_dominant_class`

## 2.4 NPZ fields for offline analysis
Key arrays in `trials/*.npz`:
- `wall_active_segment_ids`
- `wall_active_segment_force_vectors_N`
- `wall_total_force_vector_N`
- `wall_total_force_norm_N`
- `wall_force_association_method`
- `wall_force_quality_tier`
- `force_validation_status`
- `force_units`

---

## 3) Runbook / Troubleshooting

### Problem: plugin build fails or plugin not loaded
- Meaning: native explicit contact export cannot run.
- Likely cause: `SOFA_ROOT` wrong, missing build deps, plugin path not exported.
- First check:
  - run `bash scripts/build_wall_force_monitor.sh`
  - verify file exists: `native/sofa_wire_force_monitor/build/libSofaWireForceMonitor.so`
  - ensure `STEVE_WALL_FORCE_MONITOR_PLUGIN` points to that file.

### Problem: calibration not passed
- Meaning: strict compare is blocked by design.
- Likely cause: reference suite failed or probe quality failed.
- First check:
  - `python -m steve_recommender.evaluation.calibrate_cli --config ...`
  - inspect `validation_error` in `results/force_calibration/cache_hq_validated_smoke.json`.

### Problem: compare fails with “requires a passing calibration cache entry”
- Meaning: no passing cache entry for the current fingerprint.
- Likely cause: calibration not run for this exact config/tool/checkpoint/plugin hash.
- First check:
  - rerun both calibration commands in section 1.4 with current env/plugin.

### Problem: reference suite fails
- Meaning: reproducibility/validated oracle did not pass.
- Likely cause: plugin not loaded, explicit mapping missing, or runtime-level instability.
- First check:
  - inspect `reference_scene_report.json`
  - verify `pass_validated_suite=1`, `external_limit_detected=0`.

### Problem: quality is `degraded` instead of `validated`
- Meaning: strict explicit coverage/ordering/consistency gate did not pass.
- Likely cause: fallback mapping path used, missing explicit IDs, ordering/integrity flags.
- First check in `summary.csv`:
  - `wall_force_association_method`
  - `wall_force_gap_unmapped_count_sum`
  - `wall_native_contact_export_status`
  - `wall_force_error`

### Problem: native contact association missing
- Meaning: explicit contact export unavailable/incomplete in relevant steps.
- Likely cause: plugin/runtime mismatch or contact export path inactive.
- First check:
  - `wall_native_contact_export_available`
  - `wall_native_contact_export_status`
  - `wall_force_gap_dominant_class`

### Problem: many files, unclear where to start
- Start order:
  1) `summary.csv`
  2) `report.md`
  3) `cache_hq_validated_smoke.json`
  4) `reference_scene_report.json`
  5) trial `*.npz` for deep analysis.

---

## 4) Final Status

- Force magnitude source is strict and unchanged: constraint projection (`J^T * λ / dt`).
- Contact-to-wall association is explicit via native export.
- `validated` remains strict (no fallback promotion).
- Calibration is green for the target configs and strict compare runs successfully.
- Reference suite is the reproducibility/validation oracle.
- Probe trace length differences can occur in full RL episodes, but no longer falsely invalidate calibration when the reference suite passes.

This keeps the workflow scientifically defensive and operationally usable: SI-converted values, explicit association, and validated gating are separate and visible in outputs.
