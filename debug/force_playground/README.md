# Force Playground v1

Isolated, oracle-based force-debug playground for SOFA contact telemetry.

This module is intentionally separate from the productive training/evaluation/comparison pipeline.
It reuses stable force telemetry primitives (especially `SofaWallForceInfo`) but does not patch or modify production code paths.

## Goals

1. Reproduce contact forces in controlled scenes (`plane_wall`, `tube_wall`) with reproducible settings.
2. Log full per-step and per-triangle telemetry (`all triangles, all steps`).
3. Keep force metrics separated:
   - `norm(sum(v_i))`
   - `sum(norm(v_i))`
   - `peak_triangle_force`
   - `total_force_vector`
4. Keep normal/tangential decomposition explicit (`F_n`, `F_t`, `|F_n|`, `|F_t|`, `|F_t|/|F_n|`).
5. Keep `lambda` and `lambda/dt` visible in step output, persisted files, and live plot.
6. Add at least one physical plausibility oracle in v1:
   - `normal_force_balance` (plane wall + rigid probe + open-loop-force mode).

## Quick Start

```bash
cd ~/dev/Uni/master-project
source ~/miniconda3/etc/profile.d/conda.sh
conda activate master-project

export SOFA_ROOT=/home/hannes-deittert/opt/Sofa_v23.06.00_Linux/SOFA_v23.06.00_Linux
source scripts/sofa_env.sh
export MPLCONFIGDIR=/tmp/mplconfig && mkdir -p "$MPLCONFIGDIR"
export STEVE_WALL_FORCE_MONITOR_PLUGIN="$PWD/native/sofa_wire_force_monitor/build/libSofaWireForceMonitor.so"

python -m debug.force_playground.cli \
  --scene plane_wall \
  --probe rigid_probe \
  --mode open_loop_force \
  --steps 300 \
  --plot --interactive \
  --show-sofa \
  --camera-preset plane_oblique \
  --require-oracle-applicable
```

## CLI

Entry point:

```bash
python -m debug.force_playground.cli [flags]
```

Main flags:

- `--scene {plane_wall,tube_wall}`
- `--probe {rigid_probe,guidewire}`
- `--mode {displacement,open_loop_force}`
- `--tool-ref <model/wire>`: used when `probe=guidewire` (default `ArchVarJShaped/JShaped_Default`)
- `--steps <int>`
- `--seed <int>`
- `--units <length,mass,time>` (example: `mm,kg,s`)
- `--friction <float>`
- `--output-root <path>`
- `--run-name <name>`
- `--plot / --no-plot`
- `--show-sofa / --no-show-sofa`
- `--camera-preset {auto,plane_front,plane_oblique,tube_oblique}`
- `--interactive / --non-interactive`
- `--save-plot-snapshots`

Control flags:

- `--insert-action <float>`
- `--rotate-action <float>`
- `--open-loop-force-n <float>`
- `--open-loop-force-node-index <int>` (`-1` targets the distal tip node; default)
- `--open-loop-insert-action <float>`
- `--action-step-delta <float>`
- `--force-step-delta-n <float>`

Oracle flags:

- `--oracle / --no-oracle`
- `--oracle-rel-tol <float>` (default `0.10`)
- `--oracle-abs-tol-n <float>`
- `--oracle-near-zero-ref-n <float>`
- `--oracle-warmup-steps <int>`
- `--oracle-window-steps <int>`
- `--require-oracle-applicable`
- `--require-oracle-pass`

Mesh flags:

- `--plane-width-mm`, `--plane-height-mm`
- `--tube-radius-mm`, `--tube-length-mm`, `--tube-segments`, `--tube-rings`

## Hotkeys (Interactive Plot)

When running with `--plot --interactive`:

- `space` / `n`: execute one step
- `c`: toggle run/pause
- `q` / `Esc`: stop run
- `p`: save plot snapshot (when `--save-plot-snapshots`)
- `up` / `down`: adjust displacement insert action
- `+` / `-`: adjust open-loop force target

Backend note:

- The playground now avoids Qt backends by default to prevent `xcb` aborts in mixed conda/OpenCV setups.
- It prefers `TkAgg`, then falls back to `Agg`.
- If fallback `Agg` is used, interactive hotkeys are disabled automatically (run continues non-interactive).
- You can force a backend via:
  - `STEVE_FORCE_PLAYGROUND_MPL_BACKEND=TkAgg`
  - or `STEVE_FORCE_PLAYGROUND_MPL_BACKEND=Agg`

SOFA scene note:

- Use `--show-sofa` to open the SOFA/Pygame scene window in parallel to telemetry plotting.
- If you only want scene rendering, run with `--show-sofa --no-plot`.
- `--non-interactive` runs immediately and may close the scene window quickly after the last step.
- For VSCode integrated terminal, use `--plot --interactive --show-sofa` so both windows stay event-responsive.

## Data Outputs

Each run writes to:

- `results/force_playground/<timestamp>_<run_name>/config.json`
- `results/force_playground/<timestamp>_<run_name>/steps.csv`
- `results/force_playground/<timestamp>_<run_name>/steps.jsonl`
- `results/force_playground/<timestamp>_<run_name>/triangle_forces.csv`
- `results/force_playground/<timestamp>_<run_name>/oracle_report.json`
- `results/force_playground/<timestamp>_<run_name>/summary.md`
- optional: `results/force_playground/<timestamp>_<run_name>/plots/*.png`

### `steps.csv` / `steps.jsonl`

Per-step aggregated telemetry, including:

- `norm_sum_vector` = `norm(sum(v_i))`
- `sum_norm` = `sum(norm(v_i))`
- `peak_triangle_force`
- `total_force_vector`
- `sum_abs_fn`, `sum_abs_ft`
- `lambda_abs_sum`, `lambda_dt_abs_sum`, `lambda_active_rows_count`
- validation stages:
  - `si_converted`
  - `explicit_association`
  - `internal_validated`
  - `oracle_physical_pass`

`steps.jsonl` additionally stores complete step records, including `lambda_values` and `lambda_dt_values` arrays.

### `triangle_forces.csv`

All triangles for all steps (v1 default), with:

- force vector + norm
- triangle normal
- normal/tangential decomposition (`fn_*`, `ft_*`)
- `ft_over_fn`
- `active`

## Oracle Semantics

`normal_force_balance` compares measured normal wall reaction (`F_meas_n`) to reference normal load (`F_ref_n`) for compatible v1 setup:

- `scene=plane_wall`
- `probe=rigid_probe`
- `mode=open_loop_force`

Tolerances:

- relative tolerance: `--oracle-rel-tol` (default `10%`)
- absolute tolerance: `--oracle-abs-tol-n`
- near-zero reference threshold: `--oracle-near-zero-ref-n`

Always logged:

- `oracle_f_ref_n`
- `oracle_f_meas_n`
- `oracle_abs_error`
- `oracle_rel_error`
- pass/fail + reason

## Four Validation Stages (Important)

The playground keeps validation stages separated by design:

1. `si_converted`
   - explicit unit metadata was applied (`mm/kg/s` etc.)
2. `explicit_association`
   - force-to-wall-triangle association is explicit and fully covered
3. `internal_validated`
   - internal collector quality gate (`quality_tier=validated`) passed
4. `oracle_physical_pass`
   - physical plausibility check passed for the configured oracle

`internal_validated` is **not** the same as physical correctness.

## Typical Failure Patterns

- Very low forces with contact present:
  - check `friction`, units, `lambda` vs `lambda/dt`, and `command_apply_status`
- `explicit_association=false` with contact:
  - inspect `association_method`, `gap_*`, and `native_contact_export_status`
- Oracle not applicable:
  - expected unless running `plane_wall + rigid_probe + open_loop_force`
- Oracle fail with high absolute error:
  - check force command direction, units, and whether contact is sustained in the oracle window
- SOFA window does not open / `video system not initialized`:
  - this usually means the viewer was not initialized correctly by pygame/OpenGL;
  - playground now logs a viewer warning and continues telemetry;
  - re-run with `--show-sofa --no-plot` to isolate display stack issues.
- SOFA window opens but stays black:
  - prefer `--camera-preset plane_oblique` for `plane_wall`;
  - verify startup logs print camera `position` and `look_at`;
  - if still black, run `--show-sofa --no-plot` to rule out UI contention.

## Canonical Commands

Visual debug run (dual-window, manual stepping):

```bash
python -m debug.force_playground.cli \
  --scene plane_wall \
  --probe rigid_probe \
  --mode displacement \
  --steps 300 \
  --plot --interactive \
  --show-sofa \
  --camera-preset plane_oblique \
  --run-name fp_plane_rigid_disp_vscode
```

Oracle validation run (must be applicable and pass):

```bash
python -m debug.force_playground.cli \
  --scene plane_wall \
  --probe rigid_probe \
  --mode open_loop_force \
  --steps 300 \
  --open-loop-force-n 0.10 \
  --show-sofa --plot --interactive \
  --camera-preset plane_oblique \
  --require-oracle-applicable \
  --require-oracle-pass \
  --run-name fp_plane_rigid_oracle
```

Maintainer-style projection audit (`H^T * lambda/dt` vs monitor force):

```bash
python -m debug.force_playground.constraint_audit \
  --scene plane_wall \
  --probe rigid_probe \
  --mode displacement \
  --steps 120 \
  --insert-action 0.04 \
  --run-name constraint_audit_plane
```

Output:
- `results/force_playground/<timestamp>_<run_name>/constraint_audit.csv`
- per-step gap between projected force and `wall_total_force_vector_N`

Minimal maintainer-style script (small, pythonic, easy to debug):

```bash
python -m debug.force_playground.simple_contact_forces \
  --scene plane_wall \
  --probe rigid_probe \
  --mode open_loop_force \
  --steps 120 \
  --insert-action 1.0 \
  --open-loop-force-n 1.0 \
  --run-name simple_ht_lambda
```

Output:
- `results/force_playground/<timestamp>_<run_name>/simple_contact_forces.csv`
- per-step:
  - active lambda rows from `LCP.constraintForces`
  - projected wall force from `H^T*(lambda/dt)`
  - `wall_total_force_vector_N`
  - `gap_N` between both force vectors

Single-wire keyboard debug scene (single_jwire-style + tip-force CSV):

```bash
python -m debug.force_playground.single_jwire_tip_force_debug \
  --tool-ref ArchVarJShaped/JShaped_Default \
  --arch-type I \
  --force-mode passive \
  --tip-index-mode nearest \
  --max-steps 1000 \
  --run-name single_jwire_tip_force_dev
```

Output:
- `results/force_playground/<timestamp>_<run_name>/single_jwire_tip_force_debug.csv`
- per-step:
  - keyboard action (`insert`, `rotate`)
  - mapped tip-force vector (`tip_force_*_N`) and norm
  - mapping metadata (`tip_force_wire_index`, `tip_force_source`)
  - wall-force/contact telemetry (`wall_total_force_norm_N`, contact fields)

Single-wire basic scene (single_jwire-style, minimal tip-force printout):

```bash
python -m debug.force_playground.single_jwire_basic_scene \
  --tool-ref ArchVarJShaped/JShaped_Default \
  --arch-type I \
  --tip-node-count 3 \
  --max-steps 1000
```

Per step, this prints a single readable line:
- tip position from last wire node (`position[-1]`)
- tip force proxy `F_tip->wall ~= -sum(lambda_i)` over the last `tip_node_count` nodes
- fallback to wire nodal `force` if `lambda` is not exposed by the scene object

Stable-contact debug variant (preload/hold/ramp, tighter contact thresholds):

```bash
python -m debug.force_playground.constraint_audit \
  --scene plane_wall \
  --probe rigid_probe \
  --mode open_loop_force \
  --steps 180 \
  --image-frequency-hz 50 \
  --alarm-distance 0.05 \
  --contact-distance 0.01 \
  --friction 0.0 \
  --phase-preload-steps 40 \
  --phase-hold-steps 20 \
  --phase-ramp-steps 40 \
  --phase-measure-steps 80 \
  --preload-insert-action 0.04 \
  --hold-insert-action 0.0 \
  --open-loop-force-n 1.0 \
  --ramp-force-start-n 0.0 \
  --lambda-active-eps 1e-6 \
  --min-consecutive-active-steps 10 \
  --run-name constraint_audit_stable_contact
```

Key diagnostics in `constraint_audit.csv`:
- `lambda_abs_max`
- `active_step` (strict activity flag)
- `consecutive_active_steps`
- `gap_N`

## Isolation Guarantee

This module lives under `debug/force_playground/` and is invoked via:

```bash
python -m debug.force_playground.cli
```

It does not change training/eval/comparison behavior.
It only reads stable telemetry from existing collectors and writes standalone debug artifacts.
