# eval_v2

`eval_v2` is the clean-room evaluation pipeline for wire/policy comparison in the
`steve_recommender` project.

It has two user-facing entry points:

- a researcher CLI for scripted runs and reproducible benchmarks
- a PyQt GUI for interactive case selection, execution, and archive browsing

This document explains:

- the folder layout
- the main runtime architecture
- how seed management works
- how to launch and use the CLI
- how to launch and use the GUI
- current limitations and intended usage patterns

## 1. What eval_v2 is responsible for

At a high level, `eval_v2` does five things:

1. discovers anatomies, wires, policies, and targets
2. builds a fully resolved evaluation job
3. runs one or more trials for one or more candidates
4. computes trial telemetry and candidate summaries
5. writes reports that can later be reopened in the GUI archive

The design is intentionally split into small modules so that:

- the CLI and GUI both use the same service layer
- seed generation is centralized in the data model
- trial execution logic stays isolated from the UI
- report loading and archive browsing are deterministic and simple

## 2. Folder layout

The package lives in:

`steve_recommender/eval_v2/`

### Core runtime modules

- [models.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/models.py)
  - canonical dataclasses for jobs, scenarios, candidates, execution plans, telemetry, and reports
  - this is the most important file for understanding configuration

- [builders.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/builders.py)
  - converts eval_v2 models into actual stEVE / intervention runtime objects

- [runtime.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/runtime.py)
  - loads policies and prepares a `PreparedEvaluationRuntime`
  - owns the boundary from resolved model objects into executable runtime state

- [runner.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/runner.py)
  - executes exactly one trial
  - resets the environment, seeds randomness, steps the policy, captures telemetry, and returns one `TrialResult`

- [service.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/service.py)
  - service boundary used by both CLI and GUI
  - owns discovery, job execution, report writing, report loading, and archive summaries
  - contains the local serial and parallel evaluation runner

- [scoring.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/scoring.py)
  - converts trial telemetry into score breakdowns

- [force_telemetry.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/force_telemetry.py)
  - wall-force extraction during eval trials
  - handles passive monitor and fallback behavior

- [visualization.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/visualization.py)
  - visualization adapter used by the runner for Sofa/Pygame rendering

### Discovery and targeting modules

- [discovery.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/discovery.py)
  - anatomy and policy discovery helpers

- [target_discovery.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/target_discovery.py)
  - target mode support and branch/target discovery helpers

### CLI entry point

- [cli.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/cli.py)
  - `python -m steve_recommender.eval_v2.cli ...`
  - supports listing assets and running eval jobs from the shell

### GUI modules

- [app.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/app.py)
  - GUI launcher

- [ui_main.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/ui_main.py)
  - top-level window, navigation, and app shell

- [ui_controller.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/ui_controller.py)
  - UI-facing controller and worker-thread integration

- [ui_wizard.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/ui_wizard.py)
  - wizard shell and final `EvaluationJob` construction

- [ui_wizard_pages.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/ui_wizard_pages.py)
  - individual wizard pages:
    - anatomy selection
    - branch selection
    - target selection
    - wire selection
    - execution configuration
    - pipeline running
    - results

- [ui_home.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/ui_home.py)
  - GUI landing page

- [ui_archive.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/ui_archive.py)
  - historical report browsing

- [ui_view.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/ui_view.py)
  - older dashboard-style view; not the main eval_v2 entry flow

### Assets

- [assets/logo.svg](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/assets/logo.svg)

## 3. Runtime architecture

The normal execution flow is:

1. CLI or GUI creates an `EvaluationJob`
2. `DefaultEvaluationService` resolves all runtime inputs
3. `LocalEvaluationRunner` decides between serial and parallel execution
4. `prepare_evaluation_runtime(...)` builds the runtime for a candidate/scenario pair
5. `run_single_trial(...)` executes each trial
6. `summarize_trials(...)` aggregates results into candidate summaries
7. run artifacts are written:
   - `manifest.json`
   - `candidate_summaries.csv`
   - `candidate_summaries.json`
   - `trials.h5`
   - `report.md`

Important separation of concerns:

- `ExecutionPlan` decides *what seed schedule and execution mode should happen*
- `runner.py` decides *how one single trial is actually executed*
- `service.py` decides *how many runtimes/workers are used and how manifests / trial tables are written*

## 4. Seed management

This is the most important conceptual part for fair comparison.

### 4.1 Two independent seed channels

`eval_v2` now distinguishes between:

- **Environment seed**
  - controls trial setup randomness
  - for fixed anatomy/target this mainly means the initial simulation state, such as the guidewire starting rotation

- **Policy seed**
  - controls stochastic action sampling
  - only relevant when `policy_mode="stochastic"`

These schedules live in [ExecutionPlan](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/models.py:356).

### 4.2 Why the split exists

Without the split, one number would control both:

- the initial physical state
- the policy’s sampled actions

That makes some experiments impossible, for example:

- keep the initial wire orientation fixed
- run 100 stochastic policy rollouts from that same initial condition

With the split, that is now representable.

### 4.3 Default scheduling

If you pass a single base seed and `n` trials:

- environment seeds default to `base_seed + 0, +1, +2, ...`
- policy seeds default to `policy_base_seed + 0, +1, +2, ...`

Example:

```text
trial_count = 3
base_seed = 123
policy_base_seed = 1000
```

becomes:

```text
environment seeds: 123, 124, 125
policy seeds:      1000, 1001, 1002
```

### 4.4 Explicit override lists

The CLI can override either schedule with comma-separated explicit lists.

Example:

```text
--env-seeds 123,999,42
--policy-seeds 1000,2000,3000
```

These must match `--trial-count` exactly or eval_v2 raises a validation error.

### 4.5 Deterministic mode

In deterministic mode:

- actions are taken with `get_eval_action(...)`
- policy sampling randomness is not used
- `policy_seed` is effectively disabled

If `trial_count > 1`, the same environment seed sequence is given to every candidate.

Example:

```text
candidate A: env 123,124,125
candidate B: env 123,124,125
```

### 4.6 Stochastic mode

In stochastic mode:

- actions are taken with `get_exploration_action(...)`
- the policy seed controls the sampled action sequence

There are two environment behaviors:

#### `random_start`

The environment changes across trials.

Example:

```text
env seeds:    123,124,125
policy seeds: 1000,1001,1002
```

#### `fixed_start`

The environment is held constant across trials and only the policy seed changes.

Example:

```text
env seeds:    123,123,123
policy seeds: 1000,1001,1002
```

This is useful for experiments where you want to remove starting-state randomness
and isolate policy sampling variability.

### 4.7 Fairness guarantee

Within one job, all candidates receive the same seed schedule.

That means if one job contains multiple candidates, then trial `k` for every
candidate uses the same environment seed and the same policy seed.

This is required for fair wire/policy comparison.

## 5. Environment setup

Typical project-local setup before running eval_v2:

```bash
cd /home/hannes-deittert/dev/Uni/master-project
source ~/miniconda3/etc/profile.d/conda.sh
conda activate master-project
source scripts/sofa_env.sh
```

The exact details depend on your local machine, but in practice you need:

- the correct conda environment
- SOFA environment variables set
- PyQt5 available for the GUI

## 6. CLI usage

Main entry point:

```bash
python -m steve_recommender.eval_v2.cli <command> ...
```

### 6.1 Discovery commands

List anatomies:

```bash
python -m steve_recommender.eval_v2.cli list-anatomies
```

List wires:

```bash
python -m steve_recommender.eval_v2.cli list-wires
python -m steve_recommender.eval_v2.cli list-wires --startable-only
```

List policies:

```bash
python -m steve_recommender.eval_v2.cli list-policies
python -m steve_recommender.eval_v2.cli list-policies --execution-wire steve_default/standard_j
```

List candidate options for one execution wire:

```bash
python -m steve_recommender.eval_v2.cli list-candidates \
  --execution-wire steve_default/standard_j
```

Exclude cross-wire candidates:

```bash
python -m steve_recommender.eval_v2.cli list-candidates \
  --execution-wire steve_default/standard_j \
  --no-cross-wire
```

List branches for one anatomy:

```bash
python -m steve_recommender.eval_v2.cli list-branches \
  --anatomy Tree_00
```

List supported target modes:

```bash
python -m steve_recommender.eval_v2.cli list-target-modes
```

### 6.2 Target modes

The CLI supports three target definitions:

- `branch_end`
- `branch_index`
- `manual`

#### `branch_end`

```bash
--target-mode branch_end
--target-branches lcca
```

Multi-branch is also supported:

```bash
--target-mode branch_end
--target-branches lcca,rcca
```

#### `branch_index`

```bash
--target-mode branch_index
--target-branch lcca
--target-index 7
```

#### `manual`

Repeat `--manual-target` for one or more vessel-space points:

```bash
--target-mode manual
--manual-target 1.0,2.0,3.0
--manual-target 4.0,5.0,6.0
```

### 6.3 Policy selection modes

Exactly one of the following must be used in `run`:

- `--candidate-name`
- `--policy-name`
- `--policy-agent-ref`
- `--policy-checkpoint`

Examples:

By candidate name:

```bash
--candidate-name candidate_from_registry
```

By policy name:

```bash
--policy-name policy_a
```

By stable agent ref:

```bash
--policy-agent-ref steve_default/standard_j:archvar_original_best
```

By explicit checkpoint:

```bash
--policy-checkpoint /path/to/policy.everl
--policy-label my_policy
--policy-trained-on-wire steve_default/standard_j
```

### 6.4 Basic single-run example

```bash
python -m steve_recommender.eval_v2.cli run \
  --job-name eval_demo \
  --anatomy Tree_00 \
  --execution-wire steve_default/standard_j \
  --policy-agent-ref steve_default/standard_j:archvar_original_best \
  --target-mode branch_end \
  --target-branches lcca
```

### 6.5 Deterministic comparison setup

Example: 3 deterministic trials with one shared environment seed sequence:

```bash
python -m steve_recommender.eval_v2.cli run \
  --job-name det_multi_start \
  --anatomy Tree_00 \
  --execution-wire steve_default/standard_j \
  --policy-agent-ref steve_default/standard_j:archvar_original_best \
  --target-mode branch_end \
  --target-branches lcca \
  --trial-count 3 \
  --base-seed 123 \
  --policy-mode deterministic \
  --policy-device cpu
```

This yields:

```text
env seeds: 123,124,125
policy seeds: disabled
```

### 6.6 Stochastic comparison with changing start state

```bash
python -m steve_recommender.eval_v2.cli run \
  --job-name stochastic_random_start \
  --anatomy Tree_00 \
  --execution-wire steve_default/standard_j \
  --policy-agent-ref steve_default/standard_j:archvar_original_best \
  --target-mode branch_end \
  --target-branches lcca \
  --trial-count 5 \
  --base-seed 123 \
  --policy-base-seed 1000 \
  --policy-mode stochastic \
  --stochastic-env-mode random_start \
  --policy-device cpu
```

This yields:

```text
env seeds:    123,124,125,126,127
policy seeds: 1000,1001,1002,1003,1004
```

### 6.7 Stochastic comparison with fixed start state

```bash
python -m steve_recommender.eval_v2.cli run \
  --job-name stochastic_fixed_start \
  --anatomy Tree_00 \
  --execution-wire steve_default/standard_j \
  --policy-agent-ref steve_default/standard_j:archvar_original_best \
  --target-mode branch_end \
  --target-branches lcca \
  --trial-count 5 \
  --base-seed 123 \
  --policy-base-seed 1000 \
  --policy-mode stochastic \
  --stochastic-env-mode fixed_start \
  --policy-device cpu
```

This yields:

```text
env seeds:    123,123,123,123,123
policy seeds: 1000,1001,1002,1003,1004
```

### 6.8 Explicit seed list override

Explicit environment list:

```bash
python -m steve_recommender.eval_v2.cli run \
  --job-name explicit_env_seeds \
  --anatomy Tree_00 \
  --execution-wire steve_default/standard_j \
  --policy-name policy_a \
  --target-mode branch_end \
  --target-branches lcca \
  --trial-count 3 \
  --env-seeds 123,999,42 \
  --policy-mode deterministic
```

Explicit stochastic environment and policy lists:

```bash
python -m steve_recommender.eval_v2.cli run \
  --job-name explicit_both_seeds \
  --anatomy Tree_00 \
  --execution-wire steve_default/standard_j \
  --policy-name policy_a \
  --target-mode branch_end \
  --target-branches lcca \
  --trial-count 3 \
  --env-seeds 123,123,123 \
  --policy-seeds 1000,1001,1002 \
  --policy-mode stochastic
```

If the explicit list length does not match `--trial-count`, the CLI fails fast.

### 6.9 Headless parallel execution

Headless eval can use multiple CPU worker processes:

```bash
python -m steve_recommender.eval_v2.cli run \
  --job-name parallel_eval \
  --anatomy Tree_00 \
  --execution-wire steve_default/standard_j \
  --policy-agent-ref steve_default/standard_j:archvar_original_best \
  --target-mode branch_end \
  --target-branches lcca \
  --trial-count 8 \
  --workers 4 \
  --policy-device cpu
```

Rules:

- `--workers` must be `>= 1`
- parallel workers are only supported for headless runs
- parallel workers require `--policy-device cpu`

### 6.10 Visualization

You can render a subset of trials:

```bash
python -m steve_recommender.eval_v2.cli run \
  --job-name visual_eval \
  --anatomy Tree_00 \
  --execution-wire steve_default/standard_j \
  --policy-agent-ref steve_default/standard_j:archvar_original_best \
  --target-mode branch_end \
  --target-branches lcca \
  --trial-count 3 \
  --visualize \
  --visualize-trials-per-candidate 1 \
  --workers 1 \
  --policy-device cpu
```

Rules:

- visualization is serial only
- `--visualize` cannot be combined with `--workers > 1`

### 6.11 Useful additional execution options

- `--threshold-mm`
  - target success threshold

- `--max-episode-steps`
  - rollout horizon

- `--friction`
  - scene friction

- `--tip-length-mm`
  - distal tip region length in millimeters
  - defaults to `DEFAULT_TIP_THRESHOLD_MM = 3.0`
  - a wire collision DOF is counted as tip when its arc-length distance from the
    distal end is less than or equal to this threshold

- `--force-max-N`
- `--force-score-c`
- `--force-score-p`
- `--force-score-k`
- `--force-score-F50-N`
  - parameters of the default nonlinear safety score

- `--score-lambda`
- `--score-beta`
- `--score-weight-safety`
- `--score-weight-efficiency`
  - parameters of candidate-level ranking aggregation

- `--jerk-scale-mm-s3`
  - optional jerk scale for stored smoothness scoring

- `--no-write-trace`
  - disables the default per-trial HDF5 trace writer
  - by default every trial writes one trace file under `traces/`

- `--write-diagnostics`
  - enables optional heavy diagnostic datasets inside each trial trace
  - defaults to off

- `--image-frequency-hz`
- `--image-rot-z-deg`
- `--image-rot-x-deg`
  - fluoroscopy configuration

- `--normalize-action` / `--no-normalize-action`
  - whether normalized policy outputs are mapped into physical action space

- `--stop-device-at-tree-end` / `--allow-device-past-tree-end`
  - intervention behavior at vessel-tree end

- `--output-root`
  - top-level directory for generated report folders

### 6.12 About “comparison” in the CLI

The current CLI `run` command builds **one candidate per invocation**.

So CLI comparison currently means:

- run candidate A with a chosen seed schedule
- run candidate B with the same seed schedule
- compare the resulting reports

This is still fair, because the seed logic is explicit and reproducible.

If you need a multi-candidate comparison in a single shell-level workflow, the
current practical solution is a wrapper script that invokes `eval_v2.cli run`
multiple times with the same `--trial-count`, `--base-seed`, `--env-seeds`,
`--policy-base-seed`, and `--policy-seeds` choices.

### 6.13 Replay viewer

Persisted Phase E trial traces can be replayed directly from the CLI:

```bash
python -m steve_recommender.eval_v2.viewer /path/to/trial_trace.h5
python -m steve_recommender.eval_v2.viewer /path/to/job_dir
python -m steve_recommender.eval_v2.viewer /path/to/trial_trace.h5 --start-step 47 --max-force-n 0.3
```

Notes:

- passing a job directory opens the first trace under `job_dir/traces/`
- `--start-step` selects the initial replay step
- by default the viewer auto-calibrates the triangle-force colormap to the
  trace's 95th-percentile contact magnitude
- `--max-force-n` overrides that auto-calibration with a fixed triangle-force
  colormap upper bound in Newtons
- the viewer reads the persisted trace through `TraceReader` and reuses the
  anatomy registry's `simulationmesh.obj` for the vessel mesh

## 7. Output artifacts

Each run writes a folder under the selected output root.

Typical contents:

- `manifest.json`
  - stable run-level entrypoint for archive loading and analysis
  - stores execution config, seed schedule, scoring spec, artifact paths, and candidate summaries

- `candidate_summaries.csv`
  - flat candidate summary table for quick inspection

- `candidate_summaries.json`
  - structured candidate summary table for archive/UI use

- `trials.h5`
  - column-oriented per-trial table
  - stores one row per trial with seeds, hard-validity flags, telemetry scalars,
    score components, final soft score, and trace path

- `report.md`
  - human-readable summary

- `traces/`
  - one HDF5 file per trial containing scene metadata, step-major wire state,
    actions, and split wire/triangle contact tables
  - the full on-disk schema is documented in
    [`docs/persistence_schema_v2.md`](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/docs/persistence_schema_v2.md)

- `meshes/`
  - one pre-written anatomy mesh HDF5 file per unique anatomy in the job
  - trial traces reference these files via relative `mesh_ref` paths

`manifest.json` is the archive entrypoint. The archive screen loads the manifest,
then reads `trials.h5` for full per-trial detail.

### 7.1 Default ranking logic

The default hard-validity flag is `valid_for_ranking`. A trial is valid only if:

- it succeeds
- `steps_to_success` is available and within `max_episode_steps`
- force telemetry is available for scoring
- `force_total_norm_max_N <= force_max_N`

The default soft per-trial score uses only:

- `score_safety`
- `score_efficiency`

with weights `0.5 / 0.5`.

`score_success` is still stored per trial but is not part of the default soft
score, because success is already handled by `valid_for_ranking` and the valid
rate penalty.

The candidate-level score is:

- `p_w = mean(valid_for_ranking)`
- `S_bar_w = mean(soft score over valid trials)`
- `sigma_S = sample std over valid trial soft scores`
- `Score_w = p_w^lambda * max(0, S_bar_w - beta * sigma_S)`

The exact scoring parameters used by one run are persisted in `manifest.json`
under `scoring_spec`.

Trace files can be replayed through the standalone viewer CLI or the inline GUI
replay panel described in sections 6.13 and 8.10.

## 8. GUI usage

Main GUI entry point:

```bash
python -m steve_recommender.eval_v2.app
```

### 8.1 What the GUI launches

The GUI app:

1. configures Qt plugin paths for the active conda environment
2. starts a PyQt5 application
3. opens the `ClinicalMainWindow`

The main window contains three main flows:

- **Home**
- **Start New Recommendation**
- **View Archive**

### 8.2 Home screen

The home screen is defined in [ui_home.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/ui_home.py).

It offers two main actions:

- `Start New Recommendation`
- `View Archive`

### 8.3 Wizard flow

The main evaluation flow is the wizard in [ui_wizard.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/ui_wizard.py).

Current steps:

1. **Anatomy selection**
2. **Branch selection**
3. **Target selection**
4. **Wire selection**
5. **Execution configuration**
6. **Pipeline running**
7. **Results**

### 8.4 What the wizard currently compares

The GUI is wire-centric.

For each selected wire, the wizard currently resolves the first local
non-cross-wire candidate returned by the controller:

- it does not currently expose a manual policy picker
- it does not currently expose explicit seed lists

So the GUI is best understood as:

- pick anatomy/target
- pick one or more execution wires
- evaluate the default candidate associated with each selected wire

### 8.5 Execution configuration page

The execution page currently exposes:

- **Simulation behavior**
  - deterministic
  - stochastic

- **Runs per wire**
  - now available for deterministic and stochastic runs

- **Stochastic environment mode**
  - visible only when:
    - stochastic mode is selected
    - runs per wire > 1
  - options:
    - `Random Start, Randomized Policy`
    - `Fixed Start, Randomized Policy`

- **Execution mode**
  - headless
  - live visualization

- **Visualized runs count**
  - visible when:
    - live visualization is enabled
    - runs per wire > 1

### 8.6 GUI worker selection

The GUI auto-selects workers conservatively for headless runs:

```text
worker_count = max(1, min(total_trials, cpu_count - 5))
```

where:

```text
total_trials = runs_per_wire * number_of_selected_wires
```

Live visualization always forces:

```text
worker_count = 1
```

### 8.7 GUI seed behavior

The GUI currently supports the **stochastic environment mode choice**, but it
does **not** yet expose manual seed entry fields.

So the GUI currently relies on `ExecutionPlan` defaults:

- environment base seed defaults to `123`
- policy base seed defaults to `1000`

This means:

- deterministic multi-trial GUI runs use environment seeds `123,124,125,...`
- stochastic random-start GUI runs use:
  - environment seeds `123,124,125,...`
  - policy seeds `1000,1001,1002,...`
- stochastic fixed-start GUI runs use:
  - environment seeds `123,123,123,...`
  - policy seeds `1000,1001,1002,...`

If you need exact manual seed lists, use the CLI.

The GUI uses `DEFAULT_TIP_THRESHOLD_MM = 3.0` for distal-tip force aggregation.
It does not expose this as a field yet; scripted runs can configure the same
collector setting with `--tip-length-mm`.

The GUI also uses `write_full_trace=True` by default through
`ForceTelemetrySpec`. It does not currently expose the trace-writing toggles;
scripted runs can disable traces with `--no-write-trace` or enable diagnostics
with `--write-diagnostics`.

### 8.8 Live visualization

When live visualization is selected:

- eval runs serially
- the running page shows the fluoroscopy stream
- progress and status updates are emitted from the worker thread

This is useful for debugging and qualitative inspection, not maximum throughput.

### 8.9 Results and archive

After a run, the GUI results page shows the generated output and summary.

The archive screen:

- scans historical report folders
- lists previous jobs
- lets you reopen stored reports from disk

The archive flow is implemented in [ui_archive.py](/home/hannes-deittert/dev/Uni/master-project/steve_recommender/eval_v2/ui_archive.py).

### 8.10 Replay viewer

The results page now includes an inline replay path for persisted trial traces.

For each wire, the per-trial table includes a `View` action when a
`trace_h5_path` is available for that trial. Clicking `View` opens an embedded
replay panel inside the same results flow:

- the vessel mesh is loaded from the anatomy registry's `simulationmesh.obj`
- the wire is rendered step-by-step from the persisted `wire_positions`
- contacted wall triangles are colored by persisted force magnitude in Newtons
- the user can scrub through steps with the slider and rotate/zoom the 3D scene

The same results-page replay panel is also used when historical reports are
reopened from the archive, so live runs and archived runs share the same viewer
path and the same Qt replay-control widget as the standalone CLI window.

## 9. Recommended experiment patterns

### One fixed deterministic debug rollout

Use:

- fixed anatomy
- fixed target
- `trial_count=1`
- deterministic policy

Good for:

- qualitative inspection
- visualization
- debugging one exact case

### Deterministic multi-start robustness

Use:

- fixed anatomy
- fixed target
- deterministic policy
- `trial_count > 1`
- auto-incrementing environment seeds

Good for:

- fair comparison across different starting wire orientations
- patient-specific robustness evaluation

### Stochastic fixed-start policy variability

Use:

- fixed anatomy
- fixed target
- stochastic policy
- `stochastic_environment_mode=fixed_start`

Good for:

- isolating action-sampling variability
- checking whether a policy is only unstable because of stochastic action selection

### Stochastic random-start full robustness

Use:

- fixed anatomy
- fixed target
- stochastic policy
- `stochastic_environment_mode=random_start`

Good for:

- a broader robustness estimate
- stress-testing both the initial condition and sampled policy behavior

## 10. Known limitations

Current limitations to keep in mind:

1. CLI `run` currently resolves one candidate per invocation.
2. GUI currently does not expose:
   - manual candidate selection
   - explicit environment seed lists
   - explicit policy seed lists
3. GUI currently uses default base seeds unless the code is extended.
4. Visualization cannot be parallelized.
5. Parallel headless execution currently requires CPU policy inference.
6. GUI does not expose `tip_threshold_mm`; defaults to 3.0 mm via the constant.
7. GUI does not expose trace-writing toggles; uses defaults via `ForceTelemetrySpec`.
8. The replay viewer currently focuses on triangle-force magnitude heatmaps and
   wire geometry; it does not yet render force-vector arrows or synchronized
   time-series plots.

## 11. Practical checklist

Before trusting a comparison, verify:

- anatomy is fixed as intended
- target mode and branch are fixed as intended
- both candidates use the same:
  - `trial_count`
  - environment seed schedule
  - policy seed schedule
  - `policy_mode`
  - `max_episode_steps`
  - scoring setup
  - visualization/headless setting if runtime effects matter

For reproducible scripted experiments, prefer the CLI.

For interactive case setup and archive browsing, prefer the GUI.
