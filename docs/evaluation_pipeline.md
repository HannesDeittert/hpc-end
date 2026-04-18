# Evaluation Pipeline (agent benchmark)

This repo contains a **repo-local evaluation pipeline** (outside upstream stEVE repos) to benchmark **multiple trained agents** on a **fixed anatomy + start/target** setup.
It is also used by the registry-aware comparison entrypoint (`steve-compare`).

It is meant to be:
- runnable from the **terminal** (YAML config),
- callable from your **UI code** (Python API),
- and compatible with the **stEVE / SOFA** simulator used for training.

## What it evaluates

For each configured agent (checkpoint + tool), the pipeline runs **N trials** with deterministic seeds and records:
- success / reached target
- steps to reach target
- episode reward
- simulated time (`steps * dt`)
- wall-clock time
- per-step **tip kinematics** (position + derived velocity)
- per-step **actions / rewards / terminal / truncation**
- per-step **wall-contact force telemetry**:
  - per-contact force vectors
  - per-wire-segment force vectors (full wire)
  - total wire-wall force vector/norm

### Deterministic vs stochastic behavior

- Training includes exploration noise / stochastic action sampling.
- Evaluation is deterministic by default (`stochastic_eval: false`).
- Set `stochastic_eval: true` if you explicitly want sampling noise during evaluation runs.

Currently supported anatomy:
- `aortic_arch` (stEVE built-in generator)
  - start: the default `InsertionPoint`
  - target: `BranchEnd` on selected branches (endpoints)

## Files

- Code: `steve_recommender/evaluation/`
- CLI entrypoint: `steve-eval`
- Example config: `docs/eval_example.yml`

## Prerequisites (same as training)

You need:
- your conda env (e.g. `master-project`)
- SOFA installed with SofaPython3 + BeamAdapter
- the repo package installed (`pip install -e .`) to get `steve-eval`

In a fresh shell:

```bash
conda activate master-project
source scripts/sofa_env.sh
```

## Run from terminal

1) Edit the example config and set the checkpoint path(s):

- `docs/eval_example.yml`

2) Run:

```bash
steve-eval --config docs/eval_example.yml
```

Alternative (candidate resolver + same evaluator core):

```bash
steve-compare --config docs/compare_example.yml
```

List available registry refs:

```bash
steve-compare --list-agent-refs
```

Force Sofa window rendering from CLI:

```bash
steve-compare --config docs/compare_example.yml --visualize --visualize-trials-per-agent 1
steve-eval --config docs/eval_example.yml --visualize --visualize-trials-per-agent 1
```

Override force extraction mode from CLI:

```bash
steve-compare --config docs/compare_example.yml --force-mode passive --force-required
steve-eval --config docs/eval_example.yml --force-mode intrusive_lcp --force-optional
```

Outputs are written to `results/eval_runs/<timestamp>_<name>/`.

## Config format

Top-level keys:
- `name`: string (used in output folder name)
- `agents`: list of `{name, tool, checkpoint}`
- `n_trials`: trials per agent
- `base_seed`: base seed (trial seeds are `base_seed + i`)
- `seeds`: optional explicit seed list (overrides `n_trials/base_seed`)
- `max_episode_steps`: truncation limit
- `policy_device`: `"cuda"` or `"cpu"` (policy inference device)
- `use_non_mp_sim`: `true` to run SOFA in-process (required for force extraction)
- `stochastic_eval`: policy sampling during eval (default: `false`, deterministic)
- `visualize`: show Sofa window during eval
- `visualize_trials_per_agent`: render only first N trials per agent
- `force_extraction`: wall-force extraction settings (mode + availability policy)
- `output_root`: output directory root
- `anatomy`: currently only `type: aortic_arch` is supported

Anatomy (`aortic_arch`) keys:
- `arch_type`: `"I" | "II" | "III"` (stEVE arch types)
- `seed`: generator seed
- `target_mode`: currently only `"branch_end"`
- `target_branches`: e.g. `["lcca"]`
- `target_threshold_mm`: distance threshold for success
- `image_frequency_hz`: affects `dt` and velocity scaling
- `image_rot_zx_deg`: `[rot_z_deg, rot_x_deg]` for the virtual C-arm
- `friction`: SOFA friction coefficient

Force extraction keys:
- `mode`: `"passive"` (preferred) or `"intrusive_lcp"` (explicit fallback)
- `required`: fail run if force telemetry is unavailable/inconsistent
- `contact_epsilon`: threshold used by consistency checks
- `plugin_path`: optional explicit path to `libSofaWireForceMonitor.so`

## Output format

Each evaluation run folder contains:
- `config.json`: the resolved config used for the run
- `summary.csv`: one row per `(agent, trial)`
- `trials/*.npz`: per-trial time series (compressed NumPy)
- `report.json`: per-agent aggregated scores (for UI)
- `report.csv`: per-agent aggregated scores (for spreadsheets)
- `report.md`: per-agent aggregated scores (human-readable)

### `summary.csv`

Semicolon-delimited, one row per trial. Important columns:
- `success`: `0/1` (reached target at end of episode)
- `steps_total`: number of env steps executed
- `steps_to_success`: first step index where success became true (blank if never)
- `episode_reward`: total reward
- `wall_time_s`: wall-clock runtime for the episode
- `sim_time_s`: simulated time (`steps_total * dt`)
- `wall_total_force_norm_*`: aggregate full-wire force norms
- `wall_contact_count_max`: max contact count over the trial
- `wall_force_available`: `0/1` telemetry availability
- `wall_force_source`, `wall_force_error`: provenance / diagnostics
- `score`: computed scalar score for this trial
- `score_*`: score components (success/efficiency/safety/smoothness)

### `trials/*.npz`

Keys (subset):
- `tip_pos3d`: `(T,3)` tip positions (tracking coordinates)
- `tip_vel3d`: `(T,3)` finite-difference velocities (`(pos[t]-pos[t-1])/dt`)
- `inserted_length`: `(T,)`
- `rotation`: `(T,)`
- `success`: `(T,)` bool per-step success flag
- `path_ratio`: `(T,)` progress metric from the pathfinder
- `actions`: `(T, A)` actions executed (A depends on env)
- `rewards`: `(T,)`
- `terminals`, `truncations`: `(T,)` bool
- `wall_contact_force_vectors`: `(T, N_i, 3)` per-step contact vectors (object array if variable `N_i`)
- `wall_contact_segment_indices`: `(T, N_i)` mapped segment indices
- `wall_segment_force_vectors`: `(T, S, 3)` per-step full-wire segment vectors
- `wall_total_force_vector`: `(T,3)` per-step summed wire-wall vector
- `wall_total_force_norm`: `(T,)` norm of total force
- `wall_force_available`, `wall_contact_count`
- `action_dt_s`: scalar `dt` used for velocity

## Wall Forces (SOFA)

Modes:
- `passive` (preferred): uses `WireWallForceMonitor` plugin and does not toggle LCP build mode.
- `intrusive_lcp`: explicit fallback mode; enables LCP force telemetry and may affect trajectories.

Implementation note:
- passive mode is attached dynamically at runtime from the recommender package; no stEVE source patch is required.

Safety behavior:
- `required: true` -> fail fast if telemetry is missing/inconsistent.
- `required: false` -> mark force unavailable; safety term is excluded from score (weights renormalized).

Requirements:
- `use_non_mp_sim: true`
- plugin built and discoverable for passive mode:
  - install build deps once (Ubuntu): `sudo apt install libboost-all-dev cmake build-essential`
  - run `scripts/build_wall_force_monitor.sh`
  - export `STEVE_WALL_FORCE_MONITOR_PLUGIN=/.../libSofaWireForceMonitor.so`

## Scoring (per trial + per agent)

The evaluator computes a **score per trial** and aggregates it into a **per-agent report**.

By default (`mode: default_v1`) the score combines:
- `success` (reached target)
- `efficiency` (how quickly the target is reached)
- `safety` (penalty for large contact/wall force scalars)
- `smoothness` (penalty for high peak tip speed)

If force telemetry is unavailable and `force_extraction.required: false`, the
`safety` term is removed and the remaining weights are renormalized.

The default score is intended for **relative comparisons**. Because force units can depend on scene/unit conventions, the force scales are configurable.

### Scoring config (YAML)

Add a `scoring:` block:

```yaml
scoring:
  mode: default_v1
  w_success: 2.0
  w_efficiency: 1.0
  w_safety: 1.0
  w_smoothness: 0.25
  normalize_weights: true

  # Scale factors for exp(-x/scale) penalties:
  force_scale: 1.0
  lcp_scale: 1.0
  speed_scale_mm_s: 50.0
```

## UI integration

The pipeline is importable:

```python
from steve_recommender.evaluation import load_config, run_evaluation

cfg = load_config("docs/eval_example.yml")
run_dir = run_evaluation(cfg)
```

If your UI already builds a dict, you can also use:

```python
from steve_recommender.evaluation.config import config_from_dict
cfg = config_from_dict(raw_dict)
```
