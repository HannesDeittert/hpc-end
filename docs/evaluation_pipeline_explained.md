# Evaluation Pipeline (Agent Benchmark)

This pipeline benchmarks multiple trained RL agents (checkpoint + tool pairs) on a fixed anatomy + start/target setup. It lives in repo-local code (outside upstream stEVE repos) so it can be run from CLI, UI, or imported from Python.

Sources: `steve_recommender/evaluation/pipeline.py`, `steve_recommender/evaluation/scoring.py`, `steve_recommender/evaluation/config.py`, `steve_recommender/evaluation/intervention_factory.py`, `steve_recommender/evaluation/info_collectors.py`.

---

## How to run it

### CLI

```bash
conda activate master-project
source scripts/sofa_env.sh

steve-eval --config docs/eval_example.yml
```

### Python

```python
from steve_recommender.evaluation import load_config, run_evaluation

cfg = load_config("docs/eval_example.yml")
run_dir = run_evaluation(cfg)
```

---

## Config overview (YAML)

Top-level keys (see `steve_recommender/evaluation/config.py`):

- `name`: run name (used in output folder)
- `agents`: list of `{name, tool, checkpoint}`
- `n_trials`, `base_seed`
- `max_episode_steps`
- `policy_device`: `"cuda"` or `"cpu"`
- `use_non_mp_sim`: `true` to access SOFA forces
- `output_root`
- `anatomy`: currently only `type: aortic_arch`
- `scoring`: optional scoring config (default `mode: default_v1`)

Anatomy details (aortic arch):

- `arch_type`, `seed`, optional `rotation_yzx_deg`, `scaling_xyzd`, `omit_axis`
- target definition: `target_mode: branch_end`, `target_branches`, `target_threshold_mm`
- sim/fluoro: `image_frequency_hz`, `image_rot_zx_deg`, `friction`

---

## Pipeline flow (what happens when you run it)

### 1) Setup + outputs

- stEVE packages are imported via the `steve_recommender.steve_adapter` module.
- Creates `results/eval_runs/<timestamp>_<name>/`.
- Writes `config.json` and initializes `summary.csv`.

### 2) For each agent (checkpoint + tool)

- Build an aortic-arch intervention via `build_aortic_arch_intervention(...)`.
  - Uses `image_frequency_hz` to compute `action_dt_s = 1 / image_frequency_hz`.
- Switch to non-MP sim if `use_non_mp_sim: true` (required to read SOFA forces).
- Build an evaluation `eve.Env`:
  - Observation: tracking + target + last_action
  - Reward: target reached + path delta + small step penalty
  - Terminal: target reached
  - Truncation: max steps, vessel end, sim error
  - Info collectors: target reached, path ratio, steps, avg speed, trajectory length,
    plus `TipStateInfo` (tip pose/rotation) and `SofaWallForceInfo` (approx forces)
- Load an eval-only agent from checkpoint:
  - `eve_rl.agent.single.SingleEvalOnly.from_checkpoint(...)`

### 3) For each trial (same seeds for all agents)

- Seeds are `base_seed + i` (same list for each agent for fair comparison).
- Run one eval episode: `eval_agent.evaluate(episodes=1, seeds=[seed])`.
- Extract per-step series from `episode.infos`.
- Compute derived metrics:
  - `steps_to_success` is 1-based (first step where `success=True`).
  - `tip_vel[t] = (tip_pos[t] - tip_pos[t-1]) / action_dt_s`
  - Aggregate wall-force stats via max/mean over the episode.
- Score the trial (see formula below).
- Write:
  - `trials/*.npz` (time series)
  - `summary.csv` row with scalars + score + `npz_path`.

### 4) Report aggregation

- Groups rows by `(agent, tool, checkpoint)`.
- Computes per-agent means/stdevs.
- Writes:
  - `report.json` (UI consumption)
  - `report.csv` (spreadsheet)
  - `report.md` (human-readable ranking)

---

## Scoring formula (default_v1)

Implemented in `steve_recommender/evaluation/scoring.py`.
All components are clipped to `[0, 1]`, higher is better.

### Components

```text
success:
  s_success = 1.0 if success else 0.0

efficiency:
  s_eff = 0 if steps_to_success is None or max_episode_steps <= 0
  s_eff = 1 - (steps_to_success - 1) / max_episode_steps

safety:
  force_max = max(wall_wire_force_norm_max, wall_collision_force_norm_max)
  lcp_max   = wall_lcp_max_abs_max
  safety_force = exp(-force_max / force_scale)
  safety_lcp   = exp(-lcp_max / lcp_scale)
  s_safety = safety_force * safety_lcp

smoothness:
  s_smooth = exp(-tip_speed_max_mm_s / speed_scale_mm_s)
```

Notes:

- `steps_to_success` is 1-based.
- NaNs in force/speed inputs are treated as `0` (no penalty).
- `tip_speed_max_mm_s` comes from finite-difference velocity in tracking coords.

### Final score

```text
score = w_success * s_success
      + w_efficiency * s_eff
      + w_safety * s_safety
      + w_smoothness * s_smooth

if normalize_weights:
  score /= (w_success + w_efficiency + w_safety + w_smoothness)
```

Defaults (unless overridden in YAML):

- `w_success=2.0`, `w_efficiency=1.0`, `w_safety=1.0`, `w_smoothness=0.25`
- `force_scale=1.0`, `lcp_scale=1.0`, `speed_scale_mm_s=50.0`

---

## Outputs (what you get)

Run directory: `results/eval_runs/<timestamp>_<name>/`

- `summary.csv`: one row per `(agent, trial)`
- `trials/*.npz`: time series (tip_pos, tip_vel, actions, rewards, forces, etc.)
- `report.{md,csv,json}`: aggregated per-agent metrics + ranking

---

## Important force note

Wall/contact forces are best-effort proxies:

- `wall_lcp_*` from LCP constraint solver
- `wall_wire_force_norm`, `wall_collision_force_norm` from SOFA DOFs

They are only accessible if `use_non_mp_sim: true`; otherwise values are `NaN` or `0`.
