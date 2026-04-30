# train_v2

SAC-based RL training for stEVE interventions with pluggable reward functions, including optional wall-force penalties.

## Prerequisites

```bash
conda activate master-project
source scripts/sofa_env.sh   # required for SOFA to find its plugins
```

## Quick start

```bash
python -m steve_recommender.train_v2 train \
  --name my_run \
  --tool jshaped_default \
  --trainer-device cpu \
  --worker-count 2 \
  --heatup-steps 5 --training-steps 10 --eval-every 5 \
  --explore-episodes-between-updates 2 \
  --train-max-steps 5 --eval-max-steps 5 \
  --no-preflight \
  --output-root /tmp/tv2_results
```

---

## Commands

### `train` — run a training job

```
python -m steve_recommender.train_v2 train [OPTIONS]
```

### `doctor` — preflight check without training

```
python -m steve_recommender.train_v2 doctor [OPTIONS]
```

---

## CLI flags — `train`

### Required

| Flag | Description |
|------|-------------|
| `--name NAME` | Run name. Used in output file paths. |
| `--tool TOOL_REF` | Wire tool reference key (e.g. `jshaped_default`). |

### Anatomy / wire selection

| Flag | Default | Description |
|------|---------|-------------|
| `--anatomy ANATOMY_ID` | `None` | Anatomy record ID from the file-based registry. When omitted the default ArchVar anatomy is used. |
| `--tool-module MODULE` | `None` | Python module path for a custom tool class (e.g. `steve_recommender.bench.custom_tools_amplatz_tight_j_simple`). |
| `--tool-class CLASS` | `None` | Class name inside `--tool-module` (e.g. `JShapedAmplatzSuperStiffTightJSimple`). |

### Compute devices

| Flag | Default | Description |
|------|---------|-------------|
| `--trainer-device DEVICE` | `cpu` | PyTorch device for the SAC policy/critic (e.g. `cuda:0`). |
| `--worker-device DEVICE` | `cpu` | Device for worker inference. |
| `--replay-device DEVICE` | `cpu` | Device for replay-buffer samples. |

### Training schedule

| Flag | Default | Description |
|------|---------|-------------|
| `--heatup-steps N` | `500000` | Random exploration steps before SAC updates start. |
| `--training-steps N` | `20000000` | Total explore steps target (including heatup). |
| `--eval-every N` | `250000` | Evaluate every N explore steps. |
| `--eval-episodes N` | `1` | Episodes per evaluation. |
| `--eval-seeds SEEDS` | `none` | Comma-separated integer seeds for deterministic eval (e.g. `42,43`). |
| `--explore-episodes-between-updates N` | `100` | Explore episodes collected between each SAC update batch. |
| `--update-per-explore-step RATIO` | `0.05` | SAC gradient updates per explore step. |
| `--consecutive-action-steps N` | `1` | Repeat each action N times before observing. |
| `--worker-count N` | `2` | Number of parallel SOFA worker processes. |
| `--train-max-steps N` | `None` | Max steps per training episode (unlimited if omitted). |
| `--eval-max-steps N` | `None` | Max steps per eval episode (unlimited if omitted). |

### Neural network architecture

| Flag | Default | Description |
|------|---------|-------------|
| `--hidden N [N ...]` | `400 400 400` | Hidden layer sizes for policy and critic MLPs. |
| `--embedder-nodes N` | `900` | Output size of the observation embedder. |
| `--embedder-layers N` | `1` | Number of layers in the observation embedder. |

### SAC hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--learning-rate LR` | `0.000322` | Adam learning rate for all networks. |
| `--gamma GAMMA` | `0.99` | Discount factor. |
| `--reward-scaling SCALE` | `1.0` | Scalar multiplier applied to all rewards. |
| `--batch-size N` | `32` | Replay buffer sample batch size. |
| `--replay-buffer-size N` | `10000` | Maximum number of transitions stored. |
| `--lr-end-factor F` | `0.15` | Final LR = initial LR × factor (linear schedule). |
| `--lr-linear-end-steps N` | `6000000` | Steps over which LR decays to `lr-end-factor`. |

### Reward

| Flag | Default | Description |
|------|---------|-------------|
| `--reward-profile PROFILE` | `default` | `default`, `default_plus_force_penalty`, or `default_plus_excess_force_penalty`. |
| `--force-penalty-factor F` | `0.0` | Penalty weight for wall-force reward. Required and must be `> 0` when profile is `default_plus_force_penalty`. |
| `--force-threshold N` | `0.85` | Newton threshold for `default_plus_excess_force_penalty`. Force below this value is ignored. |
| `--force-divisor D` | `1000.0` | Divisor for the excess-force penalty magnitude. |
| `--force-tip-only` | off | Penalize only catheter-tip contact force instead of total wall force. |

### Resume

| Flag | Default | Description |
|------|---------|-------------|
| `--resume-from PATH` | `None` | Path to a checkpoint folder to resume from. |
| `--resume-skip-heatup` | off | Skip heatup phase when resuming. |
| `--resume-replay-buffer-from PATH` | `None` | Load a saved replay buffer `.everl` file on resume. |

### Replay buffer persistence

| Flag | Default | Description |
|------|---------|-------------|
| `--save-latest-replay-buffer` | off | Save the replay buffer to `checkpoints/latest_replay_buffer.everl` after each checkpoint. |

### Preflight

| Flag | Default | Description |
|------|---------|-------------|
| `--no-preflight` | off | Skip the doctor preflight check before training. |
| `--preflight-only` | off | Run preflight then exit without training. |

### Eval

| Flag | Default | Description |
|------|---------|-------------|
| `--stochastic-eval` | off | Use stochastic policy during evaluation (default is deterministic). |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `--output-root PATH` | `results/train_v2` | Root folder for all run outputs. |

---

## CLI flags — `doctor`

Shares most flags with `train`. Doctor-specific flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--strict` | off | Exit non-zero on warnings, not just errors. |
| `--no-boot-env` | off | Skip booting a full SOFA environment during the check. |

---

## Output structure

```
<output-root>/
  YYYY-MM-DD_HHMMSS_<name>.csv        # results file (one row per eval)
  YYYY-MM-DD_HHMMSS_<name>/
    main.log                           # training log
    env_train.yml                      # saved env config
    env_eval.yml
    checkpoints/
      *.everl                          # agent checkpoints
      latest_replay_buffer.everl       # replay buffer (if --save-latest-replay-buffer)
```

---

## Reward profiles

The reward always starts from the same three base components:

```
R_base = TargetReached × 1.0
       + PathLengthDelta × 0.001
       + Step × -0.005
```

Profiles then optionally add one force-derived penalty term.

### `default`

Standard ArchVar reward:

```
R = R_base
```

### `default_plus_force_penalty`

Adds a linear wall-force penalty on top of the default reward:

```
R = R_base - force_penalty_factor × wall_force_N
```

where `wall_force_N` is the peak wall contact force in Newtons measured during that simulation step.

### `default_plus_excess_force_penalty`

Adds a threshold-gated penalty that only activates above one force threshold:

```
R = R_base + R_force

R_force = -((wall_force_N - force_threshold_N) / force_divisor)   if wall_force_N > force_threshold_N
          0.0                                                     otherwise
```

Default parameters:

- `force_threshold_N = 0.85`
- `force_divisor = 1000.0`

This is the softer option: small contact forces are not penalized, and only the excess above the threshold reduces reward.

---

## Force telemetry integration

The force penalty reads wall contact forces directly from the SOFA physics simulation at each environment step. The pipeline is:

```
SOFA LCP solver
    └─ EvalV2ForceTelemetryCollector   (steve_recommender/eval_v2/force_telemetry.py)
           └─ ForceRuntime             (train_v2/telemetry/force_runtime.py)
                  └─ ForcePenaltyReward.step()
```

**Per environment step:**

1. `ForcePenaltyReward.step()` calls `ForceRuntime.sample_step(intervention, step_index)`.
2. `ForceRuntime` calls `EvalV2ForceTelemetryCollector.capture_step()`, which reads `constraintForces` from the SOFA LCP object and projects them to per-DOF world-space forces.
3. `build_summary()` returns a `ForceTelemetrySummary` with `total_force_norm_max_newton` (peak total wall force across all contacts this step) and `tip_force_total_norm_N` (summed tip-region contact force).
4. `ForcePenaltyReward` reads the appropriate field and computes `reward = -factor × magnitude`.

**Force telemetry modes** (configured via `RewardSpec.force_telemetry_mode`, default `constraint_projected_si_validated`):

| Mode | Description |
|------|-------------|
| `constraint_projected_si_validated` | Projects SOFA constraint rows to world-space per-DOF force vectors using LCP impulses ÷ dt, with SI unit conversion (mm/kg/s → N). Most accurate. |
| `intrusive_lcp` | Reads raw LCP `constraintForces` directly with no spatial projection or unit conversion. |
| `passive` | Reads the optional `WireWallForceMonitor` plugin if present; silently returns zero if the plugin is not available. |

When using `constraint_projected_si_validated` the unit conversion assumes SOFA scene units of mm / kg / s, which is the standard stEVE scene configuration.

**Tip-only mode** (`--force-tip-only`): classifies collision DOFs within 3 mm arc length from the distal wire end as tip contacts, and uses their summed force norm instead of the total wall force.

---

## Examples

### Minimal smoke test (CPU, no force)

```bash
source scripts/sofa_env.sh
python -m steve_recommender.train_v2 train \
  --name smoke \
  --tool jshaped_default \
  --trainer-device cpu \
  --worker-count 2 \
  --heatup-steps 5 --training-steps 10 --eval-every 5 \
  --explore-episodes-between-updates 2 \
  --train-max-steps 5 --eval-max-steps 5 \
  --no-preflight \
  --output-root /tmp/tv2_smoke
```

### Full run with linear force penalty (GPU, custom tool, replay buffer)

```bash
source scripts/sofa_env.sh

RUN_NAME="amplatz_force_01"
RESULTS="/tmp/tv2_results"

python -m steve_recommender.train_v2 train \
  --name "$RUN_NAME" \
  --tool jshaped_default \
  --tool-module steve_recommender.bench.custom_tools_amplatz_tight_j_simple \
  --tool-class JShapedAmplatzSuperStiffTightJSimple \
  --trainer-device cuda:0 \
  --worker-count 2 \
  --learning-rate 0.0003218 \
  --hidden 400 400 400 \
  --embedder-nodes 1 \
  --embedder-layers 1 \
  --heatup-steps 100 \
  --training-steps 1000 \
  --eval-every 500 \
  --explore-episodes-between-updates 2 \
  --train-max-steps 20 \
  --eval-max-steps 20 \
  --save-latest-replay-buffer \
  --reward-profile default_plus_force_penalty \
  --force-penalty-factor 0.01 \
  --output-root "$RESULTS"
```

### Full run with excess-force penalty

```bash
source scripts/sofa_env.sh

RUN_NAME="amplatz_force_threshold_01"
RESULTS="/tmp/tv2_results"

python -m steve_recommender.train_v2 train \
  --name "$RUN_NAME" \
  --tool jshaped_default \
  --tool-module steve_recommender.bench.custom_tools_amplatz_tight_j_simple \
  --tool-class JShapedAmplatzSuperStiffTightJSimple \
  --trainer-device cuda:0 \
  --worker-count 2 \
  --heatup-steps 100 \
  --training-steps 1000 \
  --eval-every 500 \
  --explore-episodes-between-updates 2 \
  --train-max-steps 20 \
  --eval-max-steps 20 \
  --reward-profile default_plus_excess_force_penalty \
  --force-threshold 0.85 \
  --force-divisor 1000 \
  --force-tip-only \
  --output-root "$RESULTS"
```

### Resume a previous run

```bash
python -m steve_recommender.train_v2 train \
  --name "$RUN_NAME" \
  --tool jshaped_default \
  --trainer-device cuda:0 \
  --worker-count 2 \
  --heatup-steps 100 \
  --training-steps 5000 \
  --eval-every 500 \
  --explore-episodes-between-updates 2 \
  --resume-from "$RESULTS/2026-04-30_120000_amplatz_force_01/checkpoints/checkpoint_1000" \
  --resume-skip-heatup \
  --resume-replay-buffer-from "$RESULTS/2026-04-30_120000_amplatz_force_01/checkpoints/latest_replay_buffer.everl" \
  --output-root "$RESULTS"
```

### Preflight check only

```bash
python -m steve_recommender.train_v2 doctor \
  --tool jshaped_default \
  --trainer-device cuda:0 \
  --output-root /tmp/tv2_results
```

### Monitor a running job

```bash
RESULTS=/tmp/tv2_results
RUN_DIR=$(find "$RESULTS" -maxdepth 1 -type d -name "*my_run*" \
  -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
tail -f "$RUN_DIR/main.log"
```
