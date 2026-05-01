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
| `--eval-episodes N` | `all eval seeds` | Episodes per evaluation; omit to run one episode per eval seed. |
| `--eval-seeds SEEDS` | ArchVar seed list | Comma-separated integer seeds for deterministic eval (defaults to the ArchVar training seed schedule). |
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
| `--reward-profile PROFILE` | `default` | `default` or `default_plus_normal_force_penalty`. |
| `--force-alpha F` | `0.1` | Per-step penalty weight: `alpha × wire_force_normal_instant_N`. Active when profile is `default_plus_normal_force_penalty`. |
| `--force-beta F` | `1.0` | Terminal/truncation penalty weight: `beta × wire_force_normal_trial_max_N`. Active when profile is `default_plus_normal_force_penalty`. |
| `--force-region REGION` | `whole_wire` | `whole_wire` or `tip`. Selects which region's normal force is used in the penalty. |

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

### `default_plus_normal_force_penalty`

Adds per-step and terminal penalties based on the surface-normal contact force component:

```
R = R_base - alpha × wire_force_normal_instant_N          (every step)

At terminal/truncation:
R_terminal = -beta × wire_force_normal_trial_max_N
```

where:
- `wire_force_normal_instant_N` is the instantaneous whole-wire surface-normal contact force at the current step (in Newtons)
- `wire_force_normal_trial_max_N` is the trial-maximum surface-normal force seen so far

Default parameters:
- `alpha = 0.1`
- `beta = 1.0`
- `region = whole_wire`

With `--force-region tip`, the tip-region quantities (`tip_force_normal_instant_N`, `tip_force_normal_trial_max_N`) are used instead.

---

## Force telemetry integration

The force penalty reads wall contact forces directly from the SOFA physics simulation at each environment step. The pipeline is:

```
SOFA LCP solver
    └─ EvalV2ForceTelemetryCollector   (steve_recommender/eval_v2/force_telemetry.py)
           └─ ForceRuntime             (train_v2/telemetry/force_runtime.py)
                  └─ ForceComponent.step()
```

**Per environment step:**

1. `ForceComponent.step()` calls `ForceRuntime.sample_step(intervention, step_index)`.
2. `ForceRuntime` calls `EvalV2ForceTelemetryCollector.capture_step()`, which reads `constraintForces` from the SOFA LCP object and projects them to per-DOF world-space forces.
3. `sample_step()` returns a `ForceRewardSample` with four fields: `wire_force_normal_instant_N`, `wire_force_normal_trial_max_N`, `tip_force_normal_instant_N`, `tip_force_normal_trial_max_N`.
4. `ForceComponent` reads the appropriate field (based on `--force-region`) and computes the penalty.

**Why surface-normal force?**

The surface-normal component of contact force (force projected onto the vessel wall's outward normal) is the mechanically relevant quantity for vessel injury risk. The vector magnitude includes tangential friction components that do not contribute to radial wall stress. Using the normal component gives a more physically meaningful safety signal.

**Field naming convention:**

| Prefix | Scope |
|--------|-------|
| `wire_` | Whole guidewire |
| `tip_` | Distal tip region only |

| Quantity | Meaning |
|----------|---------|
| `force_normal` | Surface-normal contact force component |
| `force_magnitude` | Vector norm (all components) |

| Reduction | Meaning |
|-----------|---------|
| `_instant` | Current step value |
| `_trial_max` | Trial maximum so far |
| `_trial_mean` | Trial mean so far |

All values in Newtons (`_N` suffix).

The canonical safety quantity used by both the training reward and eval scoring is `wire_force_normal_trial_max_N`.

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

### Full run with normal-force penalty (GPU)

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
  --reward-profile default_plus_normal_force_penalty \
  --force-alpha 0.1 \
  --force-beta 1.0 \
  --force-region whole_wire \
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
