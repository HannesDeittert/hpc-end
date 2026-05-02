# Surface-Normal Force vs. Vector Magnitude — Design Note

## Why the distinction matters

Contact forces between the guidewire and vessel wall have two orthogonal components:

- **Normal component** (`force_normal`): the force projected onto the outward surface normal of the vessel wall. This is the compressive force pressing the wire radially into the wall. It is directly proportional to wall stress and is the quantity most relevant to vascular injury.
- **Tangential component** (`force_tangential`): the component parallel to the wall surface. This is friction. It opposes sliding motion and does not contribute to radial wall stress.

The **vector magnitude** (`force_magnitude`, also called the "norm" in older code) combines both:

```
‖F‖ = sqrt(F_normal² + F_tangential²)
```

Using the magnitude overestimates the safety-relevant force by including friction. For a wire sliding against the wall under heavy friction, the magnitude may be large while the normal component — and therefore the true injury risk — is small.

## Canonical safety quantity

The canonical safety quantity used throughout the train_v2 reward and eval_v2 scoring is:

```
wire_force_normal_trial_max_N
```

This is the maximum surface-normal contact force component (in Newtons) across all guidewire collision DOFs, over the entire trial duration.

## Field naming convention

| Prefix | Scope |
|--------|-------|
| `wire_` | Whole guidewire (all collision DOFs) |
| `tip_` | Distal tip region only (collision DOFs within `tip_threshold_mm` arc length from the distal end) |

| Quantity | Meaning |
|----------|---------|
| `force_normal` | Surface-normal component of contact force |
| `force_magnitude` | Vector norm (normal + tangential combined) |

| Reduction suffix | Meaning |
|-----------------|---------|
| `_instant` | Value at the current simulation step |
| `_trial_max` | Maximum over all steps of the trial so far |
| `_trial_mean` | Mean over all steps of the trial so far |

All values are in Newtons (suffix `_N`).

## Where to find the implementation

- **Force extraction**: `steve_recommender/eval_v2/force_telemetry.py` — `EvalV2ForceTelemetryCollector.build_summary()` computes both normal and magnitude reductions from per-DOF contact force vectors.
- **Training reward**: `steve_recommender/train_v2/rewards/force.py` — `ForceComponent` applies `alpha × wire_force_normal_instant_N` each step and `beta × wire_force_normal_trial_max_N` at terminal/truncation.
- **Eval scoring**: `steve_recommender/eval_v2/scoring.py` — `score_safety()` and `force_within_safety_threshold()` consume `wire_force_normal_trial_max_N` from `ForceTelemetrySummary`.
- **Telemetry model**: `steve_recommender/eval_v2/models.py` — `ForceTelemetrySummary` dataclass with all four reductions.
- **Train-side telemetry**: `steve_recommender/train_v2/telemetry/force_runtime.py` — `ForceRuntime.sample_step()` returns `ForceRewardSample` with `wire_force_normal_instant_N`, `wire_force_normal_trial_max_N`, `tip_force_normal_instant_N`, `tip_force_normal_trial_max_N`.
