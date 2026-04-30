# Phase T — `train_v2` local training package

## Goal

Create a self-contained training package under `steve_recommender/train_v2` that:

- supports CLI-driven training runs
- preserves the practical features of `scripts/archvar_train_tool_resume.py`
  - resumable `.everl` checkpoints
  - replay-buffer sidecar save/load
  - tool selection from the wire registry
- allows reward composition to be changed locally, including force-aware rewards
- imports only from:
  - `third_party/*`
  - `steve_recommender.eval_v2.*`
  - `data/wire_registry` through `eval_v2` discovery/runtime code
- does not depend on our legacy `steve_recommender.training`, `steve_recommender.rl`,
  `steve_recommender.bench`, or other non-`_v2` internal packages

## Working principle

`train_v2` is a local product layer around stEVE/stEVE_rl:

1. resolve anatomy and wire/tool via `eval_v2` discovery/runtime helpers
2. build a real stEVE intervention locally
3. build the training/eval environment locally with injectable rewards
4. provide a doctor/preflight command that validates runtime, files, and a real env boot
5. provide resumable training with replay-buffer persistence without modifying third-party code

Revision note:
- 2026-04-30: simplified the intended public path back toward the working
  ArchVar training script shape. `anatomy` is no longer required for the
  default ArchVarRandom path, custom tools can be supplied by module/class,
  and the package now prefers the proven bench intervention/agent path where
  that reduces moving parts.

## Task list

### T.0 — Package and handoff scaffolding
- Status: `completed`
- Deliverables:
  - package layout
  - `README.md`
  - this `plan.md`

### T.1 — Config + CLI + doctor/preflight
- Status: `completed`
- Deliverables:
  - `train_v2.cli`
  - `train_v2.config`
  - `train_v2.doctor.*`
  - tests for config parsing, path checks, and doctor reporting

### T.2 — Runtime + env factory
- Status: `completed`
- Deliverables:
  - intervention builder using `eval_v2` anatomy/wire/runtime pieces
  - local env factory using third-party stEVE primitives
  - no dependency on upstream `util.env.BenchEnv`

### T.3 — Reward architecture
- Status: `completed`
- Deliverables:
  - reward spec/factory
  - default ArchVar reward composition
  - force-penalty extension point backed by `eval_v2.force_telemetry`

### T.4 — Resumable agent + replay buffer
- Status: `completed`
- Deliverables:
  - local copy of resumable replay-buffer and checkpoint logic
  - no dependency on `steve_recommender.training.*`

### T.5 — Training command
- Status: `in_progress`
- Deliverables:
  - `python -m steve_recommender.train_v2 train ...`
  - `python -m steve_recommender.train_v2 doctor ...`
  - preflight integration before training

### T.5.1 — Runtime hang investigation
- Status: `next`
- Goal:
  - make one real `train_v2 train ...` tiny smoke run complete
  - identify whether the hang is in env stepping, agent worker lifecycle, Runner.training_run, evaluation, or agent.close
- Deliverables:
  - a deterministic reproduction command
  - phase-level runtime logging around training execution
  - watchdog/traceback dumps for hangs
  - one opt-in SOFA smoke test that fails with the exact stuck phase
  - fix for the first real training-loop blocker

### T.6 — Verification and docs
- Status: `in_progress`
- Deliverables:
  - focused pytest coverage
  - `ruff` and `black --check`
  - `README.md` usage examples

## Deferred

- cluster submission wrappers
- TensorBoard and richer run dashboards
- multiple anatomy families beyond the initial aortic-arch path
- policy registry/agent registration parity with old training entrypoints
- broad migration of legacy training scripts to call `train_v2`

## Notes for later agents

- The critical seam is reward injection. Do not route through upstream
  `third_party/stEVE_training/training_scripts/util/env.py`; that hardcodes reward.
- Reuse `eval_v2` for:
  - anatomy discovery
  - wire/device construction
  - force telemetry runtime
- Keep `doctor` first-class. It is part of the package contract, not a debug utility.

## Current progress log

- 2026-04-29: Created `train_v2` package skeleton and established the phased implementation plan.
- 2026-04-29: Implemented validated config models, `doctor` CLI, and environment/runtime preflight checks with focused test coverage.
- 2026-04-29: Implemented local reward construction and force-penalty integration through `eval_v2.force_telemetry` without routing through the upstream hardcoded reward env builder.
- 2026-04-29: Implemented local resumable replay-buffer and checkpoint helpers, plus local agent construction for synchronous SAC training.
- 2026-04-29: Added the `train` CLI path, resume target logic, and execution scaffolding; verified `doctor` and `train --preflight-only` against the real environment.
- 2026-04-29: Ran a tiny real `train` smoke and fixed two integration mistakes in the execution path (`build_intervention` keyword use and `build_env` return shape); the remaining blocker is that the actual training loop still hangs after agent/worker initialization.
- 2026-04-30: Simplified the public ArchVar path to remove the required anatomy input, added module/class custom tool support, and verified `doctor` plus `train --preflight-only` with `steve_recommender.bench.custom_tools_amplatz_gentle_simple:JShapedAmplatzSuperStiffGentleSimple`.
- 2026-04-30: Retried a tiny real training smoke on the simplified ArchVar path; the job still stalls after trainer/worker initialization, so the remaining work is a focused runtime investigation rather than more package scaffolding.

## What remains

- Complete one real short `train_v2 train ...` smoke run and tighten any runtime issues that only show up under an actual training loop.
- Investigate why the real synchronous training loop hangs after worker/trainer initialization on the first tiny smoke run.
- Expand test coverage around execution/agent factory once the first real training smoke is green.
- Extend README usage examples with concrete `train` and `doctor` command lines once the first real training smoke is confirmed.
