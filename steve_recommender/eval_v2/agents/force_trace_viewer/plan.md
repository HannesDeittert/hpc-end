# eval_v2 Force Trace Viewer Plan

This folder is the handoff location for the Phase F force-trace replay viewer
workstream.

## Goal

Build a replay viewer for Phase E trial traces so users can inspect a trial's
vessel mesh, wire state, and wall-force distribution over time through both:

- a standalone CLI viewer
- an inline GUI replay panel integrated into the existing results/archive flow

The data source is the closed Phase E HDF5 trace schema and the single
authoritative reader is `TraceReader` in `force_trace_persistence.py`.

## Working principle

The order matters:

1. Verify trace-to-mesh alignment on a real artifact before writing viewer code.
2. Lock a pure data loader layer around `TraceReader`.
3. Build one reusable renderer component used by both CLI and GUI hosts.
4. Add navigation controls after frame rendering is stable.
5. Add CLI and GUI hosts only after the shared viewer core is stable.
6. Update docs after both access paths are working.

## Phase F - Force trace replay viewer

### Task F.0 - Reconnaissance

Goal:

- verify that persisted triangle IDs index the real simulation mesh and locate
  the existing embedded GUI visualization stack to reuse

Status:

- Completed: `tools/recon_phase_f.py` opened a real benchmark trace at
  `/tmp/eval_v2_trace_bench/trace_on_50x100/traces/trial_archvar_original_best_123_none.h5`,
  confirmed `anatomy_id=Tree_00`, `mesh_ref=../meshes/anatomy_Tree_00.h5`,
  `triangle_id` range `17..1790`, `wire_positions` shape `(100, 241, 3)`, and
  loaded `data/anatomy_registry/anatomies/Tree_00/mesh/simulationmesh.obj` via
  PyVista with `n_cells=1890`, `n_points=947`, and matching bounds; the
  `triangle_id.max() < mesh.n_cells` assertion passed. Existing embedded
  PyVista/Qt reuse target is `Anatomy3DWidget` in `ui_wizard_pages.py`.

### Task F.1 - Trace data loader

Goal:

- add a pure data-layer wrapper around `TraceReader` that resolves the anatomy
  simulation mesh path and exposes per-step frame objects for rendering

Status:

- Completed: added `viewer.trace_data.TraceData` and `TraceFrame` as a lazy
  wrapper around `TraceReader`, resolving anatomy `simulationmesh.obj` paths,
  exposing per-step triangle-force magnitudes, and keeping random step access
  free of full-trace loads.

### Task F.2 - Renderer component

Goal:

- add a single reusable PyVista renderer component that can operate standalone
  or inside an injected Qt-owned plotter

Status:

- Completed: added a reusable PyVista `TraceRenderer` that loads the anatomy
  simulation mesh once, updates wire geometry and triangle-force overlays per
  frame, supports injected plotters, and stays testable in this environment via
  structural render-state assertions instead of framebuffer screenshots.

### Task F.3 - Slider and step navigation

Goal:

- add scrub slider and step navigation on top of the shared renderer

Status:

- Completed: added slider attachment, clamped step navigation, and +/-1 key
  callbacks on the shared renderer with callback-driven tests.

### Task F.4 - CLI entry point

Goal:

- add `python -m steve_recommender.eval_v2.viewer ...` as a first-class replay
  path for standalone trace inspection

Status:

- Completed: added `python -m steve_recommender.eval_v2.viewer ...` with trace
  path and job-directory resolution, start-step/max-force options, and a hidden
  off-screen mode used only by automated tests.

### Task F.5 - GUI integration

Goal:

- integrate the viewer inline into the existing results/archive GUI flow with a
  per-trial "View" action and embedded replay panel

Status:

- Completed: added an embedded `TraceViewerPanel` and integrated it into the
  existing `ResultsPage` trial-details flow so both live results and archived
  reports can open the same inline replay panel through a per-trial `View`
  action.

### Task F.6 - Documentation

Goal:

- document the replay viewer in the CLI, GUI, and output-artifact sections of
  the eval_v2 README

Status:

- Completed: documented the standalone replay viewer, replayable trace
  artifacts, and the inline GUI replay panel in the eval_v2 README.

## Phase F.7 - Viewer polish slice

### Task F.7.1 - Dynamic colormap calibration

Goal:

- replace the fixed renderer force ceiling with a trace-derived percentile scale
  so real 0.5-6 N traces do not render as saturated or uniformly dark

Status:

- Completed: `TraceData` now computes a 95th-percentile force scale across all
  persisted triangle-contact magnitudes with a documented zero-contact fallback,
  and the renderer uses that value unless the CLI/GUI caller explicitly
  overrides `max_force_n`.

### Task F.7.2 - Live max-force readout

Goal:

- show the current-frame maximum triangle-contact magnitude directly in the 3D
  view so scrubbing immediately reveals force peaks

Status:

- Completed: the renderer now keeps a top-right `Max force: X.XX N` overlay
  updated on every frame, including explicit `0.00 N` output on zero-contact
  frames.

### Task F.7.3 - Bottom-right scalar bar

Goal:

- move the scalar bar out of the full-width bottom strip into a compact
  bottom-right location

Status:

- Completed: the scalar bar now uses named bottom-right placement constants and
  is rendered there consistently for force overlays.

### Task F.7.4 - Shared Qt replay controls

Goal:

- replace the VTK slider/button path with one shared Qt control surface used by
  both the CLI window and the embedded GUI replay panel

Status:

- Completed: added `viewer.qt_replay_widget.QtTraceReplayWidget` with an
  integer-stepped `QSlider`, play/pause button, step label, time label, and
  timer-driven playback; both the standalone CLI host and `TraceViewerPanel`
  now use the same widget.

### Task F.7.5 - Remove orphaned VTK controls

Goal:

- delete dead renderer-level slider/play code after the Qt widget takes over

Status:

- Completed: removed the renderer's old VTK slider callback path and migrated
  coverage to Qt-widget tests.

## Current progress log

### Completed so far

- Created the Phase F plan and completed F.0 reconnaissance against a real
  Phase E benchmark trace; trace triangle IDs align with the anatomy
  `simulationmesh.obj`, and `ui_wizard_pages.py` contains the existing embedded
  PyVista/Qt integration pattern to reuse.
- Completed F.1 with seven loader tests covering mesh-path resolution, frame
  shapes, zero-contact steps, partial traces, triangle-force magnitude
  derivation, lazy random-step access, and missing-mesh errors.
- Completed F.2 and F.3 with eight renderer/navigation tests covering
  plotter reuse, per-step actor/scalar updates, zero-contact handling, slider
  behavior, clamping, and keyboard step callbacks without relying on unstable
  framebuffer screenshots in this environment.
- Completed F.4 through F.6 with standalone CLI replay, embedded GUI replay,
  README updates, and a full test-suite pass (`282 passed, 1 deselected`).
- Completed F.7.1 through F.7.5 with percentile-based force scaling, live
  per-frame max-force readout, compact scalar-bar placement, shared Qt replay
  controls for both hosts, and removal of the old VTK slider/play code.

### What remains

- Nothing planned remains in Phase F.

### Deferred

- Force vector arrows on triangles or wire DOFs.
- Multi-trial side-by-side replay.
- Dedicated time-series plots under the 3D replay.

## Closing retrospective

- Trace persistence from Phase E is now directly inspectable through one shared
  replay stack instead of opaque HDF5 files.
- The viewer reuses the existing embedded PyVista pattern where possible and
  falls back to structural render-state tests where framebuffer screenshots are
  not reliable in this environment.
- Live results and archived reports share the same inline replay component, so
  there is one viewer path to maintain.

## Notes for later agents

- `TraceReader` is the only supported trace reader; do not fork the schema
  access path.
- The trace persists `mesh_ref` to the Phase E mesh copy, but F.0 confirmed the
  viewer should read the anatomy registry's `simulationmesh.obj` directly.
- Reuse the existing embedded PyVista pattern from `Anatomy3DWidget` in
  `ui_wizard_pages.py` rather than building a separate Qt plotter stack.
