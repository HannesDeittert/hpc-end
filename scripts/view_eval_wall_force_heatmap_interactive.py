#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
from matplotlib.colors import Normalize, PowerNorm, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Reuse run/mesh helpers from the static heatmap script.
from plot_eval_wall_force_heatmaps import (
    _agent_from_trial_name,
    _build_wall_mesh_from_run,
    _iter_trial_npz,
    _resolve_latest_eval_run,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Interactive wall-force heatmap viewer for one trial. "
            "Use slider to move through time; rotate/spin with mouse in 3D."
        )
    )
    p.add_argument(
        "--run-dir",
        default="",
        help=(
            "Path to one run directory (results/eval_runs/<timestamp>_<name>). "
            "If empty, latest run is used."
        ),
    )
    p.add_argument(
        "--eval-root",
        default="results/eval_runs",
        help="Root directory used when --run-dir is empty.",
    )
    p.add_argument(
        "--trial",
        default="",
        help=(
            "Trial file stem or path to trial .npz. "
            "Examples: 'archvar_default_trial0000_seed123' or full path."
        ),
    )
    p.add_argument(
        "--agent",
        default="",
        help="Optional agent filter when auto-picking a trial.",
    )
    p.add_argument(
        "--cmap",
        default="turbo",
        help="Matplotlib colormap.",
    )
    p.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="Lower color scale bound.",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Upper color scale bound. Default: max value in selected trial.",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.55,
        help=(
            "Power-law normalization gamma (<1 emphasizes low forces; 1 is linear). "
            "Must be > 0."
        ),
    )
    p.add_argument(
        "--zero-threshold",
        type=float,
        default=0.0,
        help="Values <= threshold are rendered with --zero-color.",
    )
    p.add_argument(
        "--zero-color",
        default="#d9d9d9",
        help="Color used for near-zero-force segments.",
    )
    p.add_argument("--elev", type=float, default=18.0, help="3D camera elevation.")
    p.add_argument("--azim", type=float, default=-68.0, help="3D camera azimuth.")
    return p.parse_args()


def _import_interactive_pyplot():
    """Import pyplot with an interactive backend, avoiding hard Qt dependency."""

    tried = []
    for backend in ("TkAgg", "Qt5Agg"):
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Slider

            return plt, Slider, backend
        except Exception as exc:
            tried.append(f"{backend}: {exc}")
            continue
    raise RuntimeError(
        "Failed to load interactive matplotlib backend. Tried: " + " | ".join(tried)
    )


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = _resolve_latest_eval_run(Path(args.eval_root).expanduser().resolve())
    trials_dir = run_dir / "trials"
    if not trials_dir.is_dir():
        raise FileNotFoundError(f"Missing trials dir: {trials_dir}")
    return run_dir


def _select_trial_npz(
    run_dir: Path,
    *,
    trial_arg: str,
    agent_filter: str,
) -> Path:
    trials_dir = run_dir / "trials"
    all_trials = list(_iter_trial_npz(trials_dir))
    if not all_trials:
        raise FileNotFoundError(f"No trial .npz files in: {trials_dir}")

    if trial_arg:
        p = Path(trial_arg).expanduser()
        if p.is_file():
            return p.resolve()
        candidate = trials_dir / (str(trial_arg) + ".npz")
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(
            f"Trial not found. Provide full .npz path or trial stem under {trials_dir}"
        )

    if agent_filter:
        filtered = [
            p for p in all_trials if _agent_from_trial_name(p.stem) == str(agent_filter)
        ]
        if not filtered:
            raise FileNotFoundError(
                f"No trials for agent '{agent_filter}' in {trials_dir}"
            )
        return sorted(filtered)[0].resolve()

    return sorted(all_trials)[0].resolve()


def _load_segment_series(npz_path: Path, n_triangles: int) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    if "wall_segment_force_vectors_N" in data.files:
        key = "wall_segment_force_vectors_N"
    elif "wall_segment_force_vectors" in data.files:
        key = "wall_segment_force_vectors"
    else:
        raise KeyError(f"{npz_path} has no wall_segment_force_vectors(_N)")

    arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"{npz_path} expected [T,S,3] segment-force array, got {arr.shape}"
        )

    t, s, _ = arr.shape
    if s < n_triangles:
        pad = np.zeros((t, n_triangles - s, 3), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    elif s > n_triangles:
        arr = arr[:, :n_triangles, :]
    return arr


def _set_equal_xyz_limits(ax, vertices: np.ndarray) -> None:
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    span = max(span, 1.0)
    half = 0.5 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def main() -> None:
    args = _parse_args()
    plt, Slider, backend = _import_interactive_pyplot()

    run_dir = _resolve_run_dir(args)
    npz_path = _select_trial_npz(
        run_dir,
        trial_arg=str(args.trial),
        agent_filter=str(args.agent),
    )

    vertices, triangles = _build_wall_mesh_from_run(run_dir)
    segment_series = _load_segment_series(npz_path, n_triangles=int(triangles.shape[0]))
    norms = np.linalg.norm(segment_series, axis=2).astype(np.float32)  # [T, Ntri]

    n_steps = int(norms.shape[0])
    if n_steps <= 0:
        raise RuntimeError(f"No steps in selected trial: {npz_path}")

    vmin = float(args.vmin)
    vmax = float(args.vmax) if args.vmax is not None else float(np.nanmax(norms))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = max(vmin + 1e-6, 1.0)
    gamma = float(args.gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("--gamma must be > 0")
    if abs(gamma - 1.0) < 1e-9:
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    else:
        norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(str(args.cmap))
    zero_threshold = float(args.zero_threshold)
    zero_rgba = np.asarray(to_rgba(str(args.zero_color)), dtype=np.float32)

    tri_vertices = vertices[triangles]
    step0 = norms[0]
    facecolors = cmap(norm(step0))
    if np.isfinite(zero_threshold):
        facecolors[step0 <= zero_threshold] = zero_rgba

    fig = plt.figure(figsize=(12.0, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(
        tri_vertices,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0.0,
        alpha=1.0,
    )
    ax.add_collection3d(mesh)
    _set_equal_xyz_limits(ax, vertices)
    ax.view_init(elev=float(args.elev), azim=float(args.azim))
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(step0)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Wall Segment Force Magnitude")

    title = fig.suptitle("", fontsize=10)

    slider_ax = fig.add_axes([0.16, 0.05, 0.68, 0.035])
    step_slider = Slider(
        ax=slider_ax,
        label="Step",
        valmin=0,
        valmax=max(0, n_steps - 1),
        valinit=0,
        valstep=1,
    )

    def _update(step_idx: int) -> None:
        s = int(step_idx)
        vals = norms[s]
        fc = cmap(norm(vals))
        if np.isfinite(zero_threshold):
            fc[vals <= zero_threshold] = zero_rgba
        mesh.set_facecolor(fc)
        vmax_step = float(np.nanmax(vals)) if vals.size else float("nan")
        title.set_text(
            f"{run_dir.name} | {npz_path.stem} | step={s}/{n_steps-1} "
            f"| step_max={vmax_step:.6g} | scale=[{vmin:.6g}, {vmax:.6g}] "
            f"| gamma={gamma:.3g} | backend={backend}"
        )
        fig.canvas.draw_idle()

    step_slider.on_changed(lambda val: _update(int(val)))
    _update(0)

    # Mouse drag in the 3D axis already provides spin/rotate.
    # Slider controls time-step heatmap.
    plt.show()


if __name__ == "__main__":
    main()
