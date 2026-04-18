#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Always use a file backend for non-interactive batch plotting.
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from steve_recommender.evaluation.config import AorticArchSpec
from steve_recommender.evaluation.intervention_factory import build_aortic_arch_intervention


TRIAL_NAME_RE = re.compile(r"^(?P<agent>.+)_trial(?P<trial>\d+)_seed(?P<seed>\d+)$")


def _read_data(obj: object, attr: str) -> object:
    try:
        value = getattr(obj, attr)
    except Exception:
        return None
    if hasattr(value, "value"):
        try:
            return value.value
        except Exception:
            return None
    return value


def _normalize_points(arr: object) -> np.ndarray:
    try:
        out = np.asarray(arr, dtype=np.float32)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)
    if out.ndim == 1:
        if out.size % 3 != 0:
            return np.zeros((0, 3), dtype=np.float32)
        out = out.reshape((-1, 3))
    if out.ndim != 2 or out.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    return out[:, :3].astype(np.float32)


def _read_wall_geometry(simulation: object) -> Tuple[np.ndarray, np.ndarray]:
    try:
        vessel_dofs = simulation.root.vesselTree.dofs
        vessel_topo = simulation.root.vesselTree.MeshTopology
    except Exception as exc:
        raise RuntimeError(f"Unable to access vessel wall objects on simulation root: {exc}") from exc

    vertices = _normalize_points(_read_data(vessel_dofs, "position"))
    try:
        triangles = np.asarray(_read_data(vessel_topo, "triangles"), dtype=np.int32).reshape((-1, 3))
    except Exception as exc:
        raise RuntimeError(f"Unable to read wall triangle topology: {exc}") from exc

    if vertices.size == 0 or triangles.size == 0:
        raise RuntimeError("Wall geometry is empty (vertices or triangles missing).")

    if int(np.max(triangles)) >= int(vertices.shape[0]) or int(np.min(triangles)) < 0:
        raise RuntimeError("Wall triangle indices are out of bounds.")

    return vertices.astype(np.float32), triangles.astype(np.int32)


def _resolve_latest_eval_run(root: Path) -> Path:
    candidates = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "force_reference_scene":
            continue
        if not (child / "config.json").is_file():
            continue
        if not (child / "trials").is_dir():
            continue
        candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No evaluation run directories found under: {root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render vessel-wall force heatmaps from an evaluation run directory. "
            "Reads trials/*.npz and reconstructs vessel wall mesh from run config."
        )
    )
    p.add_argument(
        "--run-dir",
        default="",
        help=(
            "Path to one run directory (results/eval_runs/<timestamp>_<name>). "
            "If empty, the latest run is used."
        ),
    )
    p.add_argument(
        "--eval-root",
        default="results/eval_runs",
        help="Root directory used to discover latest run when --run-dir is empty.",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory for PNG heatmaps. Default: <run-dir>/force_heatmaps",
    )
    p.add_argument(
        "--aggregate",
        choices=["max", "mean", "p95", "sum"],
        default="max",
        help="How to aggregate per-step segment-force magnitudes into one value per wall segment.",
    )
    p.add_argument(
        "--cmap",
        default="turbo",
        help="Matplotlib colormap.",
    )
    p.add_argument("--elev", type=float, default=18.0, help="3D camera elevation.")
    p.add_argument("--azim", type=float, default=-68.0, help="3D camera azimuth.")
    p.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    p.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Optional fixed lower color scale bound.",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional fixed upper color scale bound.",
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
    p.add_argument(
        "--agent",
        default="",
        help="Optional agent name filter (matches filename prefix before _trial).",
    )
    p.add_argument(
        "--no-agent-aggregate",
        action="store_true",
        help="Skip additional per-agent aggregate heatmaps.",
    )
    return p.parse_args()


def _load_run_context(run_dir: Path) -> Tuple[AorticArchSpec, str]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing run config: {cfg_path}")
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Run config is not a mapping: {cfg_path}")

    anatomy_raw = raw.get("anatomy", {})
    if not isinstance(anatomy_raw, dict):
        raise TypeError("config.json anatomy field is not a mapping")
    anatomy = AorticArchSpec(
        type="aortic_arch",
        arch_type=str(anatomy_raw.get("arch_type", "I")),
        seed=int(anatomy_raw.get("seed", 30)),
        rotation_yzx_deg=anatomy_raw.get("rotation_yzx_deg", None),
        scaling_xyzd=anatomy_raw.get("scaling_xyzd", None),
        omit_axis=anatomy_raw.get("omit_axis", None),
        target_mode=str(anatomy_raw.get("target_mode", "branch_end")),
        target_branches=list(anatomy_raw.get("target_branches", ["lcca"])),
        target_threshold_mm=float(anatomy_raw.get("target_threshold_mm", 5.0)),
        image_frequency_hz=float(anatomy_raw.get("image_frequency_hz", 7.5)),
        image_rot_zx_deg=tuple(anatomy_raw.get("image_rot_zx_deg", (20.0, 5.0))),
        friction=float(anatomy_raw.get("friction", 0.001)),
    )

    agents = raw.get("agents", [])
    if not isinstance(agents, list) or not agents:
        raise ValueError("config.json contains no agents.")
    first = agents[0]
    if not isinstance(first, dict) or not str(first.get("tool", "")).strip():
        raise ValueError("First agent in config.json has no tool reference.")
    tool_ref = str(first["tool"]).strip()
    return anatomy, tool_ref


def _build_wall_mesh_from_run(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    anatomy, tool_ref = _load_run_context(run_dir)
    intervention, _action_dt = build_aortic_arch_intervention(
        tool_ref=tool_ref,
        anatomy=anatomy,
        force_extraction=None,
    )
    try:
        intervention.make_non_mp()
    except Exception:
        # make_non_mp may already be active; continue.
        pass

    try:
        intervention.reset(seed=int(anatomy.seed))
        vertices, triangles = _read_wall_geometry(intervention.simulation)
    finally:
        try:
            intervention.close()
        except Exception:
            pass

    return vertices, triangles


def _iter_trial_npz(trials_dir: Path) -> Iterable[Path]:
    for p in sorted(trials_dir.glob("*.npz")):
        if p.is_file():
            yield p


def _agent_from_trial_name(stem: str) -> str:
    m = TRIAL_NAME_RE.match(stem)
    if m:
        return str(m.group("agent"))
    return stem


def _resize_segment_metric(values: np.ndarray, n_segments: int) -> np.ndarray:
    out = np.zeros((n_segments,), dtype=np.float32)
    n_copy = int(min(n_segments, values.shape[0]))
    if n_copy > 0:
        out[:n_copy] = values[:n_copy].astype(np.float32)
    return out


def _aggregate_segment_metric(
    segment_vectors: np.ndarray,
    mode: str,
) -> np.ndarray:
    if segment_vectors.ndim != 3 or segment_vectors.shape[2] != 3:
        raise ValueError(f"Expected [T, S, 3] segment vectors, got: {segment_vectors.shape}")
    norms = np.linalg.norm(segment_vectors.astype(np.float32), axis=2)
    if mode == "max":
        return np.nanmax(norms, axis=0).astype(np.float32)
    if mode == "mean":
        return np.nanmean(norms, axis=0).astype(np.float32)
    if mode == "p95":
        return np.nanpercentile(norms, 95.0, axis=0).astype(np.float32)
    if mode == "sum":
        return np.nansum(norms, axis=0).astype(np.float32)
    raise ValueError(f"Unsupported aggregate mode: {mode}")


def _read_trial_metric(npz_path: Path, aggregate: str, n_triangles: int) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    if "wall_segment_force_vectors_N" in data.files:
        key = "wall_segment_force_vectors_N"
    elif "wall_segment_force_vectors" in data.files:
        key = "wall_segment_force_vectors"
    else:
        raise KeyError("npz has no wall_segment_force_vectors(_N) dataset")

    segment_vectors = np.asarray(data[key], dtype=np.float32)
    metric = _aggregate_segment_metric(segment_vectors, aggregate)
    return _resize_segment_metric(metric, n_triangles)


def _plot_wall_heatmap(
    *,
    vertices: np.ndarray,
    triangles: np.ndarray,
    segment_values: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    gamma: float,
    zero_threshold: float,
    zero_color: str,
    elev: float,
    azim: float,
    dpi: int,
) -> None:
    tri_vertices = vertices[triangles]
    if abs(float(gamma) - 1.0) < 1e-9:
        norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
    else:
        norm = PowerNorm(gamma=float(gamma), vmin=float(vmin), vmax=float(vmax), clip=True)
    cmap_obj = plt.get_cmap(cmap)
    facecolors = cmap_obj(norm(segment_values))
    if np.isfinite(float(zero_threshold)):
        facecolors[segment_values <= float(zero_threshold)] = np.asarray(
            to_rgba(str(zero_color)),
            dtype=np.float32,
        )

    fig = plt.figure(figsize=(10.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(
        tri_vertices,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0.0,
        alpha=1.0,
    )
    ax.add_collection3d(mesh)

    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    span = max(span, 1.0)
    half = 0.5 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    # Keep visual clean for shape-focused force maps.
    ax.set_axis_off()
    ax.view_init(elev=float(elev), azim=float(azim))

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array(segment_values)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Wall Segment Force Magnitude")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        eval_root = Path(args.eval_root).expanduser().resolve()
        run_dir = _resolve_latest_eval_run(eval_root)

    trials_dir = run_dir / "trials"
    if not trials_dir.is_dir():
        raise FileNotFoundError(f"Missing trials directory: {trials_dir}")

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (run_dir / "force_heatmaps").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vertices, triangles = _build_wall_mesh_from_run(run_dir)
    n_triangles = int(triangles.shape[0])

    trial_metrics: Dict[str, np.ndarray] = {}
    by_agent: Dict[str, List[np.ndarray]] = {}

    for npz_path in _iter_trial_npz(trials_dir):
        stem = npz_path.stem
        agent = _agent_from_trial_name(stem)
        if args.agent and agent != str(args.agent):
            continue
        metric = _read_trial_metric(npz_path, args.aggregate, n_triangles)
        trial_metrics[stem] = metric
        by_agent.setdefault(agent, []).append(metric)

    if not trial_metrics:
        raise RuntimeError("No trial files matched. Check --run-dir/--agent.")

    auto_vmax = float(max(np.nanmax(v) for v in trial_metrics.values()))
    if not np.isfinite(auto_vmax) or auto_vmax <= 0.0:
        auto_vmax = 1.0
    auto_vmin = 0.0

    vmin = float(args.vmin) if args.vmin is not None else float(auto_vmin)
    vmax = float(args.vmax) if args.vmax is not None else float(auto_vmax)
    gamma = float(args.gamma)
    zero_threshold = float(args.zero_threshold)
    zero_color = str(args.zero_color)
    if not np.isfinite(vmin):
        raise ValueError("--vmin must be finite when provided")
    if not np.isfinite(vmax):
        raise ValueError("--vmax must be finite when provided")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("--gamma must be > 0")
    if vmax <= vmin:
        raise ValueError("--vmax must be greater than --vmin")

    written: List[str] = []

    for stem, metric in sorted(trial_metrics.items()):
        out_path = out_dir / f"{stem}__{args.aggregate}.png"
        title = f"{run_dir.name} | {stem} | agg={args.aggregate}"
        _plot_wall_heatmap(
            vertices=vertices,
            triangles=triangles,
            segment_values=metric,
            out_path=out_path,
            title=title,
            cmap=str(args.cmap),
            vmin=vmin,
            vmax=vmax,
            gamma=gamma,
            zero_threshold=zero_threshold,
            zero_color=zero_color,
            elev=float(args.elev),
            azim=float(args.azim),
            dpi=int(args.dpi),
        )
        written.append(str(out_path))

    if not bool(args.no_agent_aggregate):
        for agent, items in sorted(by_agent.items()):
            stacked = np.stack(items, axis=0).astype(np.float32)
            metric = np.nanmean(stacked, axis=0).astype(np.float32)
            out_path = out_dir / f"{agent}__aggregate_mean_over_trials__{args.aggregate}.png"
            title = (
                f"{run_dir.name} | {agent} | aggregate=mean(trials) "
                f"| per-trial-agg={args.aggregate}"
            )
            _plot_wall_heatmap(
                vertices=vertices,
                triangles=triangles,
                segment_values=metric,
                out_path=out_path,
                title=title,
                cmap=str(args.cmap),
                vmin=vmin,
                vmax=vmax,
                gamma=gamma,
                zero_threshold=zero_threshold,
                zero_color=zero_color,
                elev=float(args.elev),
                azim=float(args.azim),
                dpi=int(args.dpi),
            )
            written.append(str(out_path))

    index = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "aggregate": str(args.aggregate),
        "cmap": str(args.cmap),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "gamma": float(gamma),
        "zero_threshold": float(zero_threshold),
        "zero_color": str(zero_color),
        "n_triangles": int(n_triangles),
        "files": written,
    }
    (out_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[heatmap] run_dir={run_dir}")
    print(f"[heatmap] out_dir={out_dir}")
    print(f"[heatmap] wrote={len(written)}")
    print(f"[heatmap] index={out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
