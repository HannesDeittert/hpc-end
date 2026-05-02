from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from steve_recommender.eval_v2.builders import build_aortic_arch
from steve_recommender.eval_v2.service import DefaultEvaluationService
from third_party.stEVE.eve.intervention.vesseltree.util.meshing import load_mesh


@dataclass(frozen=True)
class AggregationStats:
    minimum: float
    median: float
    p95: float
    maximum: float


def _build_triangle_geometry(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = load_mesh(str(mesh_path)).extract_surface().triangulate()
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape((-1, 4))
    triangle_indices = np.asarray(faces[:, 1:4], dtype=np.int64)
    vertices = np.asarray(mesh.points, dtype=np.float64).reshape((-1, 3))
    p0 = vertices[triangle_indices[:, 0]]
    p1 = vertices[triangle_indices[:, 1]]
    p2 = vertices[triangle_indices[:, 2]]
    centroids = (p0 + p1 + p2) / 3.0
    normals = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0.0
    normals[valid] /= lengths[valid, None]
    return centroids, normals


def _infer_orientation_sign(
    *,
    centroids: np.ndarray,
    normals: np.ndarray,
    centerline: np.ndarray,
) -> dict[str, float]:
    nearest_idx = np.argmin(
        np.sum((centroids[:, None, :] - centerline[None, :, :]) ** 2, axis=2),
        axis=1,
    )
    nearest_centerline = centerline[nearest_idx]
    radial = centroids - nearest_centerline
    signed = np.einsum("ij,ij->i", normals, radial)
    return {
        "positive_fraction": float(np.mean(signed > 0.0)),
        "median": float(np.median(signed)),
        "p05": float(np.percentile(signed, 5.0)),
        "p95": float(np.percentile(signed, 95.0)),
    }


def _signed_volume(mesh_path: Path) -> float:
    mesh = load_mesh(str(mesh_path)).extract_surface().triangulate()
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape((-1, 4))
    triangle_indices = np.asarray(faces[:, 1:4], dtype=np.int64)
    vertices = np.asarray(mesh.points, dtype=np.float64).reshape((-1, 3))
    p0 = vertices[triangle_indices[:, 0]]
    p1 = vertices[triangle_indices[:, 1]]
    p2 = vertices[triangle_indices[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", p0, np.cross(p1, p2))) / 6.0)


def _stats(values: np.ndarray) -> AggregationStats:
    if values.size == 0:
        return AggregationStats(0.0, 0.0, 0.0, 0.0)
    return AggregationStats(
        minimum=float(np.min(values)),
        median=float(np.median(values)),
        p95=float(np.percentile(values, 95.0)),
        maximum=float(np.max(values)),
    )


def _stepwise_max(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    per_step: list[float] = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        lo = int(start)
        hi = int(end)
        per_step.append(float(np.max(values[lo:hi])) if hi > lo else 0.0)
    return np.asarray(per_step, dtype=np.float64)


def _representative_step(offsets: np.ndarray) -> int:
    nonempty = np.flatnonzero(offsets[1:] > offsets[:-1])
    if nonempty.size == 0:
        return 0
    return int(nonempty[len(nonempty) // 2])


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose signed normal-force aggregation from one persisted eval_v2 trace.",
    )
    parser.add_argument("trace_h5", type=Path, help="Path to one eval_v2 trial trace .h5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/force_sign_diagnostics"),
        help="Directory for the CSV and JSON diagnostic outputs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    trace_path = Path(args.trace_h5)

    with h5py.File(trace_path, "r") as handle:
        scenario = dict(handle["scenario"].attrs)
        triangle_ids = np.asarray(handle["contacts/triangle/triangle_id"], dtype=np.int64)
        triangle_forces_N = np.asarray(
            handle["contacts/triangle/triangle_force_xyz_N"], dtype=np.float64
        ).reshape((-1, 3))
        step_offsets = np.asarray(handle["contacts/triangle/step_offsets"], dtype=np.int64)

    anatomy_id = str(scenario["anatomy_id"])
    service = DefaultEvaluationService()
    anatomy = service.get_anatomy(record_id=anatomy_id)
    vessel_tree = build_aortic_arch(anatomy)
    vessel_tree.reset()
    centerline = np.asarray(vessel_tree.centerline_coordinates, dtype=np.float64).reshape(
        (-1, 3)
    )
    mesh_path = Path(anatomy.simulation_mesh_path)

    centroids, normals = _build_triangle_geometry(mesh_path)
    signed_volume = _signed_volume(mesh_path)
    orientation = _infer_orientation_sign(
        centroids=centroids,
        normals=normals,
        centerline=centerline,
    )

    raw_normals = normals[triangle_ids]
    raw_dot_N = np.einsum("ij,ij->i", triangle_forces_N, raw_normals)
    wire_to_wall_force_N = -triangle_forces_N
    signed_dot_N = np.einsum("ij,ij->i", wire_to_wall_force_N, raw_normals)
    absolute_dot_N = np.abs(signed_dot_N)
    current_clamp_N = np.maximum(0.0, signed_dot_N)
    outward_compression_clamp_N = np.maximum(0.0, -signed_dot_N)
    magnitude_N = np.linalg.norm(wire_to_wall_force_N, axis=1)

    per_step_current = _stepwise_max(current_clamp_N, step_offsets)
    per_step_absolute = _stepwise_max(absolute_dot_N, step_offsets)
    per_step_outward = _stepwise_max(outward_compression_clamp_N, step_offsets)

    representative_step = _representative_step(step_offsets)
    rep_start = int(step_offsets[representative_step])
    rep_end = int(step_offsets[representative_step + 1])
    representative_rows: list[dict[str, object]] = []
    for index in range(rep_start, rep_end):
        representative_rows.append(
            {
                "trace_path": str(trace_path),
                "step_index": representative_step,
                "triangle_id": int(triangle_ids[index]),
                "force_vector_N": tuple(float(v) for v in wire_to_wall_force_N[index]),
                "surface_normal": tuple(float(v) for v in raw_normals[index]),
                "force_magnitude_N": float(magnitude_N[index]),
                "signed_dot_N": float(signed_dot_N[index]),
                "absolute_dot_N": float(absolute_dot_N[index]),
                "current_clamp_N": float(current_clamp_N[index]),
                "outward_compression_clamp_N": float(outward_compression_clamp_N[index]),
                "raw_triangle_force_N": tuple(float(v) for v in triangle_forces_N[index]),
                "raw_dot_N": float(raw_dot_N[index]),
            }
        )

    sign_counts = {
        "dot_gt_zero": int(np.sum(signed_dot_N > 0.0)),
        "dot_lt_zero": int(np.sum(signed_dot_N < 0.0)),
        "dot_abs_lt_1e-8": int(np.sum(np.abs(signed_dot_N) < 1e-8)),
        "total_contacts": int(signed_dot_N.size),
    }

    summary = {
        "trace_path": str(trace_path),
        "anatomy_id": anatomy_id,
        "mesh_path": str(mesh_path),
        "mesh_signed_volume": signed_volume,
        "orientation_centerline_check": orientation,
        "orientation_conclusion": (
            "triangle winding normals point inward toward the lumen"
            if orientation["positive_fraction"] < 0.5 and signed_volume < 0.0
            else "triangle winding normals do not appear inward-dominant"
        ),
        "sign_counts": sign_counts,
        "per_contact_stats": {
            "current_max0_dot": asdict(_stats(current_clamp_N)),
            "absolute_abs_dot": asdict(_stats(absolute_dot_N)),
            "outward_compression_max0_negdot": asdict(_stats(outward_compression_clamp_N)),
        },
        "per_step_stats": {
            "current_max0_dot": asdict(_stats(per_step_current)),
            "absolute_abs_dot": asdict(_stats(per_step_absolute)),
            "outward_compression_max0_negdot": asdict(_stats(per_step_outward)),
        },
        "representative_step_index": representative_step,
        "representative_contact_count": len(representative_rows),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = trace_path.stem
    csv_path = output_dir / f"{stem}_representative_contacts.csv"
    json_path = output_dir / f"{stem}_summary.json"
    _write_csv(csv_path, representative_rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"representative_contacts_csv={csv_path}")
    print(f"summary_json={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
