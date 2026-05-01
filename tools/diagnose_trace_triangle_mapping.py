#!/usr/bin/env python3
"""Inspect persisted triangle-force mapping quality for one trace step."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from steve_recommender.eval_v2.force_trace_persistence import TraceReader


ANATOMY_MESH_ROOT = Path("data/anatomy_registry/anatomies")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose triangle-force mapping for one eval_v2 trace step.",
    )
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("--step", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with TraceReader(args.trace_path) as reader:
        step = reader.step(args.step)
        triangle_contacts = reader.triangle_contacts_for_step(args.step)
        anatomy_id = str(reader._file["scenario"].attrs["anatomy_id"])
        mesh_path = ANATOMY_MESH_ROOT / anatomy_id / "mesh" / "simulationmesh.obj"
        mesh = pv.read(str(mesh_path))
        collision_positions_mm = np.asarray(step.wire_collision_positions_mm, dtype=np.float64)
        cell_centers_mm = np.asarray(mesh.cell_centers().points, dtype=np.float64)

        print(f"trace={args.trace_path}")
        print(f"step={args.step}")
        print(f"anatomy_id={anatomy_id}")
        print(f"mesh_path={mesh_path}")
        print(f"triangle_contact_count={len(triangle_contacts)}")
        print("")
        print("triangle_id,norm_N,nearest_collision_dof,min_dist_mm,centroid_mm")
        for record in triangle_contacts:
            centroid_mm = cell_centers_mm[int(record.triangle_id)]
            distances_mm = np.linalg.norm(collision_positions_mm - centroid_mm, axis=1)
            nearest_collision_dof = int(np.argmin(distances_mm))
            min_dist_mm = float(np.min(distances_mm))
            centroid_list = [round(float(value), 3) for value in centroid_mm.tolist()]
            print(
                f"{int(record.triangle_id)},{float(record.norm_N):.6f},"
                f"{nearest_collision_dof},{min_dist_mm:.3f},{centroid_list}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
