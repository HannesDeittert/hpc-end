from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pyvista as pv


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_CANDIDATES = (
    Path("/tmp/eval_v2_trace_bench/trace_on_50x100/traces"),
    Path("/tmp/eval_v2_trace_smoke/eval_v2_job/traces"),
)
ANATOMY_BASE = REPO_ROOT / "data" / "anatomy_registry" / "anatomies"


def _find_trace() -> Path:
    for directory in TRACE_CANDIDATES:
        if directory.exists():
            traces = sorted(directory.glob("*.h5"))
            if traces:
                return traces[0]
    raise FileNotFoundError("No real trace file found in known benchmark directories.")


def main() -> None:
    trace_path = _find_trace()
    print(f"trace_path={trace_path}")
    with h5py.File(trace_path, "r") as handle:
        scenario = handle["scenario"].attrs
        mesh_ref = str(scenario["mesh_ref"])
        anatomy_id = str(scenario["anatomy_id"])
        triangle_ids = np.asarray(
            handle["contacts/triangle/triangle_id"][:], dtype=np.int64
        )
        wire_positions_shape = tuple(
            int(dim) for dim in handle["steps/wire_positions"].shape
        )
        print(f"anatomy_id={anatomy_id}")
        print(f"mesh_ref={mesh_ref}")
        if triangle_ids.size > 0:
            print(f"triangle_id_min={int(triangle_ids.min())}")
            print(f"triangle_id_max={int(triangle_ids.max())}")
            print(f"triangle_id_unique={int(np.unique(triangle_ids).size)}")
        else:
            print("triangle_id_min=None")
            print("triangle_id_max=None")
            print("triangle_id_unique=0")
        print(f"wire_positions_shape={wire_positions_shape}")

    mesh_path = ANATOMY_BASE / anatomy_id / "mesh" / "simulationmesh.obj"
    print(f"simulation_mesh_path={mesh_path}")
    mesh = pv.read(mesh_path)
    print(f"mesh_n_cells={mesh.n_cells}")
    print(f"mesh_n_points={mesh.n_points}")
    print(f"mesh_bounds={tuple(float(value) for value in mesh.bounds)}")

    if triangle_ids.size > 0:
        assert int(triangle_ids.max()) < int(
            mesh.n_cells
        ), f"triangle_id.max()={int(triangle_ids.max())} is out of range for mesh.n_cells={int(mesh.n_cells)}"
    print("triangle_id_assertion=passed")


if __name__ == "__main__":
    main()
