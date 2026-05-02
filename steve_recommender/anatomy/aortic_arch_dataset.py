from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from steve_recommender.storage import repo_root
from third_party.stEVE.eve.intervention.vesseltree import (
    AorticArch,
    ArchType,
)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


ARCHVAR_SCALE_WIDTH_ARRAY = np.linspace(0.7, 1.3, 61, endpoint=True)
ARCHVAR_SCALE_HEIGHT_ARRAY = np.linspace(0.7, 1.3, 61, endpoint=True)
ARCHVAR_SCALE_DIAMETER_ARRAY = np.array([0.85], dtype=np.float32)
ARCHVAR_ROTATION_ARRAY = np.array([0.0], dtype=np.float32)
ARCHVAR_DEFAULT_ARCH_TYPES = ("I",)
DEFAULT_ANATOMY_REGISTRY_ROOT = repo_root() / "data" / "anatomy_registry"


@dataclass(frozen=True)
class AorticArchRecord:
    """One reproducible aortic arch anatomy definition.

    This stores the parameters needed to recreate the vessel tree, plus an optional
    precomputed centerline bundle for fast UI preview.
    """

    record_id: str
    arch_type: str
    seed: int
    rotation_yzx_deg: Optional[Tuple[float, float, float]] = None
    scaling_xyzd: Optional[Tuple[float, float, float, float]] = None
    omit_axis: Optional[str] = None
    created_at: str = ""

    # Relative paths inside the dataset (optional)
    centerline_npz: Optional[str] = None
    simulation_mesh_path: Optional[str] = None
    visualization_mesh_path: Optional[str] = None

    @property
    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.record_id,
            "arch_type": self.arch_type,
            "seed": int(self.seed),
            "rotation_yzx_deg": list(self.rotation_yzx_deg) if self.rotation_yzx_deg else None,
            "scaling_xyzd": list(self.scaling_xyzd) if self.scaling_xyzd else None,
            "omit_axis": self.omit_axis,
            "created_at": self.created_at,
            "centerline_npz": self.centerline_npz,
            "simulation_mesh_path": self.simulation_mesh_path,
            "visualization_mesh_path": self.visualization_mesh_path,
        }


@dataclass(frozen=True)
class AorticArchDataset:
    """Filesystem-backed dataset of AorticArchRecord entries."""

    root: Path

    @property
    def records_dir(self) -> Path:
        return self.root / "records"

    @property
    def index_path(self) -> Path:
        return self.root / "index.jsonl"

    def iter_index(self) -> Iterator[AorticArchRecord]:
        if not self.index_path.exists():
            return iter(())

        def _iter() -> Iterator[AorticArchRecord]:
            with self.index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    yield _record_from_dict(raw)

        return _iter()

    def record_dir(self, record_id: str) -> Path:
        return self.records_dir / record_id


def default_aortic_arch_dataset_root() -> Path:
    """Default location for large anatomy banks (ignored by git via `results/`)."""

    return repo_root() / "results" / "anatomies" / "aortic_arch"


def load_aortic_arch_dataset(root: Optional[Union[str, Path]] = None) -> AorticArchDataset:
    root_path = Path(root) if root is not None else default_aortic_arch_dataset_root()
    return AorticArchDataset(root=root_path)


def _make_record_id(i: int) -> str:
    return f"arch_{i:06d}"


_RECORD_ID_RE = re.compile(r"^arch_(\d+)$")


def next_record_index(dataset: AorticArchDataset) -> int:
    """Return the next free integer index for `arch_XXXXXX` record ids."""

    max_idx = -1
    if dataset.records_dir.exists():
        for child in dataset.records_dir.iterdir():
            if not child.is_dir():
                continue
            m = _RECORD_ID_RE.match(child.name)
            if not m:
                continue
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _record_from_dict(raw: Dict[str, Any]) -> AorticArchRecord:
    return AorticArchRecord(
        record_id=str(raw["id"]),
        arch_type=str(raw["arch_type"]),
        seed=int(raw["seed"]),
        rotation_yzx_deg=tuple(raw["rotation_yzx_deg"]) if raw.get("rotation_yzx_deg") else None,
        scaling_xyzd=tuple(raw["scaling_xyzd"]) if raw.get("scaling_xyzd") else None,
        omit_axis=raw.get("omit_axis"),
        created_at=str(raw.get("created_at", "")),
        centerline_npz=raw.get("centerline_npz"),
        simulation_mesh_path=raw.get("simulation_mesh_path"),
        visualization_mesh_path=raw.get("visualization_mesh_path"),
    )


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _append_index_line(index_path: Path, record: AorticArchRecord) -> None:
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.as_dict, sort_keys=True) + "\n")


def _write_registry_index(root_path: Path, entries: Sequence[Dict[str, Any]]) -> None:
    """Write the eval_v2 anatomy registry index file."""

    _write_json(
        root_path / "index.json",
        {
            "version": 1,
            "anatomies": list(entries),
        },
    )


def _registry_entry_from_description(
    description_path: Path,
    *,
    root_path: Path,
) -> Dict[str, Any]:
    raw = json.loads(description_path.read_text(encoding="utf-8"))
    record_id = str(raw.get("record_id") or description_path.parent.name)
    return {
        "record_id": record_id,
        "created_at": str(raw.get("created_at", "")),
        "description_path": str(description_path.relative_to(root_path)),
    }


def _description_payload_from_record(record: AorticArchRecord) -> Dict[str, Any]:
    """Emit the eval_v2 anatomy-registry schema for one record."""

    mesh_path = "mesh/simulationmesh.obj" if record.simulation_mesh_path else None
    return {
        "anatomy_type": "aortic_arch",
        "record_id": record.record_id,
        "arch_type": record.arch_type,
        "seed": int(record.seed),
        "rotation_yzx_deg": list(record.rotation_yzx_deg) if record.rotation_yzx_deg else None,
        "scaling_xyzd": list(record.scaling_xyzd) if record.scaling_xyzd else None,
        "omit_axis": record.omit_axis,
        "simulation_mesh_path": mesh_path,
        "visualization_mesh_path": mesh_path,
        "centerline_npz": "centerline.npz" if record.centerline_npz else None,
        "centerline_bundle_path": "centerline.npz" if record.centerline_npz else None,
    }


def _polyline_bundle_from_vesseltree(vessel_tree) -> Dict[str, np.ndarray]:
    """Extract centerline data from an stEVE `AorticArch` vessel tree.

    We store branch coordinates + radii so UI preview does not need meshing.
    """

    branches = list(vessel_tree.branches)
    branch_names = np.asarray([b.name for b in branches], dtype=object)

    arrays: Dict[str, np.ndarray] = {
        "branch_names": branch_names,
        "centerline_coordinates": np.asarray(vessel_tree.centerline_coordinates, dtype=np.float32),
        "insertion_position": np.asarray(vessel_tree.insertion.position, dtype=np.float32),
        "insertion_direction": np.asarray(vessel_tree.insertion.direction, dtype=np.float32),
        "coord_low": np.asarray(vessel_tree.coordinate_space.low, dtype=np.float32),
        "coord_high": np.asarray(vessel_tree.coordinate_space.high, dtype=np.float32),
    }

    for b in branches:
        arrays[f"branch_{b.name}_coords"] = np.asarray(b.coordinates, dtype=np.float32)
        if hasattr(b, "radii"):
            arrays[f"branch_{b.name}_radii"] = np.asarray(b.radii, dtype=np.float32)
    return arrays


def _sample_archvar_record(
    *,
    record_id: str,
    dataset_seed: int,
    index: int,
    chosen_arch_types: Sequence[str],
    width_values: Sequence[float],
    height_values: Sequence[float],
    diameter_values: Sequence[float],
    rotate_y_values: Sequence[float],
    rotate_z_values: Sequence[float],
    rotate_x_values: Sequence[float],
    omit_axis: Optional[str],
    write_centerlines: bool,
    write_meshes: bool,
) -> tuple[AorticArchRecord, AorticArch]:
    local_rng = np.random.default_rng(int(dataset_seed + index))

    arch_type = str(local_rng.choice(tuple(chosen_arch_types)))
    seed = int(local_rng.integers(0, 2**31 - 1))

    xy_scaling = float(local_rng.choice(tuple(width_values)))
    z_scaling = float(local_rng.choice(tuple(height_values)))
    diameter_scaling = float(local_rng.choice(tuple(diameter_values)))
    rot_y = float(local_rng.choice(tuple(rotate_y_values)))
    rot_z = float(local_rng.choice(tuple(rotate_z_values)))
    rot_x = float(local_rng.choice(tuple(rotate_x_values)))

    record = AorticArchRecord(
        record_id=record_id,
        arch_type=arch_type,
        seed=seed,
        rotation_yzx_deg=(rot_y, rot_z, rot_x),
        scaling_xyzd=(xy_scaling, xy_scaling, z_scaling, diameter_scaling),
        omit_axis=omit_axis,
        created_at=_now_iso(),
        centerline_npz="centerline.npz" if write_centerlines else None,
        simulation_mesh_path="mesh/simulationmesh.obj" if write_meshes else None,
        visualization_mesh_path="mesh/simulationmesh.obj" if write_meshes else None,
    )

    vessel_tree = AorticArch(
        arch_type=ArchType(record.arch_type),
        seed=record.seed,
        rotation_yzx_deg=record.rotation_yzx_deg,
        scaling_xyzd=record.scaling_xyzd,
        omit_axis=record.omit_axis,
    )
    vessel_tree.reset()
    return record, vessel_tree


def _write_archvar_record(
    *,
    out_dir: Path,
    record: AorticArchRecord,
    vessel_tree: AorticArch,
    write_centerlines: bool,
    write_meshes: bool,
) -> None:
    _ensure_dir(out_dir)
    _write_json(out_dir / "description.json", _description_payload_from_record(record))

    if write_meshes:
        mesh_dir = out_dir / "mesh"
        _ensure_dir(mesh_dir)
        sim_src = Path(vessel_tree.mesh_path)
        sim_dst = mesh_dir / "simulationmesh.obj"
        shutil.copy2(sim_src, sim_dst)

    if write_centerlines:
        arrays = _polyline_bundle_from_vesseltree(vessel_tree)
        np.savez_compressed(out_dir / "centerline.npz", **arrays)


def generate_aortic_arch_records(
    *,
    dataset: AorticArchDataset,
    start_index: int,
    count: int,
    dataset_seed: int,
    arch_types: Sequence[str] = ARCHVAR_DEFAULT_ARCH_TYPES,
    scale_width_array: Sequence[float] = ARCHVAR_SCALE_WIDTH_ARRAY,
    scale_heigth_array: Sequence[float] = ARCHVAR_SCALE_HEIGHT_ARRAY,
    scale_diameter_array: Sequence[float] = ARCHVAR_SCALE_DIAMETER_ARRAY,
    rotate_y_deg_array: Sequence[float] = ARCHVAR_ROTATION_ARRAY,
    rotate_z_deg_array: Sequence[float] = ARCHVAR_ROTATION_ARRAY,
    rotate_x_deg_array: Sequence[float] = ARCHVAR_ROTATION_ARRAY,
    omit_axis: Optional[str] = None,
    write_centerlines: bool = True,
    write_meshes: bool = True,
    overwrite: bool = False,
    progress_every: int = 100,
) -> List[AorticArchRecord]:
    """Generate a batch of AorticArchRecord entries and store them on disk.

    This is designed to be run in multiple batches, e.g.:
    - `start_index=0, count=1000`
    - `start_index=1000, count=1000`
    - ...
    """

    if count <= 0:
        return []

    _ensure_dir(dataset.root)
    _ensure_dir(dataset.records_dir)

    # Normalize/validate arch types once.
    allowed = {t.value for t in ArchType}
    chosen = [t for t in arch_types if t in allowed]
    if not chosen:
        raise ValueError(f"No valid arch_types provided. Allowed: {sorted(allowed)}")

    width_values = tuple(float(v) for v in scale_width_array)
    height_values = tuple(float(v) for v in scale_heigth_array)
    diameter_values = tuple(float(v) for v in scale_diameter_array)
    rotate_y_values = tuple(float(v) for v in rotate_y_deg_array)
    rotate_z_values = tuple(float(v) for v in rotate_z_deg_array)
    rotate_x_values = tuple(float(v) for v in rotate_x_deg_array)
    if not width_values or not height_values or not diameter_values:
        raise ValueError("scale arrays must not be empty")
    if not rotate_y_values or not rotate_z_values or not rotate_x_values:
        raise ValueError("rotation arrays must not be empty")

    indexed_ids = {r.record_id for r in dataset.iter_index()}
    records: List[AorticArchRecord] = []

    for offset in range(count):
        i = int(start_index + offset)
        record_id = _make_record_id(i)
        out_dir = dataset.record_dir(record_id)

        if out_dir.exists() and not overwrite:
            # Repair a missing index entry if the record directory exists (e.g. after an interrupted run).
            if record_id not in indexed_ids:
                desc_path = out_dir / "description.json"
                if desc_path.exists():
                    raw = json.loads(desc_path.read_text(encoding="utf-8"))
                    record = _record_from_dict(raw)
                    _append_index_line(dataset.index_path, record)
                    indexed_ids.add(record_id)
                    records.append(record)
            continue

        # Deterministic per-record RNG stream.
        # Using a spawned seed avoids drift when batches are generated separately.
        record, vessel_tree = _sample_archvar_record(
            record_id=record_id,
            dataset_seed=dataset_seed,
            index=i,
            chosen_arch_types=chosen,
            width_values=width_values,
            height_values=height_values,
            diameter_values=diameter_values,
            rotate_y_values=rotate_y_values,
            rotate_z_values=rotate_z_values,
            rotate_x_values=rotate_x_values,
            omit_axis=omit_axis,
            write_centerlines=write_centerlines,
            write_meshes=write_meshes,
        )

        # Write to disk
        _write_archvar_record(
            out_dir=out_dir,
            record=record,
            vessel_tree=vessel_tree,
            write_centerlines=write_centerlines,
            write_meshes=write_meshes,
        )

        _append_index_line(dataset.index_path, record)
        records.append(record)

        if progress_every > 0 and (offset + 1) % progress_every == 0:
            print(f"[aortic_arch_dataset] generated {offset + 1}/{count} (latest={record_id})")

    return records


def generate_anatomy_registry_from_archvar(
    *,
    root: Optional[Union[str, Path]] = None,
    count: int,
    dataset_seed: int = 123,
    start_index: int = 0,
    arch_types: Sequence[str] = ARCHVAR_DEFAULT_ARCH_TYPES,
    write_centerlines: bool = True,
    overwrite: bool = True,
    progress_every: int = 10,
) -> Path:
    """Generate a complete eval_v2 anatomy registry in the current on-disk format."""

    root_path = Path(root) if root is not None else DEFAULT_ANATOMY_REGISTRY_ROOT
    anatomies_root = root_path / "anatomies"
    if overwrite and root_path.exists():
        for child in anatomies_root.iterdir() if anatomies_root.exists() else []:
            if child.is_dir():
                shutil.rmtree(child)
            elif child.exists():
                child.unlink()
        index_path = root_path / "index.json"
        if index_path.exists():
            index_path.unlink()
    _ensure_dir(root_path)
    _ensure_dir(anatomies_root)

    allowed = {t.value for t in ArchType}
    chosen = [t for t in arch_types if t in allowed]
    if not chosen:
        raise ValueError(f"No valid arch_types provided. Allowed: {sorted(allowed)}")

    entries: List[Dict[str, Any]] = []
    for offset in range(max(0, int(count))):
        record_index = int(start_index + offset)
        record_id = f"Tree_{record_index:02d}"
        out_dir = anatomies_root / record_id
        record, vessel_tree = _sample_archvar_record(
            record_id=record_id,
            dataset_seed=dataset_seed,
            index=record_index,
            chosen_arch_types=chosen,
            width_values=ARCHVAR_SCALE_WIDTH_ARRAY,
            height_values=ARCHVAR_SCALE_HEIGHT_ARRAY,
            diameter_values=ARCHVAR_SCALE_DIAMETER_ARRAY,
            rotate_y_values=ARCHVAR_ROTATION_ARRAY,
            rotate_z_values=ARCHVAR_ROTATION_ARRAY,
            rotate_x_values=ARCHVAR_ROTATION_ARRAY,
            omit_axis=None,
            write_centerlines=write_centerlines,
            write_meshes=True,
        )
        _write_archvar_record(
            out_dir=out_dir,
            record=record,
            vessel_tree=vessel_tree,
            write_centerlines=write_centerlines,
            write_meshes=True,
        )
        entries.append(
            {
                "record_id": record_id,
                "created_at": record.created_at,
                "description_path": f"anatomies/{record_id}/description.json",
            }
        )
        if progress_every > 0 and (offset + 1) % progress_every == 0:
            print(
                f"[anatomy_registry] generated {offset + 1}/{count} "
                f"(latest={record_id})"
            )

    _write_registry_index(root_path, entries)
    return root_path


def rebuild_anatomy_registry_index(
    *,
    root: Optional[Union[str, Path]] = None,
) -> Path:
    """Rebuild `index.json` from anatomy folders already on disk.

    This is useful when a generation run is stopped early or interrupted after
    producing anatomy folders but before writing the registry index.
    """

    root_path = Path(root) if root is not None else DEFAULT_ANATOMY_REGISTRY_ROOT
    anatomies_root = root_path / "anatomies"
    if not anatomies_root.exists():
        raise FileNotFoundError(f"Anatomies directory does not exist: {anatomies_root}")

    def _sort_key(path: Path) -> tuple[int, str]:
        name = path.name
        if name.startswith("Tree_"):
            suffix = name[5:]
            if suffix.isdigit():
                return (0, f"{int(suffix):08d}")
        return (1, name)

    entries: List[Dict[str, Any]] = []
    for child in sorted(anatomies_root.iterdir(), key=_sort_key):
        if not child.is_dir():
            continue
        desc_path = child / "description.json"
        if not desc_path.exists():
            continue
        entries.append(_registry_entry_from_description(desc_path, root_path=root_path))

    _write_registry_index(root_path, entries)
    return root_path
