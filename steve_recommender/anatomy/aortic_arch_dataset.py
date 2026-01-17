from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from steve_recommender.storage import repo_root


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def load_aortic_arch_dataset(root: Optional[str | Path] = None) -> AorticArchDataset:
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
    )


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _append_index_line(index_path: Path, record: AorticArchRecord) -> None:
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.as_dict, sort_keys=True) + "\n")


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


def generate_aortic_arch_records(
    *,
    dataset: AorticArchDataset,
    start_index: int,
    count: int,
    dataset_seed: int,
    arch_types: Sequence[str],
    write_centerlines: bool = True,
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

    from steve_recommender.adapters import eve

    # Normalize/validate arch types once.
    allowed = {t.value for t in eve.intervention.vesseltree.ArchType}
    chosen = [t for t in arch_types if t in allowed]
    if not chosen:
        raise ValueError(f"No valid arch_types provided. Allowed: {sorted(allowed)}")

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
        local_rng = np.random.default_rng(int(dataset_seed + i))

        arch_type = str(local_rng.choice(chosen))
        seed = int(local_rng.integers(0, 2**31 - 1))

        # Optional geometric variation (kept small; can be expanded later).
        rotation_yzx_deg = (
            float(local_rng.uniform(-5.0, 5.0)),
            float(local_rng.uniform(-5.0, 5.0)),
            float(local_rng.uniform(-5.0, 5.0)),
        )
        scaling_xyzd = (
            float(local_rng.uniform(0.9, 1.1)),
            float(local_rng.uniform(0.9, 1.1)),
            float(local_rng.uniform(0.9, 1.1)),
            float(local_rng.uniform(0.9, 1.1)),
        )

        # Keep omit_axis off for 3D preview by default.
        omit_axis: Optional[str] = None

        record = AorticArchRecord(
            record_id=record_id,
            arch_type=arch_type,
            seed=seed,
            rotation_yzx_deg=rotation_yzx_deg,
            scaling_xyzd=scaling_xyzd,
            omit_axis=omit_axis,
            created_at=_now_iso(),
            centerline_npz=f"records/{record_id}/centerline.npz" if write_centerlines else None,
        )

        # Write to disk
        _ensure_dir(out_dir)
        _write_json(out_dir / "description.json", record.as_dict)

        if write_centerlines:
            vessel_tree = eve.intervention.vesseltree.AorticArch(
                arch_type=eve.intervention.vesseltree.ArchType(record.arch_type),
                seed=record.seed,
                rotation_yzx_deg=record.rotation_yzx_deg,
                scaling_xyzd=record.scaling_xyzd,
                omit_axis=record.omit_axis,
            )
            vessel_tree.reset()
            arrays = _polyline_bundle_from_vesseltree(vessel_tree)
            np.savez_compressed(out_dir / "centerline.npz", **arrays)

        _append_index_line(dataset.index_path, record)
        records.append(record)

        if progress_every > 0 and (offset + 1) % progress_every == 0:
            print(f"[aortic_arch_dataset] generated {offset + 1}/{count} (latest={record_id})")

    return records
