from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

SCHEMA_VERSION = 1


def _records_to_array(records: Sequence[dict[str, Any]], key: str, *, dtype: Any) -> np.ndarray:
    return np.asarray([record.get(key) for record in records], dtype=dtype).reshape((-1,))


def _records_to_json(records: Sequence[dict[str, Any]]) -> str:
    return json.dumps([dict(record) for record in records], sort_keys=True)


def write_force_trace_npz(
    path: Path | str,
    *,
    triangle_records: Sequence[dict[str, Any]] = (),
    wire_records: Sequence[dict[str, Any]] = (),
    metadata: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    triangle_records = tuple(triangle_records)
    wire_records = tuple(wire_records)
    np.savez_compressed(
        out_path,
        schema_version=np.asarray([SCHEMA_VERSION], dtype=np.int64),
        triangle_timestep=_records_to_array(triangle_records, "timestep", dtype=np.int64),
        triangle_id=_records_to_array(triangle_records, "triangle_id", dtype=np.int64),
        triangle_fx_N=_records_to_array(triangle_records, "fx_N", dtype=np.float64),
        triangle_fy_N=_records_to_array(triangle_records, "fy_N", dtype=np.float64),
        triangle_fz_N=_records_to_array(triangle_records, "fz_N", dtype=np.float64),
        triangle_norm_N=_records_to_array(triangle_records, "norm_N", dtype=np.float64),
        triangle_contributing_rows=_records_to_array(triangle_records, "contributing_rows", dtype=np.int64),
        wire_timestep=_records_to_array(wire_records, "timestep", dtype=np.int64),
        wire_collision_dof=_records_to_array(wire_records, "wire_collision_dof", dtype=np.int64),
        wire_row_idx=_records_to_array(wire_records, "row_idx", dtype=np.int64),
        wire_fx_N=_records_to_array(wire_records, "fx_N", dtype=np.float64),
        wire_fy_N=_records_to_array(wire_records, "fy_N", dtype=np.float64),
        wire_fz_N=_records_to_array(wire_records, "fz_N", dtype=np.float64),
        wire_norm_N=_records_to_array(wire_records, "norm_N", dtype=np.float64),
        triangle_records_json=np.asarray(_records_to_json(triangle_records)),
        wire_records_json=np.asarray(_records_to_json(wire_records)),
        metadata_json=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    )
    return out_path


def read_force_trace_npz(path: Path | str) -> dict[str, Any]:
    in_path = Path(path)
    with np.load(in_path, allow_pickle=True) as data:
        schema_version = int(np.asarray(data["schema_version"]).reshape((-1,))[0])
        triangle_records = json.loads(str(np.asarray(data["triangle_records_json"]).reshape(())))
        wire_records = json.loads(str(np.asarray(data["wire_records_json"]).reshape(())))
        metadata = json.loads(str(np.asarray(data["metadata_json"]).reshape(())))
        return {
            "schema_version": schema_version,
            "triangle_records": triangle_records,
            "wire_records": wire_records,
            "metadata": metadata,
            "triangle_timestep": np.asarray(data["triangle_timestep"]),
            "wire_timestep": np.asarray(data["wire_timestep"]),
        }


def write_force_trace_jsonl(
    path: Path | str,
    records: Iterable[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "record_type": "header",
                    "schema_version": SCHEMA_VERSION,
                    "metadata": metadata or {},
                },
                sort_keys=True,
            )
            + "\n"
        )
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
    return out_path


def read_force_trace_jsonl(path: Path | str) -> dict[str, Any]:
    in_path = Path(path)
    header: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("record_type") == "header" and header is None:
                header = payload
                continue
            records.append(payload)
    return {
        "schema_version": int(header.get("schema_version", SCHEMA_VERSION)) if header else SCHEMA_VERSION,
        "metadata": (header or {}).get("metadata", {}),
        "records": records,
    }
