from __future__ import annotations

import json
import unittest
from pathlib import Path
import tempfile

from steve_recommender.eval_v2.force_trace_persistence import (
    read_force_trace_jsonl,
    read_force_trace_npz,
    write_force_trace_jsonl,
    write_force_trace_npz,
)


class ForceTracePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.triangle_records = [
            {"timestep": 1, "triangle_id": 12, "fx_N": 0.0, "fy_N": 0.0, "fz_N": 0.004, "norm_N": 0.004, "contributing_rows": 1},
        ]
        self.wire_records = [
            {"timestep": 1, "wire_collision_dof": 7, "row_idx": 0, "fx_N": 0.0, "fy_N": 0.0, "fz_N": 0.004, "norm_N": 0.004},
        ]

    def test_persistence_roundtrip_preserves_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "trace.npz"
            jsonl_path = Path(tmp) / "trace.jsonl"
            write_force_trace_npz(npz_path, triangle_records=self.triangle_records, wire_records=self.wire_records, metadata={"seed": 123})
            write_force_trace_jsonl(jsonl_path, self.triangle_records + self.wire_records, metadata={"seed": 123})

            npz = read_force_trace_npz(npz_path)
            jsonl = read_force_trace_jsonl(jsonl_path)

            self.assertEqual(npz["schema_version"], 1)
            self.assertEqual(npz["metadata"], {"seed": 123})
            self.assertEqual(npz["triangle_records"], self.triangle_records)
            self.assertEqual(npz["wire_records"], self.wire_records)
            self.assertEqual(jsonl["schema_version"], 1)
            self.assertEqual(jsonl["metadata"], {"seed": 123})
            self.assertEqual(jsonl["records"], self.triangle_records + self.wire_records)

    def test_persistence_schema_version_in_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "trace.npz"
            write_force_trace_npz(npz_path, triangle_records=self.triangle_records, wire_records=self.wire_records)
            with npz_path.open("rb") as handle:
                payload = handle.read(16)
            self.assertTrue(payload.startswith(b"PK"), "NPZ bundle should be a zip archive")

            npz = read_force_trace_npz(npz_path)
            self.assertEqual(npz["schema_version"], 1)

    def test_jsonl_header_contains_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "trace.jsonl"
            write_force_trace_jsonl(jsonl_path, self.triangle_records, metadata={"seed": 123})
            header = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(header["record_type"], "header")
            self.assertEqual(header["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
