from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable


_STEP_FIELDS = [
    "step",
    "time_s",
    "scene",
    "probe",
    "mode",
    "insert_action",
    "rotate_action",
    "controller_status",
    "commanded_force_scalar_n",
    "commanded_force_vector_n",
    "commanded_force_vector_scene",
    "command_apply_status",
    "norm_sum_vector",
    "sum_norm",
    "peak_triangle_force",
    "peak_triangle_id",
    "total_force_vector",
    "total_force_norm",
    "sum_force_vector",
    "sum_force_gap_norm",
    "sum_abs_fn",
    "sum_abs_ft",
    "fn_vector_sum",
    "ft_vector_sum",
    "decomposition_gap_norm",
    "lambda_abs_sum",
    "lambda_abs_max",
    "lambda_dt_abs_sum",
    "lambda_dt_abs_max",
    "lambda_active_rows_count",
    "si_converted",
    "explicit_association",
    "internal_validated",
    "oracle_physical_pass",
    "oracle_reason",
    "oracle_f_ref_n",
    "oracle_f_meas_n",
    "oracle_abs_error",
    "oracle_rel_error",
    "oracle_abs_tol",
    "oracle_rel_tol",
    "quality_tier",
    "association_method",
    "association_coverage",
    "association_explicit_force_coverage",
    "association_ordering_stable",
    "active_constraint_step",
    "gap_active_projected_count",
    "gap_explicit_mapped_count",
    "gap_unmapped_count",
    "gap_contact_mode",
    "native_contact_export_status",
    "wall_contact_count",
    "wall_contact_detected",
    "dt_s",
    "wall_reference_normal",
]


_TRIANGLE_FIELDS = [
    "step",
    "triangle_id",
    "active",
    "force_x",
    "force_y",
    "force_z",
    "force_norm",
    "normal_x",
    "normal_y",
    "normal_z",
    "fn_scalar",
    "fn_x",
    "fn_y",
    "fn_z",
    "fn_abs",
    "ft_x",
    "ft_y",
    "ft_z",
    "ft_abs",
    "ft_over_fn",
]


class PlaygroundRunIO:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._steps_csv = (self.run_dir / "steps.csv").open("w", encoding="utf-8", newline="")
        self._steps_writer = csv.DictWriter(self._steps_csv, fieldnames=_STEP_FIELDS, delimiter=";")
        self._steps_writer.writeheader()

        self._tri_csv = (self.run_dir / "triangle_forces.csv").open("w", encoding="utf-8", newline="")
        self._tri_writer = csv.DictWriter(self._tri_csv, fieldnames=_TRIANGLE_FIELDS, delimiter=";")
        self._tri_writer.writeheader()

        self._steps_jsonl = (self.run_dir / "steps.jsonl").open("w", encoding="utf-8")

    @staticmethod
    def build_run_dir(output_root: str, run_name: str) -> Path:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        return Path(output_root) / f"{ts}_{run_name}"

    def write_config(self, config: Dict[str, Any]) -> None:
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if isinstance(value, (list, dict, tuple)):
            return json.dumps(value, sort_keys=True)
        if value is None:
            return ""
        return value

    def append_step(self, step_record: Dict[str, Any]) -> None:
        row = {k: self._csv_value(step_record.get(k, "")) for k in _STEP_FIELDS}
        self._steps_writer.writerow(row)
        self._steps_csv.flush()

        self._steps_jsonl.write(json.dumps(step_record, sort_keys=True) + "\n")
        self._steps_jsonl.flush()

    def append_triangles(self, triangle_rows: Iterable[Dict[str, Any]]) -> None:
        for row in triangle_rows:
            self._tri_writer.writerow({k: self._csv_value(row.get(k, "")) for k in _TRIANGLE_FIELDS})
        self._tri_csv.flush()

    def write_oracle_report(self, oracle_report: Dict[str, Any]) -> None:
        with (self.run_dir / "oracle_report.json").open("w", encoding="utf-8") as f:
            json.dump(oracle_report, f, indent=2, sort_keys=True)

    def write_summary(self, text: str) -> None:
        (self.run_dir / "summary.md").write_text(text, encoding="utf-8")

    def close(self) -> None:
        self._steps_jsonl.close()
        self._steps_csv.close()
        self._tri_csv.close()
