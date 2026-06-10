#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("results/master_thesis/e2_tinyfat_finished_readable_20260515/metadata/job_manifest_work.json")
DEFAULT_FORCE_AWARE = Path("results/master_thesis/e2_force_aware_20260515/metadata/job_manifest_force_aware.json")

UNCHANGED_FIELDS = (
    "anatomy_id",
    "anatomy_index",
    "target_index",
    "seed_base",
    "target_spec",
    "config_id",
    "config_spec",
    "friction",
    "write_full_trace",
    "write_diagnostics",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify E2 force-aware manifest preserves the completed E2 source jobs.")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--force-aware", type=Path, default=DEFAULT_FORCE_AWARE)
    ap.add_argument("--job-count", type=int, default=1700)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    source_jobs = _read(args.source).get("jobs", [])
    force_payload = _read(args.force_aware)
    force_jobs = force_payload.get("jobs", [])
    errors: list[str] = []
    if len(force_jobs) != int(args.job_count):
        errors.append(f"force-aware job count {len(force_jobs)} != expected {args.job_count}")
    if len(source_jobs) < int(args.job_count):
        errors.append(f"source job count {len(source_jobs)} < expected {args.job_count}")

    for idx in range(min(int(args.job_count), len(source_jobs), len(force_jobs))):
        src = source_jobs[idx]
        dst = force_jobs[idx]
        if dst.get("source_e2_array_index") != idx:
            errors.append(f"job {idx}: source_e2_array_index={dst.get('source_e2_array_index')!r}")
        for field in UNCHANGED_FIELDS:
            if src.get(field) != dst.get(field):
                errors.append(f"job {idx}: field {field} changed")
        candidates = dst.get("force_aware_candidates", [])
        if len(candidates) != 9:
            errors.append(f"job {idx}: force_aware_candidates count {len(candidates)} != 9")
        for candidate in candidates:
            rel = candidate.get("checkpoint_project_relative")
            if rel and not Path(rel).exists():
                errors.append(f"job {idx}: local checkpoint missing: {rel}")

    if errors:
        for error in errors[:100]:
            print(f"ERROR {error}")
        if len(errors) > 100:
            print(f"... {len(errors) - 100} more errors")
        return 1

    print(f"OK: verified {args.job_count} force-aware jobs against {args.source}")
    print("Preserved fields:", ", ".join(UNCHANGED_FIELDS))
    print("Force-aware candidates per job: 9")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
