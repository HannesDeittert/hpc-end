from __future__ import annotations

import argparse
from pathlib import Path

from steve_recommender.evaluation.config import load_config
from steve_recommender.evaluation.force_calibration import run_force_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and cache SI force calibration")
    parser.add_argument("--config", required=True, help="Path to evaluation YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    rec = run_force_calibration(cfg)
    print(
        "[force-calibration] key={key} passed={passed} status={status}".format(
            key=rec.key,
            passed=int(rec.passed),
            status=rec.validation_status,
        )
    )
    print(f"[force-calibration] cache: {cfg.force_extraction.calibration.cache_path}")
    probe_a = rec.metrics.get("probe_a", {}) if isinstance(rec.metrics, dict) else {}
    probe_b = rec.metrics.get("probe_b", {}) if isinstance(rec.metrics, dict) else {}
    repro = rec.metrics.get("reproducibility", {}) if isinstance(rec.metrics, dict) else {}
    ref_suite = rec.metrics.get("reference_suite", {}) if isinstance(rec.metrics, dict) else {}
    print(f"[force-calibration] probe_a: {probe_a.get('run_dir', '')}")
    print(f"[force-calibration] probe_b: {probe_b.get('run_dir', '')}")
    print(
        "[force-calibration] reproducibility passed={passed} error={err}".format(
            passed=repro.get("passed", ""),
            err=repro.get("error", ""),
        )
    )
    if ref_suite:
        print(
            "[force-calibration] reference_suite passed={passed} external_limit={limit} reason={reason}".format(
                passed=ref_suite.get("pass_validated_suite", ""),
                limit=ref_suite.get("external_limit_detected", ""),
                reason=ref_suite.get("external_limit_reason", ""),
            )
        )


if __name__ == "__main__":
    main()
