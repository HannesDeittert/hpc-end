#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_DIR = SCRIPT_DIR.parent / "experiments" / "master-thesis" / "notebook_helpers"
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

from e1 import target_equivalence_report  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an E1 job manifest locally")
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    report = target_equivalence_report(payload)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["same_targets_across_configs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
