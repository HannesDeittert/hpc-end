#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_DIR = SCRIPT_DIR.parent / "experiments" / "master-thesis" / "notebook_helpers"
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

from e1 import probe_cluster_partitions  # noqa: E402


def main() -> int:
    payload = {
        "schema_version": 1,
        "partitions": [
            {
                "partition": row.partition,
                "nodes_total": row.nodes_total,
                "states": row.states,
                "gres": list(row.gres),
                "cpus_per_node": row.cpus_per_node,
            }
            for row in probe_cluster_partitions()
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
