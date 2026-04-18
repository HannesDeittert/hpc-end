from __future__ import annotations

import json
from pathlib import Path

from .types import RegistryIndex


DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "index.json"


def load_registry(index_path: Path = DEFAULT_INDEX_PATH) -> RegistryIndex:
    """Load the registry index from disk.

    Returns an empty index when the file does not exist yet.
    """

    if not index_path.exists():
        return RegistryIndex()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Registry index must be a JSON object: {index_path}")
    return RegistryIndex.from_dict(payload)


def save_registry(index: RegistryIndex, index_path: Path = DEFAULT_INDEX_PATH) -> Path:
    """Persist the registry index to disk."""

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(index.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return index_path

