"""Canonical wire/agent registry under ``data/wire_registry``."""

from .bootstrap import bootstrap_registry, bootstrap_to_disk
from .store import DEFAULT_INDEX_PATH, load_registry, save_registry
from .types import AgentEntry, ModelEntry, RegistryIndex, WireEntry

__all__ = [
    "AgentEntry",
    "ModelEntry",
    "RegistryIndex",
    "WireEntry",
    "DEFAULT_INDEX_PATH",
    "load_registry",
    "save_registry",
    "bootstrap_registry",
    "bootstrap_to_disk",
]
