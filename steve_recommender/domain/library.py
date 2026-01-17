"""Domain models for the local device/agent library."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelInfo:
    name: str
    description: str
    path: Path


@dataclass(frozen=True)
class WireInfo:
    name: str
    model: str
    description: str
    path: Path
    tool_py: Path
    definition_path: Path


@dataclass(frozen=True)
class AgentInfo:
    name: str
    model: str
    wire: str
    path: Path
    checkpoint_path: Optional[Path]
    metadata_path: Optional[Path]
