"""Service helpers for browsing local models, wires, and agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from steve_recommender.domain import AgentInfo, ModelInfo, WireInfo
from steve_recommender.storage import (
    list_models as _list_models,
    list_wires as _list_wires,
    model_definition_path,
    wire_agents_dir,
    wire_definition_path,
    wire_dir,
)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_checkpoint(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    candidates = []
    for ext in (".everl", ".pth", ".pt"):
        candidates.extend(path.rglob(f"*{ext}"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def list_models() -> List[ModelInfo]:
    models: List[ModelInfo] = []
    for name in _list_models():
        definition = model_definition_path(name)
        payload = _read_json(definition)
        models.append(
            ModelInfo(
                name=name,
                description=str(payload.get("description", "")),
                path=definition.parent,
            )
        )
    return models


def list_wires(model_name: str) -> List[WireInfo]:
    wires: List[WireInfo] = []
    for wire in _list_wires(model_name):
        definition = wire_definition_path(model_name, wire)
        payload = _read_json(definition)
        wires.append(
            WireInfo(
                name=wire,
                model=model_name,
                description=str(payload.get("description", "")),
                path=wire_dir(model_name, wire),
                tool_py=wire_dir(model_name, wire) / "tool.py",
                definition_path=definition,
            )
        )
    return wires


def list_agents(model_name: str, wire_name: str) -> List[AgentInfo]:
    agents: List[AgentInfo] = []
    root = wire_agents_dir(model_name, wire_name)
    if not root.exists():
        return agents
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        metadata_path = child / "agent.json"
        checkpoint = _latest_checkpoint(child)
        if metadata_path.exists():
            payload = _read_json(metadata_path)
            checkpoint_val = payload.get("checkpoint")
            if checkpoint_val:
                checkpoint = Path(checkpoint_val)
        agents.append(
            AgentInfo(
                name=child.name,
                model=model_name,
                wire=wire_name,
                path=child,
                checkpoint_path=checkpoint if checkpoint and checkpoint.exists() else None,
                metadata_path=metadata_path if metadata_path.exists() else None,
            )
        )
    return agents
