from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .store import DEFAULT_INDEX_PATH, load_registry, save_registry
from .types import AgentEntry, ModelEntry, RegistryIndex, WireEntry


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _discover_repo_root(start: Path) -> Path:
    start_dir = start if start.is_dir() else start.parent
    for parent in (start_dir, *start_dir.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return start_dir


def _read_model_description(model_definition_path: Path) -> str:
    try:
        payload = json.loads(model_definition_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if isinstance(payload, dict):
        return str(payload.get("description", ""))
    return ""


def bootstrap_registry(repo_root: Optional[Path] = None) -> RegistryIndex:
    """Build a registry snapshot from canonical wire-registry folders.

    Expected layout:
      data/wire_registry/<model>/model_definition.json
      data/wire_registry/<model>/wire_versions/<version>/...
    """

    this_file = Path(__file__).resolve()
    repo = repo_root or _discover_repo_root(this_file)
    registry_root = repo / "data" / "wire_registry"

    models: Dict[str, ModelEntry] = {}
    wires: Dict[str, WireEntry] = {}
    agents: Dict[str, AgentEntry] = {}

    if not registry_root.exists():
        now = _now_utc_iso()
        return RegistryIndex(
            schema_version=1,
            created_at_utc=now,
            updated_at_utc=now,
            models=models,
            wires=wires,
            agents=agents,
        )

    for child in sorted(registry_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("__"):
            continue

        model_definition = child / "model_definition.json"
        versions_root = child / "wire_versions"
        if not model_definition.exists() or not versions_root.exists():
            continue

        model_name = child.name
        models[model_name] = ModelEntry(
            name=model_name,
            description=_read_model_description(model_definition),
            model_definition_path=str(model_definition.resolve()),
            wires_root=str(versions_root.resolve()),
        )

        for version_dir in sorted(versions_root.iterdir()):
            if not version_dir.is_dir() or version_dir.name.startswith("__"):
                continue
            tool_py = version_dir / "tool.py"
            if not tool_py.exists():
                continue

            version_name = version_dir.name
            tool_ref = f"{model_name}/{version_name}"
            tool_definition = version_dir / "tool_definition.json"
            agents_dir = version_dir / "agents"

            wires[tool_ref] = WireEntry(
                tool_ref=tool_ref,
                model=model_name,
                name=version_name,
                wire_dir=str(version_dir.resolve()),
                tool_py_path=str(tool_py.resolve()),
                tool_definition_path=(
                    str(tool_definition.resolve()) if tool_definition.exists() else None
                ),
                agents_dir=str(agents_dir.resolve()) if agents_dir.exists() else None,
            )

            if not agents_dir.exists():
                continue

            for agent_dir in sorted(agents_dir.iterdir()):
                if not agent_dir.is_dir():
                    continue
                agent_json = agent_dir / "agent.json"
                if not agent_json.exists():
                    continue

                try:
                    payload = json.loads(agent_json.read_text(encoding="utf-8"))
                except Exception:
                    payload = {}

                checkpoint_path = payload.get("checkpoint")
                run_dir = payload.get("run_dir")
                checkpoint_exists = False
                if checkpoint_path:
                    checkpoint_exists = Path(str(checkpoint_path)).exists()

                agent_name = agent_dir.name
                agent_ref = f"{tool_ref}:{agent_name}"
                agents[agent_ref] = AgentEntry(
                    agent_ref=agent_ref,
                    model=model_name,
                    wire=version_name,
                    name=agent_name,
                    agent_json_path=str(agent_json.resolve()),
                    checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                    run_dir=str(run_dir) if run_dir else None,
                    checkpoint_exists=checkpoint_exists,
                )

    now = _now_utc_iso()
    return RegistryIndex(
        schema_version=1,
        created_at_utc=now,
        updated_at_utc=now,
        models=models,
        wires=wires,
        agents=agents,
    )


def bootstrap_to_disk(index_path: Path = DEFAULT_INDEX_PATH) -> Path:
    """Rebuild the registry index and write it to disk."""

    existing = load_registry(index_path)
    rebuilt = bootstrap_registry()
    if existing.created_at_utc:
        rebuilt = replace(rebuilt, created_at_utc=existing.created_at_utc)
    rebuilt = replace(rebuilt, updated_at_utc=_now_utc_iso())
    return save_registry(rebuilt, index_path)

