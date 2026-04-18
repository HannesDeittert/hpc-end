from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class ModelEntry:
    name: str
    description: str
    model_definition_path: str
    wires_root: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ModelEntry":
        return cls(
            name=str(payload.get("name", "")),
            description=str(payload.get("description", "")),
            model_definition_path=str(payload.get("model_definition_path", "")),
            wires_root=str(payload.get("wires_root", "")),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "model_definition_path": self.model_definition_path,
            "wires_root": self.wires_root,
        }


@dataclass(frozen=True)
class WireEntry:
    tool_ref: str
    model: str
    name: str
    wire_dir: str
    tool_py_path: str
    tool_definition_path: Optional[str]
    agents_dir: Optional[str]

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "WireEntry":
        tool_def = payload.get("tool_definition_path")
        agents_dir = payload.get("agents_dir")
        return cls(
            tool_ref=str(payload.get("tool_ref", "")),
            model=str(payload.get("model", "")),
            name=str(payload.get("name", "")),
            wire_dir=str(payload.get("wire_dir", "")),
            tool_py_path=str(payload.get("tool_py_path", "")),
            tool_definition_path=(str(tool_def) if tool_def is not None else None),
            agents_dir=(str(agents_dir) if agents_dir is not None else None),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "tool_ref": self.tool_ref,
            "model": self.model,
            "name": self.name,
            "wire_dir": self.wire_dir,
            "tool_py_path": self.tool_py_path,
            "tool_definition_path": self.tool_definition_path,
            "agents_dir": self.agents_dir,
        }


@dataclass(frozen=True)
class AgentEntry:
    agent_ref: str
    model: str
    wire: str
    name: str
    agent_json_path: str
    checkpoint_path: Optional[str]
    run_dir: Optional[str]
    checkpoint_exists: bool

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AgentEntry":
        checkpoint_path = payload.get("checkpoint_path")
        run_dir = payload.get("run_dir")
        return cls(
            agent_ref=str(payload.get("agent_ref", "")),
            model=str(payload.get("model", "")),
            wire=str(payload.get("wire", "")),
            name=str(payload.get("name", "")),
            agent_json_path=str(payload.get("agent_json_path", "")),
            checkpoint_path=(
                str(checkpoint_path) if checkpoint_path is not None else None
            ),
            run_dir=(str(run_dir) if run_dir is not None else None),
            checkpoint_exists=bool(payload.get("checkpoint_exists", False)),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent_ref": self.agent_ref,
            "model": self.model,
            "wire": self.wire,
            "name": self.name,
            "agent_json_path": self.agent_json_path,
            "checkpoint_path": self.checkpoint_path,
            "run_dir": self.run_dir,
            "checkpoint_exists": self.checkpoint_exists,
        }


@dataclass(frozen=True)
class RegistryIndex:
    schema_version: int = 1
    created_at_utc: str = ""
    updated_at_utc: str = ""
    models: Dict[str, ModelEntry] = field(default_factory=dict)
    wires: Dict[str, WireEntry] = field(default_factory=dict)
    agents: Dict[str, AgentEntry] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RegistryIndex":
        models_raw = payload.get("models") or {}
        wires_raw = payload.get("wires") or {}
        agents_raw = payload.get("agents") or {}

        models: Dict[str, ModelEntry] = {}
        if isinstance(models_raw, Mapping):
            for key, value in models_raw.items():
                if isinstance(key, str) and isinstance(value, Mapping):
                    models[key] = ModelEntry.from_dict(value)

        wires: Dict[str, WireEntry] = {}
        if isinstance(wires_raw, Mapping):
            for key, value in wires_raw.items():
                if isinstance(key, str) and isinstance(value, Mapping):
                    wires[key] = WireEntry.from_dict(value)

        agents: Dict[str, AgentEntry] = {}
        if isinstance(agents_raw, Mapping):
            for key, value in agents_raw.items():
                if isinstance(key, str) and isinstance(value, Mapping):
                    agents[key] = AgentEntry.from_dict(value)

        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            created_at_utc=str(payload.get("created_at_utc", "")),
            updated_at_utc=str(payload.get("updated_at_utc", "")),
            models=models,
            wires=wires,
            agents=agents,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "models": {name: entry.to_dict() for name, entry in self.models.items()},
            "wires": {name: entry.to_dict() for name, entry in self.wires.items()},
            "agents": {name: entry.to_dict() for name, entry in self.agents.items()},
        }

