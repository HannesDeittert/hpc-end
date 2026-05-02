from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .models import AgentRef, AorticArchAnatomy, PolicySpec, WireRef


DEFAULT_ANATOMY_REGISTRY_PATH = Path("data/anatomy_registry/index.json")
DEFAULT_WIRE_REGISTRY_PATH = Path("data/wire_registry/index.json")
DEFAULT_EXPLICIT_POLICY_MANIFEST_PATH = Path("data/wire_registry/archvar_inventory_manifest.json")


def _read_json_file(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Registry file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Registry root must be an object, got {type(payload)}")
    return payload


def _optional_tuple(
    value: Any,
    *,
    expected_length: int,
    field_name: str,
) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list/tuple or null, got {type(value)}")
    if len(value) != expected_length:
        raise ValueError(
            f"{field_name} must have length {expected_length}, got {len(value)}"
        )
    return tuple(float(item) for item in value)


def _resolve_optional_path(value: Any, *, base_dir: Path) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_required_path(value: Any, *, base_dir: Path, field_name: str) -> Path:
    path = _resolve_optional_path(value, base_dir=base_dir)
    if path is None:
        raise ValueError(f"{field_name} must be provided")
    return path


def _read_description_file(path: Path) -> Mapping[str, Any]:
    payload = _read_json_file(path)
    return payload


def _map_anatomy_entry(
    index_entry: Mapping[str, Any],
    *,
    registry_dir: Path,
) -> AorticArchAnatomy:
    description_path = _resolve_required_path(
        index_entry.get("description_path"),
        base_dir=registry_dir,
        field_name="description_path",
    )
    raw = _read_description_file(description_path)
    anatomy_dir = description_path.parent
    return AorticArchAnatomy(
        anatomy_type=str(raw.get("anatomy_type", index_entry.get("anatomy_type", "aortic_arch"))),
        arch_type=str(raw["arch_type"]),
        seed=int(raw["seed"]),
        rotation_yzx_deg=_optional_tuple(
            raw.get("rotation_yzx_deg"),
            expected_length=3,
            field_name="rotation_yzx_deg",
        ),
        scaling_xyzd=_optional_tuple(
            raw.get("scaling_xyzd"),
            expected_length=4,
            field_name="scaling_xyzd",
        ),
        omit_axis=raw.get("omit_axis"),
        record_id=str(index_entry.get("record_id", raw.get("record_id", ""))) or None,
        created_at=str(index_entry.get("created_at", raw.get("created_at", ""))),
        centerline_bundle_path=_resolve_optional_path(
            raw.get("centerline_bundle_path"),
            base_dir=anatomy_dir,
        ),
        simulation_mesh_path=_resolve_optional_path(
            raw.get("simulation_mesh_path", raw.get("simulation_mesh")),
            base_dir=anatomy_dir,
        ),
        visualization_mesh_path=_resolve_optional_path(
            raw.get("visualization_mesh_path", raw.get("visu_mesh")),
            base_dir=anatomy_dir,
        ),
    )


class FileBasedAnatomyDiscovery:
    """Read anatomies from a self-sufficient anatomy registry directory.

    The registry uses an `index.json` file that points to per-anatomy
    `description.json` specs plus attached mesh assets. No legacy recommender
    code, dataset classes, or service helpers are required.
    """

    def __init__(self, *, registry_path: Path = DEFAULT_ANATOMY_REGISTRY_PATH) -> None:
        self._registry_path = Path(registry_path)

    @property
    def registry_path(self) -> Path:
        return self._registry_path

    def list_anatomies(
        self,
        *,
        registry_path: Optional[Path] = None,
        limit: Optional[int] = None,
        random_sample: bool = False,
    ) -> Tuple[AorticArchAnatomy, ...]:
        """Return all anatomies from the configured registry index."""

        path = Path(registry_path) if registry_path is not None else self._registry_path
        payload = _read_json_file(path)
        raw_anatomies = payload.get("anatomies", [])
        if not isinstance(raw_anatomies, list):
            raise TypeError(
                f"Anatomy registry 'anatomies' field must be a list, got {type(raw_anatomies)}"
            )
        registry_dir = path.parent
        if limit is not None:
            limit = max(0, int(limit))
            if limit < len(raw_anatomies):
                if random_sample:
                    rng = np.random.default_rng()
                    indices = np.sort(rng.choice(len(raw_anatomies), size=limit, replace=False))
                    raw_anatomies = [raw_anatomies[int(idx)] for idx in indices]
                else:
                    raw_anatomies = raw_anatomies[:limit]
        return tuple(
            _map_anatomy_entry(raw, registry_dir=registry_dir) for raw in raw_anatomies
        )

    def get_anatomy(
        self,
        *,
        record_id: str,
        registry_path: Optional[Path] = None,
    ) -> AorticArchAnatomy:
        """Return one anatomy by its stable registry record id."""

        for anatomy in self.list_anatomies(registry_path=registry_path):
            if anatomy.record_id == record_id:
                return anatomy
        raise KeyError(f"Unknown anatomy record_id: {record_id}")


def _map_wire_entry(raw: Mapping[str, Any]) -> WireRef:
    model = str(raw.get("model", "")).strip()
    wire = str(raw.get("name", raw.get("wire", ""))).strip()
    return WireRef(model=model, wire=wire)


def _load_agent_metadata(
    raw: Mapping[str, Any],
    *,
    registry_dir: Path,
) -> Mapping[str, Any]:
    agent_json_path = _resolve_required_path(
        raw.get("agent_json_path"),
        base_dir=registry_dir,
        field_name="agent_json_path",
    )
    if not agent_json_path.exists():
        raise FileNotFoundError(f"Agent metadata file does not exist: {agent_json_path}")
    return _read_json_file(agent_json_path)


def _map_registry_policy_entry(
    raw: Mapping[str, Any],
    *,
    registry_dir: Path,
) -> Optional[PolicySpec]:
    metadata_path = _resolve_required_path(
        raw.get("agent_json_path"),
        base_dir=registry_dir,
        field_name="agent_json_path",
    )
    agent_payload = _load_agent_metadata(raw, registry_dir=registry_dir)
    trained_on_wire = WireRef(
        model=str(raw.get("model", "")).strip(),
        wire=str(raw.get("wire", raw.get("name", ""))).strip(),
    )
    registry_agent = AgentRef(
        wire=trained_on_wire,
        agent=str(agent_payload.get("name", raw.get("name", ""))).strip(),
    )
    checkpoint_path = _resolve_optional_path(
        agent_payload.get("checkpoint", raw.get("checkpoint_path")),
        base_dir=metadata_path.parent,
    )
    if checkpoint_path is None or not checkpoint_path.exists():
        return None
    run_dir = _resolve_optional_path(
        agent_payload.get("run_dir", raw.get("run_dir")),
        base_dir=metadata_path.parent,
    )
    return PolicySpec(
        name=registry_agent.agent,
        checkpoint_path=checkpoint_path,
        source="registry",
        trained_on_wire=trained_on_wire,
        registry_agent=registry_agent,
        metadata_path=metadata_path,
        run_dir=run_dir,
    )


class FileBasedWireRegistryDiscovery:
    """Read execution wires and registry-backed agents from `data/wire_registry`.

    The adapter only consumes raw registry files on disk. A wire is considered
    startable when it has at least one loadable agent checkpoint.
    """

    def __init__(self, *, registry_path: Path = DEFAULT_WIRE_REGISTRY_PATH) -> None:
        self._registry_path = Path(registry_path)

    @property
    def registry_path(self) -> Path:
        return self._registry_path

    def list_execution_wires(self) -> Tuple[WireRef, ...]:
        payload = _read_json_file(self._registry_path)
        raw_wires = payload.get("wires", {})
        if not isinstance(raw_wires, dict):
            raise TypeError(
                f"Wire registry 'wires' field must be an object, got {type(raw_wires)}"
            )
        wires = tuple(
            _map_wire_entry(raw_wires[key]) for key in sorted(raw_wires.keys())
        )
        return wires

    def list_startable_wires(self) -> Tuple[WireRef, ...]:
        seen: set[WireRef] = set()
        ordered: list[WireRef] = []
        for policy in self.list_registry_policies():
            wire = policy.trained_on_wire
            if wire is None or wire in seen:
                continue
            seen.add(wire)
            ordered.append(wire)
        return tuple(ordered)

    def list_registry_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        payload = _read_json_file(self._registry_path)
        raw_agents = payload.get("agents", {})
        if not isinstance(raw_agents, dict):
            raise TypeError(
                f"Wire registry 'agents' field must be an object, got {type(raw_agents)}"
            )
        registry_dir = self._registry_path.parent
        policies: list[PolicySpec] = []
        for key in sorted(raw_agents.keys()):
            policy = _map_registry_policy_entry(raw_agents[key], registry_dir=registry_dir)
            if policy is None:
                continue
            if execution_wire is not None and policy.trained_on_wire != execution_wire:
                continue
            policies.append(policy)
        return tuple(policies)

    def list_explicit_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        _ = execution_wire
        return ()

    def resolve_policy_from_agent_ref(self, agent_ref: AgentRef) -> PolicySpec:
        for policy in self.list_registry_policies():
            if policy.registry_agent == agent_ref:
                return policy
        raise KeyError(f"Unknown registry agent: {agent_ref.agent_ref}")


def _map_explicit_policy_entry(
    version_raw: Mapping[str, Any],
    agent_raw: Mapping[str, Any],
    *,
    manifest_dir: Path,
) -> Optional[PolicySpec]:
    new_model = str(version_raw.get("new_model", "")).strip()
    new_version = str(version_raw.get("new_version", "")).strip()
    if not new_model or not new_version:
        return None

    source_agent_json = _resolve_required_path(
        agent_raw.get("agent_json"),
        base_dir=manifest_dir,
        field_name="agent_json",
    )
    if not source_agent_json.exists():
        return None

    agent_payload = _read_json_file(source_agent_json)
    checkpoint_path = _resolve_optional_path(
        agent_payload.get("checkpoint", agent_raw.get("source_checkpoint")),
        base_dir=source_agent_json.parent,
    )
    if checkpoint_path is None or not checkpoint_path.exists():
        return None

    agent_name = str(agent_payload.get("name", agent_raw.get("agent_name", ""))).strip()
    if not agent_name:
        return None

    trained_on_wire = WireRef(model=new_model, wire=new_version)
    return PolicySpec(
        name=agent_name,
        checkpoint_path=checkpoint_path,
        source="explicit",
        trained_on_wire=trained_on_wire,
        registry_agent=AgentRef(wire=trained_on_wire, agent=agent_name),
        metadata_path=source_agent_json,
        run_dir=_resolve_optional_path(
            agent_payload.get("run_dir", agent_raw.get("run_dir")),
            base_dir=source_agent_json.parent,
        ),
    )


class FileBasedExplicitPolicyDiscovery:
    """Read explicit policies from the wire-registry inventory manifest."""

    def __init__(
        self,
        *,
        manifest_path: Path = DEFAULT_EXPLICIT_POLICY_MANIFEST_PATH,
    ) -> None:
        self._manifest_path = Path(manifest_path)

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def list_explicit_policies(
        self,
        *,
        execution_wire: Optional[WireRef] = None,
    ) -> Tuple[PolicySpec, ...]:
        payload = _read_json_file(self._manifest_path)
        raw_versions = payload.get("versions", [])
        if not isinstance(raw_versions, list):
            raise TypeError(
                f"Explicit policy manifest 'versions' field must be a list, got {type(raw_versions)}"
            )

        manifest_dir = self._manifest_path.parent
        policies: list[PolicySpec] = []
        for version_raw in raw_versions:
            if not isinstance(version_raw, dict):
                raise TypeError(
                    f"Explicit policy manifest version entries must be objects, got {type(version_raw)}"
                )
            raw_agents = version_raw.get("agents", [])
            if not isinstance(raw_agents, list):
                raise TypeError(
                    f"Explicit policy manifest 'agents' field must be a list, got {type(raw_agents)}"
                )
            for agent_raw in raw_agents:
                if not isinstance(agent_raw, dict):
                    raise TypeError(
                        f"Explicit policy manifest agent entries must be objects, got {type(agent_raw)}"
                    )
                policy = _map_explicit_policy_entry(
                    version_raw,
                    agent_raw,
                    manifest_dir=manifest_dir,
                )
                if policy is None:
                    continue
                if execution_wire is not None and policy.trained_on_wire != execution_wire:
                    continue
                policies.append(policy)
        return tuple(policies)

    def resolve_policy_from_agent_ref(self, agent_ref: AgentRef) -> PolicySpec:
        for policy in self.list_explicit_policies():
            if policy.registry_agent == agent_ref:
                return policy
        raise KeyError(f"Unknown explicit policy: {agent_ref.agent_ref}")


__all__ = [
    "DEFAULT_ANATOMY_REGISTRY_PATH",
    "DEFAULT_EXPLICIT_POLICY_MANIFEST_PATH",
    "DEFAULT_WIRE_REGISTRY_PATH",
    "FileBasedAnatomyDiscovery",
    "FileBasedExplicitPolicyDiscovery",
    "FileBasedWireRegistryDiscovery",
]
