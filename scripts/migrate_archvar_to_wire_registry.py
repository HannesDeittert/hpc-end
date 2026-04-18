#!/usr/bin/env python3
"""Migrate ArchVarJShaped wires into the canonical data/wire_registry layout.

This script performs Phases 1-3 from the agreed migration plan:
- inventory manifest (freeze source facts)
- canonical skeleton + tool/tool_definition/model_definition creation
- full agent/checkpoint copy with rewritten agent.json and parity checks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class MappingEntry:
    old_wire: str
    new_model: str
    new_version: str
    model_display_name: str
    model_description: str
    producer: str


MAPPINGS: Tuple[MappingEntry, ...] = (
    MappingEntry(
        old_wire="JShaped_Default",
        new_model="steve_default",
        new_version="default",
        model_display_name="stEVE Default",
        model_description="stEVE baseline J-shaped guidewire family.",
        producer="stEVE",
    ),
    MappingEntry(
        old_wire="JShaped_Default_StraightTip",
        new_model="steve_default",
        new_version="straight_tip",
        model_display_name="stEVE Default",
        model_description="stEVE baseline J-shaped guidewire family.",
        producer="stEVE",
    ),
    MappingEntry(
        old_wire="j_shaped_AmplatzSuperStiff",
        new_model="amplatz_super_stiff",
        new_version="default",
        model_display_name="Amplatz Super Stiff",
        model_description="Amplatz Super Stiff J-shaped guidewire family.",
        producer="Boston Scientific",
    ),
    MappingEntry(
        old_wire="j_shaped_UniversalII",
        new_model="universal_ii",
        new_version="default",
        model_display_name="Universal II",
        model_description="Universal II J-shaped guidewire family.",
        producer="Abbott",
    ),
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _touch_init(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "__init__.py").touch(exist_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _inventory_manifest(src_wires_root: Path, out_path: Path) -> Dict[str, object]:
    version_entries: List[Dict[str, object]] = []
    total_agents = 0
    total_checkpoints = 0

    for m in MAPPINGS:
        src_wire_dir = src_wires_root / m.old_wire
        agents_root = src_wire_dir / "agents"
        agent_entries: List[Dict[str, object]] = []

        for agent_dir in sorted([p for p in agents_root.iterdir() if p.is_dir()]):
            checkpoints = sorted(agent_dir.glob("*.everl"))
            total_agents += 1
            total_checkpoints += len(checkpoints)
            agent_entries.append(
                {
                    "agent_name": agent_dir.name,
                    "agent_json": str((agent_dir / "agent.json").resolve()),
                    "checkpoint_count": len(checkpoints),
                    "best_checkpoint_exists": (agent_dir / "best_checkpoint.everl").exists(),
                    "checkpoints": [str(p.resolve()) for p in checkpoints],
                }
            )

        version_entries.append(
            {
                "legacy_wire": m.old_wire,
                "new_model": m.new_model,
                "new_version": m.new_version,
                "source_wire_dir": str(src_wire_dir.resolve()),
                "source_tool_py": str((src_wire_dir / "tool.py").resolve()),
                "source_tool_definition": str((src_wire_dir / "tool_definition.json").resolve()),
                "agents": agent_entries,
            }
        )

    manifest: Dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": _now_iso(),
        "source_model": "ArchVarJShaped",
        "versions": version_entries,
        "baseline_targets": {
            "version_count": 4,
            "agent_count": 4,
            "checkpoint_count": 324,
        },
        "observed_counts": {
            "version_count": len(version_entries),
            "agent_count": total_agents,
            "checkpoint_count": total_checkpoints,
        },
    }
    _save_json(out_path, manifest)
    return manifest


def _steve_default_tool_definitions() -> Dict[str, Dict[str, object]]:
    default_spec: Dict[str, object] = {
        "name": "guidewire",
        "velocity_limit": [35.0, 3.14],
        "length": 450.0,
        "tip_radius": 12.1,
        "tip_angle": 0.4 * math.pi,
        "tip_outer_diameter": 0.7,
        "tip_inner_diameter": 0.0,
        "straight_outer_diameter": 0.89,
        "straight_inner_diameter": 0.0,
        "poisson_ratio": 0.49,
        "young_modulus_tip": 17e3,
        "young_modulus_straight": 80e3,
        "mass_density_tip": 0.000021,
        "mass_density_straight": 0.000021,
        "visu_edges_per_mm": 0.5,
        "collis_edges_per_mm_tip": 2.0,
        "collis_edges_per_mm_straight": 0.1,
        "beams_per_mm_tip": 1.4,
        "beams_per_mm_straight": 0.5,
        "color": [0.0, 0.0, 0.0],
    }
    straight_spec = dict(default_spec)
    default_tip_length = 12.1 * 0.4 * math.pi
    overstraight_tip_radius = 1000.0
    straight_spec["tip_radius"] = overstraight_tip_radius
    straight_spec["tip_angle"] = default_tip_length / overstraight_tip_radius

    return {
        "default": {
            "name": "default",
            "description": "Default stEVE J-shaped wire variant.",
            "type": "procedural",
            "model": "steve_default",
            "version": "default",
            "status": "trained",
            "policy_available": True,
            "spec": default_spec,
        },
        "straight_tip": {
            "name": "straight_tip",
            "description": "Straight-tip variant of stEVE default J-shaped wire.",
            "type": "procedural",
            "model": "steve_default",
            "version": "straight_tip",
            "status": "trained",
            "policy_available": True,
            "spec": straight_spec,
        },
    }


def _rewrite_straight_tip_import(text: str) -> str:
    old = "from data.ArchVarJShaped.wires.JShaped_Default.tool import JShaped_Default"
    new = (
        "from data.wire_registry.steve_default.wire_versions.default.tool "
        "import JShaped_Default"
    )
    return text.replace(old, new)


def _copy_tool_and_definition(src_wire_dir: Path, dst_version_dir: Path, mapping: MappingEntry) -> None:
    tool_text = (src_wire_dir / "tool.py").read_text(encoding="utf-8")
    if mapping.new_model == "steve_default" and mapping.new_version == "straight_tip":
        tool_text = _rewrite_straight_tip_import(tool_text)
    (dst_version_dir / "tool.py").write_text(tool_text, encoding="utf-8")

    src_tool_def = src_wire_dir / "tool_definition.json"
    if src_tool_def.exists():
        payload = _load_json(src_tool_def)
        payload["model"] = mapping.new_model
        payload["version"] = mapping.new_version
        payload["status"] = "trained"
        payload["policy_available"] = True
        _save_json(dst_version_dir / "tool_definition.json", payload)


def _create_model_definitions(registry_root: Path) -> None:
    steve_defs = _steve_default_tool_definitions()

    model_payloads: Dict[str, Dict[str, object]] = {
        "steve_default": {
            "name": "steve_default",
            "display_name": "stEVE Default",
            "description": "stEVE baseline J-shaped guidewire family with default and straight-tip variants.",
            "producer": "stEVE",
            "family": "j_shaped_guidewire",
            "default_version": "default",
            "wire_versions": ["default", "straight_tip"],
            "overall_properties": {
                "length_mm": steve_defs["default"]["spec"]["length"],
                "velocity_limit": steve_defs["default"]["spec"]["velocity_limit"],
            },
            "source": {
                "legacy_model": "ArchVarJShaped",
                "legacy_wires": ["JShaped_Default", "JShaped_Default_StraightTip"],
                "migrated_at_utc": _now_iso(),
            },
        },
        "amplatz_super_stiff": {
            "name": "amplatz_super_stiff",
            "display_name": "Amplatz Super Stiff",
            "description": "Amplatz Super Stiff J-shaped guidewire family.",
            "producer": "Boston Scientific",
            "family": "j_shaped_guidewire",
            "default_version": "default",
            "wire_versions": ["default"],
            "overall_properties": {
                "length_mm": 450.0,
                "velocity_limit": [35.0, 3.14],
            },
            "source": {
                "legacy_model": "ArchVarJShaped",
                "legacy_wires": ["j_shaped_AmplatzSuperStiff"],
                "migrated_at_utc": _now_iso(),
            },
        },
        "universal_ii": {
            "name": "universal_ii",
            "display_name": "Universal II",
            "description": "Universal II J-shaped guidewire family.",
            "producer": "Abbott",
            "family": "j_shaped_guidewire",
            "default_version": "default",
            "wire_versions": ["default"],
            "overall_properties": {
                "length_mm": 450.0,
                "velocity_limit": [35.0, 3.14],
            },
            "source": {
                "legacy_model": "ArchVarJShaped",
                "legacy_wires": ["j_shaped_UniversalII"],
                "migrated_at_utc": _now_iso(),
            },
        },
    }

    for model, payload in model_payloads.items():
        _save_json(registry_root / model / "model_definition.json", payload)


def _write_steve_tool_definitions(registry_root: Path) -> None:
    defs = _steve_default_tool_definitions()
    _save_json(
        registry_root
        / "steve_default"
        / "wire_versions"
        / "default"
        / "tool_definition.json",
        defs["default"],
    )
    _save_json(
        registry_root
        / "steve_default"
        / "wire_versions"
        / "straight_tip"
        / "tool_definition.json",
        defs["straight_tip"],
    )


def _copy_agents_and_checkpoints(
    src_wire_dir: Path,
    dst_version_dir: Path,
    tool_ref: str,
) -> Dict[str, object]:
    agents_root = src_wire_dir / "agents"
    dst_agents_root = dst_version_dir / "agents"
    _touch_init(dst_agents_root)

    copied_agents = 0
    copied_checkpoints = 0
    size_checks: List[Dict[str, object]] = []
    best_hash_checks: List[Dict[str, object]] = []

    for src_agent_dir in sorted([p for p in agents_root.iterdir() if p.is_dir()]):
        copied_agents += 1
        dst_agent_dir = dst_agents_root / src_agent_dir.name
        dst_ckpt_dir = dst_agent_dir / "checkpoints"

        _touch_init(dst_agent_dir)
        _touch_init(dst_ckpt_dir)

        checkpoints = sorted(src_agent_dir.glob("*.everl"))
        for src_ckpt in checkpoints:
            dst_ckpt = dst_ckpt_dir / src_ckpt.name
            shutil.copy2(src_ckpt, dst_ckpt)
            copied_checkpoints += 1
            size_checks.append(
                {
                    "src": str(src_ckpt.resolve()),
                    "dst": str(dst_ckpt.resolve()),
                    "src_size": src_ckpt.stat().st_size,
                    "dst_size": dst_ckpt.stat().st_size,
                    "size_match": src_ckpt.stat().st_size == dst_ckpt.stat().st_size,
                }
            )

        src_agent_json = src_agent_dir / "agent.json"
        payload = _load_json(src_agent_json) if src_agent_json.exists() else {}
        source_checkpoint = payload.get("checkpoint")

        dst_best = dst_ckpt_dir / "best_checkpoint.everl"
        if dst_best.exists():
            selected = dst_best
        elif isinstance(source_checkpoint, str) and source_checkpoint:
            basename = Path(source_checkpoint).name
            candidate = dst_ckpt_dir / basename
            selected = candidate if candidate.exists() else None
        else:
            selected = None

        if selected is None:
            all_ckpts = sorted(dst_ckpt_dir.glob("*.everl"))
            selected = all_ckpts[0] if all_ckpts else None

        if dst_best.exists():
            src_best = src_agent_dir / "best_checkpoint.everl"
            best_hash_checks.append(
                {
                    "agent": src_agent_dir.name,
                    "src_best": str(src_best.resolve()),
                    "dst_best": str(dst_best.resolve()),
                    "src_sha256": _sha256(src_best),
                    "dst_sha256": _sha256(dst_best),
                    "hash_match": _sha256(src_best) == _sha256(dst_best),
                }
            )

        new_payload = dict(payload)
        new_payload["name"] = str(new_payload.get("name", src_agent_dir.name))
        new_payload["tool"] = tool_ref
        new_payload["checkpoint"] = str(selected.resolve()) if selected else None
        new_payload["run_dir"] = str(dst_ckpt_dir.resolve())
        new_payload["source_agent_json"] = (
            str(src_agent_json.resolve()) if src_agent_json.exists() else None
        )
        new_payload["source_checkpoint"] = (
            str(source_checkpoint) if isinstance(source_checkpoint, str) else None
        )
        _save_json(dst_agent_dir / "agent.json", new_payload)

    return {
        "copied_agents": copied_agents,
        "copied_checkpoints": copied_checkpoints,
        "size_checks": size_checks,
        "best_hash_checks": best_hash_checks,
    }


def migrate(clean: bool) -> Dict[str, object]:
    root = _repo_root()
    data_root = root / "data"
    src_model_root = data_root / "ArchVarJShaped"
    src_wires_root = src_model_root / "wires"
    registry_root = data_root / "wire_registry"

    if clean:
        for model in ("steve_default", "amplatz_super_stiff", "universal_ii"):
            target = registry_root / model
            if target.exists():
                shutil.rmtree(target)

    # Ensure package roots
    _touch_init(registry_root)

    inventory_manifest_path = registry_root / "archvar_inventory_manifest.json"
    inventory_manifest = _inventory_manifest(src_wires_root, inventory_manifest_path)

    for model in ("steve_default", "amplatz_super_stiff", "universal_ii"):
        _touch_init(registry_root / model)
        _touch_init(registry_root / model / "wire_versions")

    for mapping in MAPPINGS:
        src_wire_dir = src_wires_root / mapping.old_wire
        dst_version_dir = (
            registry_root / mapping.new_model / "wire_versions" / mapping.new_version
        )
        _touch_init(dst_version_dir)
        _copy_tool_and_definition(src_wire_dir, dst_version_dir, mapping)

    _write_steve_tool_definitions(registry_root)
    _create_model_definitions(registry_root)

    total_agents = 0
    total_checkpoints = 0
    all_size_checks: List[Dict[str, object]] = []
    all_best_hash_checks: List[Dict[str, object]] = []

    for mapping in MAPPINGS:
        src_wire_dir = src_wires_root / mapping.old_wire
        dst_version_dir = (
            registry_root / mapping.new_model / "wire_versions" / mapping.new_version
        )
        tool_ref = f"{mapping.new_model}/{mapping.new_version}"
        copied = _copy_agents_and_checkpoints(src_wire_dir, dst_version_dir, tool_ref)
        total_agents += int(copied["copied_agents"])
        total_checkpoints += int(copied["copied_checkpoints"])
        all_size_checks.extend(copied["size_checks"])  # type: ignore[arg-type]
        all_best_hash_checks.extend(copied["best_hash_checks"])  # type: ignore[arg-type]

    size_ok = all(item["size_match"] for item in all_size_checks)
    best_hash_ok = all(item["hash_match"] for item in all_best_hash_checks)

    migration_manifest: Dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": _now_iso(),
        "scope": "ArchVarJShaped",
        "targets": {
            "version_count": 4,
            "agent_count": 4,
            "checkpoint_count": 324,
        },
        "observed": {
            "version_count": 4,
            "agent_count": total_agents,
            "checkpoint_count": total_checkpoints,
        },
        "parity": {
            "agent_count_ok": total_agents == 4,
            "checkpoint_count_ok": total_checkpoints == 324,
            "all_sizes_match": size_ok,
            "best_checkpoint_hash_match": best_hash_ok,
        },
        "inventory_manifest": str(inventory_manifest_path.resolve()),
        "size_checks": all_size_checks,
        "best_checkpoint_hash_checks": all_best_hash_checks,
    }
    _save_json(registry_root / "archvar_migration_manifest.json", migration_manifest)
    return migration_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing target wire_registry model folders before migration.",
    )
    args = parser.parse_args()

    manifest = migrate(clean=args.clean)
    print("[ok] archvar migration complete")
    print("[ok] versions:", manifest["observed"]["version_count"])  # type: ignore[index]
    print("[ok] agents:", manifest["observed"]["agent_count"])  # type: ignore[index]
    print(
        "[ok] checkpoints:",
        manifest["observed"]["checkpoint_count"],  # type: ignore[index]
    )
    print("[ok] parity:", manifest["parity"])  # type: ignore[index]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

