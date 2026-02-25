from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


HERE = Path(__file__).resolve()


def repo_root() -> Path:
    markers = (".git", "pyproject.toml", "README.md")
    for parent in (HERE.parent, *HERE.parents):
        for marker in markers:
            if (parent / marker).exists():
                return parent
    return HERE.parent


def data_root() -> Path:
    return repo_root() / "data"


def model_dir(model_name: str) -> Path:
    return data_root() / model_name


def model_definition_path(model_name: str) -> Path:
    return model_dir(model_name) / "model_definition.json"


def wires_root(model_name: str) -> Path:
    return model_dir(model_name) / "wires"


def wire_dir(model_name: str, wire_name: str) -> Path:
    return wires_root(model_name) / wire_name


def wire_definition_path(model_name: str, wire_name: str) -> Path:
    return wire_dir(model_name, wire_name) / "tool_definition.json"


def wire_py_path(model_name: str, wire_name: str) -> Path:
    return wire_dir(model_name, wire_name) / "tool.py"


def wire_agents_dir(model_name: str, wire_name: str) -> Path:
    return wire_dir(model_name, wire_name) / "agents"


def parse_wire_ref(ref: str) -> Tuple[Optional[str], str]:
    if "/" in ref:
        model, wire = ref.split("/", 1)
        return model, wire
    return None, ref


def list_models() -> List[str]:
    root = data_root()
    if not root.exists():
        return []
    names: List[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "model_definition.json").exists():
            names.append(child.name)
    return names


def list_wires(model_name: str) -> List[str]:
    root = wires_root(model_name)
    if not root.exists():
        return []
    wires: List[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "tool.py").exists():
            wires.append(child.name)
    return wires


def ensure_model(model_name: str, description: str = "") -> Path:
    mdir = model_dir(model_name)
    mdir.mkdir(parents=True, exist_ok=True)
    (data_root() / "__init__.py").touch(exist_ok=True)
    (mdir / "__init__.py").touch(exist_ok=True)
    wires_root(model_name).mkdir(parents=True, exist_ok=True)
    (wires_root(model_name) / "__init__.py").touch(exist_ok=True)

    definition = model_definition_path(model_name)
    if not definition.exists():
        definition.write_text(
            json.dumps({"name": model_name, "description": description}, indent=2),
            encoding="utf-8",
        )
    return mdir


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
