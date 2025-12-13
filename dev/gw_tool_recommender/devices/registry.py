from __future__ import annotations

import importlib.util
import inspect
from dataclasses import is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Type


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]

# Canonical location for stored tools/devices.
# The UI writes to <repo>/data/<tool_name>/tool.py.
DATA_DIR = PROJECT_ROOT / "data"

# Backward-compatible fallback if older runs wrote under dev/...
LEGACY_DATA_DIR = PROJECT_ROOT / "dev" / "gw_tool_recommender" / "data"


def _load_module_from_path(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[call-arg]
    return mod


def list_devices(data_dir: Path = DATA_DIR) -> List[str]:
    """Return all device/tool names available under data/<name>/tool.py.

    Searches canonical data_dir first, then legacy fallback.
    """
    names: List[str] = []
    for base in (data_dir, LEGACY_DATA_DIR):
        if not base.exists():
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "tool.py").exists():
                if child.name not in names:
                    names.append(child.name)
    return names


def load_device_class(
    tool_name: str,
    data_dir: Path = DATA_DIR,
    expected_class_name: Optional[str] = None,
) -> Type[Any]:
    """Load the device class from data/<tool_name>/tool.py.

    By default it looks for a class matching tool_name. If not found,
    it returns the first dataclass-like class in the module.
    """
    tool_py = data_dir / tool_name / "tool.py"
    if not tool_py.exists():
        legacy_py = LEGACY_DATA_DIR / tool_name / "tool.py"
        if legacy_py.exists():
            tool_py = legacy_py
        else:
            raise FileNotFoundError(f"No tool.py for '{tool_name}' at {tool_py}")

    mod = _load_module_from_path(f"gw_tool_{tool_name}", tool_py)

    class_name = expected_class_name or tool_name
    if hasattr(mod, class_name):
        cls = getattr(mod, class_name)
        if inspect.isclass(cls):
            return cls

    # Fallback: first class that looks like a device dataclass.
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ != mod.__name__:
            continue
        if is_dataclass(obj):
            return obj

    raise ImportError(
        f"Could not find a suitable device class in {tool_py}. "
        f"Expected '{class_name}' or a dataclass."
    )


def make_device(
    tool_name: str,
    overrides: Optional[Dict[str, Any]] = None,
    data_dir: Path = DATA_DIR,
    expected_class_name: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Instantiate a stored device, with optional field overrides.

    Overrides can be passed either via 'overrides' dict or kwargs.
    """
    cls = load_device_class(
        tool_name, data_dir=data_dir, expected_class_name=expected_class_name
    )
    init_kwargs: Dict[str, Any] = {}
    if overrides:
        init_kwargs.update(overrides)
    init_kwargs.update(kwargs)
    return cls(**init_kwargs)
