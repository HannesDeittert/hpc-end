import importlib.util
import inspect
import sys
import re
from dataclasses import is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Type


from steve_recommender.storage import (
    data_root,
    list_models,
    parse_wire_ref,
    wire_py_path,
    wires_root,
)


DATA_DIR = data_root()


_NON_IDENT = re.compile(r"[^0-9A-Za-z_]")


def _safe_module_name(name: str) -> str:
    return _NON_IDENT.sub("_", name)


def _load_module_from_path(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Ensure the module is visible during execution (required by dataclasses/typing).
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[call-arg]
    return mod


def list_devices(data_dir: Path = DATA_DIR) -> List[str]:
    """Return available device refs.

    Canonical layout: data/wire_registry/<model>/wire_versions/<version>/tool.py
    -> "model/version"
    """
    names: List[str] = []

    for model in list_models():
        versions_root = wires_root(model)
        if not versions_root.exists():
            continue
        for wire in sorted(versions_root.iterdir()):
            if wire.is_dir() and (wire / "tool.py").exists():
                names.append(f"{model}/{wire.name}")
    return names


def load_device_class(
    tool_name: str,
    data_dir: Path = DATA_DIR,
    expected_class_name: Optional[str] = None,
) -> Type[Any]:
    """Load the device class from the stored tool module.

    Accepts "model/version" refs, resolved through storage paths.
    """
    model, wire = parse_wire_ref(tool_name)

    if model:
        tool_py = wire_py_path(model, wire)
        if not tool_py.exists():
            raise FileNotFoundError(f"No tool.py for '{tool_name}' at {tool_py}")

        # Prefer a normal import so pickling works with multiprocessing ("spawn").
        try:
            mod = __import__(
                f"data.wire_registry.{model}.wire_versions.{wire}.tool",
                fromlist=["*"],
            )
        except Exception:
            mod = _load_module_from_path(
                f"steve_tool_{_safe_module_name(tool_name)}",
                tool_py,
            )
    else:
        raise FileNotFoundError(
            f"Tool ref '{tool_name}' must be fully qualified as 'model/version'."
        )

    class_name = expected_class_name or wire
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
