import importlib.util
import inspect
from dataclasses import is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Type


from dev.gw_tool_recommender.storage import data_root, list_models, parse_wire_ref, wire_py_path


DATA_DIR = data_root()

# Legacy fallback if older runs wrote tools directly under data/<tool>/tool.py
LEGACY_FLAT_DATA_DIR = DATA_DIR


def _load_module_from_path(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[call-arg]
    return mod


def list_devices(data_dir: Path = DATA_DIR) -> List[str]:
    """Return available device refs.

    New layout: data/<model>/wires/<wire>/tool.py -> "model/wire"
    Legacy layout: data/<wire>/tool.py -> "wire"
    """
    names: List[str] = []

    for model in list_models():
        model_dir = data_dir / model / "wires"
        if not model_dir.exists():
            continue
        for wire in sorted(model_dir.iterdir()):
            if wire.is_dir() and (wire / "tool.py").exists():
                names.append(f"{model}/{wire.name}")

    if LEGACY_FLAT_DATA_DIR.exists():
        for child in sorted(LEGACY_FLAT_DATA_DIR.iterdir()):
            if child.is_dir() and (child / "tool.py").exists():
                if child.name not in names:
                    names.append(child.name)
    return names


def load_device_class(
    tool_name: str,
    data_dir: Path = DATA_DIR,
    expected_class_name: Optional[str] = None,
) -> Type[Any]:
    """Load the device class from the stored tool module.

    Accepts either:
    - "model/wire" (preferred): data/<model>/wires/<wire>/tool.py
    - "wire" (legacy): data/<wire>/tool.py
    """
    model, wire = parse_wire_ref(tool_name)

    if model:
        tool_py = wire_py_path(model, wire)
        if not tool_py.exists():
            raise FileNotFoundError(f"No tool.py for '{tool_name}' at {tool_py}")
    else:
        tool_py = data_dir / wire / "tool.py"
        if not tool_py.exists():
            matches: List[Path] = []
            for m in list_models():
                candidate = wire_py_path(m, wire)
                if candidate.exists():
                    matches.append(candidate)
            if len(matches) == 1:
                tool_py = matches[0]
            elif len(matches) > 1:
                refs = ", ".join(f"{p.parents[1].name}/{p.parent.name}" for p in matches)
                raise FileNotFoundError(
                    f"Ambiguous wire name '{wire}'. Use one of: {refs}"
                )
            else:
                raise FileNotFoundError(f"No tool.py for '{tool_name}' at {tool_py}")

    mod = _load_module_from_path(f"gw_tool_{tool_name}", tool_py)

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
