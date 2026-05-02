from __future__ import annotations

import importlib
import inspect
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from third_party.stEVE.eve.intervention import MonoPlaneStatic
from third_party.stEVE.eve.intervention.device import Device, JShaped
from third_party.stEVE.eve.intervention.simulation import Simulation, SofaBeamAdapter

from .builders import build_aortic_arch, build_fluoroscopy, build_target
from .discovery import DEFAULT_WIRE_REGISTRY_PATH
from .models import EvaluationCandidate, EvaluationScenario, PolicySpec, WireRef


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_STEVE_PATH = PROJECT_ROOT / "third_party" / "stEVE"
LOCAL_STEVE_RL_PATH = PROJECT_ROOT / "third_party" / "stEVE_rl"


class PlayPolicy(Protocol):
    """Runtime policy handle used by eval_v2 runners."""

    device: object

    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        ...

    def reset(self) -> None:
        ...

    def close(self) -> None:
        ...

    def to(self, device: object) -> None:
        ...


@dataclass(frozen=True)
class PreparedEvaluationRuntime:
    """Fully assembled runtime bundle for one candidate/scenario pair."""

    candidate: EvaluationCandidate
    scenario: EvaluationScenario
    device: Device
    intervention: MonoPlaneStatic
    play_policy: PlayPolicy


def _read_json_file(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Registry file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Registry root must be an object, got {type(payload)}")
    return payload


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


def _load_wire_entry(
    wire_ref: WireRef,
    *,
    registry_path: Path,
) -> Mapping[str, Any]:
    payload = _read_json_file(registry_path)
    raw_wires = payload.get("wires", {})
    if not isinstance(raw_wires, dict):
        raise TypeError(
            f"Wire registry 'wires' field must be an object, got {type(raw_wires)}"
        )
    key = wire_ref.tool_ref
    if key not in raw_wires:
        raise KeyError(f"Unknown wire ref: {key}")
    raw = raw_wires[key]
    if not isinstance(raw, dict):
        raise TypeError(f"Wire entry must be an object, got {type(raw)}")
    return raw


def build_device(
    wire_ref: WireRef,
    *,
    registry_path: Path = DEFAULT_WIRE_REGISTRY_PATH,
) -> Device:
    """Build a real stEVE device from the file-based wire registry."""

    raw = _load_wire_entry(wire_ref, registry_path=Path(registry_path))
    tool_definition_path = _resolve_required_path(
        raw.get("tool_definition_path"),
        base_dir=Path(registry_path).parent,
        field_name="tool_definition_path",
    )
    tool_payload = _read_json_file(tool_definition_path)
    wire_type = str(tool_payload.get("type", "")).strip()
    if wire_type != "procedural":
        raise NotImplementedError(f"Unsupported wire type: {wire_type!r}")
    spec = tool_payload.get("spec")
    if not isinstance(spec, dict):
        raise TypeError(f"Tool definition 'spec' must be an object, got {type(spec)}")
    return JShaped(
        name=str(spec["name"]),
        velocity_limit=tuple(float(item) for item in spec["velocity_limit"]),
        length=float(spec["length"]),
        tip_radius=float(spec["tip_radius"]),
        tip_angle=float(spec["tip_angle"]),
        tip_outer_diameter=float(spec["tip_outer_diameter"]),
        tip_inner_diameter=float(spec["tip_inner_diameter"]),
        straight_outer_diameter=float(spec["straight_outer_diameter"]),
        straight_inner_diameter=float(spec["straight_inner_diameter"]),
        poisson_ratio=float(spec["poisson_ratio"]),
        young_modulus_tip=float(spec["young_modulus_tip"]),
        young_modulus_straight=float(spec["young_modulus_straight"]),
        mass_density_tip=float(spec["mass_density_tip"]),
        mass_density_straight=float(spec["mass_density_straight"]),
        visu_edges_per_mm=float(spec["visu_edges_per_mm"]),
        collis_edges_per_mm_tip=float(spec["collis_edges_per_mm_tip"]),
        collis_edges_per_mm_straight=float(spec["collis_edges_per_mm_straight"]),
        beams_per_mm_tip=float(spec["beams_per_mm_tip"]),
        beams_per_mm_straight=float(spec["beams_per_mm_straight"]),
        color=tuple(float(item) for item in spec["color"]),
    )


def _purge_modules(*, prefix: str) -> None:
    for name in list(sys.modules.keys()):
        if name == prefix or name.startswith(f"{prefix}."):
            sys.modules.pop(name, None)


def _prepend_sys_path(path: Path) -> None:
    text = str(path)
    sys.path[:] = [item for item in sys.path if item != text]
    sys.path.insert(0, text)


def _is_relative_to(path: Path, other: Path) -> bool:
    """Backport Path.is_relative_to for the Python 3.8 SOFA runtime."""

    try:
        path.resolve().relative_to(other.resolve())
    except ValueError:
        return False
    return True


def _module_is_local(module_name: str, *, expected_root: Path) -> bool:
    module = sys.modules.get(module_name)
    if module is None:
        return False
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    return _is_relative_to(Path(module_file), expected_root)


def _ensure_local_steve_imports() -> None:
    _prepend_sys_path(LOCAL_STEVE_RL_PATH)
    _prepend_sys_path(LOCAL_STEVE_PATH)

    if not _module_is_local("eve", expected_root=LOCAL_STEVE_PATH):
        _purge_modules(prefix="eve")
    if not _module_is_local("eve_rl", expected_root=LOCAL_STEVE_RL_PATH):
        _purge_modules(prefix="eve_rl")

    eve = importlib.import_module("eve")
    eve_rl = importlib.import_module("eve_rl")

    if not _is_relative_to(Path(eve.__file__), LOCAL_STEVE_PATH):
        raise ImportError(f"Failed to import local stEVE package from {LOCAL_STEVE_PATH}")
    if not _is_relative_to(Path(eve_rl.__file__), LOCAL_STEVE_RL_PATH):
        raise ImportError(
            f"Failed to import local stEVE_rl package from {LOCAL_STEVE_RL_PATH}"
        )


@contextmanager
def _legacy_checkpoint_load_context() -> Iterator[None]:
    original_torch_load = torch.load
    patched_scheduler_inits: dict[type[object], object] = {}

    base_scheduler_types = []
    for name in ("LRScheduler", "_LRScheduler"):
        base = getattr(lr_scheduler, name, None)
        if isinstance(base, type):
            base_scheduler_types.append(base)
    base_scheduler_types = tuple(base_scheduler_types)

    def _compat_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    def _patch_scheduler_verbose_kwargs() -> None:
        if not base_scheduler_types:
            return
        for _, cls in vars(lr_scheduler).items():
            if not isinstance(cls, type):
                continue
            if not issubclass(cls, base_scheduler_types):
                continue
            init = getattr(cls, "__init__", None)
            if init is None:
                continue
            try:
                sig = inspect.signature(init)
            except Exception:
                continue
            if "verbose" in sig.parameters:
                continue

            def _wrapped_init(self, *args, __orig=init, **kwargs):
                kwargs.pop("verbose", None)
                return __orig(self, *args, **kwargs)

            patched_scheduler_inits[cls] = init
            cls.__init__ = _wrapped_init  # type: ignore[assignment]

    torch.load = _compat_torch_load  # type: ignore[assignment]
    _patch_scheduler_verbose_kwargs()
    try:
        yield
    finally:
        for cls, init in patched_scheduler_inits.items():
            cls.__init__ = init  # type: ignore[assignment]
        torch.load = original_torch_load  # type: ignore[assignment]


def load_play_policy(
    policy: PolicySpec,
    *,
    device: str = "cpu",
) -> PlayPolicy:
    """Load a play-only policy runtime from a trusted local checkpoint."""

    _ensure_local_steve_imports()
    with _legacy_checkpoint_load_context():
        from eve_rl.algo import AlgoPlayOnly  # type: ignore[import-not-found]

        algo = AlgoPlayOnly.from_checkpoint(str(policy.checkpoint_path))
    algo.to(torch.device(device))
    return algo


def build_intervention(
    *,
    candidate: EvaluationCandidate,
    scenario: EvaluationScenario,
    simulation: Optional[Simulation] = None,
    registry_path: Path = DEFAULT_WIRE_REGISTRY_PATH,
) -> Tuple[MonoPlaneStatic, Device]:
    """Assemble a full stEVE intervention from one candidate and one scenario."""

    device = build_device(candidate.execution_wire, registry_path=registry_path)
    vessel_tree = build_aortic_arch(scenario.anatomy)
    sim = simulation if simulation is not None else SofaBeamAdapter(friction=scenario.friction)
    fluoroscopy = build_fluoroscopy(
        spec=scenario.fluoroscopy,
        vessel_tree=vessel_tree,
        simulation=sim,
    )
    target = build_target(
        scenario.target,
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
    )
    intervention = MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=sim,
        fluoroscopy=fluoroscopy,
        target=target,
        stop_device_at_tree_end=scenario.stop_device_at_tree_end,
        normalize_action=scenario.normalize_action,
    )
    return intervention, device


def safe_reset_intervention(
    intervention: MonoPlaneStatic,
    *,
    episode_number: int = 0,
    seed: Optional[int] = None,
) -> None:
    """Reset a real stEVE intervention with Python-native seeds.

    stEVE's current `MonoPlaneStatic.reset(seed=...)` forwards NumPy integer
    scalars into target resets. On current Python this breaks the
    `random.Random(seed)` path used by branch-random and manual targets.
    eval_v2 keeps the behavior local and deterministic by normalizing all
    derived seeds to plain Python `int`.
    """

    if seed is not None:
        intervention._np_random = np.random.default_rng(int(seed))

    vessel_seed = (
        None
        if seed is None
        else int(intervention._np_random.integers(0, 2**31))
    )
    target_seed_override = getattr(intervention.target, "_eval_v2_seed_override", None)
    if target_seed_override is not None:
        target_seed = int(target_seed_override)
    elif seed is None:
        target_seed = None
    else:
        target_seed = int(intervention._np_random.integers(0, 2**31))
    simulation_seed = (
        None
        if seed is None
        else int(intervention._np_random.integers(0, 2**31))
    )
    intervention.vessel_tree.reset(episode_number, vessel_seed)

    insertion = intervention.vessel_tree.insertion
    intervention.simulation.reset(
        insertion_point=insertion.position,
        insertion_direction=insertion.direction,
        mesh_path=intervention.vessel_tree.mesh_path,
        devices=intervention.devices,
        coords_low=intervention.vessel_tree.coordinate_space.low,
        coords_high=intervention.vessel_tree.coordinate_space.high,
        vessel_visual_path=intervention.vessel_tree.visu_mesh_path,
        seed=simulation_seed,
    )

    intervention.target.reset(episode_number, target_seed)
    intervention.fluoroscopy.reset(episode_number)
    intervention.last_action *= 0.0


def prepare_evaluation_runtime(
    *,
    candidate: EvaluationCandidate,
    scenario: EvaluationScenario,
    simulation: Optional[Simulation] = None,
    registry_path: Path = DEFAULT_WIRE_REGISTRY_PATH,
    policy_device: str = "cpu",
) -> PreparedEvaluationRuntime:
    """Build the runnable intervention and load the matching play policy."""

    intervention, device = build_intervention(
        candidate=candidate,
        scenario=scenario,
        simulation=simulation,
        registry_path=registry_path,
    )
    play_policy = load_play_policy(candidate.policy, device=policy_device)
    return PreparedEvaluationRuntime(
        candidate=candidate,
        scenario=scenario,
        device=device,
        intervention=intervention,
        play_policy=play_policy,
    )


__all__ = [
    "PlayPolicy",
    "PreparedEvaluationRuntime",
    "build_device",
    "build_intervention",
    "load_play_policy",
    "prepare_evaluation_runtime",
    "safe_reset_intervention",
]
