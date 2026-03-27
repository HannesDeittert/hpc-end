from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type
import importlib
import math

import eve
from steve_recommender.bench.straight_eps import StraightEps


@dataclass(frozen=True)
class ToolSpec:
    device_cls: Type
    kwargs: Dict[str, Any]
    description: str = ""


def _jshaped_default_kwargs() -> Dict[str, Any]:
    return dict(
        name="guidewire",
        velocity_limit=(35, 3.14),
        length=450.0,
        tip_radius=12.1,
        tip_angle=0.4 * math.pi,
        tip_outer_diameter=0.7,
        tip_inner_diameter=0.0,
        straight_outer_diameter=0.89,
        straight_inner_diameter=0.0,
        poisson_ratio=0.49,
        young_modulus_tip=17e3,
        young_modulus_straight=80e3,
        mass_density_tip=0.000021,
        mass_density_straight=0.000021,
        visu_edges_per_mm=0.5,
        collis_edges_per_mm_tip=2.0,
        collis_edges_per_mm_straight=0.1,
        beams_per_mm_tip=1.4,
        beams_per_mm_straight=0.5,
        color=(0.0, 0.0, 0.0),
    )


def _universal_common_kwargs() -> Dict[str, Any]:
    return dict(
        name="guidewire",
        velocity_limit=(20.0, 6.0),
        length=450.0,
        tip_outer_diameter=0.3556,
        tip_inner_diameter=0.0,
        straight_outer_diameter=0.3556,
        straight_inner_diameter=0.0,
        poisson_ratio=0.33,
        young_modulus_tip=1.27e9,
        young_modulus_straight=8.93e9,
        mass_density_tip=4000.0,
        mass_density_straight=2500.0,
        visu_edges_per_mm=0.5,
        collis_edges_per_mm_tip=2.0,
        collis_edges_per_mm_straight=0.1,
        beams_per_mm_tip=1.4,
        beams_per_mm_straight=0.5,
        color=(0.0, 0.0, 0.0),
    )


TOOL_SPECS: Dict[str, ToolSpec] = {
    "jshaped_default": ToolSpec(
        eve.intervention.device.JShaped,
        _jshaped_default_kwargs(),
        "Original ArchVar J-shaped guidewire.",
    ),
    "jshaped_default_tipradius_min": ToolSpec(
        eve.intervention.device.JShaped,
        {
            **_jshaped_default_kwargs(),
            # Keep everything identical to jshaped_default, except a minimal bend radius.
            "tip_radius": 0.1,
        },
        "ArchVar default J-shaped guidewire with minimal tip radius (0.1mm).",
    ),
    "jshaped_longtip_40mm": ToolSpec(
        eve.intervention.device.JShaped,
        {
            **_jshaped_default_kwargs(),
            # Match ~40mm arc length: tip_radius * tip_angle
            "tip_radius": 40.0 / (0.4 * math.pi),
        },
        "J-shaped geometry with ~40mm tip arc length.",
    ),
    "jshaped_dense_x2": ToolSpec(
        eve.intervention.device.JShaped,
        {
            **_jshaped_default_kwargs(),
            # Increase discretisation density to raise complexity
            "visu_edges_per_mm": _jshaped_default_kwargs()["visu_edges_per_mm"] * 2.0,
            "collis_edges_per_mm_tip": _jshaped_default_kwargs()["collis_edges_per_mm_tip"] * 2.0,
            "collis_edges_per_mm_straight": _jshaped_default_kwargs()["collis_edges_per_mm_straight"] * 2.0,
            "beams_per_mm_tip": _jshaped_default_kwargs()["beams_per_mm_tip"] * 2.0,
            "beams_per_mm_straight": _jshaped_default_kwargs()["beams_per_mm_straight"] * 2.0,
        },
        "J-shaped with 2x discretisation density (more beams/collisions).",
    ),
    "jshaped_dense_x4": ToolSpec(
        eve.intervention.device.JShaped,
        {
            **_jshaped_default_kwargs(),
            # Increase discretisation density to raise complexity
            "visu_edges_per_mm": _jshaped_default_kwargs()["visu_edges_per_mm"] * 4.0,
            "collis_edges_per_mm_tip": _jshaped_default_kwargs()["collis_edges_per_mm_tip"] * 4.0,
            "collis_edges_per_mm_straight": _jshaped_default_kwargs()["collis_edges_per_mm_straight"] * 4.0,
            "beams_per_mm_tip": _jshaped_default_kwargs()["beams_per_mm_tip"] * 4.0,
            "beams_per_mm_straight": _jshaped_default_kwargs()["beams_per_mm_straight"] * 4.0,
        },
        "J-shaped with 4x discretisation density (much more beams/collisions).",
    ),
    "jshaped_universal": ToolSpec(
        eve.intervention.device.JShaped,
        {
            **_universal_common_kwargs(),
            "tip_radius": 12.1,
            "tip_angle": 0.4 * math.pi,
        },
        "J-shaped geometry with Universal material/discretisation.",
    ),
    "jshaped_universal_near_straight": ToolSpec(
        eve.intervention.device.JShaped,
        {
            **_universal_common_kwargs(),
            # Nearly straight: tiny bend angle with same arc length as default J
            "tip_angle": 0.02 * math.pi,
            "tip_radius": (12.1 * 0.4 * math.pi) / (0.02 * math.pi),
        },
        "Universal physics with near-straight J geometry (tiny bend).",
    ),
    "straight_universal": ToolSpec(
        eve.intervention.device.Straight,
        {
            **_universal_common_kwargs(),
            # Match J-shaped arc length: tip_radius * tip_angle
            "tip_length": 12.1 * 0.4 * math.pi,
        },
        "Straight geometry with Universal material/discretisation.",
    ),
    "straight_universal_simple": ToolSpec(
        eve.intervention.device.Straight,
        {
            **_universal_common_kwargs(),
            # Match J-shaped arc length: tip_radius * tip_angle
            "tip_length": 12.1 * 0.4 * math.pi,
            # Coarser discretisation for faster simulation
            "visu_edges_per_mm": 0.25,
            "collis_edges_per_mm_tip": 0.5,
            "collis_edges_per_mm_straight": 0.025,
            "beams_per_mm_tip": 0.35,
            "beams_per_mm_straight": 0.125,
        },
        "Straight Universal with very coarse discretisation (fastest).",
    ),
    "straight_universal_ultra": ToolSpec(
        eve.intervention.device.Straight,
        {
            **_universal_common_kwargs(),
            # Match J-shaped arc length: tip_radius * tip_angle
            "tip_length": 12.1 * 0.4 * math.pi,
            # Ultra coarse discretisation (diagnostic baseline)
            "visu_edges_per_mm": 0.1,
            "collis_edges_per_mm_tip": 0.2,
            "collis_edges_per_mm_straight": 0.01,
            "beams_per_mm_tip": 0.1,
            "beams_per_mm_straight": 0.05,
        },
        "Straight Universal with ultra coarse discretisation (diagnostic).",
    ),
    "straight_universal_eps": ToolSpec(
        StraightEps,
        {
            **_universal_common_kwargs(),
            # Match J-shaped arc length: tip_radius * tip_angle
            "tip_length": 12.1 * 0.4 * math.pi,
            # Ultra coarse discretisation (diagnostic baseline)
            "visu_edges_per_mm": 0.1,
            "collis_edges_per_mm_tip": 0.2,
            "collis_edges_per_mm_straight": 0.01,
            "beams_per_mm_tip": 0.1,
            "beams_per_mm_straight": 0.05,
            # Tiny non-zero spire diameter to avoid degeneracy
            "spire_diameter": 0.45,
        },
        "Straight Universal with tiny spire diameter (degeneracy test).",
    ),
    "straight_universal_eps_01": ToolSpec(
        StraightEps,
        {
            **_universal_common_kwargs(),
            # Match J-shaped arc length: tip_radius * tip_angle
            "tip_length": 12.1 * 0.4 * math.pi,
            # Ultra coarse discretisation (diagnostic baseline)
            "visu_edges_per_mm": 0.1,
            "collis_edges_per_mm_tip": 0.2,
            "collis_edges_per_mm_straight": 0.01,
            "beams_per_mm_tip": 0.1,
            "beams_per_mm_straight": 0.05,
            # Even smaller spire diameter
            "spire_diameter": 0.1,
        },
        "Straight Universal with spire diameter 0.1mm (degeneracy test).",
    ),
}


def register_tool(name: str, device_cls: Type, **kwargs: Any) -> None:
    TOOL_SPECS[name] = ToolSpec(device_cls, dict(kwargs))


def list_tools() -> List[str]:
    return sorted(TOOL_SPECS.keys())


def build_device(tool: str) -> Any:
    if tool not in TOOL_SPECS:
        raise ValueError(
            f"Unknown tool '{tool}'. Available: {', '.join(list_tools())}"
        )
    spec = TOOL_SPECS[tool]
    return spec.device_cls(**spec.kwargs)


def load_device_class(module_path: str, class_name: str) -> Type:
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_path}'"
        ) from exc


def resolve_device(
    tool: str,
    tool_module: Optional[str] = None,
    tool_class: Optional[str] = None,
) -> Tuple[Any, str]:
    if tool_module or tool_class:
        if not tool_module or not tool_class:
            raise ValueError("--tool-module and --tool-class must be provided together")
        cls = load_device_class(tool_module, tool_class)
        return cls(), f"{tool_module}:{tool_class}"

    if ":" in tool:
        module_path, class_name = tool.rsplit(":", 1)
        cls = load_device_class(module_path, class_name)
        return cls(), tool

    return build_device(tool), tool
