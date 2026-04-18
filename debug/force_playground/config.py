from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from steve_recommender.evaluation.config import ForceUnitsConfig

SceneType = Literal["plane_wall", "tube_wall"]
ProbeType = Literal["rigid_probe", "guidewire"]
ControlMode = Literal["displacement", "open_loop_force"]
OracleType = Literal["normal_force_balance"]
CameraPreset = Literal["auto", "plane_front", "plane_oblique", "tube_oblique"]


@dataclass(frozen=True)
class MeshConfig:
    plane_width_mm: float = 220.0
    plane_height_mm: float = 120.0
    tube_radius_mm: float = 0.6
    tube_length_mm: float = 50.0
    tube_segments: int = 12
    tube_rings: int = 24


@dataclass(frozen=True)
class ControlConfig:
    insert_action: float = 0.2
    rotate_action: float = 0.0
    open_loop_force_n: float = 0.10
    open_loop_force_node_index: int = -1
    open_loop_insert_action: float = 0.0
    action_step_delta: float = 0.05
    force_step_delta_n: float = 0.01


@dataclass(frozen=True)
class OracleConfig:
    oracle_type: OracleType = "normal_force_balance"
    enabled: bool = True
    rel_tol: float = 0.10
    abs_tol_n: float = 0.01
    near_zero_ref_n: float = 0.02
    warmup_steps: int = 40
    window_steps: int = 80


@dataclass(frozen=True)
class ForcePlaygroundConfig:
    scene: SceneType = "plane_wall"
    probe: ProbeType = "rigid_probe"
    mode: ControlMode = "displacement"
    tool_ref: str = "ArchVarJShaped/JShaped_Default"
    steps: int = 400
    seed: int = 123
    friction: float = 0.1
    image_frequency_hz: float = 7.5
    alarm_distance: float = 0.5
    contact_distance: float = 0.3
    contact_epsilon: float = 1e-7
    plugin_path: Optional[str] = None
    units: ForceUnitsConfig = field(
        default_factory=lambda: ForceUnitsConfig(length_unit="mm", mass_unit="kg", time_unit="s")
    )
    interactive: bool = True
    plot: bool = True
    show_sofa: bool = False
    camera_preset: CameraPreset = "auto"
    save_plot_snapshots: bool = False
    require_oracle_applicable: bool = False
    require_oracle_pass: bool = False
    output_root: str = "results/force_playground"
    run_name: Optional[str] = None
    mesh: MeshConfig = field(default_factory=MeshConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)

    def __post_init__(self) -> None:
        if int(self.steps) <= 0:
            raise ValueError("steps must be > 0")
        if float(self.contact_epsilon) < 0.0:
            raise ValueError("contact_epsilon must be >= 0")
        if float(self.friction) < 0.0:
            raise ValueError("friction must be >= 0")
        if float(self.image_frequency_hz) <= 0.0:
            raise ValueError("image_frequency_hz must be > 0")
        if float(self.alarm_distance) <= 0.0:
            raise ValueError("alarm_distance must be > 0")
        if float(self.contact_distance) <= 0.0:
            raise ValueError("contact_distance must be > 0")
        if float(self.alarm_distance) <= float(self.contact_distance):
            raise ValueError("alarm_distance must be > contact_distance")
        if self.mode == "open_loop_force" and self.probe != "rigid_probe":
            raise ValueError(
                "open_loop_force is v1-only for probe='rigid_probe'. "
                "Use probe='rigid_probe' or switch mode='displacement'."
            )

    def effective_run_name(self) -> str:
        if self.run_name:
            return str(self.run_name)
        return f"{self.scene}_{self.probe}_{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def output_root_path(self) -> Path:
        return Path(self.output_root)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["units"] = {
            "length_unit": self.units.length_unit,
            "mass_unit": self.units.mass_unit,
            "time_unit": self.units.time_unit,
        }
        return out


def parse_units(value: str) -> ForceUnitsConfig:
    raw = str(value).strip().lower().replace(" ", "")
    parts = raw.split(",")
    if len(parts) != 3:
        raise ValueError("--units must be '<length>,<mass>,<time>' (example: mm,kg,s)")
    length_unit, mass_unit, time_unit = parts
    if length_unit not in {"mm", "m"}:
        raise ValueError("length unit must be 'mm' or 'm'")
    if mass_unit not in {"kg", "g"}:
        raise ValueError("mass unit must be 'kg' or 'g'")
    if time_unit not in {"s", "ms"}:
        raise ValueError("time unit must be 's' or 'ms'")
    return ForceUnitsConfig(
        length_unit=length_unit,
        mass_unit=mass_unit,
        time_unit=time_unit,
    )
