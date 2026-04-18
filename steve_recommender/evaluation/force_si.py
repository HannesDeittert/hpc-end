from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from .config import ForceUnitsConfig


def unit_scale_to_si_newton(units: ForceUnitsConfig) -> float:
    """Return multiplicative factor to convert scene force units to Newton.

    Scene force unit is M * L / T^2.
    """

    length_scale = {"m": 1.0, "mm": 1e-3}[units.length_unit]
    mass_scale = {"kg": 1.0, "g": 1e-3}[units.mass_unit]
    time_scale = {"s": 1.0, "ms": 1e-3}[units.time_unit]
    return float(mass_scale * length_scale / (time_scale * time_scale))


def units_to_dict(units: ForceUnitsConfig) -> Dict[str, str]:
    return {
        "length_unit": str(units.length_unit),
        "mass_unit": str(units.mass_unit),
        "time_unit": str(units.time_unit),
    }


def units_label(units: ForceUnitsConfig) -> str:
    d = asdict(units)
    return f"{d['mass_unit']}*{d['length_unit']}/{d['time_unit']}^2 -> N"
