"""Thin adapter for stEVE imports and integration points."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Optional

_INSTALL_HINT = (
    "Install stEVE dependencies with: "
    "pip install -e . (installs eve + eve_rl via pyproject.toml)"
)

_EVE: Optional[ModuleType] = None
_EVE_RL: Optional[ModuleType] = None


def _load_module(name: str) -> ModuleType:
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing stEVE dependency '{name}'. {_INSTALL_HINT}"
        ) from exc


def _get_eve() -> ModuleType:
    global _EVE
    if _EVE is None:
        _EVE = _load_module("eve")
    return _EVE


def _get_eve_rl() -> ModuleType:
    global _EVE_RL
    if _EVE_RL is None:
        _EVE_RL = _load_module("eve_rl")
    return _EVE_RL


def __getattr__(name: str) -> ModuleType:
    if name == "eve":
        return _get_eve()
    if name == "eve_rl":
        return _get_eve_rl()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["eve", "eve_rl"]
