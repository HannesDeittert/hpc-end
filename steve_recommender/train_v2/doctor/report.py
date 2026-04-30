"""Formatting for train_v2 doctor results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


CheckLevel = Literal["ok", "warning", "error"]


@dataclass(frozen=True)
class CheckResult:
    """One doctor/preflight result row."""

    level: CheckLevel
    code: str
    message: str


def render_report(results: Iterable[CheckResult]) -> str:
    """Render one human-readable doctor report."""

    lines = []
    for result in results:
        lines.append(f"[{result.level.upper()}] {result.code}: {result.message}")
    return "\n".join(lines)


def exit_code(results: Iterable[CheckResult], *, strict: bool) -> int:
    """Return the correct shell exit code for one result set."""

    rows = list(results)
    if any(result.level == "error" for result in rows):
        return 1
    if strict and any(result.level == "warning" for result in rows):
        return 2
    return 0
