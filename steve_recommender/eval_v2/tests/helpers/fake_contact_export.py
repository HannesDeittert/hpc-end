from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class FakeContactExport:
    """Small stand-in for the native WireWallContactExport component.

    The collector reads these fields via getattr / .value, so plain Python
    values are enough for unit tests.
    """

    constraintRowIndices: list[int] = field(default_factory=list)
    constraintRowValidFlags: list[int] = field(default_factory=list)
    wallTriangleIds: list[int] = field(default_factory=list)
    triangleIdValidFlags: list[int] = field(default_factory=list)
    collisionDofIndices: list[int] = field(default_factory=list)
    collisionDofValidFlags: list[int] = field(default_factory=list)
    contactCount: int = 0
    explicitCoverage: float = 1.0
    available: bool = True
    source: str = "synthetic_contact_export"
    status: str = "ok"
    orderingStable: bool = True

    def __post_init__(self) -> None:
        self.constraintRowIndices = [int(value) for value in self.constraintRowIndices]
        self.constraintRowValidFlags = self._normalize_flags(self.constraintRowValidFlags, len(self.constraintRowIndices))
        self.wallTriangleIds = [int(value) for value in self.wallTriangleIds]
        self.triangleIdValidFlags = self._normalize_flags(self.triangleIdValidFlags, len(self.wallTriangleIds))
        self.collisionDofIndices = [int(value) for value in self.collisionDofIndices]
        self.collisionDofValidFlags = self._normalize_flags(self.collisionDofValidFlags, len(self.collisionDofIndices))
        self.contactCount = int(self.contactCount)
        self.explicitCoverage = float(self.explicitCoverage)
        self.available = bool(self.available)
        self.orderingStable = bool(self.orderingStable)

    @staticmethod
    def _normalize_flags(values: Iterable[int], target_length: int) -> list[int]:
        flags = [int(value) for value in values]
        if not flags and target_length > 0:
            return [1 for _ in range(target_length)]
        if len(flags) < target_length:
            flags = flags + [1 for _ in range(target_length - len(flags))]
        return flags[:target_length]
