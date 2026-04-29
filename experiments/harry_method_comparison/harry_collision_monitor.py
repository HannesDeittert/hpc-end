"""Faithful port of the foreign per-DOF constraint-direction aggregation."""

from __future__ import annotations

import numpy as np

from steve_recommender.eval_v2.force_telemetry import _parse_constraint_rows


def aggregate_constraint_directions_per_dof(
    constraint_string: str,
    n_dofs: int,
) -> np.ndarray:
    """Replicates the per-DOF aggregation method used by Robertshaw et al.

    Uses the existing eval_v2 `_parse_constraint_rows(...)` helper to read the
    SOFA constraint-string rows. That parser is broader than the original
    foreign implementation because it supports both the legacy SOFA v22.12 text
    format and the newer v23.06 format, but the aggregation logic below is the
    deliberate foreign method: for every parsed row entry, sum the row's 3D
    coefficient vector onto each DOF touched by that row and return a dense
    `(n_dofs, 3)` accumulator.

    NOTE: This is a deliberate, faithful port of the foreign method. It
    intentionally does NOT multiply by lambda from LCP.constraintForces,
    does NOT divide by dt, and does NOT apply the mm->m scaling. The
    output is a sum of geometric direction vectors in constraint units,
    not a force in Newtons. Do not "fix" this function to compute real
    forces -- its purpose is to faithfully reproduce the foreign method
    so we can compare against our calibrated pipeline.
    """

    count = int(n_dofs)
    if count < 0:
        raise ValueError(f"n_dofs must be >= 0, got {n_dofs!r}")

    accumulator = np.zeros((count, 3), dtype=np.float32)
    for _row_idx, dof_idx, coeff_xyz in _parse_constraint_rows(constraint_string):
        if 0 <= int(dof_idx) < count:
            accumulator[int(dof_idx)] += np.asarray(coeff_xyz, dtype=np.float32).reshape(3)
    return accumulator
