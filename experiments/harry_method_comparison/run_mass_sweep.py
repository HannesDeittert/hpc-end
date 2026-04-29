"""Mass-sweep comparison between eval_v2 force projection and Harry's method.

If the calibrated active contact DOF changes across mass samples, the script
prints a note before the table so the comparison is visibly cross-index.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.harry_method_comparison.harry_collision_monitor import (
    aggregate_constraint_directions_per_dof,
)
from steve_recommender.eval_v2.force_telemetry import _project_constraint_forces
from steve_recommender.eval_v2.tests.scenes.validation_sphere_on_plane import createScene


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_CSV = RESULTS_DIR / "mass_sweep.csv"
MASS_SWEEP_KG = (0.5e-3, 1e-3, 2e-3, 4e-3, 8e-3)
GRAVITY_M_PER_S2 = 9.81
SCENE_FORCE_TO_NEWTON = 1e-3
DEFAULT_FRICTION_MU = 0.0
DEFAULT_DT_S = 0.01
SETTLE_STEPS = 200
AVERAGE_LAST_STEPS = 30


@dataclass(frozen=True)
class MassSweepRow:
    mass_kg: float
    expected_mg_N: float
    effective_mass_kg: float
    active_dof: int
    our_force_N: float
    our_error_pct: float
    harrys_at_active_dof: float
    harrys_total_norm: float
    harrys_ratio_to_baseline: float


def _capture_equilibrium_snapshot(
    root: object,
    *,
    dt_s: float,
    settle_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    import Sofa  # type: ignore

    for step_index in range(int(settle_steps)):
        Sofa.Simulation.animate(root, float(dt_s))
        if step_index == int(settle_steps - 1):
            break

    collision_positions = np.asarray(
        root.Sphere.Collision.dofs.position.value,
        dtype=np.float64,
    ).reshape(-1, 3)
    constraint_raw = getattr(root.Sphere.Collision.dofs.constraint, "value", "")
    lcp_forces = np.asarray(root.LCP.constraintForces.value, dtype=np.float64).reshape(-1)
    return collision_positions, lcp_forces


def _build_scene_for_mass(
    *,
    mass_kg: float,
    friction_mu: float,
    dt_s: float,
) -> tuple[object, float]:
    import Sofa  # type: ignore

    root = Sofa.Core.Node("root")
    createScene(root, friction_mu=float(friction_mu), dt_s=float(dt_s))
    root.Sphere.UniformMass.findData("totalMass").value = float(mass_kg)
    Sofa.Simulation.init(root)
    effective_mass_kg = float(root.Sphere.UniformMass.findData("totalMass").value)
    return root, effective_mass_kg


def _run_single_mass(
    *,
    mass_kg: float,
    dt_s: float,
    friction_mu: float,
    settle_steps: int,
    average_last_steps: int,
) -> MassSweepRow:
    root, effective_mass_kg = _build_scene_for_mass(
        mass_kg=mass_kg,
        friction_mu=friction_mu,
        dt_s=dt_s,
    )
    collision_positions, lcp_forces = _capture_equilibrium_snapshot(
        root,
        dt_s=dt_s,
        settle_steps=settle_steps,
    )
    n_dofs = int(collision_positions.shape[0])
    constraint_raw = getattr(root.Sphere.Collision.dofs.constraint, "value", "")
    projected, _rows = _project_constraint_forces(
        lcp_forces=lcp_forces,
        constraint_raw=constraint_raw,
        n_points=n_dofs,
        dt_s=float(dt_s),
    )
    projected = np.asarray(projected, dtype=np.float64).reshape(-1, 3)
    our_norms = np.linalg.norm(projected, axis=1)
    active_dof = int(np.argmax(our_norms))
    our_force_N = float(our_norms[active_dof] * SCENE_FORCE_TO_NEWTON)

    harry_matrix = np.asarray(
        aggregate_constraint_directions_per_dof(str(constraint_raw), n_dofs=n_dofs),
        dtype=np.float64,
    ).reshape(-1, 3)
    harrys_at_active_dof = float(np.linalg.norm(harry_matrix[active_dof]))
    harrys_total_norm = float(sum(np.linalg.norm(row) for row in harry_matrix))
    expected_mg_N = float(mass_kg) * GRAVITY_M_PER_S2
    our_error_pct = abs(our_force_N - expected_mg_N) / expected_mg_N * 100.0
    return MassSweepRow(
        mass_kg=float(mass_kg),
        expected_mg_N=expected_mg_N,
        effective_mass_kg=effective_mass_kg,
        active_dof=active_dof,
        our_force_N=our_force_N,
        our_error_pct=our_error_pct,
        harrys_at_active_dof=harrys_at_active_dof,
        harrys_total_norm=harrys_total_norm,
        harrys_ratio_to_baseline=float("nan"),
    )


def _format_table(rows: Iterable[MassSweepRow]) -> str:
    lines = [
        "mass_kg,expected_mg_N,active_dof,our_force_N,our_error_pct,harrys_at_active_dof,harrys_total_norm,harrys_ratio_to_baseline"
    ]
    for row in rows:
        lines.append(
            ",".join(
                [
                    f"{row.mass_kg:.9g}",
                    f"{row.expected_mg_N:.9g}",
                    str(row.active_dof),
                    f"{row.our_force_N:.9g}",
                    f"{row.our_error_pct:.6f}",
                    f"{row.harrys_at_active_dof:.9g}",
                    f"{row.harrys_total_norm:.9g}",
                    f"{row.harrys_ratio_to_baseline:.6f}",
                ]
            )
        )
    return "\n".join(lines)


def run_mass_sweep() -> tuple[MassSweepRow, ...]:
    rows_without_ratio = tuple(
        _run_single_mass(
            mass_kg=float(mass_kg),
            dt_s=DEFAULT_DT_S,
            friction_mu=DEFAULT_FRICTION_MU,
            settle_steps=SETTLE_STEPS,
            average_last_steps=AVERAGE_LAST_STEPS,
        )
        for mass_kg in MASS_SWEEP_KG
    )

    baseline = rows_without_ratio[0].harrys_at_active_dof
    rows = tuple(
        MassSweepRow(
            mass_kg=row.mass_kg,
            expected_mg_N=row.expected_mg_N,
            effective_mass_kg=row.effective_mass_kg,
            active_dof=row.active_dof,
            our_force_N=row.our_force_N,
            our_error_pct=row.our_error_pct,
            harrys_at_active_dof=row.harrys_at_active_dof,
            harrys_total_norm=row.harrys_total_norm,
            harrys_ratio_to_baseline=(
                float("nan") if baseline == 0.0 else row.harrys_at_active_dof / baseline
            ),
        )
        for row in rows_without_ratio
    )
    return rows


def _write_csv(rows: Iterable[MassSweepRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mass_kg",
                "expected_mg_N",
                "active_dof",
                "our_force_N",
                "our_error_pct",
                "harrys_at_active_dof",
                "harrys_total_norm",
                "harrys_ratio_to_baseline",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    f"{row.mass_kg:.9g}",
                    f"{row.expected_mg_N:.9g}",
                    str(row.active_dof),
                    f"{row.our_force_N:.9g}",
                    f"{row.our_error_pct:.6f}",
                    f"{row.harrys_at_active_dof:.9g}",
                    f"{row.harrys_total_norm:.9g}",
                    f"{row.harrys_ratio_to_baseline:.6f}",
                ]
            )


def main() -> None:
    rows = run_mass_sweep()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, RESULTS_CSV)
    for row in rows:
        print(
            "mass_check "
            f"requested={row.mass_kg:.9g}kg "
            f"effective={row.effective_mass_kg:.9g}kg"
        )
    print()
    active_dofs = {row.active_dof for row in rows}
    if len(active_dofs) > 1:
        print(
            "NOTE: active_dof changed across mass samples: "
            + ", ".join(str(idx) for idx in sorted(active_dofs))
        )
        print()
    print(_format_table(rows))
    print(f"\nWrote CSV to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
