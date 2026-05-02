from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from steve_recommender.eval_v2.experimental_prep_scripts.sample_anatomies import (
    _mean_curvature_from_points,
    sample_anatomies_from_registry,
)


def _write_fake_registry(root: Path, tortuosities: list[float]) -> Path:
    anatomies_root = root / "anatomies"
    anatomies_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, tortuosity in enumerate(tortuosities):
        record_id = f"Tree_{index:02d}"
        anatomy_dir = anatomies_root / record_id
        anatomy_dir.mkdir(parents=True, exist_ok=True)
        centerline_path = anatomy_dir / "centerline.npz"
        description_path = anatomy_dir / "description.json"
        if tortuosity == 0.0:
            coords = np.stack(
                [np.linspace(0.0, 40.0, 32), np.zeros(32), np.zeros(32)],
                axis=1,
            )
        else:
            theta = np.linspace(0.0, np.pi / 2.0, 64)
            radius = 20.0 / (tortuosity * 20.0)
            coords = np.stack(
                [radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)],
                axis=1,
            )
        np.savez_compressed(
            centerline_path,
            branch_names=np.asarray(["bct"], dtype=object),
            branch_bct_coords=coords.astype(np.float32),
            branch_bct_radii=np.ones(coords.shape[0], dtype=np.float32),
        )
        description = {
            "anatomy_type": "aortic_arch",
            "arch_type": "I",
            "record_id": record_id,
            "seed": index,
            "rotation_yzx_deg": [0.0, 0.0, 0.0],
            "scaling_xyzd": [1.0, 1.0, 1.0, 1.0],
            "omit_axis": None,
            "centerline_bundle_path": "centerline.npz",
            "simulation_mesh_path": None,
            "visualization_mesh_path": None,
        }
        description_path.write_text(json.dumps(description), encoding="utf-8")
        entries.append(
            {
                "record_id": record_id,
                "description_path": f"anatomies/{record_id}/description.json",
            }
        )
    (root / "index.json").write_text(
        json.dumps({"version": 1, "anatomies": entries}),
        encoding="utf-8",
    )
    return root


def test_mean_curvature_of_half_circle_is_close_to_inverse_radius() -> None:
    radius = 20.0
    theta = np.linspace(0.0, np.pi, 200)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)],
        axis=1,
    )
    curvature = _mean_curvature_from_points(points)
    assert curvature == pytest.approx(1.0 / radius, rel=5e-2, abs=5e-3)


def test_sampling_reproducible_and_spans_range(tmp_path: Path) -> None:
    root = _write_fake_registry(tmp_path / "registry", [float(i) for i in range(20)])
    payload_a = sample_anatomies_from_registry(
        pool_path=root,
        n=5,
        seed=0,
        strata="none",
        sampling_method="random",
        branches=("bct",),
        workers=1,
    )
    payload_b = sample_anatomies_from_registry(
        pool_path=root,
        n=5,
        seed=0,
        strata="none",
        sampling_method="random",
        branches=("bct",),
        workers=1,
    )
    text_a = json.dumps(payload_a, sort_keys=True, indent=2)
    text_b = json.dumps(payload_b, sort_keys=True, indent=2)
    assert hashlib.sha256(text_a.encode("utf-8")).hexdigest() == hashlib.sha256(
        text_b.encode("utf-8")
    ).hexdigest()
    selected_ids = [item["record_id"] for item in payload_a["selected_anatomies"]]
    selected_tortuosity = [item["tortuosity"] for item in payload_a["selected_anatomies"]]
    assert len(selected_ids) == 5
    assert min(selected_tortuosity) < 5.0
    assert max(selected_tortuosity) > 14.0


def test_sampling_edge_cases(tmp_path: Path) -> None:
    root = _write_fake_registry(tmp_path / "registry", [float(i) for i in range(4)])
    payload = sample_anatomies_from_registry(
        pool_path=root,
        n=4,
        seed=123,
        strata="none",
        sampling_method="random",
        branches=("bct",),
        workers=1,
    )
    assert len(payload["selected_anatomies"]) == 4

    with pytest.raises(ValueError):
        sample_anatomies_from_registry(
            pool_path=root,
            n=5,
            seed=123,
            strata="none",
            sampling_method="random",
            branches=("bct",),
            workers=1,
        )
