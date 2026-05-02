from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import splev, splprep

from steve_recommender.storage import repo_root


DEFAULT_POOL_PATH = repo_root() / "data" / "anatomy_registry"
DEFAULT_OUTPUT_PATH = repo_root() / "results" / "experimental_prep" / "sampled_anatomies.json"
DEFAULT_BRANCHES = ("bct", "lcca", "lsa")
SUPPORTED_STRATA = ("none", "tortuosity_tertiles", "tortuosity_quartiles")
SUPPORTED_METHODS = ("random", "maximin")
_TREE_ID_PREFIX = "Tree_"


@dataclass(frozen=True)
class AnatomyEntry:
    record_id: str
    description_path: Path
    centerline_bundle_path: Path
    arch_type: str
    seed: int
    simulation_mesh_path: Optional[Path]
    visualization_mesh_path: Optional[Path]
    tortuosity: float = 0.0
    branch_curvatures: Tuple[Tuple[str, float], ...] = ()
    stratum: str = "all"

    def with_metrics(
        self,
        *,
        tortuosity: float,
        branch_curvatures: Mapping[str, float],
        stratum: str,
    ) -> "AnatomyEntry":
        return AnatomyEntry(
            record_id=self.record_id,
            description_path=self.description_path,
            centerline_bundle_path=self.centerline_bundle_path,
            arch_type=self.arch_type,
            seed=self.seed,
            simulation_mesh_path=self.simulation_mesh_path,
            visualization_mesh_path=self.visualization_mesh_path,
            tortuosity=float(tortuosity),
            branch_curvatures=tuple(
                (name, float(value)) for name, value in sorted(branch_curvatures.items())
            ),
            stratum=str(stratum),
        )


def _round_float(value: float, *, digits: int = 12) -> float:
    return float(round(float(value), digits))


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}, got {type(payload)}")
    return payload


def _anatomy_sort_key(record_id: str) -> Tuple[int, str]:
    text = str(record_id)
    if text.startswith(_TREE_ID_PREFIX):
        suffix = text[len(_TREE_ID_PREFIX) :]
        if suffix.isdigit():
            return (0, f"{int(suffix):08d}")
    return (1, text)


def _resolve_registry_root(pool_path: Path) -> Path:
    pool_path = Path(pool_path)
    if pool_path.is_file():
        if pool_path.name != "index.json":
            raise ValueError(
                f"Unsupported pool file: {pool_path}. Expected an anatomy registry index.json"
            )
        return pool_path.parent
    if (pool_path / "index.json").exists():
        return pool_path
    raise FileNotFoundError(
        f"Could not find anatomy registry index.json at {pool_path} or {pool_path / 'index.json'}"
    )


def _resolve_optional_path(value: Any, *, base_dir: Path) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base_dir / path


def _load_registry_index(pool_path: Path) -> Tuple[Path, List[Mapping[str, Any]]]:
    root = _resolve_registry_root(pool_path)
    index_path = root / "index.json"
    payload = _read_json(index_path)
    anatomies = payload.get("anatomies", [])
    if not isinstance(anatomies, list):
        raise TypeError(
            f"Registry index field 'anatomies' must be a list, got {type(anatomies)}"
        )
    return root, anatomies


def _load_entry_from_description(
    *,
    root: Path,
    index_entry: Mapping[str, Any],
) -> AnatomyEntry:
    description_path = _resolve_optional_path(
        index_entry.get("description_path"),
        base_dir=root,
    )
    if description_path is None:
        raise ValueError(f"Registry entry is missing description_path: {index_entry}")
    desc = _read_json(description_path)
    record_id = str(desc.get("record_id") or index_entry.get("record_id") or description_path.parent.name)
    centerline_value = desc.get("centerline_bundle_path") or desc.get("centerline_npz")
    centerline_bundle_path = _resolve_optional_path(centerline_value, base_dir=description_path.parent)
    if centerline_bundle_path is None:
        centerline_bundle_path = description_path.parent / "centerline.npz"
    sim_path = _resolve_optional_path(desc.get("simulation_mesh_path"), base_dir=description_path.parent)
    vis_path = _resolve_optional_path(desc.get("visualization_mesh_path"), base_dir=description_path.parent)
    if sim_path is not None and not sim_path.exists():
        raise FileNotFoundError(f"Simulation mesh does not exist: {sim_path}")
    if centerline_bundle_path is not None and not centerline_bundle_path.exists():
        raise FileNotFoundError(f"Centerline bundle does not exist: {centerline_bundle_path}")
    return AnatomyEntry(
        record_id=record_id,
        description_path=description_path,
        centerline_bundle_path=centerline_bundle_path,
        arch_type=str(desc.get("arch_type", "")),
        seed=int(desc.get("seed", 0)),
        simulation_mesh_path=sim_path,
        visualization_mesh_path=vis_path,
    )


def _normalize_consecutive_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return points
    keep = [0]
    for idx in range(1, points.shape[0]):
        if not np.allclose(points[idx], points[keep[-1]]):
            keep.append(idx)
    return points[np.asarray(keep, dtype=int)]


def _mean_curvature_from_points(points: np.ndarray, *, n_samples: int = 1000) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) coordinates, got shape {pts.shape}")
    pts = _normalize_consecutive_points(pts)
    if pts.shape[0] < 2:
        raise ValueError("A branch must contain at least two points")
    if pts.shape[0] == 2:
        return 0.0

    deltas = np.diff(pts, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 0.0:
        return 0.0

    u = np.concatenate([[0.0], np.cumsum(segment_lengths) / total_length])
    k = min(3, pts.shape[0] - 1)
    tck, _ = splprep(pts.T, u=u, s=0.0, k=k)
    sample_u = np.linspace(0.0, 1.0, int(n_samples), endpoint=True)
    deriv_1 = np.asarray(splev(sample_u, tck, der=1), dtype=float).T
    if k >= 2:
        deriv_2 = np.asarray(splev(sample_u, tck, der=2), dtype=float).T
    else:
        deriv_2 = np.zeros_like(deriv_1)

    numerator = np.linalg.norm(np.cross(deriv_1, deriv_2), axis=1)
    denominator = np.linalg.norm(deriv_1, axis=1) ** 3
    curvature = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return float(curvature.mean())


def _compute_entry_metrics(
    payload: Tuple[str, str, Tuple[str, ...]],
) -> Tuple[str, float, Dict[str, float]]:
    description_path_str, centerline_path_str, branch_names = payload
    description_path = Path(description_path_str)
    centerline_path = Path(centerline_path_str)
    with np.load(centerline_path, allow_pickle=True) as data:
        branch_curvatures: Dict[str, float] = {}
        for branch_name in branch_names:
            key = f"branch_{branch_name}_coords"
            if key not in data.files:
                raise KeyError(
                    f"Missing branch coordinates {key!r} in centerline bundle {centerline_path}"
                )
            branch_curvatures[branch_name] = _mean_curvature_from_points(data[key])
    tortuosity = float(np.mean(list(branch_curvatures.values())))
    return description_path.as_posix(), tortuosity, branch_curvatures


def _compute_pool_metrics(
    entries: Sequence[AnatomyEntry],
    *,
    branch_names: Sequence[str],
    workers: Optional[int] = None,
) -> List[AnatomyEntry]:
    if not entries:
        return []

    n_workers = workers if workers is not None else multiprocessing.cpu_count() or 1
    n_workers = max(1, min(int(n_workers), len(entries)))
    payloads = tuple(
        (
            entry.description_path.as_posix(),
            entry.centerline_bundle_path.as_posix(),
            tuple(branch_names),
        )
        for entry in entries
    )

    if n_workers == 1:
        metrics = [_compute_entry_metrics(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            metrics = list(executor.map(_compute_entry_metrics, payloads))

    metric_by_description = {description_path: (tortuosity, branch_curvatures) for description_path, tortuosity, branch_curvatures in metrics}
    enriched: List[AnatomyEntry] = []
    for entry in entries:
        tortuosity, branch_curvatures = metric_by_description[entry.description_path.as_posix()]
        enriched.append(
            entry.with_metrics(
                tortuosity=tortuosity,
                branch_curvatures=branch_curvatures,
                stratum="all",
            )
        )
    return enriched


def _compute_strata_metadata(
    tortuosities: Sequence[float],
    strata: str,
) -> Tuple[List[str], List[float]]:
    if strata == "none":
        return ["all"], []
    if not tortuosities:
        raise ValueError("Cannot compute strata without pool metrics")
    if strata == "tortuosity_tertiles":
        cuts = [float(x) for x in np.percentile(np.asarray(tortuosities, dtype=float), [33.3333333333, 66.6666666667])]
        return ["tertile_1", "tertile_2", "tertile_3"], cuts
    if strata == "tortuosity_quartiles":
        cuts = [float(x) for x in np.percentile(np.asarray(tortuosities, dtype=float), [25.0, 50.0, 75.0])]
        return ["quartile_1", "quartile_2", "quartile_3", "quartile_4"], cuts
    raise ValueError(f"Unsupported strata mode: {strata!r}")


def _assign_stratum(value: float, strata: str, cuts: Sequence[float]) -> str:
    if strata == "none":
        return "all"
    if strata == "tortuosity_tertiles":
        if value <= cuts[0]:
            return "tertile_1"
        if value <= cuts[1]:
            return "tertile_2"
        return "tertile_3"
    if strata == "tortuosity_quartiles":
        if value <= cuts[0]:
            return "quartile_1"
        if value <= cuts[1]:
            return "quartile_2"
        if value <= cuts[2]:
            return "quartile_3"
        return "quartile_4"
    raise ValueError(f"Unsupported strata mode: {strata!r}")


def _stratum_order(strata: str) -> Dict[str, int]:
    if strata == "none":
        return {"all": 0}
    if strata == "tortuosity_tertiles":
        return {"tertile_1": 0, "tertile_2": 1, "tertile_3": 2}
    if strata == "tortuosity_quartiles":
        return {"quartile_1": 0, "quartile_2": 1, "quartile_3": 2, "quartile_4": 3}
    raise ValueError(f"Unsupported strata mode: {strata!r}")


def _allocate_counts_by_stratum(
    groups: Mapping[str, Sequence[AnatomyEntry]],
    *,
    n: int,
    stratum_order: Mapping[str, int],
) -> Dict[str, int]:
    total = sum(len(items) for items in groups.values())
    if n > total:
        raise ValueError(f"Requested n={n} anatomies but pool size is only {total}")
    if total == 0:
        return {key: 0 for key in groups.keys()}
    quotas: Dict[str, float] = {
        key: (len(items) * float(n)) / float(total) for key, items in groups.items()
    }
    counts: Dict[str, int] = {key: int(math.floor(value)) for key, value in quotas.items()}
    remainder = n - sum(counts.values())
    non_empty = [key for key, items in groups.items() if len(items) > 0]
    tie_break = sorted(
        non_empty,
        key=lambda key: (- (quotas[key] - counts[key]), stratum_order[key], key),
    )
    for key in tie_break[:remainder]:
        counts[key] += 1
    for key, items in groups.items():
        if not items:
            counts[key] = 0
    return counts


def _sample_group_random(
    group: Sequence[AnatomyEntry],
    *,
    count: int,
    rng: random.Random,
) -> List[AnatomyEntry]:
    if count <= 0:
        return []
    if count > len(group):
        raise ValueError(f"Cannot sample {count} entries from a group of size {len(group)}")
    indices = rng.sample(list(range(len(group))), count)
    return [group[index] for index in indices]


def _sample_group_maximin(
    group: Sequence[AnatomyEntry],
    *,
    count: int,
    rng: random.Random,
) -> List[AnatomyEntry]:
    if count <= 0:
        return []
    if count > len(group):
        raise ValueError(f"Cannot sample {count} entries from a group of size {len(group)}")
    remaining = list(group)
    selected = [remaining.pop(rng.randrange(len(remaining)))]
    while len(selected) < count:
        best_entry = None
        best_score = None
        for entry in remaining:
            score = min(abs(entry.tortuosity - other.tortuosity) for other in selected)
            key = (score, -entry.tortuosity, entry.record_id)
            if best_score is None or key > best_score:
                best_score = key
                best_entry = entry
        assert best_entry is not None
        selected.append(best_entry)
        remaining.remove(best_entry)
    return selected


def _sample_from_groups(
    groups: Mapping[str, Sequence[AnatomyEntry]],
    *,
    n: int,
    seed: int,
    method: str,
    strata: str,
) -> List[AnatomyEntry]:
    rng = random.Random(int(seed))
    order = _stratum_order(strata)
    counts = _allocate_counts_by_stratum(groups, n=int(n), stratum_order=order)
    selected: List[AnatomyEntry] = []
    for stratum in sorted(groups.keys(), key=lambda key: (order[key], key)):
        group = groups[stratum]
        count = counts[stratum]
        if method == "random":
            chosen = _sample_group_random(group, count=count, rng=rng)
        elif method == "maximin":
            chosen = _sample_group_maximin(group, count=count, rng=rng)
        else:
            raise ValueError(f"Unsupported sampling method: {method!r}")
        selected.extend(chosen)
    return selected


def _build_groups(
    entries: Sequence[AnatomyEntry],
    *,
    strata: str,
    cuts: Sequence[float],
) -> Dict[str, List[AnatomyEntry]]:
    groups: Dict[str, List[AnatomyEntry]] = {}
    for entry in entries:
        stratum = _assign_stratum(entry.tortuosity, strata, cuts)
        groups.setdefault(stratum, []).append(
            entry.with_metrics(
                tortuosity=entry.tortuosity,
                branch_curvatures=dict(entry.branch_curvatures),
                stratum=stratum,
            )
        )
    order = _stratum_order(strata)
    return {
        key: groups.get(key, [])
        for key in sorted(order.keys(), key=lambda item: (order[item], item))
    }


def _entry_to_json(entry: AnatomyEntry) -> Dict[str, Any]:
    return {
        "record_id": entry.record_id,
        "arch_type": entry.arch_type,
        "seed": int(entry.seed),
        "stratum": entry.stratum,
        "tortuosity": _round_float(entry.tortuosity),
        "branch_curvatures": {
            name: _round_float(value) for name, value in entry.branch_curvatures
        },
        "description_path": entry.description_path.as_posix(),
        "centerline_bundle_path": entry.centerline_bundle_path.as_posix(),
        "simulation_mesh_path": None
        if entry.simulation_mesh_path is None
        else entry.simulation_mesh_path.as_posix(),
        "visualization_mesh_path": None
        if entry.visualization_mesh_path is None
        else entry.visualization_mesh_path.as_posix(),
    }


def sample_anatomies_from_registry(
    *,
    pool_path: Path,
    n: int,
    seed: int,
    strata: str = "none",
    sampling_method: str = "random",
    branches: Sequence[str] = DEFAULT_BRANCHES,
    workers: Optional[int] = None,
) -> Dict[str, Any]:
    if strata not in SUPPORTED_STRATA:
        raise ValueError(f"Unsupported strata mode: {strata!r}")
    if sampling_method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported sampling method: {sampling_method!r}")
    if n <= 0:
        raise ValueError("--n must be > 0")

    root, index_entries = _load_registry_index(pool_path)
    sorted_index_entries = sorted(
        index_entries,
        key=lambda raw: _anatomy_sort_key(
            str(raw.get("record_id") or raw.get("id") or "")
        ),
    )
    entries = [
        _load_entry_from_description(root=root, index_entry=index_entry)
        for index_entry in sorted_index_entries
    ]
    entries = _compute_pool_metrics(entries, branch_names=branches, workers=workers)
    tortuosities = [entry.tortuosity for entry in entries]
    stratum_labels, cuts = _compute_strata_metadata(tortuosities, strata)
    grouped = _build_groups(entries, strata=strata, cuts=cuts)
    selected = _sample_from_groups(
        grouped,
        n=int(n),
        seed=int(seed),
        method=sampling_method,
        strata=strata,
    )
    order = _stratum_order(strata)
    selected = sorted(
        selected,
        key=lambda entry: (order[entry.stratum], _anatomy_sort_key(entry.record_id)),
    )

    group_sizes = {key: len(value) for key, value in grouped.items()}
    selected_counts = {key: 0 for key in grouped.keys()}
    for entry in selected:
        selected_counts[entry.stratum] = selected_counts.get(entry.stratum, 0) + 1

    metadata = {
        "pool_root": root.as_posix(),
        "pool_index_path": (root / "index.json").as_posix(),
        "pool_size": len(entries),
        "sample_size": int(n),
        "seed": int(seed),
        "branches": list(branches),
        "strata": strata,
        "stratum_labels": stratum_labels,
        "stratum_thresholds": [_round_float(value) for value in cuts],
        "sampling_method": sampling_method,
        "group_sizes": group_sizes,
        "selected_counts": selected_counts,
    }
    return {
        "metadata": metadata,
        "selected_anatomies": [_entry_to_json(entry) for entry in selected],
    }


def _parse_csv_list(text: str) -> Tuple[str, ...]:
    values = [item.strip() for item in str(text).split(",")]
    values = [item for item in values if item]
    if not values:
        raise ValueError("List argument must not be empty")
    return tuple(values)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample anatomies from an eval_v2 anatomy registry for an experiment."
    )
    parser.add_argument(
        "--pool-path",
        type=Path,
        default=DEFAULT_POOL_PATH,
        help="Anatomy registry root or index.json path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON file for the sampled experiment anatomy set.",
    )
    parser.add_argument("--n", type=int, required=True, help="Number of anatomies to sample.")
    parser.add_argument("--seed", type=int, required=True, help="Deterministic sampling seed.")
    parser.add_argument(
        "--strata",
        choices=SUPPORTED_STRATA,
        default="none",
        help="Stratification mode. 'none' disables stratification and uses global random sampling.",
    )
    parser.add_argument(
        "--sampling-method",
        choices=SUPPORTED_METHODS,
        default="random",
        help="Sampling method inside each stratum.",
    )
    parser.add_argument(
        "--branches",
        type=_parse_csv_list,
        default=DEFAULT_BRANCHES,
        help="Comma-separated branch names used for tortuosity measurement.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional worker count for tortuosity computation. Defaults to cpu_count().",
    )
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    payload = sample_anatomies_from_registry(
        pool_path=args.pool_path,
        n=int(args.n),
        seed=int(args.seed),
        strata=str(args.strata),
        sampling_method=str(args.sampling_method),
        branches=tuple(args.branches),
        workers=args.workers,
    )
    _write_json(args.output, payload)
    print(
        f"[sample_anatomies] wrote {len(payload['selected_anatomies'])} anatomies to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
