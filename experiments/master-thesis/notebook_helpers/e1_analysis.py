from __future__ import annotations

import csv
import json
import math
import random
import re
from pathlib import Path
from statistics import mean
from statistics import pstdev
from typing import Any, Iterable, Sequence

from scipy.stats import kendalltau

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_E1_RESULTS_ROOT = PROJECT_ROOT / "results" / "master_thesis" / "e1"
DEFAULT_E1_ANALYSIS_ROOT = PROJECT_ROOT / "results" / "master_thesis" / "e1_analysis"




def _infer_config_id(run_dir: Path, payload: dict[str, Any] | None = None) -> int | None:
    if payload and payload.get("config_id") is not None:
        return _maybe_int(payload.get("config_id"))
    for part in run_dir.parts:
        match = re.fullmatch(r"config_(\d+)", part)
        if match:
            return int(match.group(1))
    execution = dict(payload.get("execution_plan", {})) if payload else {}
    policy_mode = execution.get("policy_mode")
    env_mode = execution.get("stochastic_environment_mode")
    if policy_mode == "deterministic" and env_mode == "fixed_start":
        return 1
    if policy_mode == "deterministic" and env_mode == "random_start":
        return 2
    if policy_mode == "stochastic" and env_mode == "fixed_start":
        return 3
    if policy_mode == "stochastic" and env_mode == "random_start":
        return 4
    return None


def _parse_job_name(job_name: str) -> dict[str, Any]:
    match = re.fullmatch(r"(?P<anatomy>.+)__target_(?P<target>\d+)__seedbase_(?P<seedbase>\d+)", str(job_name))
    if not match:
        return {}
    return {
        "anatomy_id": match.group("anatomy"),
        "target_index": int(match.group("target")),
        "seed_base": int(match.group("seedbase")),
    }


def _run_dir_from_manifest_output(result_root: Path, output_dir: Any, job_name: str | None = None, config_id: int | None = None) -> Path:
    if output_dir:
        path = Path(str(output_dir))
        if path.exists():
            return path.resolve()
        parts = path.parts
        if "runs" in parts:
            idx = parts.index("runs")
            return (result_root / "runs" / Path(*parts[idx + 1 :])).resolve()
    if job_name and config_id is not None:
        return (result_root / "runs" / f"config_{int(config_id)}" / str(job_name)).resolve()
    if job_name:
        matches = list((result_root / "runs").glob(f"config_*/{job_name}"))
        if matches:
            return matches[0].resolve()
    return result_root.resolve()


def _load_expected_run_rows(result_root: Path) -> dict[str, dict[str, Any]]:
    metadata_root = result_root / "metadata"
    manifest_paths = sorted(metadata_root.glob("job_manifest_*.json"))
    if not manifest_paths and (metadata_root / "job_manifest.json").exists():
        manifest_paths = [metadata_root / "job_manifest.json"]

    rows: dict[str, dict[str, Any]] = {}
    for manifest_path in manifest_paths:
        payload = _load_json(manifest_path)
        default_friction = payload.get("friction")
        for array_index, job in enumerate(payload.get("jobs", [])):
            job_name = str(job.get("job_name") or Path(str(job.get("output_dir", ""))).name)
            config_id = _maybe_int(job.get("config_id"))
            run_dir = _run_dir_from_manifest_output(result_root, job.get("output_dir"), job_name, config_id)
            parsed = _parse_job_name(job_name)
            target_spec = dict(job.get("target_spec", {}))
            config_spec = dict(job.get("config_spec", {}))
            row = {
                "job_name": job_name,
                "config_id": config_id,
                "anatomy_id": job.get("anatomy_id") or parsed.get("anatomy_id"),
                "target_index": target_spec.get("target_index", parsed.get("target_index")),
                "target_seed": target_spec.get("target_seed"),
                "seed_base": job.get("seed_base", parsed.get("seed_base")),
                "trial_count": config_spec.get("trial_count"),
                "max_episode_steps": config_spec.get("max_episode_steps"),
                "run_dir": str(run_dir),
                "manifest_path": "",
                "generated_time": None,
                "n_candidates": None,
                "n_trials_total": None,
                "array_index": array_index,
                "partition": payload.get("partition"),
                "friction": job.get("friction", default_friction),
                "write_full_trace": job.get("write_full_trace"),
                "load_status": "expected_from_job_manifest",
                "trials_h5_status": "not_checked",
                "trials_h5_error": "",
            }
            rows[str(run_dir)] = row
    return rows

def _resolve_path(base: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        if path.exists():
            return path
        # Synced cluster manifests contain absolute /home/woody/... artifact paths.
        # When analyzing locally, the artifact filename still lives in the local run dir.
        local_sibling = Path(base) / path.name
        if local_sibling.exists() or Path(base).exists():
            return local_sibling.resolve()
        return path
    return (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _decode_h5_column(array: Any) -> list[Any]:
    import numpy as np

    values = np.asarray(array)
    if values.dtype.kind in {"S", "O"}:
        decoded: list[Any] = []
        for value in values:
            if isinstance(value, (bytes, bytearray)):
                decoded.append(value.decode("utf-8"))
            elif value is None:
                decoded.append(None)
            else:
                decoded.append(value)
        return decoded
    return values.tolist()


def _maybe_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if value in {"1", 1, "true", "True", "TRUE"}:
        return True
    if value in {"0", 0, "false", "False", "FALSE"}:
        return False
    return value


def _maybe_int(value: Any) -> Any:
    if value in ("", None):
        return None
    try:
        return int(value)
    except Exception:
        return value


def _maybe_float(value: Any) -> Any:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return value


def load_trials_h5(path: Path) -> list[dict[str, Any]]:
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as handle:
        group = handle["trials"]
        columns = {name: _decode_h5_column(group[name][...]) for name in group.keys()}

    n_rows = len(next(iter(columns.values()))) if columns else 0
    rows: list[dict[str, Any]] = []
    for idx in range(n_rows):
        row = {name: values[idx] for name, values in columns.items()}
        for key in [
            "success",
            "valid_for_ranking",
            "force_within_safety_threshold",
            "force_available_for_score",
        ]:
            if key in row:
                row[key] = _maybe_bool(row[key])
        for key in ["trial_index", "env_seed", "max_episode_steps", "steps_total", "config_id", "target_index"]:
            if key in row:
                row[key] = _maybe_int(row[key])
        for key in [
            "policy_seed",
            "episode_reward",
            "sim_time_s",
            "wall_time_s",
            "tip_speed_max_mm_s",
            "tip_speed_mean_mm_s",
            "tip_total_distance_mm",
            "steps_to_success",
            "wire_force_magnitude_instant_N",
            "wire_force_magnitude_trial_max_N",
            "wire_force_magnitude_trial_mean_N",
            "wire_force_normal_instant_N",
            "wire_force_normal_trial_max_N",
            "wire_force_normal_trial_mean_N",
            "tip_force_magnitude_instant_N",
            "tip_force_magnitude_trial_max_N",
            "tip_force_magnitude_trial_mean_N",
            "tip_force_normal_instant_N",
            "tip_force_normal_trial_max_N",
            "tip_force_normal_trial_mean_N",
            "tip_length_mm",
            "tip_acc_p95",
            "tip_acc_max",
            "tip_jerk_p95",
            "tip_jerk_max",
            "score_success",
            "score_efficiency",
            "score_safety",
            "score_smoothness",
            "score_total",
        ]:
            if key in row:
                row[key] = _maybe_float(row[key])
        rows.append(row)
    return rows


def load_e1_result_rows(result_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load E1 rows from a result tree.

    The loader is intentionally tolerant of live/partial TinyFat runs:
    - expected run folders are read from `metadata/job_manifest*.json` when present;
    - completed `manifest.json` files enrich those expected rows;
    - corrupt or currently unreadable `trials.h5` files are skipped and recorded in
      the returned manifest row via `trials_h5_status` / `trials_h5_error`.
    """

    result_root = Path(result_root).resolve()
    expected_rows = _load_expected_run_rows(result_root)
    manifest_paths = sorted(result_root.rglob("manifest.json"))
    if not manifest_paths and not expected_rows:
        raise FileNotFoundError(
            f"No manifest.json files or metadata/job_manifest*.json files found under {result_root}. "
            "Sync the E1 results there or set E1_RESULTS_ROOT."
        )

    manifest_rows_by_dir: dict[str, dict[str, Any]] = dict(expected_rows)
    summary_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    for manifest_path in manifest_paths:
        payload = _load_json(manifest_path)
        run_dir = manifest_path.parent.resolve()
        job_name = str(payload.get("job_name", run_dir.name))
        parsed = _parse_job_name(job_name)
        execution = dict(payload.get("execution_plan", {}))
        counts = dict(payload.get("counts", {}))
        config_id = _infer_config_id(run_dir, payload)
        anatomy_metadata = payload.get("anatomy_metadata") or []
        anatomy_id = payload.get("anatomy_id") or parsed.get("anatomy_id")
        if anatomy_id is None and anatomy_metadata:
            anatomy_id = anatomy_metadata[0].get("record_id")
        target_payload = (payload.get("targets") or [{}])[0]

        previous = manifest_rows_by_dir.get(str(run_dir), {})
        manifest_row = {
            **previous,
            "job_name": job_name,
            "config_id": config_id,
            "config_name": payload.get("config_name"),
            "anatomy_id": anatomy_id,
            "target_index": payload.get("target_index", parsed.get("target_index")),
            "target_seed": payload.get("target_seed", target_payload.get("seed")),
            "seed_base": execution.get("base_seed", parsed.get("seed_base")),
            "trial_count": payload.get("trial_count", execution.get("trials_per_candidate")),
            "max_episode_steps": payload.get("max_episode_steps", execution.get("max_episode_steps")),
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
            "generated_time": payload.get("generated_time"),
            "n_candidates": len(payload.get("candidate_names", [])) or counts.get("n_candidates"),
            "n_trials_total": counts.get("n_trials_total"),
            "friction": payload.get("friction", previous.get("friction")),
            "load_status": "completed_manifest",
            "trials_h5_status": previous.get("trials_h5_status", "not_checked"),
            "trials_h5_error": previous.get("trials_h5_error", ""),
        }
        manifest_rows_by_dir[str(run_dir)] = manifest_row

    for run_dir_text, manifest_row in sorted(manifest_rows_by_dir.items(), key=lambda item: (item[1].get("config_id") or 0, item[1].get("job_name") or item[0])):
        run_dir = Path(run_dir_text)
        manifest_path_text = manifest_row.get("manifest_path") or ""
        payload: dict[str, Any] = {}
        if manifest_path_text:
            try:
                payload = _load_json(Path(manifest_path_text))
            except Exception:
                payload = {}
        artifact_paths = payload.get("artifact_paths", {}) if payload else {}

        candidate_csv = _resolve_path(
            run_dir,
            artifact_paths.get("candidate_summaries_csv", run_dir / "candidate_summaries.csv"),
        )
        trials_h5 = _resolve_path(run_dir, artifact_paths.get("trials_h5", run_dir / "trials.h5"))

        if candidate_csv.exists():
            with candidate_csv.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    row.update(
                        {
                            "job_name": manifest_row["job_name"],
                            "config_id": manifest_row.get("config_id"),
                            "anatomy_id": manifest_row.get("anatomy_id"),
                            "target_index": manifest_row.get("target_index"),
                            "target_seed": manifest_row.get("target_seed"),
                            "friction": manifest_row.get("friction"),
                            "run_dir": str(run_dir),
                            "manifest_path": manifest_path_text,
                        }
                    )
                    summary_rows.append(row)

        if trials_h5.exists():
            try:
                loaded_trial_rows = load_trials_h5(trials_h5)
            except Exception as exc:
                manifest_row["trials_h5_status"] = "unreadable"
                manifest_row["trials_h5_error"] = f"{type(exc).__name__}: {exc}"
            else:
                manifest_row["trials_h5_status"] = "loaded"
                manifest_row["trials_h5_error"] = ""
                manifest_row["trial_rows_loaded"] = len(loaded_trial_rows)
                for row in loaded_trial_rows:
                    row.update(
                        {
                            "job_name": manifest_row["job_name"],
                            "config_id": manifest_row.get("config_id"),
                            "anatomy_id": manifest_row.get("anatomy_id"),
                            "target_index": manifest_row.get("target_index"),
                            "target_seed": manifest_row.get("target_seed"),
                            "seed_base": manifest_row.get("seed_base"),
                            "friction": manifest_row.get("friction"),
                            "run_dir": str(run_dir),
                            "manifest_path": manifest_path_text,
                        }
                    )
                    trial_rows.append(row)
        else:
            manifest_row["trials_h5_status"] = "missing"
            manifest_row["trials_h5_error"] = ""
            manifest_row.setdefault("trial_rows_loaded", 0)

    return list(manifest_rows_by_dir.values()), summary_rows, trial_rows

def iqm(values: Sequence[float] | Iterable[float]) -> float:
    samples = sorted(float(value) for value in values if value is not None)
    if not samples:
        return float("nan")
    if len(samples) < 4:
        return float(mean(samples))
    lower = len(samples) // 4
    upper = len(samples) - lower
    if upper <= lower:
        return float(mean(samples))
    return float(mean(samples[lower:upper]))


def summarize_trials_by_config(trial_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[int, list[float]] = {}
    for row in trial_rows:
        config_id = row.get("config_id")
        score = row.get("score_total")
        if config_id is None or score is None:
            continue
        buckets.setdefault(int(config_id), []).append(float(score))
    summaries: list[dict[str, Any]] = []
    for config_id in sorted(buckets):
        scores = buckets[config_id]
        summaries.append(
            {
                "config_id": config_id,
                "n_trials": len(scores),
                "mean_score": float(mean(scores)) if scores else float("nan"),
                "iqm_score": iqm(scores),
                "median_score": float(__import__("statistics").median(scores)) if scores else float("nan"),
                "score_min": min(scores) if scores else float("nan"),
                "score_max": max(scores) if scores else float("nan"),
            }
        )
    return summaries


def _group_trial_scores(
    trial_rows: Sequence[dict[str, Any]],
) -> dict[tuple[int, str, int], dict[str, list[float]]]:
    grouped: dict[tuple[int, str, int], dict[str, list[float]]] = {}
    for row in trial_rows:
        config_id = row.get("config_id")
        anatomy_id = row.get("anatomy_id")
        target_index = row.get("target_index")
        candidate_name = row.get("candidate_name") or row.get("candidate")
        score = row.get("score_total")
        if config_id is None or anatomy_id is None or target_index is None or candidate_name is None or score is None:
            continue
        key = (int(config_id), str(anatomy_id), int(target_index))
        grouped.setdefault(key, {}).setdefault(str(candidate_name), []).append(float(score))
    return grouped


def _bootstrap_metric(
    samples: Sequence[float],
    *,
    rng: random.Random,
    n_resamples: int,
    metric,
) -> list[float]:
    if not samples:
        return []
    values = list(samples)
    outputs: list[float] = []
    for _ in range(int(n_resamples)):
        resampled = [values[rng.randrange(len(values))] for _ in values]
        outputs.append(float(metric(resampled)))
    return outputs


def compute_snr_table(
    trial_rows: Sequence[dict[str, Any]],
    *,
    aggregate_fn=iqm,
    n_boot: int = 500,
    seed: int = 0,
) -> list[dict[str, Any]]:
    grouped = _group_trial_scores(trial_rows)
    rows: list[dict[str, Any]] = []
    for (config_id, anatomy_id, target_index), wire_scores in sorted(grouped.items()):
        wire_aggregates: list[float] = []
        wire_within: list[float] = []
        for scores in wire_scores.values():
            if not scores:
                continue
            wire_aggregates.append(float(aggregate_fn(scores)))
            wire_within.append(float(pstdev(scores)) if len(scores) > 1 else 0.0)
        if len(wire_aggregates) < 2:
            continue
        between_sigma = float(pstdev(wire_aggregates))
        within_sigma = float(mean(wire_within)) if wire_within else 0.0
        snr = float(between_sigma / within_sigma) if within_sigma > 0 else float("nan")

        boot_rng = random.Random((seed + config_id * 10_000 + target_index * 97) & 0xFFFFFFFF)
        boot_values: list[float] = []
        if n_boot > 0:
            wire_items = list(wire_scores.values())
            for _ in range(int(n_boot)):
                boot_wire_aggregates: list[float] = []
                boot_within: list[float] = []
                for scores in wire_items:
                    resampled = [scores[boot_rng.randrange(len(scores))] for _ in scores]
                    boot_wire_aggregates.append(float(aggregate_fn(resampled)))
                    boot_within.append(float(pstdev(resampled)) if len(resampled) > 1 else 0.0)
                if len(boot_wire_aggregates) >= 2:
                    boot_between = float(pstdev(boot_wire_aggregates))
                    boot_within_sigma = float(mean(boot_within)) if boot_within else 0.0
                    boot_values.append(float(boot_between / boot_within_sigma) if boot_within_sigma > 0 else float("nan"))
        boot_values = [value for value in boot_values if not math.isnan(value)]
        boot_low = float("nan")
        boot_high = float("nan")
        if boot_values:
            boot_values.sort()
            lo_idx = max(0, int(round(0.025 * (len(boot_values) - 1))))
            hi_idx = min(len(boot_values) - 1, int(round(0.975 * (len(boot_values) - 1))))
            boot_low = float(boot_values[lo_idx])
            boot_high = float(boot_values[hi_idx])

        rows.append(
            {
                "config_id": config_id,
                "anatomy_id": anatomy_id,
                "target_index": target_index,
                "n_wires": len(wire_aggregates),
                "between_sigma": between_sigma,
                "within_sigma": within_sigma,
                "snr": snr,
                "snr_boot_low": boot_low,
                "snr_boot_high": boot_high,
            }
        )
    return rows


def build_group_candidate_table(
    trial_rows: Sequence[dict[str, Any]],
    *,
    aggregate_fn=iqm,
) -> list[dict[str, Any]]:
    grouped = _group_trial_scores(trial_rows)
    rows: list[dict[str, Any]] = []
    for (config_id, anatomy_id, target_index), wire_scores in sorted(grouped.items()):
        for candidate_name, scores in sorted(wire_scores.items()):
            if not scores:
                continue
            rows.append(
                {
                    "config_id": config_id,
                    "anatomy_id": anatomy_id,
                    "target_index": target_index,
                    "candidate_name": candidate_name,
                    "n_trials": len(scores),
                    "aggregate_score": float(aggregate_fn(scores)),
                    "mean_score": float(mean(scores)),
                    "median_score": float(__import__("statistics").median(scores)),
                    "within_sigma": float(pstdev(scores)) if len(scores) > 1 else 0.0,
                }
            )
    return rows


def compute_kendall_matrix(
    trial_rows: Sequence[dict[str, Any]],
    *,
    aggregate_fn=iqm,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows = build_group_candidate_table(trial_rows, aggregate_fn=aggregate_fn)
    if not candidate_rows:
        return [], []

    by_group: dict[tuple[str, int], dict[int, list[dict[str, Any]]]] = {}
    for row in candidate_rows:
        group_key = (str(row["anatomy_id"]), int(row["target_index"]))
        config_key = int(row["config_id"])
        by_group.setdefault(group_key, {}).setdefault(config_key, []).append(row)

    config_ids = sorted({int(row["config_id"]) for row in candidate_rows})
    matrix: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for left in config_ids:
        for right in config_ids:
            taus: list[float] = []
            for (anatomy_id, target_index), rows_by_config in by_group.items():
                left_rows = rows_by_config.get(left, [])
                right_rows = rows_by_config.get(right, [])
                if not left_rows or not right_rows:
                    continue
                left_map = {row["candidate_name"]: float(row["aggregate_score"]) for row in left_rows}
                right_map = {row["candidate_name"]: float(row["aggregate_score"]) for row in right_rows}
                shared = sorted(set(left_map) & set(right_map))
                if len(shared) < 2:
                    continue
                left_scores = [left_map[name] for name in shared]
                right_scores = [right_map[name] for name in shared]
                tau = kendalltau(left_scores, right_scores, nan_policy="omit").statistic
                if tau is not None and not math.isnan(float(tau)):
                    taus.append(float(tau))
            pair_rows.append(
                {
                    "config_left": left,
                    "config_right": right,
                    "mean_tau": float(mean(taus)) if taus else float("nan"),
                    "median_tau": float(__import__("statistics").median(taus)) if taus else float("nan"),
                    "n_groups": len(taus),
                }
            )
            matrix.append(pair_rows[-1])
    return pair_rows, candidate_rows


def compute_k_convergence_table(
    trial_rows: Sequence[dict[str, Any]],
    *,
    aggregate_fn=iqm,
    k_values: Sequence[int] = (1, 2, 5, 10, 20, 30, 50, 100),
    n_boot: int = 200,
    seed: int = 0,
) -> list[dict[str, Any]]:
    grouped = _group_trial_scores(trial_rows)
    rows: list[dict[str, Any]] = []
    for config_id in sorted({int(key[0]) for key in grouped}):
        config_groups = {
            key: value
            for key, value in grouped.items()
            if int(key[0]) == config_id
        }
        for k in k_values:
            group_within: list[float] = []
            group_between: list[float] = []
            for (_cfg, anatomy_id, target_index), wire_scores in config_groups.items():
                wire_values = [scores for scores in wire_scores.values() if scores]
                if len(wire_values) < 2:
                    continue
                boot_rng = random.Random((seed + config_id * 10_000 + k * 97 + target_index) & 0xFFFFFFFF)
                boot_between_vals: list[float] = []
                boot_within_vals: list[float] = []
                for _ in range(int(n_boot)):
                    wire_aggregates: list[float] = []
                    wire_within: list[float] = []
                    for scores in wire_values:
                        resampled = [scores[boot_rng.randrange(len(scores))] for _ in range(int(k))]
                        wire_aggregates.append(float(aggregate_fn(resampled)))
                        wire_within.append(float(pstdev(resampled)) if len(resampled) > 1 else 0.0)
                    if len(wire_aggregates) >= 2:
                        boot_between_vals.append(float(pstdev(wire_aggregates)))
                        boot_within_vals.append(float(mean(wire_within)) if wire_within else 0.0)
                if boot_between_vals:
                    group_between.append(float(mean(boot_between_vals)))
                if boot_within_vals:
                    group_within.append(float(mean(boot_within_vals)))
            rows.append(
                {
                    "config_id": config_id,
                    "k": int(k),
                    "within_sigma": float(mean(group_within)) if group_within else float("nan"),
                    "between_sigma": float(mean(group_between)) if group_between else float("nan"),
                    "snr": float(mean(group_between) / mean(group_within)) if group_within and mean(group_within) > 0 else float("nan"),
                }
            )
    return rows


def _safe_mean(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return mean(values) if values else None


def _safe_std(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return pstdev(values) if len(values) > 1 else None


def _safe_percentile(values: Iterable[float], q: float) -> float | None:
    import numpy as np

    values = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q * 100.0))


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def analyze_e1_e0_style_results(
    trial_rows: list[dict[str, Any]],
    *,
    analysis_root: Path,
    prefix: str,
    title_prefix: str,
    current_max_default: int = 1000,
) -> dict[str, Any]:
    """Run the same step-budget-oriented summaries used for E0 on E1 trial rows."""
    import matplotlib.pyplot as plt
    import numpy as np

    analysis_root = Path(analysis_root).resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)

    if not trial_rows:
        raise ValueError("trial_rows is empty; there is nothing to analyze.")

    for row in trial_rows:
        row["steps_total"] = _maybe_float(row.get("steps_total"))
        row["steps_to_success"] = _maybe_float(row.get("steps_to_success"))
        row["score_total"] = _maybe_float(row.get("score_total"))
        row["score_safety"] = _maybe_float(row.get("score_safety"))
        row["success"] = _maybe_bool(row.get("success"))
        row["max_episode_steps"] = _maybe_int(row.get("max_episode_steps"))

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in trial_rows:
        wire = str(row.get("execution_wire") or row.get("candidate_name") or "")
        groups.setdefault(wire, []).append(row)

    candidate_rows: list[dict[str, Any]] = []
    for wire, rows in groups.items():
        score_values = [row.get("score_total") for row in rows]
        safety_values = [row.get("score_safety") for row in rows]
        success_values = [row.get("success") for row in rows]
        steps_total_values = [row.get("steps_total") for row in rows]
        steps_to_success_values = [row.get("steps_to_success") for row in rows if row.get("success") and row.get("steps_to_success") is not None]
        wire_force_normal_trial_max_values = [row.get("wire_force_normal_trial_max_N") for row in rows]
        tip_force_normal_trial_max_values = [row.get("tip_force_normal_trial_max_N") for row in rows]

        success_rate = _safe_mean([1.0 if bool(v) else 0.0 for v in success_values])
        candidate_rows.append(
            {
                "execution_wire": wire,
                "n_trials": len(rows),
                "success_rate": success_rate,
                "score_mean": _safe_mean(score_values),
                "score_std": _safe_std(score_values),
                "score_safety_mean": _safe_mean(safety_values),
                "steps_total_mean": _safe_mean(steps_total_values),
                "steps_total_p95": _safe_percentile(steps_total_values, 0.95),
                "steps_to_success_mean": _safe_mean(steps_to_success_values),
                "steps_to_success_p95": _safe_percentile(steps_to_success_values, 0.95),
                "wire_force_normal_trial_max_mean_N": _safe_mean(wire_force_normal_trial_max_values),
                "tip_force_normal_trial_max_mean_N": _safe_mean(tip_force_normal_trial_max_values),
            }
        )

    candidate_rows.sort(
        key=lambda r: (
            r["score_mean"] is None,
            -(r["score_mean"] if r["score_mean"] is not None else float("-inf")),
            -(r["success_rate"] if r["success_rate"] is not None else float("-inf")),
        )
    )

    successful_steps = [float(row["steps_to_success"]) for row in trial_rows if row.get("success") and row.get("steps_to_success") is not None]
    all_steps = [float(row["steps_total"]) for row in trial_rows if row.get("steps_total") is not None]
    current_max = max((int(row["max_episode_steps"]) for row in trial_rows if row.get("max_episode_steps") is not None), default=current_max_default)

    if successful_steps:
        percentiles = [0.5, 0.75, 0.9, 0.95, 0.99]
        percentile_rows = [
            {"quantile": int(q * 100), "steps_to_success": _safe_percentile(successful_steps, q)}
            for q in percentiles
        ]
        p95 = next((row["steps_to_success"] for row in percentile_rows if row["quantile"] == 95), None)
        recommended_cutoff = int(math.ceil(float(p95) / 25.0) * 25) if p95 is not None else None
    else:
        percentile_rows = []
        recommended_cutoff = None

    candidate_fields = [
        "execution_wire",
        "n_trials",
        "success_rate",
        "score_mean",
        "score_std",
        "score_safety_mean",
        "steps_total_mean",
        "steps_total_p95",
        "steps_to_success_mean",
        "steps_to_success_p95",
        "wire_force_normal_trial_max_mean_N",
        "tip_force_normal_trial_max_mean_N",
    ]
    _write_rows_csv(analysis_root / f"{prefix}_candidate_aggregate.csv", candidate_rows, candidate_fields)
    _write_rows_csv(analysis_root / f"{prefix}_trials_flat.csv", trial_rows, sorted({key for row in trial_rows for key in row.keys()}))

    if candidate_rows:
        labels = [row["execution_wire"] for row in candidate_rows]
        success_rates = [row["success_rate"] if row["success_rate"] is not None else float("nan") for row in candidate_rows]
        steps_success_mean = [row["steps_to_success_mean"] if row["steps_to_success_mean"] is not None else float("nan") for row in candidate_rows]

        fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
        y = np.arange(len(labels))
        axes[0].barh(y, success_rates, color="#2a6fdb")
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels)
        axes[0].set_xlim(0, 1)
        axes[0].set_xlabel("success rate")
        axes[0].set_title(f"{title_prefix} candidate success rate")
        axes[0].invert_yaxis()

        axes[1].barh(y, steps_success_mean, color="#e07a2d")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels([])
        axes[1].set_xlabel("mean steps to success")
        axes[1].set_title("Mean steps needed for successful rollouts")
        axes[1].invert_yaxis()

        fig.savefig(analysis_root / f"{prefix}_candidate_summary.png", dpi=200, bbox_inches="tight")
        fig.savefig(analysis_root / f"{prefix}_candidate_summary.pdf", bbox_inches="tight")
        plt.close(fig)

    step_axis_max = int(max(current_max, max(all_steps, default=0), max(successful_steps, default=0)))
    if successful_steps:
        cutoffs = list(range(25, step_axis_max + 25, 25))
        completion_curve = [
            {
                "step_cutoff": cutoff,
                "share_of_successes_finished": sum(1 for step in successful_steps if step <= cutoff) / len(successful_steps),
            }
            for cutoff in cutoffs
        ]

        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        ax.plot(
            [row["step_cutoff"] for row in completion_curve],
            [row["share_of_successes_finished"] for row in completion_curve],
            color="#0f766e",
            linewidth=2.5,
        )
        ax.axvline(current_max, color="#6b7280", linestyle="--", linewidth=1.8, label=f"current max = {current_max}")
        if recommended_cutoff is not None:
            ax.axvline(recommended_cutoff, color="#dc2626", linestyle="--", linewidth=1.8, label=f"95th percentile = {recommended_cutoff}")
        ax.set_xlim(0, step_axis_max)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("step budget cutoff")
        ax.set_ylabel("share of successful trials completed")
        ax.set_title(f"{title_prefix}: how much of the success distribution is captured by a step budget?")
        ax.legend(loc="lower right")
        fig.savefig(analysis_root / f"{prefix}_step_budget_curve.png", dpi=200, bbox_inches="tight")
        fig.savefig(analysis_root / f"{prefix}_step_budget_curve.pdf", bbox_inches="tight")
        plt.close(fig)

        def _ecdf(values: list[float]):
            data = np.sort(np.asarray(values, dtype=float))
            y = np.arange(1, len(data) + 1) / len(data)
            return data, y

        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        all_x, all_y = _ecdf(all_steps)
        succ_x, succ_y = _ecdf(successful_steps)
        ax.step(all_x, all_y, where="post", color="#2563eb", linewidth=2, label="all trials: steps_total")
        ax.step(succ_x, succ_y, where="post", color="#16a34a", linewidth=2, label="successful trials: steps_to_success")
        ax.axvline(current_max, color="#6b7280", linestyle="--", linewidth=1.5)
        if recommended_cutoff is not None:
            ax.axvline(recommended_cutoff, color="#dc2626", linestyle="--", linewidth=1.5)
        ax.set_xlabel("steps")
        ax.set_ylabel("cumulative share")
        ax.set_title(f"{title_prefix}: episode-length distribution")
        ax.legend(loc="lower right")
        fig.savefig(analysis_root / f"{prefix}_episode_length_ecdf.png", dpi=200, bbox_inches="tight")
        fig.savefig(analysis_root / f"{prefix}_episode_length_ecdf.pdf", bbox_inches="tight")
        plt.close(fig)

        hist_max = min(1000, step_axis_max)
        hist_bins = np.arange(0.5, hist_max + 1.5, 1.0)
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        ax.hist(successful_steps, bins=hist_bins, color="#7c3aed", edgecolor="white", linewidth=0.5)
        ax.set_xlim(1, hist_max)
        ax.set_xlabel("steps to success")
        ax.set_ylabel("number of successful runs")
        ax.set_title(f"{title_prefix}: distribution of steps-to-success over successful trials")
        fig.savefig(analysis_root / f"{prefix}_success_steps_hist.png", dpi=200, bbox_inches="tight")
        fig.savefig(analysis_root / f"{prefix}_success_steps_hist.pdf", bbox_inches="tight")
        plt.close(fig)

    return {
        "candidate_rows": candidate_rows,
        "successful_steps": successful_steps,
        "all_steps": all_steps,
        "percentiles_rows": percentile_rows,
        "recommended_cutoff": recommended_cutoff,
        "current_max": current_max,
        "analysis_root": analysis_root,
        "step_axis_max": step_axis_max,
    }


def analyze_e1_e0_style_results_by_config(
    trial_rows: list[dict[str, Any]],
    *,
    analysis_root: Path,
    prefix: str,
    title_prefix: str,
    current_max_default: int = 1000,
) -> dict[int, dict[str, Any]]:
    """Run the E0-style step-budget analysis separately for each E1 config."""
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in trial_rows:
        config_id = row.get("config_id")
        if config_id is None:
            continue
        try:
            config_key = int(config_id)
        except Exception:
            continue
        buckets.setdefault(config_key, []).append(dict(row))

    analyses: dict[int, dict[str, Any]] = {}
    for config_id in sorted(buckets):
        analyses[config_id] = analyze_e1_e0_style_results(
            buckets[config_id],
            analysis_root=analysis_root,
            prefix=f"{prefix}_config_{config_id}",
            title_prefix=f"{title_prefix} config {config_id}",
            current_max_default=current_max_default,
        )
    return analyses


def build_e1_visual_selection_table(
    trial_rows: list[dict[str, Any]],
    *,
    max_rows_per_config: int | None = None,
    representative: str = "worst_score",
) -> list[dict[str, Any]]:
    """Build selectable rows where each row maps to one exact E1 trial."""
    def _as_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return float("nan")

    def _as_int_or_none(value: Any) -> int | None:
        try:
            if value is None or math.isnan(float(value)):
                return None
            return int(float(value))
        except Exception:
            return None

    rows: list[dict[str, Any]] = []
    for source in trial_rows:
        config_id = _as_int_or_none(source.get("config_id"))
        anatomy_id = source.get("anatomy_id")
        target_index = _as_int_or_none(source.get("target_index"))
        execution_wire = source.get("execution_wire")
        candidate_name = source.get("candidate_name")
        env_seed = _as_int_or_none(source.get("env_seed"))
        trial_index = _as_int_or_none(source.get("trial_index"))
        if (
            config_id is None
            or anatomy_id is None
            or target_index is None
            or execution_wire is None
            or candidate_name is None
            or env_seed is None
        ):
            continue

        rows.append(
            {
                "selection_id": len(rows),
                "config_id": config_id,
                "anatomy_id": str(anatomy_id),
                "target_index": target_index,
                "target_seed": _as_int_or_none(source.get("target_seed")),
                "execution_wire": str(execution_wire),
                "candidate_name": str(candidate_name),
                "trained_on_wire": source.get("trained_on_wire"),
                "trial_index": trial_index,
                "env_seed": env_seed,
                "policy_seed": _as_int_or_none(source.get("policy_seed")),
                "seed_base": _as_int_or_none(source.get("seed_base")),
                "max_episode_steps": _as_int_or_none(source.get("max_episode_steps")),
                "success": bool(source.get("success")),
                "valid_for_ranking": bool(source.get("valid_for_ranking")),
                "score_total": source.get("score_total"),
                "score_success": source.get("score_success"),
                "score_efficiency": source.get("score_efficiency"),
                "score_safety": source.get("score_safety"),
                "score_smoothness": source.get("score_smoothness"),
                "steps_total": source.get("steps_total"),
                "steps_to_success": source.get("steps_to_success"),
                "end_reason": source.get("end_reason"),
                "wire_force_normal_trial_max_N": source.get("wire_force_normal_trial_max_N"),
                "wire_force_normal_trial_mean_N": source.get("wire_force_normal_trial_mean_N"),
                "tip_force_normal_trial_max_N": source.get("tip_force_normal_trial_max_N"),
                "tip_force_normal_trial_mean_N": source.get("tip_force_normal_trial_mean_N"),
                "wire_force_magnitude_trial_max_N": source.get("wire_force_magnitude_trial_max_N"),
                "wall_time_s": source.get("wall_time_s"),
                # Backward-compatible aliases used by older notebook command cells.
                "rep_trial_index": trial_index,
                "rep_env_seed": env_seed,
                "rep_policy_seed": _as_int_or_none(source.get("policy_seed")),
                "rep_success": bool(source.get("success")),
                "rep_score_total": source.get("score_total"),
                "rep_steps_total": source.get("steps_total"),
                "rep_wire_force_normal_trial_max_N": source.get("wire_force_normal_trial_max_N"),
            }
        )

    def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        score = _as_float(row.get("score_total"))
        force = _as_float(row.get("wire_force_normal_trial_max_N"))
        steps = _as_float(row.get("steps_total"))
        if representative == "max_force":
            return (row["config_id"], -force, row["success"], score)
        if representative == "first_failure":
            return (row["config_id"], row["success"], row["anatomy_id"], row["target_index"], row["execution_wire"], row.get("trial_index") or -1)
        return (row["config_id"], row["success"], score, -force, -steps)

    rows = sorted(rows, key=_sort_key)
    if max_rows_per_config is not None:
        limited: list[dict[str, Any]] = []
        by_config: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_config.setdefault(int(row["config_id"]), []).append(row)
        for config_id in sorted(by_config):
            limited.extend(by_config[config_id][: int(max_rows_per_config)])
        rows = limited

    for idx, row in enumerate(rows):
        row["selection_id"] = idx
    return rows

def e1_visual_command_from_selection(
    selection_row: dict[str, Any],
    *,
    output_root: str | Path,
    friction: float,
    python_executable: str = "python3",
    target_branches: str = "bct,lcca,lsa",
    force_debug: bool = False,
) -> str:
    """Generate a one-trial visual eval_v2 CLI command from a selection table row."""
    import shlex

    config_id = int(selection_row["config_id"])
    policy_mode = "deterministic" if config_id in {1, 2} else "stochastic"
    stochastic_env_mode = "fixed_start" if config_id in {1, 3} else "random_start"
    env_seed_value = selection_row.get("env_seed", selection_row.get("rep_env_seed"))
    env_seed = int(float(env_seed_value))
    policy_seed = selection_row.get("policy_seed", selection_row.get("rep_policy_seed"))
    candidate_name = str(selection_row["candidate_name"])
    execution_wire = str(selection_row["execution_wire"])
    anatomy_id = str(selection_row["anatomy_id"])
    target_seed = int(selection_row["target_seed"])
    max_episode_steps = int(selection_row.get("max_episode_steps") or 1000)

    args = [
        python_executable,
        "-m",
        "steve_recommender.eval_v2.cli",
        "run",
        "--job-name",
        f"visual_e1_cfg{config_id}_{anatomy_id}_target{int(selection_row['target_index'])}_{execution_wire.replace('/', '_')}",
        "--scenario-name",
        f"{anatomy_id}__target_{int(selection_row['target_index'])}",
        "--anatomy",
        anatomy_id,
        "--execution-wire",
        execution_wire,
        "--candidate-name",
        candidate_name,
        "--target-mode",
        "centerline_random",
        "--target-branches",
        target_branches,
        "--target-seed",
        str(target_seed),
        "--threshold-mm",
        "5.0",
        "--trial-count",
        "1",
        "--env-seeds",
        str(env_seed),
        "--base-seed",
        str(env_seed),
        "--policy-mode",
        policy_mode,
        "--stochastic-env-mode",
        stochastic_env_mode,
        "--max-episode-steps",
        str(max_episode_steps),
        "--workers",
        "1",
        "--policy-device",
        "cpu",
        "--friction",
        str(float(friction)),
        "--output-root",
        str(output_root),
        "--visualize",
        "--visualize-trials-per-candidate",
        "1",
    ]
    try:
        if policy_mode == "stochastic" and policy_seed is not None and not math.isnan(float(policy_seed)):
            args.extend(["--policy-seeds", str(int(float(policy_seed))), "--policy-base-seed", str(int(float(policy_seed)))])
    except Exception:
        pass
    if force_debug:
        args.append("--write-diagnostics")
    return " \\\n  ".join(shlex.quote(str(arg)) for arg in args)
